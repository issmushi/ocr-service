from __future__ import annotations

import unittest
from dataclasses import replace
from io import BytesIO
from unittest.mock import patch

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import settings
from app.core.errors import AppError, api_error_response
from app.core.images.printed_content import crop_printed_content
from app.modules.v1.crop.router import router


def synthetic_scan() -> np.ndarray:
    image = np.full((1600, 1200, 3), 30, np.uint8)
    image[25:1575, 35:1165] = 240
    cv2.putText(
        image,
        "SYNTHETIC EXTRACT",
        (130, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (60, 60, 60),
        2,
    )
    cv2.rectangle(image, (125, 500), (1080, 960), (100, 100, 100), 2)
    cv2.line(image, (550, 500), (550, 960), (100, 100, 100), 2)
    for y in range(550, 960, 50):
        cv2.line(image, (125, y), (1080, y), (100, 100, 100), 2)
        cv2.putText(
            image,
            "Field       Example value",
            (145, y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (60, 60, 60),
            1,
        )
    for x in (125, 475):
        cv2.rectangle(image, (x, 1100), (x + 250, 1200), (60, 60, 60), 3)
        cv2.putText(
            image,
            "SIGNATURE",
            (x + 15, 1150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (60, 60, 60),
            2,
        )
    cv2.putText(
        image,
        "Page 1 of 4",
        (880, 1150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (60, 60, 60),
        1,
    )
    cv2.putText(
        image,
        "Updated 01.01.2000",
        (850, 1200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (60, 60, 60),
        1,
    )
    # Dust in the empty margin must not expand the crop.
    cv2.circle(image, (900, 1400), 2, (60, 60, 60), -1)
    return image


def png_bytes(image: np.ndarray) -> bytes:
    return cv2.imencode(".png", image)[1].tobytes()


class PrintedContentTests(unittest.TestCase):
    def test_preserves_all_printed_blocks_and_removes_margins(self):
        source = synthetic_scan()
        cropped = crop_printed_content(source)
        self.assertTrue(750 < cropped.shape[0] < 820)
        self.assertTrue(950 < cropped.shape[1] < 1020)
        # Header, table, both signatures and detached footer must survive.
        for x1, y1, x2, y2 in (
            (125, 435, 500, 470),
            (120, 495, 1085, 965),
            (120, 1095, 380, 1205),
            (470, 1095, 730, 1205),
            (845, 1130, 1070, 1205),
        ):
            similarity = cv2.matchTemplate(
                cropped, source[y1:y2, x1:x2], cv2.TM_CCOEFF_NORMED
            )
            self.assertGreater(float(similarity.max()), 0.999)

    def test_deskews_translated_scans_without_losing_footer(self):
        source = synthetic_scan()
        reference = crop_printed_content(source)
        for angle, dx, dy in ((3, -20, 80), (-4, 25, -90)):
            with self.subTest(angle=angle):
                matrix = cv2.getRotationMatrix2D((600, 800), angle, 1)
                matrix[:, 2] += (dx, dy)
                rotated = cv2.warpAffine(
                    source, matrix, (1200, 1600), borderValue=(30, 30, 30)
                )
                cropped = crop_printed_content(rotated)
                for actual, expected in zip(cropped.shape, reference.shape):
                    self.assertLessEqual(abs(actual - expected), 15)
                for x1, y1, x2, y2 in (
                    (125, 435, 500, 470),
                    (120, 1095, 380, 1205),
                    (845, 1130, 1070, 1205),
                ):
                    similarity = cv2.matchTemplate(
                        cropped,
                        source[y1:y2, x1:x2],
                        cv2.TM_CCOEFF_NORMED,
                    )
                    self.assertGreater(float(similarity.max()), 0.8)

    def test_scales_with_input_resolution(self):
        source = synthetic_scan()
        reference = crop_printed_content(source)
        for scale in (0.75, 2.0):
            with self.subTest(scale=scale):
                resized = cv2.resize(source, None, fx=scale, fy=scale)
                cropped = crop_printed_content(resized)
                for actual, expected in zip(
                    cropped.shape[:2], reference.shape[:2]
                ):
                    self.assertLess(abs(actual / scale - expected), 15)

    def test_blank_scan_and_scanner_frame_are_not_content(self):
        for color in (0, 200, 255):
            with self.subTest(color=color):
                blank = np.full((1600, 1200, 3), color, np.uint8)
                blank[:15] = 20
                blank[:, :30] = 20
                with self.assertRaises(ValueError):
                    crop_printed_content(blank)


class EgrnCropEndpointTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(router, prefix="/v1/ocr")

        @app.exception_handler(AppError)
        async def handle_error(_request: Request, error: AppError):
            return JSONResponse(
                status_code=error.status_code,
                content=api_error_response(error.code, error.message),
            )

        self.client = TestClient(app)

    def post(self, content: bytes, filename: str = "scan.png"):
        return self.client.post(
            "/v1/ocr/crop/egrn", files={"image": (filename, content)}
        )

    def test_returns_uncached_png_and_documents_binary_response(self):
        response = self.post(png_bytes(synthetic_scan()))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertEqual(response.headers["cache-control"], "no-store")
        with Image.open(BytesIO(response.content)) as image:
            self.assertEqual(image.format, "PNG")
            self.assertLess(image.height, 820)
        schema = self.client.get("/openapi.json").json()
        operation = schema["paths"]["/v1/ocr/crop/egrn"]["post"]
        self.assertEqual(operation["tags"], ["Crop Sync"])
        self.assertIn("image/png", operation["responses"]["200"]["content"])

    def test_invalid_blank_and_oversized_files_use_error_envelope(self):
        for content, name, status, code in (
            (b"", "scan.png", 400, "EMPTY_IMAGE"),
            (b"invalid", "scan.png", 400, "IMAGE_READ_ERROR"),
            (b"invalid", "scan.pdf", 400, "UNSUPPORTED_FORMAT"),
            (
                png_bytes(np.full((500, 500, 3), 255, np.uint8)),
                "scan.png",
                422,
                "OCR_FAILED",
            ),
        ):
            with self.subTest(code=code):
                response = self.post(content, name)
                self.assertEqual(response.status_code, status)
                self.assertFalse(response.json()["success"])
                self.assertIsNone(response.json()["data"])
                self.assertEqual(response.json()["error"]["code"], code)
        with patch(
            "app.modules.v1.crop.router.settings",
            replace(settings, ocr_max_image_size_mb=0),
        ):
            self.assertEqual(self.post(b"oversized").status_code, 413)
