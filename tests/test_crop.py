from __future__ import annotations

import base64
import math
import unittest
from dataclasses import replace
from io import BytesIO
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.core.config import settings
from app.core.errors import AppError, ErrorCode, api_error_response
from app.core.images.orientation import rotate_document
from app.modules.v1.crop.router import router
from app.modules.v1.crop.service import (
    crop_document_bytes,
    crop_document_image_bytes,
    crop_mrz_bytes,
)
from app.ocr_engine.contracts import DocumentOrientation
from tests.test_mrz_detection import document_with_rows

DOCUMENT_CORNERS = [(150, 90), (680, 160), (620, 500), (90, 440)]
SERVICE = "app.modules.v1.crop.service"


def document_image() -> Image.Image:
    image = Image.new("RGB", (800, 600), (15, 20, 28))
    ImageDraw.Draw(image).polygon(DOCUMENT_CORNERS, fill=(240, 225, 180))
    return image


def encode_image(
    image: Image.Image, *, orientation: int | None = None
) -> bytes:
    output = BytesIO()
    if orientation is None:
        image.save(output, format="PNG")
    else:
        exif = Image.Exif()
        exif[274] = orientation
        image.save(output, format="JPEG", quality=98, exif=exif)
    return output.getvalue()


class CropAssertions:
    def decoded_preview(self, value: str) -> Image.Image:
        self.assertTrue(value.startswith("data:image/jpeg;base64,"))
        content = base64.b64decode(value.split(",", 1)[1], validate=True)
        with Image.open(BytesIO(content)) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertEqual(image.mode, "RGB")
            image.load()
            return image.copy()

    def assert_preview_contract(
        self, result: dict[str, Any]
    ) -> tuple[Image.Image, Image.Image]:
        self.assertIsInstance(result["document_detected"], bool)
        self.assertIsInstance(result["duration_seconds"], (int, float))
        self.assertTrue(math.isfinite(result["duration_seconds"]))
        self.assertGreaterEqual(result["duration_seconds"], 0)
        original = self.decoded_preview(result["original_image"])
        cropped = self.decoded_preview(result["cropped_image"])
        self.assertEqual(
            original.size,
            (result["original_width"], result["original_height"]),
        )
        self.assertEqual(
            cropped.size,
            (result["cropped_width"], result["cropped_height"]),
        )
        self.assert_mrz_contract(result)
        self.assertIsInstance(result["orientation_detected"], bool)
        self.assertIn(result["document_rotation"], (0, 90, 180, 270))
        confidence = result["orientation_confidence"]
        if confidence is not None:
            self.assertTrue(math.isfinite(confidence))
            self.assertTrue(0 <= confidence <= 1)
        if not result["orientation_detected"]:
            self.assertEqual(result["document_rotation"], 0)
        return original, cropped

    def assert_mrz_contract(self, result: dict[str, Any]) -> None:
        self.assertIsInstance(result["mrz_detected"], bool)
        if not result["mrz_detected"]:
            for key in ("mrz_image", "mrz_bbox", "mrz_width", "mrz_height"):
                self.assertIsNone(result[key], key)
            return

        bbox = result["mrz_bbox"]
        self.assertIsInstance(bbox, list)
        self.assertEqual(len(bbox), 4)
        for coordinate in bbox:
            self.assertIs(type(coordinate), int)
        x1, y1, x2, y2 = bbox
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLess(x1, x2)
        self.assertLess(y1, y2)
        self.assertLessEqual(x2, result["cropped_width"])
        self.assertLessEqual(y2, result["cropped_height"])
        self.assertIs(type(result["mrz_width"]), int)
        self.assertIs(type(result["mrz_height"]), int)
        self.assertEqual(result["mrz_width"], x2 - x1)
        self.assertEqual(result["mrz_height"], y2 - y1)
        mrz = self.decoded_preview(result["mrz_image"])
        self.assertEqual(mrz.size, (x2 - x1, y2 - y1))

    def assert_mrz_pixels(
        self, result: dict[str, Any], warped_bgr: np.ndarray
    ) -> None:
        x1, y1, x2, y2 = result["mrz_bbox"]
        expected = warped_bgr[y1:y2, x1:x2, ::-1].astype(float)
        actual = np.asarray(self.decoded_preview(result["mrz_image"])).astype(
            float
        )
        self.assertEqual(actual.shape, expected.shape)
        # JPEG is lossy; compare against the selected pre-encoding RGB pixels.
        self.assertLess(float(np.abs(actual - expected).mean()), 3.0)

    def assert_detected_corners(
        self, result: dict[str, Any], expected: list[tuple[int, int]]
    ) -> None:
        self.assertIs(result["document_detected"], True)
        quad = np.asarray(result["document_quad"], dtype=float)
        self.assertEqual(quad.shape, (4, 2))
        self.assertTrue(np.isfinite(quad).all())
        self.assertTrue((quad >= 0).all())
        self.assertTrue((quad[:, 0] < result["original_width"]).all())
        self.assertTrue((quad[:, 1] < result["original_height"]).all())
        # Match the four known physical corners without imposing a list order.
        distances = np.linalg.norm(
            quad[:, None, :] - np.asarray(expected)[None, :, :], axis=2
        )
        self.assertLess(float(distances.min(axis=0).max()), 10)
        self.assertLess(float(distances.min(axis=1).max()), 10)


class CropDocumentTests(CropAssertions, unittest.TestCase):
    def test_refinement_preserves_known_mrz_when_redetection_loses_end_glyphs(
        self,
    ) -> None:
        from app.core.images.mrz_detection import find_mrz_bbox
        from app.modules.v1.crop.service import (
            _PreparedDocument,
            _refine_document,
        )

        upright, _ = document_with_rows()
        source_bbox = find_mrz_bbox(upright)
        selected = np.array(
            [[20, 100], [980, 100], [980, 620], [20, 620]], np.float32
        )
        for rotation in (0, 90, 180, 270):
            for redetected in (None, (100, 385, 800, 450)):
                with self.subTest(rotation=rotation, redetected=redetected):
                    original = rotate_document(upright, (-rotation) % 360)
                    prepared = _PreparedDocument(original, original, None, 0)
                    with (
                        patch(
                            f"{SERVICE}.find_document_quad",
                            return_value=selected,
                        ),
                        patch(
                            f"{SERVICE}.find_mrz_bbox", return_value=redetected
                        ),
                    ):
                        refined, quad, bbox = _refine_document(
                            prepared, rotation, upright, source_bbox
                        )
                    self.assertEqual(refined.shape[:2], (520, 960))
                    self.assertIsNotNone(quad)
                    self.assertIsNotNone(bbox)
                    # All dark pixels in this generated page belong to MRZ.
                    # Verify image content rather than duplicating the
                    # coordinate formula.
                    y, x = np.where(refined.min(axis=2) < 70)
                    self.assertGreater(int(x.max() - x.min()), 800)
                    self.assertGreater(int(y.min()), 350)
                    self.assertLessEqual(bbox[0], x.min())
                    self.assertLessEqual(bbox[1], y.min())
                    self.assertGreater(bbox[2], x.max())
                    self.assertGreater(bbox[3], y.max())
                    self.assertGreaterEqual(bbox[0], 0)
                    self.assertGreaterEqual(bbox[1], 0)
                    self.assertLessEqual(bbox[2], 960)
                    self.assertLessEqual(bbox[3], 520)

    def test_full_frame_passport_and_id_find_complete_mrz_in_all_orientations(
        self,
    ) -> None:
        for rows in (2, 3):
            upright, ink = document_with_rows(rows)
            for angle in (0, 90, 180, 270):
                with self.subTest(rows=rows, angle=angle):
                    source = rotate_document(upright, angle)
                    predictor = Mock(
                        return_value=DocumentOrientation((-angle) % 360, 0.95)
                    )
                    result = crop_document_bytes(
                        encode_image(Image.fromarray(source[:, :, ::-1])),
                        orientation_predictor=predictor,
                    )
                    self.assert_preview_contract(result)
                    self.assertFalse(result["document_detected"])
                    self.assertIsNone(result["document_quad"])
                    np.testing.assert_array_equal(
                        predictor.call_args.args[0], source
                    )
                    self.assertEqual(
                        (result["cropped_width"], result["cropped_height"]),
                        (1000, 640),
                    )
                    self.assertEqual(
                        result["document_rotation"], (-angle) % 360
                    )
                    x1, y1, x2, y2 = result["mrz_bbox"]
                    self.assertLessEqual(x1, ink[0])
                    self.assertLessEqual(y1, ink[1])
                    self.assertGreaterEqual(x2, ink[2])
                    self.assertGreaterEqual(y2, ink[3])
                    self.assert_mrz_pixels(result, upright)

    def test_internal_borders_cannot_remove_mrz_rows_or_line_beginnings(
        self,
    ) -> None:
        from app.core.images.preprocessing import find_document_quad

        for rows, bad_quad in (
            (3, [[0, 0], [999, 0], [999, 400], [0, 400]]),
            (2, [[340, 0], [999, 0], [999, 639], [340, 639]]),
        ):
            with self.subTest(rows=rows):
                image, ink = document_with_rows(rows)

                def candidates(source, **constraints):
                    # Reproduce a printed internal rectangle winning the first
                    # pass; the MRZ-constrained second pass uses the real
                    # detector.
                    if not constraints:
                        return np.asarray(bad_quad, dtype=np.float32)
                    return find_document_quad(source, **constraints)

                with patch(
                    f"{SERVICE}.find_document_quad", side_effect=candidates
                ):
                    result = crop_document_bytes(
                        encode_image(Image.fromarray(image[:, :, ::-1]))
                    )
                self.assert_preview_contract(result)
                self.assertFalse(result["document_detected"])
                self.assertEqual(
                    (result["cropped_width"], result["cropped_height"]),
                    (1000, 640),
                )
                x1, y1, x2, y2 = result["mrz_bbox"]
                self.assertLessEqual(x1, ink[0])
                self.assertLessEqual(y1, ink[1])
                self.assertGreaterEqual(x2, ink[2])
                self.assertGreaterEqual(y2, ink[3])
                self.assert_mrz_pixels(result, image)

    def test_orientation_is_corrected_before_mrz_in_all_four_positions(
        self,
    ) -> None:
        upright = np.zeros((160, 240, 3), dtype=np.uint8)
        upright[:] = (30, 80, 140)
        upright[:65, :100] = (190, 50, 20)
        upright[105:, 90:] = (10, 210, 180)
        quad = np.asarray(DOCUMENT_CORNERS, dtype=np.float32)
        for input_angle in (0, 90, 180, 270):
            with self.subTest(input_angle=input_angle):
                warped = rotate_document(upright, input_angle)
                correction = (360 - input_angle) % 360
                predictor = Mock(
                    return_value=DocumentOrientation(correction, 0.93)
                )

                def locate_mrz(oriented: np.ndarray):
                    np.testing.assert_array_equal(oriented, upright)
                    return (10, 110, 230, 150)

                with (
                    patch(f"{SERVICE}.find_document_quad", return_value=quad),
                    patch(f"{SERVICE}.warp_document", return_value=warped),
                    patch(
                        f"{SERVICE}.find_mrz_bbox", side_effect=locate_mrz
                    ) as locate,
                    patch(
                        f"{SERVICE}._refine_document",
                        side_effect=lambda prepared, rotation, crop, bbox: (
                            crop,
                            prepared.quad,
                            bbox,
                        ),
                    ),
                ):
                    result = crop_document_bytes(
                        encode_image(document_image()),
                        orientation_predictor=predictor,
                    )
                predictor.assert_called_once()
                self.assertIs(predictor.call_args.args[0], warped)
                locate.assert_called_once()
                _, cropped = self.assert_preview_contract(result)
                self.assertEqual(cropped.size, (240, 160))
                self.assertTrue(result["orientation_detected"])
                self.assertEqual(result["document_rotation"], correction)
                self.assertEqual(result["orientation_confidence"], 0.93)
                self.assertEqual(
                    result["document_quad"], quad.astype(float).tolist()
                )
                self.assert_mrz_pixels(result, upright)

    def test_uncertain_orientation_keeps_crop_and_still_searches_mrz(
        self,
    ) -> None:
        predictor = Mock(return_value=DocumentOrientation(90, 0.50))
        with patch(f"{SERVICE}.find_mrz_bbox", return_value=None) as locate:
            result = crop_document_bytes(
                encode_image(document_image()),
                orientation_predictor=predictor,
            )
        self.assert_preview_contract(result)
        self.assertFalse(result["orientation_detected"])
        self.assertEqual(result["document_rotation"], 0)
        self.assertEqual(result["orientation_confidence"], 0.50)
        self.assertIs(
            locate.call_args_list[0].args[0], predictor.call_args.args[0]
        )

    def test_orientation_is_skipped_for_uniform_image(self) -> None:
        predictor = Mock()
        result = crop_document_bytes(
            encode_image(Image.new("RGB", (480, 320), "gray")),
            orientation_predictor=predictor,
        )
        predictor.assert_not_called()
        self.assert_preview_contract(result)

    def test_removes_background_and_rectifies_a_slanted_document(self) -> None:
        result = crop_document_bytes(encode_image(document_image()))

        original, cropped = self.assert_preview_contract(result)
        self.assertEqual(original.size, (800, 600))
        self.assert_detected_corners(result, DOCUMENT_CORNERS)
        self.assertTrue(510 <= cropped.width <= 560, cropped.size)
        self.assertTrue(330 <= cropped.height <= 380, cropped.size)
        self.assertLess(cropped.width * cropped.height, 0.5 * 800 * 600)

        original_pixels = np.asarray(original)
        cropped_pixels = np.asarray(cropped)
        self.assertLess(
            float((original_pixels.mean(axis=2) > 160).mean()), 0.5
        )
        self.assertGreater(
            float((cropped_pixels.mean(axis=2) > 160).mean()), 0.95
        )
        # The warm document color also catches accidental RGB/BGR swaps.
        center = cropped_pixels[
            cropped.height // 2, cropped.width // 2
        ].astype(int)
        self.assertGreater(int(center[0] - center[2]), 40)

    def test_uniform_image_is_returned_without_a_document_crop(self) -> None:
        result = crop_document_bytes(
            encode_image(Image.new("RGB", (480, 320), (90, 130, 170)))
        )

        original, cropped = self.assert_preview_contract(result)
        self.assertIs(result["document_detected"], False)
        self.assertIsNone(result["document_quad"])
        self.assertEqual(original.size, (480, 320))
        self.assertEqual(cropped.size, original.size)
        np.testing.assert_array_equal(
            np.asarray(cropped), np.asarray(original)
        )

    def test_exif_rotation_updates_previews_dimensions_and_coordinates(
        self,
    ) -> None:
        def bottom_band(warped: np.ndarray) -> tuple[int, int, int, int]:
            height, width = warped.shape[:2]
            return (12, height - 80, width - 15, height - 6)

        with (
            patch(
                f"{SERVICE}.find_mrz_bbox", side_effect=bottom_band
            ) as find_mrz,
            patch(
                f"{SERVICE}._refine_document",
                side_effect=lambda prepared, rotation, crop, bbox: (
                    crop,
                    prepared.quad,
                    bbox,
                ),
            ),
        ):
            result = crop_document_bytes(
                encode_image(document_image(), orientation=6)
            )

        original, cropped = self.assert_preview_contract(result)
        self.assertEqual(original.size, (600, 800))
        # EXIF 6 rotates the stored 800x600 pixels clockwise for display.
        rotated_corners = [(599 - y, x) for x, y in DOCUMENT_CORNERS]
        self.assert_detected_corners(result, rotated_corners)
        self.assertTrue(330 <= cropped.width <= 380, cropped.size)
        self.assertTrue(510 <= cropped.height <= 560, cropped.size)
        self.assertIn(original.getexif().get(274), (None, 1))
        find_mrz.assert_called_once()
        warped = find_mrz.call_args.args[0]
        self.assertEqual(warped.shape[:2], (cropped.height, cropped.width))
        self.assertNotEqual(
            warped.shape[:2], (original.height, original.width)
        )
        self.assertEqual(
            result["mrz_bbox"],
            [12, cropped.height - 80, cropped.width - 15, cropped.height - 6],
        )
        self.assert_mrz_pixels(result, warped)


class CropMrzBytesTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_decoded_mrz_jpeg_from_shared_crop_result(
        self,
    ) -> None:
        expected = b"jpeg-mrz-bytes"
        result = {
            "mrz_detected": True,
            "mrz_image": (
                "data:image/jpeg;base64,"
                + base64.b64encode(expected).decode("ascii")
            ),
        }
        with patch(
            f"{SERVICE}.crop_document",
            new=AsyncMock(return_value=result),
        ) as crop_document:
            actual = await crop_mrz_bytes(b"full-page")

        self.assertEqual(actual, expected)
        crop_document.assert_awaited_once_with(b"full-page")

    async def test_missing_mrz_is_reported_as_ocr_failure(self) -> None:
        with patch(
            f"{SERVICE}.crop_document",
            new=AsyncMock(
                return_value={"mrz_detected": False, "mrz_image": None}
            ),
        ):
            with self.assertRaises(AppError) as raised:
                await crop_mrz_bytes(b"full-page")

        self.assertEqual(raised.exception.code, ErrorCode.OCR_FAILED)
        self.assertEqual(
            raised.exception.message, "Не удалось найти MRZ на изображении"
        )

    async def test_invalid_mrz_preview_is_reported_as_internal_error(
        self,
    ) -> None:
        with patch(
            f"{SERVICE}.crop_document",
            new=AsyncMock(
                return_value={
                    "mrz_detected": True,
                    "mrz_image": "data:image/jpeg;base64,not-base64!",
                }
            ),
        ):
            with self.assertRaises(AppError) as raised:
                await crop_mrz_bytes(b"full-page")

        self.assertEqual(raised.exception.code, ErrorCode.INTERNAL_ERROR)


class CropDocumentImageBytesTests(unittest.IsolatedAsyncioTestCase):
    async def test_crops_and_orients_document_without_searching_for_mrz(
        self,
    ) -> None:
        content = encode_image(document_image())
        with (
            patch(
                f"{SERVICE}.paddle_runtime.run",
                new=AsyncMock(
                    return_value={
                        "rotation_degrees": 0,
                        "confidence": 0.95,
                    }
                ),
            ) as orientation,
            patch(f"{SERVICE}.find_mrz_bbox") as find_mrz,
        ):
            result = await crop_document_image_bytes(content)

        find_mrz.assert_not_called()
        orientation.assert_awaited_once()
        with Image.open(BytesIO(result)) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertLess(image.width * image.height, 800 * 600)


class CropRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        route_app = FastAPI()
        route_app.include_router(router, prefix="/v1/ocr")

        @route_app.exception_handler(AppError)
        async def handle_app_error(
            _request: Request, error: AppError
        ) -> JSONResponse:
            return JSONResponse(
                status_code=error.status_code,
                content=api_error_response(error.code, error.message),
            )

        self.client = TestClient(route_app)
        self.addCleanup(self.client.close)

    def upload(
        self,
        endpoint: str,
        content: bytes | None = None,
        filename: str = "document.png",
    ):
        return self.client.post(
            endpoint,
            files={
                "image": (
                    filename,
                    content
                    if content is not None
                    else encode_image(document_image()),
                    "image/png",
                ),
            },
        )

    def test_document_returns_jpeg_from_document_only_pipeline(self) -> None:
        expected = b"document-jpeg"
        content = encode_image(document_image())
        with patch(
            "app.modules.v1.crop.router.crop_document_image_bytes",
            new=AsyncMock(return_value=expected),
        ) as operation:
            response = self.upload("/v1/ocr/crop/document", content)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, expected)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertEqual(response.headers["cache-control"], "no-store")
        operation.assert_awaited_once_with(content)

    def test_mrz_returns_jpeg_from_mrz_pipeline(self) -> None:
        expected = b"mrz-jpeg"
        content = encode_image(document_image())
        with patch(
            "app.modules.v1.crop.router.crop_mrz_bytes",
            new=AsyncMock(return_value=expected),
        ) as operation:
            response = self.upload("/v1/ocr/crop/mrz", content)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, expected)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertEqual(response.headers["cache-control"], "no-store")
        operation.assert_awaited_once_with(content)

    def test_only_two_crop_routes_are_exposed(self) -> None:
        paths = set(self.client.app.openapi()["paths"])
        self.assertEqual(
            paths,
            {"/v1/ocr/crop/document", "/v1/ocr/crop/mrz"},
        )
        for path in paths:
            self.assertEqual(
                self.client.app.openapi()["paths"][path]["post"]["tags"],
                ["Crop"],
            )

    def test_rejects_unsupported_extension(self) -> None:
        response = self.upload(
            "/v1/ocr/crop/document",
            filename="document.txt",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["error"]["code"], "UNSUPPORTED_FORMAT"
        )

    def test_rejects_upload_over_configured_limit(self) -> None:
        with patch(
            "app.modules.v1.crop.router.settings",
            new=replace(settings, ocr_max_image_size_mb=1),
        ):
            response = self.upload(
                "/v1/ocr/crop/mrz",
                content=b"x" * (1024 * 1024 + 1),
            )

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error"]["code"], "IMAGE_TOO_LARGE")


if __name__ == "__main__":
    unittest.main()
