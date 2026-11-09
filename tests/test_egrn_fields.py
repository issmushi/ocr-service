from __future__ import annotations

import base64
import unittest
from typing import get_args
from unittest.mock import AsyncMock, patch

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.core.errors import AppError, ErrorCode, api_error_response
from app.modules.v1.crop.egrn_layout import FIELD_LABELS, locate_egrn_fields
from app.modules.v1.crop.egrn_schemas import EgrnFieldName
from app.modules.v1.crop.router import router


def synthetic_extract(address_height: int = 70) -> tuple[np.ndarray, dict]:
    image = np.full((1400, 1600, 3), 245, np.uint8)
    color = (70, 70, 70)

    def text(value, x, y, size=0.6):
        cv2.putText(
            image, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1
        )

    text("SYNTHETIC EXTRACT", 220, 45, 0.9)
    text("Request No.123456", 1350, 45)
    for y in (100, 140, 180):
        cv2.line(image, (20, y), (1580, y), color, 2)
    for x in (20, 695, 1580):
        cv2.line(image, (x, 100), (x, 180), color, 2)
    text("Object type", 40, 127)
    text("Room", 710, 127)
    text("Cadastral number", 40, 167)
    text("00:00:0000000:999", 710, 167)
    y = 240
    rows = []
    for index in range(15):
        rows.append(y)
        cv2.line(image, (20, y), (1580, y), color, 2)
        text(f"Label {index}", 40, y + 26)
        if index == 3:
            text("Synthetic city, test street, building 123", 710, y + 26)
            text("Apartment 456, second address line", 710, y + 52)
        elif index == 7:
            text("Apartment", 710, y + 26)
        else:
            text(f"Other value {index}", 710, y + 26)
        y += address_height if index == 3 else 40
    for x in (20, 695, 1580):
        cv2.line(image, (x, 240), (x, y), color, 2)
    for x in (20, 450):
        cv2.rectangle(image, (x, 1000), (x + 310, 1160), color, 3)
        text("SIGNATURE", x + 25, 1080)
    text("Page 1 of 4", 1460, 1120, 0.55)
    text("Data current as of 01.01.2000", 1240, 1180)
    regions = {
        "cadastral_number": (700, 145, 1000, 177),
        "address": (700, rows[3] + 5, 1160, rows[4] - 5),
        "object_name": (700, rows[7] + 5, 840, rows[8] - 5),
    }
    return image, regions


def encode(image: np.ndarray) -> bytes:
    return cv2.imencode(".png", image)[1].tobytes()


class EgrnLayoutTests(unittest.TestCase):
    def test_each_table_value_stays_in_its_row_without_labels(self):
        image, _ = synthetic_extract()
        boxes = locate_egrn_fields(image)
        main_fields = [
            "cadastral_quarter_number",
            "cadastral_number_assignment_date",
            "previous_registration_number",
            "address",
            "floor",
            "area",
            "purpose",
            "object_name",
            "residential_premises_type",
            "cadastral_value",
            "parent_cadastral_numbers",
            "cadastral_engineer",
            "cultural_heritage",
            "claims",
            "specialized_housing",
        ]
        cells = [("object_type", 100, 140), ("cadastral_number", 140, 180)]
        top = 240
        for index, name in enumerate(main_fields):
            bottom = top + (70 if index == 3 else 40)
            cells.append((name, top, bottom))
            top = bottom
        for name, top, bottom in cells:
            with self.subTest(field=name):
                x1, y1, x2, y2 = boxes[name]
                self.assertGreater(x1, 695)
                self.assertLess(x2, 1580)
                self.assertGreater(y1, top)
                self.assertLess(y2, bottom)
        self.assertEqual(set(get_args(EgrnFieldName)), set(boxes))

    def test_empty_area_does_not_capture_label_or_neighbour(self):
        image, _ = synthetic_extract()
        # The area row follows the taller address and the floor row.
        image[471:509, 701:1575] = 245
        boxes = locate_egrn_fields(image)
        self.assertNotIn("area", boxes)
        self.assertIn("floor", boxes)
        self.assertIn("purpose", boxes)

    def test_tracks_real_rows_when_multiline_address_grows(self):
        for height in (70, 110):
            with self.subTest(address_height=height):
                image, expected = synthetic_extract(height)
                boxes = locate_egrn_fields(image)
                self.assertEqual(set(boxes), set(FIELD_LABELS))
                for name, (left, top, right, bottom) in expected.items():
                    x1, y1, x2, y2 = boxes[name]
                    self.assertTrue(left <= x1 < x2 <= right, (name, boxes))
                    # PNG crops include two extra analysis pixels of padding.
                    self.assertTrue(
                        top - 2 <= y1 < y2 <= bottom + 2, (name, boxes)
                    )
                address = boxes["address"]
                self.assertGreater(address[3] - address[1], 40)
                self.assertLess(boxes["application_number"][3], 100)
                self.assertGreater(boxes["actuality_date"][1], 1150)

    def test_missing_text_is_not_replaced_by_neighbouring_field(self):
        image, regions = synthetic_extract()
        x1, y1, x2, y2 = regions["object_name"]
        image[y1:y2, x1:x2] = 245
        image[1140:1210, 1100:] = 245  # Date absent; page number remains.
        boxes = locate_egrn_fields(image)
        self.assertNotIn("object_name", boxes)
        self.assertNotIn("actuality_date", boxes)
        self.assertIn("address", boxes)

    def test_incomplete_grid_does_not_shift_row_meanings(self):
        for closed_bottom in (False, True):
            with self.subTest(closed_bottom=closed_bottom):
                image, _ = synthetic_extract()
                if closed_bottom:
                    cv2.line(image, (20, 870), (1580, 870), (70, 70, 70), 2)
                self.assertEqual(
                    set(locate_egrn_fields(image)), set(FIELD_LABELS)
                )
                image[278:283, 15:1585] = 245  # Erase an interior rule.
                self.assertEqual(locate_egrn_fields(image), {})

    def test_scale_preserves_semantic_regions(self):
        image, _ = synthetic_extract()
        original = locate_egrn_fields(image)
        for scale in (0.75, 1.5):
            resized = cv2.resize(image, None, fx=scale, fy=scale)
            boxes = locate_egrn_fields(resized)
            self.assertEqual(set(boxes), set(original))
            for name, box in boxes.items():
                self.assertTrue(
                    all(
                        abs(a / scale - b) < 8
                        for a, b in zip(box, original[name])
                    )
                )

    def test_blurred_downsampled_scan_keeps_all_regions(self):
        image, _ = synthetic_extract()
        blurred = cv2.GaussianBlur(image, (3, 3), 0.6)
        small = cv2.resize(
            blurred, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA
        )
        self.assertEqual(set(locate_egrn_fields(small)), set(FIELD_LABELS))


class EgrnFieldsRouteTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(router, prefix="/v1/ocr")

        @app.exception_handler(AppError)
        async def error_handler(_request: Request, error: AppError):
            return JSONResponse(
                status_code=error.status_code,
                content=api_error_response(error.code, error.message),
            )

        self.client = TestClient(app)

    def post(self, image):
        return self.client.post(
            "/v1/ocr/crop/egrn-fields",
            files={"image": ("synthetic.png", encode(image), "image/png")},
        )

    def test_actual_png_pixels_match_boxes_after_crop_and_deskew(self):
        source, _ = synthetic_extract()
        for angle in (0, 2):
            with self.subTest(angle=angle):
                matrix = cv2.getRotationMatrix2D((800, 700), angle, 1)
                matrix[:, 2] += (100, 240)
                scan = cv2.warpAffine(
                    source,
                    matrix,
                    (1800, 1900),
                    borderValue=(245, 245, 245),
                )
                response = self.post(scan)
                self.assertEqual(
                    response.status_code, 200, response.text[:200]
                )
                self.assertEqual(response.headers["cache-control"], "no-store")
                data = response.json()["data"]
                self.assertEqual(data["missing_fields"], [])
                self.assertEqual(len(data["fields"]), 19)
                self.assertIsNone(data["meta"]["ocr"])
                self.assertTrue(
                    all(field["ocr"] is None for field in data["fields"])
                )

                def decode(preview):
                    self.assertTrue(
                        preview["image"].startswith("data:image/png;base64,")
                    )
                    content = base64.b64decode(preview["image"].split(",")[1])
                    return cv2.imdecode(np.frombuffer(content, np.uint8), 1)

                document = decode(data["document"])
                for field in data["fields"]:
                    x1, y1, x2, y2 = field["bbox"]
                    np.testing.assert_array_equal(
                        decode(field), document[y1:y2, x1:x2]
                    )
                    self.assertEqual(field["width"], x2 - x1)
                    self.assertEqual(field["height"], y2 - y1)

    def test_both_models_read_exact_pngs_and_isolate_field_failures(self):
        for model, operation in (
            ("paddle", "field_text"),
            ("paddle-cyrillic", "field_text_cyrillic"),
        ):
            with (
                self.subTest(model=model),
                patch(
                    "app.modules.v1.crop.egrn.ensure_ocr_model_enabled"
                ) as enabled,
                patch(
                    "app.modules.v1.crop.field_ocr.paddle_runtime.run",
                    new_callable=AsyncMock,
                ) as read,
            ):
                ok = {
                    "text": "ТЕСТ\nСТРОКА",
                    "confidence": 0.9,
                    "device": "cpu",
                }
                read.side_effect = [
                    ok,
                    AppError(ErrorCode.OCR_FAILED, "Ошибка поля"),
                    *([ok] * 17),
                ]
                source, _ = synthetic_extract()
                response = self.client.post(
                    "/v1/ocr/crop/egrn-fields",
                    files={
                        "image": ("synthetic.png", encode(source), "image/png")
                    },
                    data={"ocr_model": model},
                )
                self.assertEqual(response.status_code, 200)
                enabled.assert_called_once_with("paddle")
                data = response.json()["data"]
                self.assertEqual(read.await_count, 19)
                for field, call in zip(data["fields"], read.await_args_list):
                    self.assertEqual(
                        call.args,
                        (
                            operation,
                            base64.b64decode(field["image"].split(",")[1]),
                        ),
                    )
                    self.assertTrue(call.kwargs["split_lines"])
                    self.assertEqual(field["ocr"]["model"], model)
                self.assertEqual(data["fields"][1]["ocr"]["status"], "error")
                self.assertEqual(
                    data["fields"][2]["ocr"]["text"], "ТЕСТ\nСТРОКА"
                )
                self.assertEqual(data["meta"]["ocr"]["failed_fields"], 1)

    def test_disabled_ocr_and_invalid_model(self):
        source, _ = synthetic_extract()
        with patch(
            "app.modules.v1.crop.egrn.ensure_ocr_model_enabled"
        ) as enabled:
            enabled.side_effect = AppError(
                ErrorCode.OCR_ENGINE_UNAVAILABLE, "Отключено"
            )
            response = self.client.post(
                "/v1/ocr/crop/egrn-fields",
                files={
                    "image": ("synthetic.png", encode(source), "image/png")
                },
                data={"ocr_model": "paddle-cyrillic"},
            )
            self.assertEqual(response.status_code, 503)
            self.assertEqual(
                response.json()["error"]["code"], "OCR_ENGINE_UNAVAILABLE"
            )
            # Crop-only mode must still work without a Paddle installation.
            enabled.reset_mock()
            self.assertEqual(self.post(source).status_code, 200)
            enabled.assert_not_called()
        response = self.client.post(
            "/v1/ocr/crop/egrn-fields",
            files={"image": ("synthetic.png", encode(source), "image/png")},
            data={"ocr_model": "qwen3-vl-8b"},
        )
        self.assertEqual(response.status_code, 422)

    def test_unsupported_layout_and_unreadable_image(self):
        response = self.post(np.full((600, 800, 3), 255, np.uint8))
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "OCR_FAILED")
        response = self.client.post(
            "/v1/ocr/crop/egrn-fields",
            files={"image": ("invalid.png", b"invalid", "image/png")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "IMAGE_READ_ERROR")
