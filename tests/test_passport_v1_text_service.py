from __future__ import annotations

import unittest
from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import settings
from app.core.errors import AppError, ErrorCode
from app.main import app

PATH = "/v1/ocr/passport/page"
FILES = {"image": ("passport.jpg", b"fake-image", "image/jpeg")}
SERVICE = "app.modules.v1.passport.service"
MODELS = (
    ("russian-ocr", "russian_runtime.run"),
    ("qwen3-vl-4b", "recognize_vlm_image"),
    ("qwen3-vl-8b", "recognize_vlm_image"),
)
encoded = BytesIO()
Image.new("RGB", (120, 60), "white").save(encoded, format="JPEG")
RUSSIAN_IMAGE_BYTES = encoded.getvalue()
RUSSIAN_FILES = {
    "image": ("passport.jpg", RUSSIAN_IMAGE_BYTES, "image/jpeg"),
}
EMPTY_PASSPORT_DATA = {
    "mrz": None,
    "normalization": None,
    "passport_type": "internal",
    "document_code": "PN",
    "issuing_country": "RUS",
    "surname": None,
    "name": None,
    "patronymic": None,
    "sex": None,
    "date_of_birth": None,
    "passport_series": None,
    "passport_number": None,
    "nationality": "RUS",
    "issue_date": None,
    "expiry_date": None,
    "department_code": None,
    "personal_number": None,
    "mrz_valid": None,
}


def text_data(meta, **passport):
    return {**EMPTY_PASSPORT_DATA, **passport, "meta": meta}


def page3_form(**values: str) -> dict[str, str]:
    return {"page": "3", **values}


def files_for_model(model: str) -> dict[str, tuple[str, bytes, str]]:
    return RUSSIAN_FILES if model == "russian-ocr" else FILES


class PassportV1PageServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        crop_patch = patch(
            f"{SERVICE}.crop_document_image_bytes",
            new=AsyncMock(side_effect=lambda content: content),
        )
        self.crop_document = crop_patch.start()
        self.addCleanup(crop_patch.stop)

    def test_response_does_not_expose_raw_engine_items(self) -> None:
        items = [
            {
                "text": "  РеСПУБЛИКА, Город № 123-45 / ТЕСТ!  ",
                "confidence": 0.97,
                "box": [
                    [12.5, 20.0],
                    [110.0, 21.0],
                    [109.0, 44.0],
                    [12.0, 43.0],
                ],
                "detection_confidence": 0.91,
            },
            {
                "text": "P<4RMTEST<<NAME<",
                "confidence": 0.81,
                "box": [
                    [9.0, 80.0],
                    [190.0, 80.0],
                    [190.0, 97.0],
                    [9.0, 97.0],
                ],
                "detection_confidence": None,
            },
            {
                "text": '{"surname":"ТЕСТОВ"}',
                "confidence": None,
                "box": None,
                "detection_confidence": None,
            },
        ]
        result = {
            "items": items,
            "image_width": 240,
            "image_height": 120,
            "diagnostics": {"internal": "not-public"},
        }
        with (
            patch(
                f"{SERVICE}.recognize_vlm_image",
                new=AsyncMock(return_value=result),
            ) as run,
            patch(
                "app.modules.paddle.runtime.paddle_runtime.run",
                new_callable=AsyncMock,
            ) as paddle_run,
            patch(
                f"{SERVICE}.mrz_scanner_runtime.run", new_callable=AsyncMock
            ) as mrz_run,
            patch(f"{SERVICE}.perf_counter", side_effect=[10.0, 10.25]),
        ):
            response = self.client.post(
                PATH,
                files=FILES,
                data=page3_form(model="qwen3-vl-4b"),
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "success": True,
                "data": text_data(
                    {
                        "engine": "qwen3-vl",
                        "model": "qwen3-vl-4b",
                        "duration_seconds": 0.25,
                    },
                    surname="ТЕСТОВ",
                ),
                "error": None,
            },
        )
        run.assert_awaited_once_with(
            b"fake-image",
            model="qwen3-vl-4b",
            filename="passport-cropped.jpg",
            content_type="image/jpeg",
            task="passport_fields",
        )
        self.crop_document.assert_awaited_once_with(b"fake-image")
        mrz_run.assert_not_awaited()
        paddle_run.assert_not_awaited()

    def test_explicit_qwen4_returns_structured_contract(self) -> None:
        item = {
            "text": '{"surname":"ТЕСТОВ"}',
            "confidence": None,
            "box": [[1, 2], [100, 2], [100, 22], [1, 22]],
            "detection_confidence": None,
        }
        result = {"items": [item], "image_width": 120, "image_height": 60}
        for form in ({"model": "qwen3-vl-4b"},):
            with (
                self.subTest(form=form),
                patch(
                    f"{SERVICE}.recognize_vlm_image",
                    new=AsyncMock(return_value=result),
                ) as vlm_run,
                patch(
                    "app.modules.paddle.runtime.paddle_runtime.run",
                    new_callable=AsyncMock,
                ) as paddle_run,
                patch(
                    f"{SERVICE}.mrz_scanner_runtime.run",
                    new_callable=AsyncMock,
                ) as mrz_run,
                patch(f"{SERVICE}.perf_counter", side_effect=[5.0, 5.75]),
            ):
                response = self.client.post(
                    PATH,
                    files=FILES,
                    data=page3_form(**form),
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json(),
                {
                    "success": True,
                    "data": text_data(
                        {
                            "engine": "qwen3-vl",
                            "model": "qwen3-vl-4b",
                            "duration_seconds": 0.75,
                        },
                        surname="ТЕСТОВ",
                    ),
                    "error": None,
                },
            )
            vlm_run.assert_awaited_once_with(
                b"fake-image",
                model="qwen3-vl-4b",
                filename="passport-cropped.jpg",
                content_type="image/jpeg",
                task="passport_fields",
            )
            paddle_run.assert_not_awaited()
            mrz_run.assert_not_awaited()

    def test_russian_ocr_returns_passport_fields(self) -> None:
        items = [
            {
                "text": "ТЕСТОВ",
                "confidence": None,
                "box": None,
                "detection_confidence": None,
            },
            {
                "text": "ТЕСТ",
                "confidence": None,
                "box": None,
                "detection_confidence": None,
            },
        ]
        result = {
            "items": items,
            "image_width": 120,
            "image_height": 60,
            "passport": {
                "passport_type": "internal",
                "document_code": "PN",
                "issuing_country": "RUS",
                "surname": "ТЕСТОВ",
                "name": "ТЕСТ",
                "patronymic": "ТЕСТОВИЧ",
                "sex": "M",
                "date_of_birth": "1990-01-15",
                "passport_series": "4515",
                "passport_number": "123456",
                "nationality": "RUS",
                "issue_date": "2015-06-20",
                "expiry_date": None,
                "department_code": "770001",
                "personal_number": None,
                "mrz_valid": None,
            },
        }
        with (
            patch(
                f"{SERVICE}.russian_runtime.run",
                new=AsyncMock(return_value=result),
            ) as russian_run,
            patch(
                f"{SERVICE}.recognize_vlm_image",
                new_callable=AsyncMock,
            ) as vlm_run,
            patch(f"{SERVICE}.perf_counter", side_effect=[10.0, 10.25]),
        ):
            response = self.client.post(
                PATH,
                files=RUSSIAN_FILES,
                data=page3_form(),
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "success": True,
                "data": text_data(
                    {
                        "engine": "russian-docs-ocr",
                        "model": "russian-ocr",
                        "duration_seconds": 0.25,
                    },
                    passport_type="internal",
                    document_code="PN",
                    issuing_country="RUS",
                    surname="ТЕСТОВ",
                    name="ТЕСТ",
                    patronymic="ТЕСТОВИЧ",
                    sex="МУЖ.",
                    date_of_birth="15.01.1990",
                    passport_series="45 15",
                    passport_number="123456",
                    nationality="RUS",
                    issue_date="20.06.2015",
                    department_code="770-001",
                ),
                "error": None,
            },
        )
        russian_run.assert_awaited_once_with(
            "passport_text",
            RUSSIAN_IMAGE_BYTES,
            timeout_seconds=settings.russian_ocr_document_hard_timeout_seconds,
        )
        vlm_run.assert_not_awaited()

    def test_qwen_json_populates_the_same_passport_fields_as_mrz(self) -> None:
        item = {
            "text": (
                "```json\n"
                '{"passport_type":"internal","document_code":"PN",'
                '"issuing_country":"RUS","surname":"ТЕСТОВ",'
                '"name":"ТЕСТ","patronymic":"ТЕСТОВИЧ",'
                '"sex":"МУЖ.","date_of_birth":"15.01.1990",'
                '"passport_series":"45","passport_number":"15 123456",'
                '"nationality":"RUS","issue_date":"20.06.2015",'
                '"expiry_date":null,"department_code":"770001",'
                '"personal_number":null}\n'
                "```"
            ),
            "confidence": None,
            "box": None,
            "detection_confidence": None,
        }
        result = {"items": [item], "image_width": 120, "image_height": 60}
        with (
            patch(
                f"{SERVICE}.recognize_vlm_image",
                new=AsyncMock(return_value=result),
            ),
            patch(f"{SERVICE}.perf_counter", side_effect=[1.0, 1.5]),
        ):
            response = self.client.post(
                PATH,
                files=FILES,
                data=page3_form(model="qwen3-vl-8b"),
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertIsNone(data["mrz"])
        self.assertIsNone(data["normalization"])
        self.assertEqual(data["surname"], "ТЕСТОВ")
        self.assertEqual(data["name"], "ТЕСТ")
        self.assertEqual(data["sex"], "МУЖ.")
        self.assertEqual(data["date_of_birth"], "15.01.1990")
        self.assertEqual(data["passport_series"], "45 15")
        self.assertEqual(data["passport_number"], "123456")
        self.assertEqual(data["department_code"], "770-001")
        self.assertIsNone(data["mrz_valid"])

    def test_qwen_skips_empty_json_before_useful_object(self) -> None:
        result = {
            "items": [
                {
                    "text": '{}\n{"surname":"ИВАНОВ","name":"ИВАН"}',
                    "confidence": None,
                    "box": None,
                    "detection_confidence": None,
                }
            ],
            "image_width": 120,
            "image_height": 60,
        }
        with (
            patch(
                f"{SERVICE}.recognize_vlm_image",
                new=AsyncMock(return_value=result),
            ),
            patch(f"{SERVICE}.perf_counter", side_effect=[2.0, 2.5]),
        ):
            response = self.client.post(
                PATH,
                files=FILES,
                data=page3_form(model="qwen3-vl-4b"),
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "success": True,
                "data": text_data(
                    {
                        "engine": "qwen3-vl",
                        "model": "qwen3-vl-4b",
                        "duration_seconds": 0.5,
                    },
                    surname="ИВАНОВ",
                    name="ИВАН",
                ),
                "error": None,
            },
        )

    def test_qwen_prefers_fields_that_remain_useful_after_normalization(
        self,
    ) -> None:
        for first_candidate in (
            '{"surname":"   "}',
            '{"date_of_birth":"garbage"}',
        ):
            with (
                self.subTest(first_candidate=first_candidate),
                patch(
                    f"{SERVICE}.recognize_vlm_image",
                    new=AsyncMock(
                        return_value={
                            "items": [
                                {
                                    "text": (
                                        first_candidate
                                        + '\n{"surname":"ИВАНОВ",'
                                        '"name":"ИВАН"}'
                                    ),
                                    "confidence": None,
                                    "box": None,
                                    "detection_confidence": None,
                                }
                            ],
                            "image_width": 120,
                            "image_height": 60,
                        }
                    ),
                ),
            ):
                response = self.client.post(
                    PATH,
                    files=FILES,
                    data=page3_form(model="qwen3-vl-4b"),
                )

            self.assertEqual(response.status_code, 200)
            data = response.json()["data"]
            self.assertEqual(data["surname"], "ИВАНОВ")
            self.assertEqual(data["name"], "ИВАН")
            self.assertIsNone(data["date_of_birth"])

    def test_qwen_ignores_unknown_keys_in_otherwise_useful_object(
        self,
    ) -> None:
        result = {
            "items": [
                {
                    "text": (
                        '{"surname":"ИВАНОВ","unexpected":"НЕ ПОЛЕ ПАСПОРТА"}'
                    ),
                    "confidence": None,
                    "box": None,
                    "detection_confidence": None,
                }
            ],
            "image_width": 120,
            "image_height": 60,
        }
        with patch(
            f"{SERVICE}.recognize_vlm_image",
            new=AsyncMock(return_value=result),
        ):
            response = self.client.post(
                PATH,
                files=FILES,
                data=page3_form(model="qwen3-vl-4b"),
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertEqual(data["surname"], "ИВАНОВ")
        self.assertNotIn("unexpected", data)

    def test_qwen_rejects_json_with_only_server_owned_fields(self) -> None:
        result = {
            "items": [
                {
                    "text": (
                        '{"passport_type":"internal","document_code":"PN",'
                        '"issuing_country":"RUS","nationality":"RUS",'
                        '"expiry_date":null,"personal_number":null}'
                    ),
                    "confidence": None,
                    "box": None,
                    "detection_confidence": None,
                }
            ],
            "image_width": 120,
            "image_height": 60,
        }
        with patch(
            f"{SERVICE}.recognize_vlm_image",
            new=AsyncMock(return_value=result),
        ):
            response = self.client.post(
                PATH,
                files=FILES,
                data=page3_form(model="qwen3-vl-4b"),
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "OCR_FAILED")

    def test_qwen_accepts_each_recognized_field_even_when_null(self) -> None:
        recognized_fields = (
            "surname",
            "name",
            "patronymic",
            "sex",
            "date_of_birth",
            "passport_series",
            "passport_number",
            "issue_date",
            "department_code",
        )
        for field in recognized_fields:
            with (
                self.subTest(field=field),
                patch(
                    f"{SERVICE}.recognize_vlm_image",
                    new=AsyncMock(
                        return_value={
                            "items": [
                                {
                                    "text": f'{{"{field}":null}}',
                                    "confidence": None,
                                    "box": None,
                                    "detection_confidence": None,
                                }
                            ],
                            "image_width": 120,
                            "image_height": 60,
                        }
                    ),
                ),
            ):
                response = self.client.post(
                    PATH,
                    files=FILES,
                    data=page3_form(model="qwen3-vl-4b"),
                )

            self.assertEqual(response.status_code, 200)
            data = response.json()["data"]
            self.assertIsNone(data[field])
            self.assertEqual(data["passport_type"], "internal")
            self.assertEqual(data["document_code"], "PN")
            self.assertEqual(data["issuing_country"], "RUS")
            self.assertEqual(data["nationality"], "RUS")

    def test_qwen_partial_expected_object_keeps_nulls_and_service_defaults(
        self,
    ) -> None:
        result = {
            "items": [
                {
                    "text": '{"surname":"ИВАНОВ","passport_number":"123456"}',
                    "confidence": None,
                    "box": None,
                    "detection_confidence": None,
                }
            ],
            "image_width": 120,
            "image_height": 60,
        }
        with (
            patch(
                f"{SERVICE}.recognize_vlm_image",
                new=AsyncMock(return_value=result),
            ),
            patch(f"{SERVICE}.perf_counter", side_effect=[3.0, 3.25]),
        ):
            response = self.client.post(
                PATH,
                files=FILES,
                data=page3_form(model="qwen3-vl-8b"),
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "success": True,
                "data": text_data(
                    {
                        "engine": "qwen3-vl",
                        "model": "qwen3-vl-8b",
                        "duration_seconds": 0.25,
                    },
                    surname="ИВАНОВ",
                    passport_number="123456",
                ),
                "error": None,
            },
        )

    def test_qwen_text_without_json_is_ocr_failed(self) -> None:
        result = {
            "items": [
                {
                    "text": "ФАМИЛИЯ ИВАНОВ, ИМЯ ИВАН",
                    "confidence": None,
                    "box": None,
                    "detection_confidence": None,
                }
            ],
            "image_width": 120,
            "image_height": 60,
        }
        with patch(
            f"{SERVICE}.recognize_vlm_image",
            new=AsyncMock(return_value=result),
        ):
            response = self.client.post(
                PATH,
                files=FILES,
                data=page3_form(model="qwen3-vl-4b"),
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "OCR_FAILED")

    def test_qwen_rejects_only_empty_or_irrelevant_json(self) -> None:
        for text in ("{}", '{"raw":true}', '{"unexpected":"ИВАНОВ"}'):
            with (
                self.subTest(text=text),
                patch(
                    f"{SERVICE}.recognize_vlm_image",
                    new=AsyncMock(
                        return_value={
                            "items": [
                                {
                                    "text": text,
                                    "confidence": None,
                                    "box": None,
                                    "detection_confidence": None,
                                }
                            ],
                            "image_width": 120,
                            "image_height": 60,
                        }
                    ),
                ),
            ):
                response = self.client.post(
                    PATH,
                    files=FILES,
                    data=page3_form(model="qwen3-vl-4b"),
                )

            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.json()["error"]["code"], "OCR_FAILED")

    def test_vlm_models_reject_json_with_only_unknown_keys(self) -> None:
        item = {
            "text": (
                "  **ПАСПОРТ**\n\n"
                '<div data-bbox="1,2,3,4">ФАМИЛИЯ: ТЕСТОВ</div>\n'
                '```json\n{"raw": true}\n```  \n'
            ),
            "confidence": None,
            "box": None,
            "detection_confidence": None,
        }
        result = {"items": [item], "image_width": 120, "image_height": 60}
        for model in ("qwen3-vl-4b", "qwen3-vl-8b"):
            with (
                self.subTest(model=model),
                patch(
                    f"{SERVICE}.recognize_vlm_image",
                    new=AsyncMock(return_value=result),
                ) as vlm_run,
                patch(
                    "app.modules.paddle.runtime.paddle_runtime.run",
                    new_callable=AsyncMock,
                ) as paddle_run,
                patch(
                    f"{SERVICE}.mrz_scanner_runtime.run",
                    new_callable=AsyncMock,
                ) as mrz_run,
                patch(f"{SERVICE}.perf_counter", side_effect=[4.0, 6.25]),
            ):
                response = self.client.post(
                    PATH,
                    files=FILES,
                    data=page3_form(model=model),
                )

            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.json()["error"]["code"], "OCR_FAILED")
            vlm_run.assert_awaited_once_with(
                b"fake-image",
                model=model,
                filename="passport-cropped.jpg",
                content_type="image/jpeg",
                task="passport_fields",
            )
            paddle_run.assert_not_awaited()
            mrz_run.assert_not_awaited()

    def test_crop_false_passes_original_image_to_qwen(self) -> None:
        result = {
            "items": [
                {
                    "text": '{"surname":"ИВАНОВ"}',
                    "confidence": None,
                    "box": None,
                    "detection_confidence": None,
                }
            ],
            "image_width": 120,
            "image_height": 60,
        }
        with patch(
            f"{SERVICE}.recognize_vlm_image",
            new=AsyncMock(return_value=result),
        ) as vlm_run:
            response = self.client.post(
                PATH,
                files={
                    "image": ("passport.png", b"source-image", "image/png")
                },
                data=page3_form(model="qwen3-vl-4b", crop="false"),
            )

        self.assertEqual(response.status_code, 200)
        self.crop_document.assert_not_awaited()
        vlm_run.assert_awaited_once_with(
            b"source-image",
            model="qwen3-vl-4b",
            filename="passport.png",
            content_type="image/png",
            task="passport_fields",
        )

    def test_russian_ocr_empty_text_is_a_successful_result(self) -> None:
        with patch(
            f"{SERVICE}.russian_runtime.run",
            new=AsyncMock(
                return_value={
                    "items": [],
                    "image_width": 120,
                    "image_height": 60,
                }
            ),
        ):
            response = self.client.post(
                PATH,
                files=RUSSIAN_FILES,
                data=page3_form(model="russian-ocr"),
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertNotIn("items", body["data"])
        self.assertIsNone(body["error"])
        self.assertGreaterEqual(body["data"]["meta"]["duration_seconds"], 0)

    def test_qwen_empty_items_are_ocr_failed(self) -> None:
        for model in ("qwen3-vl-4b", "qwen3-vl-8b"):
            with (
                self.subTest(model=model),
                patch(
                    f"{SERVICE}.recognize_vlm_image",
                    new=AsyncMock(
                        return_value={
                            "items": [],
                            "image_width": 120,
                            "image_height": 60,
                        }
                    ),
                ),
            ):
                response = self.client.post(
                    PATH,
                    files=FILES,
                    data=page3_form(model=model),
                )

            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.json()["error"]["code"], "OCR_FAILED")

    def test_page_is_required_before_ocr_dispatch(self) -> None:
        with (
            patch(
                f"{SERVICE}.recognize_vlm_image", new_callable=AsyncMock
            ) as vlm_run,
            patch(
                f"{SERVICE}.russian_runtime.run", new_callable=AsyncMock
            ) as russian_run,
        ):
            response = self.client.post(PATH, files=FILES)

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "VALIDATION_ERROR")
        vlm_run.assert_not_awaited()
        russian_run.assert_not_awaited()

    def test_unsupported_page_is_rejected_before_ocr_dispatch(self) -> None:
        with (
            patch(
                f"{SERVICE}.recognize_vlm_image", new_callable=AsyncMock
            ) as vlm_run,
            patch(
                f"{SERVICE}.russian_runtime.run", new_callable=AsyncMock
            ) as russian_run,
        ):
            response = self.client.post(
                PATH,
                files=FILES,
                data={"page": "2"},
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "VALIDATION_ERROR")
        vlm_run.assert_not_awaited()
        russian_run.assert_not_awaited()

    def test_upload_errors_do_not_call_ocr(self) -> None:
        for files, status, code in (
            ({}, 400, "NO_IMAGE"),
            (
                {"image": ("passport.jpg", b"", "image/jpeg")},
                400,
                "EMPTY_IMAGE",
            ),
            (
                {"image": ("passport.pdf", b"document", "application/pdf")},
                400,
                "UNSUPPORTED_FORMAT",
            ),
            (FILES, 413, "IMAGE_TOO_LARGE"),
        ):
            with self.subTest(code=code):
                with (
                    patch(
                        "app.modules.paddle.runtime.paddle_runtime.run",
                        new_callable=AsyncMock,
                    ) as run,
                    patch(
                        f"{SERVICE}.recognize_vlm_image",
                        new_callable=AsyncMock,
                    ) as vlm_run,
                    patch(
                        f"{SERVICE}.russian_runtime.run",
                        new_callable=AsyncMock,
                    ) as russian_run,
                    patch(
                        f"{SERVICE}.settings",
                        new=SimpleNamespace(ocr_max_image_size_bytes=3),
                    ),
                ):
                    response = self.client.post(
                        PATH,
                        files=files,
                        data=page3_form(),
                    )

                self.assertEqual(response.status_code, status)
                body = response.json()
                self.assertFalse(body["success"])
                self.assertIsNone(body["data"])
                self.assertEqual(body["error"]["code"], code)
                run.assert_not_awaited()
                vlm_run.assert_not_awaited()
                russian_run.assert_not_awaited()

    def test_russian_ocr_rejects_invalid_image_before_runtime(self) -> None:
        with patch(
            f"{SERVICE}.russian_runtime.run",
            new_callable=AsyncMock,
        ) as russian_run:
            response = self.client.post(
                PATH,
                files={"image": ("broken.jpg", b"not-an-image", "image/jpeg")},
                data=page3_form(model="russian-ocr"),
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "IMAGE_READ_ERROR")
        russian_run.assert_not_awaited()

    def test_russian_ocr_rejects_excessive_resolution_before_runtime(
        self,
    ) -> None:
        with (
            patch(
                "app.core.images.validation.settings",
                new=replace(settings, ocr_max_image_width=100),
            ),
            patch(
                f"{SERVICE}.russian_runtime.run",
                new_callable=AsyncMock,
            ) as russian_run,
        ):
            response = self.client.post(
                PATH,
                files=RUSSIAN_FILES,
                data=page3_form(model="russian-ocr"),
            )

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error"]["code"], "IMAGE_TOO_LARGE")
        russian_run.assert_not_awaited()

    def test_runtime_errors_keep_the_api_envelope(self) -> None:
        for code, status in (
            (ErrorCode.IMAGE_READ_ERROR, 400),
            (ErrorCode.OCR_FAILED, 422),
            (ErrorCode.OCR_ENGINE_UNAVAILABLE, 503),
            (ErrorCode.OCR_TIMEOUT, 504),
        ):
            for model, target in MODELS:
                with (
                    self.subTest(code=code, model=model),
                    patch(
                        f"{SERVICE}.{target}",
                        new=AsyncMock(
                            side_effect=AppError(code, "Ошибка распознавания")
                        ),
                    ),
                ):
                    response = self.client.post(
                        PATH,
                        files=files_for_model(model),
                        data=page3_form(model=model),
                    )

                self.assertEqual(response.status_code, status)
                self.assertEqual(
                    response.json(),
                    {
                        "success": False,
                        "data": None,
                        "error": {
                            "code": code.value,
                            "message": "Ошибка распознавания",
                        },
                    },
                )

    def test_malformed_runtime_result_is_reported_as_ocr_failure(self) -> None:
        for result in (["invalid"],):
            for model, target in MODELS:
                with (
                    self.subTest(result=result, model=model),
                    patch(
                        f"{SERVICE}.{target}",
                        new=AsyncMock(return_value=result),
                    ),
                ):
                    response = self.client.post(
                        PATH,
                        files=files_for_model(model),
                        data=page3_form(model=model),
                    )

                self.assertEqual(response.status_code, 422)
                self.assertEqual(
                    response.json()["error"]["code"], "OCR_FAILED"
                )

    def test_unsupported_models_are_rejected_before_dispatch(self) -> None:
        for model in (
            "removed-model",
            "russian",
            "paddle",
            "mrz",
            "paddleocr-vl-1.6",
            "chandra-ocr-2",
            "unknown",
        ):
            with (
                self.subTest(model=model),
                patch(
                    f"{SERVICE}.recognize_vlm_image", new_callable=AsyncMock
                ) as vlm_run,
                patch(
                    f"{SERVICE}.russian_runtime.run", new_callable=AsyncMock
                ) as russian_run,
                patch(
                    "app.modules.paddle.runtime.paddle_runtime.run",
                    new_callable=AsyncMock,
                ) as paddle_run,
            ):
                response = self.client.post(
                    PATH,
                    files=FILES,
                    data=page3_form(model=model),
                )
            self.assertEqual(response.status_code, 422)
            self.assertEqual(
                response.json()["error"]["code"], "VALIDATION_ERROR"
            )
            vlm_run.assert_not_awaited()
            russian_run.assert_not_awaited()
            paddle_run.assert_not_awaited()

    def test_openapi_documents_page3_passport_response(self) -> None:
        schema = app.openapi()
        self.assertIn(PATH, schema["paths"])
        operation = schema["paths"][PATH]["post"]
        self.assertEqual(
            operation["operationId"],
            "recognize_passport_page_v1_ocr_passport_page_post",
        )
        request = operation["requestBody"]["content"]["multipart/form-data"][
            "schema"
        ]
        self.assertIn("recognize_passport_page", request["$ref"])
        body_schema = schema["components"]["schemas"][
            request["$ref"].rsplit("/", 1)[-1]
        ]
        image_schema = body_schema["properties"]["image"]

        self.assertEqual(
            set(body_schema["properties"]),
            {"image", "page", "model", "crop"},
        )
        self.assertEqual(body_schema["required"], ["image", "page"])
        self.assertEqual(image_schema["type"], "string")
        self.assertEqual(image_schema["format"], "binary")
        page_schema = body_schema["properties"]["page"]
        self.assertEqual(
            page_schema["$ref"],
            "#/components/schemas/PassportPage",
        )
        self.assertEqual(
            schema["components"]["schemas"]["PassportPage"],
            {"type": "integer", "enum": [3], "title": "PassportPage"},
        )
        model_schema = body_schema["properties"]["model"]
        self.assertEqual(model_schema["default"], "russian-ocr")
        self.assertEqual(
            model_schema["$ref"],
            "#/components/schemas/PassportPageOcrModel",
        )
        enum_schema = schema["components"]["schemas"][
            model_schema["$ref"].rsplit("/", 1)[-1]
        ]
        self.assertEqual(
            set(enum_schema["enum"]), {model for model, _ in MODELS}
        )
        self.assertIs(body_schema["properties"]["crop"]["default"], True)
        meta_schema = schema["components"]["schemas"]["PassportOcrMetaSchema"]
        self.assertEqual(
            set(meta_schema["required"]),
            {"engine", "model", "duration_seconds"},
        )
        self.assertEqual(
            set(meta_schema["properties"]),
            {"engine", "model", "duration_seconds"},
        )
        self.assertEqual(
            set(meta_schema["properties"]["engine"]["enum"]),
            {"mrz-scanner", "qwen3-vl", "russian-docs-ocr"},
        )
        self.assertNotIn(
            "PassportTextDataSchema", schema["components"]["schemas"]
        )
        self.assertNotIn(
            "PassportTextResponseSchema", schema["components"]["schemas"]
        )
        page_data_schema = schema["components"]["schemas"][
            "PassportPageDataSchema"
        ]
        common_fields = set(EMPTY_PASSPORT_DATA)
        self.assertTrue(common_fields <= set(page_data_schema["properties"]))
        self.assertTrue(common_fields <= set(page_data_schema["required"]))
        self.assertNotIn("items", page_data_schema["properties"])
        self.assertEqual(page_data_schema["properties"]["mrz"]["type"], "null")
        self.assertEqual(
            page_data_schema["properties"]["normalization"]["type"],
            "null",
        )
        self.assertEqual(
            page_data_schema["properties"]["passport_type"]["const"],
            "internal",
        )
        self.assertEqual(
            page_data_schema["properties"]["issuing_country"]["const"],
            "RUS",
        )
        self.assertNotIn("PassportOcrModel", schema["components"]["schemas"])
        self.assertIn("страницу внутреннего паспорта РФ", operation["summary"])
        self.assertEqual(
            operation["responses"]["200"]["content"]["application/json"][
                "schema"
            ]["$ref"],
            "#/components/schemas/PassportPageResponseSchema",
        )
        self.assertTrue(
            {"400", "413", "422", "503", "504"}
            <= operation["responses"].keys()
        )


if __name__ == "__main__":
    unittest.main()
