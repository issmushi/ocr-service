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

PATH = "/v1/ocr/passport/internal"
SERVICE = "app.modules.v1.passport.service"
encoded = BytesIO()
Image.new("RGB", (120, 60), "white").save(encoded, format="JPEG")
IMAGE_BYTES = encoded.getvalue()
FILES = {"image": ("passport.jpg", IMAGE_BYTES, "image/jpeg")}
EMPTY_PASSPORT_DATA = {
    "mrz": None,
    "passport_type": "internal",
    "surname": None,
    "name": None,
    "patronymic": None,
    "sex": None,
    "date_of_birth": None,
    "passport_series": None,
    "passport_number": None,
    "issue_date": None,
    "Issue_organization_ru": None,
    "department_code": None,
}
INTERNAL_FIELDS = set(EMPTY_PASSPORT_DATA) | {"meta"}
MRZ_ONLY_FIELDS = {
    "normalization",
    "mrz_valid",
    "personal_number",
    "expiry_date",
    "document_code",
    "issuing_country",
    "nationality",
}


def text_data(meta: dict, **passport: str | None) -> dict:
    return {**EMPTY_PASSPORT_DATA, **passport, "meta": meta}


class PassportV1InternalServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_russian_ocr_returns_passport_fields_and_issue_organization(
        self,
    ) -> None:
        result = {
            "items": [
                {"text": "ТЕСТОВ", "confidence": None, "box": None},
                {
                    "text": "ГУ МВД РОССИИ ПО Г. МОСКВЕ",
                    "confidence": None,
                    "box": None,
                },
            ],
            "image_width": 120,
            "image_height": 60,
            "diagnostics": {"internal": "not-public"},
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
                "Issue_organization_ru": "ГУ МВД РОССИИ ПО Г. МОСКВЕ",
                "department_code": "770001",
            },
        }
        with (
            patch(
                f"{SERVICE}.russian_runtime.run",
                new=AsyncMock(return_value=result),
            ) as russian_run,
            patch(f"{SERVICE}.perf_counter", side_effect=[10.0, 10.25]),
        ):
            response = self.client.post(PATH, files=FILES)

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
                    surname="ТЕСТОВ",
                    name="ТЕСТ",
                    patronymic="ТЕСТОВИЧ",
                    sex="МУЖ.",
                    date_of_birth="15.01.1990",
                    passport_series="45 15",
                    passport_number="123456",
                    issue_date="20.06.2015",
                    Issue_organization_ru="ГУ МВД РОССИИ ПО Г. МОСКВЕ",
                    department_code="770-001",
                ),
                "error": None,
            },
        )
        russian_run.assert_awaited_once_with(
            "passport_text",
            IMAGE_BYTES,
            timeout_seconds=settings.russian_ocr_document_hard_timeout_seconds,
        )

    def test_empty_ocr_result_keeps_nullable_fields(self) -> None:
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
            response = self.client.post(PATH, files=FILES)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertIsNone(body["error"])
        self.assertIsNone(body["data"]["Issue_organization_ru"])
        self.assertIsNone(body["data"]["surname"])
        self.assertNotIn("items", body["data"])
        self.assertEqual(set(body["data"]), INTERNAL_FIELDS)
        self.assertGreaterEqual(body["data"]["meta"]["duration_seconds"], 0)

    def test_russian_ocr_mrz_rows_are_returned_without_mrz_scanner(
        self,
    ) -> None:
        lines = [
            "PNRUSTESTOV<<TEST".ljust(44, "<"),
            "4511234569RUS9001158M<<<<<<<5150620770001<88",
        ]
        with patch(
            f"{SERVICE}.russian_runtime.run",
            new=AsyncMock(return_value={"mrz": lines, "passport": {}}),
        ):
            response = self.client.post(PATH, files=FILES)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["mrz"], lines)

    def test_blank_issue_organization_is_null(self) -> None:
        with patch(
            f"{SERVICE}.russian_runtime.run",
            new=AsyncMock(
                return_value={"passport": {"Issue_organization_ru": "  "}}
            ),
        ):
            response = self.client.post(PATH, files=FILES)

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["data"]["Issue_organization_ru"])

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
            with (
                self.subTest(code=code),
                patch(
                    f"{SERVICE}.russian_runtime.run", new_callable=AsyncMock
                ) as run,
                patch(
                    f"{SERVICE}.settings",
                    new=SimpleNamespace(ocr_max_image_size_bytes=3),
                ),
            ):
                response = self.client.post(PATH, files=files)

            self.assertEqual(response.status_code, status)
            self.assertEqual(response.json()["error"]["code"], code)
            run.assert_not_awaited()

    def test_invalid_image_is_rejected_before_runtime(self) -> None:
        with patch(
            f"{SERVICE}.russian_runtime.run", new_callable=AsyncMock
        ) as russian_run:
            response = self.client.post(
                PATH,
                files={"image": ("broken.jpg", b"not-an-image", "image/jpeg")},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "IMAGE_READ_ERROR")
        russian_run.assert_not_awaited()

    def test_excessive_resolution_is_rejected_before_runtime(self) -> None:
        with (
            patch(
                "app.core.images.validation.settings",
                new=replace(settings, ocr_max_image_width=100),
            ),
            patch(
                f"{SERVICE}.russian_runtime.run", new_callable=AsyncMock
            ) as run,
        ):
            response = self.client.post(PATH, files=FILES)

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error"]["code"], "IMAGE_TOO_LARGE")
        run.assert_not_awaited()

    def test_runtime_errors_keep_the_api_envelope(self) -> None:
        for code, status in (
            (ErrorCode.IMAGE_READ_ERROR, 400),
            (ErrorCode.OCR_FAILED, 422),
            (ErrorCode.OCR_ENGINE_UNAVAILABLE, 503),
            (ErrorCode.OCR_TIMEOUT, 504),
        ):
            with (
                self.subTest(code=code),
                patch(
                    f"{SERVICE}.russian_runtime.run",
                    new=AsyncMock(
                        side_effect=AppError(code, "Ошибка распознавания")
                    ),
                ),
            ):
                response = self.client.post(PATH, files=FILES)

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

    def test_malformed_runtime_result_is_ocr_failure(self) -> None:
        with patch(
            f"{SERVICE}.russian_runtime.run",
            new=AsyncMock(return_value=["invalid"]),
        ):
            response = self.client.post(PATH, files=FILES)

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "OCR_FAILED")

    def test_openapi_documents_image_only_and_issuer_field(self) -> None:
        schema = app.openapi()
        self.assertIn(PATH, schema["paths"])
        self.assertNotIn("/v1/ocr/passport/page", schema["paths"])
        operation = schema["paths"][PATH]["post"]
        request = operation["requestBody"]["content"]["multipart/form-data"][
            "schema"
        ]
        body_schema = schema["components"]["schemas"][
            request["$ref"].rsplit("/", 1)[-1]
        ]
        self.assertEqual(set(body_schema["properties"]), {"image"})
        self.assertEqual(body_schema["required"], ["image"])
        image_schema = body_schema["properties"]["image"]
        self.assertEqual(image_schema["type"], "string")
        self.assertEqual(image_schema["format"], "binary")

        response_ref = operation["responses"]["200"]["content"][
            "application/json"
        ]["schema"]["$ref"]
        self.assertEqual(
            response_ref,
            "#/components/schemas/PassportInternalResponseSchema",
        )
        response_schema = schema["components"]["schemas"][
            "PassportInternalResponseSchema"
        ]
        data_ref = next(
            option["$ref"]
            for option in response_schema["properties"]["data"]["anyOf"]
            if "$ref" in option
        )
        data_schema = schema["components"]["schemas"][
            data_ref.rsplit("/", 1)[-1]
        ]
        self.assertEqual(set(data_schema["properties"]), INTERNAL_FIELDS)
        self.assertTrue(MRZ_ONLY_FIELDS.isdisjoint(data_schema["properties"]))
        self.assertIn("Issue_organization_ru", data_schema["properties"])
        self.assertIn("Issue_organization_ru", data_schema["required"])
        self.assertIn(
            "Кем выдан",
            data_schema["properties"]["Issue_organization_ru"]["description"],
        )
        self.assertNotIn("items", data_schema["properties"])
        self.assertTrue(
            {"400", "413", "422", "503", "504"}
            <= operation["responses"].keys()
        )


if __name__ == "__main__":
    unittest.main()
