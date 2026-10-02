from __future__ import annotations

import unittest
from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.errors import AppError, ErrorCode
from app.main import app

MRZ = [
    "P<ARMHAYRAPETYAN<<ARAM".ljust(44, "<"),
    "AA12345678ARM9001158M3001156<<<<<<<<<<<<<<06",
]
PARSED_MRZ = {
    "passport_type": "international",
    "document_code": "P",
    "issuing_country": "ARM",
    "surname": "HAYRAPETYAN",
    "name": "ARAM",
    "patronymic": None,
    "sex": "МУЖ.",
    "date_of_birth": "15.01.1990",
    "passport_series": None,
    "passport_number": "AA1234567",
    "nationality": "ARM",
    "issue_date": "15.01.2020",
    "expiry_date": "15.01.2030",
    "department_code": None,
    "personal_number": None,
    "mrz_valid": True,
}
PATH = "/v1/ocr/passport/mrz"


def expected_meta(
    *,
    duration_seconds: float = 0.25,
) -> dict[str, Any]:
    return {
        "engine": "mrz-scanner",
        "model": "mrz",
        "duration_seconds": duration_seconds,
    }


class PassportV1ServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def recognize(
        self,
        runtime_result: dict[str, Any],
        *,
        normalization: bool = True,
    ) -> dict[str, Any]:
        mrz_run = AsyncMock(return_value=runtime_result)
        with (
            patch(
                "app.modules.v1.passport.service.mrz_scanner_runtime.run",
                new=mrz_run,
            ),
            patch(
                "app.modules.v1.passport.service.settings",
                new=replace(
                    settings,
                    mrz_scanner_document_hard_timeout_seconds=17.0,
                ),
            ),
            patch(
                "app.modules.v1.passport.service.perf_counter",
                side_effect=[10.0, 10.25],
            ),
        ):
            response = self.client.post(
                PATH,
                data={
                    "crop": "false",
                    "normalization": str(normalization).lower(),
                },
                files={"image": ("passport.jpg", b"fake-image", "image/jpeg")},
            )

        self.assertEqual(response.status_code, 200, response.text)
        mrz_run.assert_awaited_once_with(
            "passport_v1", b"fake-image", timeout_seconds=17.0
        )
        return response.json()

    def test_scanner_preserves_each_correction_and_counts(
        self,
    ) -> None:
        normalization = {
            "corrections_count": 3,
            "replacements_count": 1,
            "insertions_count": 2,
            "corrections": [
                {
                    "line": 1,
                    "position": 1,
                    "source_position": None,
                    "field": "Тип документа",
                    "operation": "вставка",
                    "source": "",
                    "replacement": "P",
                    "reason": (
                        "Восстановлен отсутствующий символ типа документа."
                    ),
                },
                {
                    "line": 2,
                    "position": 11,
                    "source_position": 11,
                    "field": "Гражданство",
                    "operation": "замена",
                    "source": "4",
                    "replacement": "A",
                    "reason": "Цифра заменена похожей буквой.",
                },
                {
                    "line": 2,
                    "position": 44,
                    "source_position": None,
                    "field": "Общая контрольная цифра",
                    "operation": "вставка",
                    "source": "",
                    "replacement": "6",
                    "reason": "Вычислена отсутствующая контрольная цифра.",
                },
            ],
        }
        body = self.recognize(
            {
                "mrz": MRZ,
                "raw_mrz": [
                    MRZ[0][1:],
                    MRZ[1][:10] + "4" + MRZ[1][11:-1],
                ],
                "diagnostics": {
                    "normalization": normalization,
                    "header_normalization": {"country": "ARM"},
                    "preprocessing": {"internal": "not-public"},
                },
            },
        )

        self.assertEqual(
            body,
            {
                "success": True,
                "data": {
                    "mrz": MRZ,
                    "normalization": {**normalization, "issues": []},
                    **PARSED_MRZ,
                    "meta": expected_meta(),
                },
                "error": None,
            },
        )

    def test_legacy_model_values_cannot_select_another_engine(self) -> None:
        for legacy_model in (
            "paddle",
            "mrz",
            "qwen3-vl-4b",
            "qwen3-vl-8b",
            "removed-model",
        ):
            for crop in (False, True):
                raw_result = {"items": [{"text": line} for line in MRZ]}
                with (
                    self.subTest(model=legacy_model, crop=crop),
                    patch(
                        "app.modules.v1.passport.service.crop_mrz_bytes",
                        new=AsyncMock(return_value=b"crop"),
                    ) as crop_mrz,
                    patch(
                        "app.modules.v1.passport.service.mrz_scanner_runtime.run",
                        new=AsyncMock(
                            return_value=raw_result if crop else {"mrz": MRZ}
                        ),
                    ) as mrz_run,
                    patch(
                        "app.modules.paddle.runtime.paddle_runtime.run",
                        new_callable=AsyncMock,
                    ) as paddle_run,
                    patch(
                        "app.modules.v1.passport.service.recognize_vlm_image",
                        new_callable=AsyncMock,
                    ) as vlm_run,
                    patch(
                        "app.modules.v1.passport.service.ensure_ocr_model_enabled"
                    ) as enabled,
                ):
                    response = self.client.post(
                        PATH,
                        data={
                            "model": legacy_model,
                            "crop": str(crop).lower(),
                        },
                        files={
                            "image": ("passport.jpg", b"image", "image/jpeg")
                        },
                    )
                self.assertEqual(response.status_code, 200, response.text)
                meta = response.json()["data"]["meta"]
                self.assertEqual(meta["engine"], "mrz-scanner")
                self.assertEqual(meta["model"], "mrz")
                mrz_run.assert_awaited_once_with(
                    "mrz_crop_text" if crop else "passport_v1",
                    b"crop" if crop else b"image",
                    timeout_seconds=(
                        settings.mrz_scanner_document_hard_timeout_seconds
                    ),
                )
                if crop:
                    crop_mrz.assert_awaited_once_with(b"image")
                else:
                    crop_mrz.assert_not_awaited()
                enabled.assert_called_once_with("mrz")
                paddle_run.assert_not_awaited()
                vlm_run.assert_not_awaited()

    def test_default_crop_sends_only_the_crop_to_mrz_scanner(self) -> None:
        raw_result = {
            "items": [
                {"text": MRZ[0], "confidence": 0.99, "box": None},
                {"text": MRZ[1], "confidence": 0.99, "box": None},
            ],
            "image_width": 900,
            "image_height": 140,
        }
        with (
            patch(
                "app.modules.v1.passport.service.crop_mrz_bytes",
                new=AsyncMock(return_value=b"cropped-mrz-jpeg"),
            ) as crop_mrz,
            patch(
                "app.modules.v1.passport.service.mrz_scanner_runtime.run",
                new=AsyncMock(return_value=raw_result),
            ) as mrz_run,
            patch(
                "app.modules.v1.passport.service.settings",
                new=replace(
                    settings,
                    mrz_scanner_document_hard_timeout_seconds=17.0,
                ),
            ),
            patch(
                "app.modules.v1.passport.service.perf_counter",
                side_effect=[5.0, 5.5],
            ),
        ):
            response = self.client.post(
                PATH,
                files={"image": ("passport.png", b"full-page", "image/png")},
            )

        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()["data"]
        self.assertEqual(data["mrz"], MRZ)
        self.assertEqual(data["meta"], expected_meta(duration_seconds=0.5))
        crop_mrz.assert_awaited_once_with(b"full-page")
        mrz_run.assert_awaited_once_with(
            "mrz_crop_text", b"cropped-mrz-jpeg", timeout_seconds=17.0
        )

    def test_crop_failure_stops_before_mrz_scanner(self) -> None:
        crop_mrz = AsyncMock(
            side_effect=AppError(
                ErrorCode.OCR_FAILED,
                "Не удалось найти MRZ на изображении",
            )
        )
        with (
            patch(
                "app.modules.v1.passport.service.crop_mrz_bytes",
                new=crop_mrz,
            ),
            patch(
                "app.modules.paddle.runtime.paddle_runtime.run",
                new_callable=AsyncMock,
            ) as paddle_run,
            patch(
                "app.modules.v1.passport.service.mrz_scanner_runtime.run",
                new_callable=AsyncMock,
            ) as mrz_run,
            patch(
                "app.modules.v1.passport.service.recognize_vlm_image",
                new_callable=AsyncMock,
            ) as vlm_run,
        ):
            response = self.client.post(
                PATH,
                files={"image": ("passport.jpg", b"full-page", "image/jpeg")},
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json()["error"],
            {
                "code": "OCR_FAILED",
                "message": "Не удалось найти MRZ на изображении",
            },
        )
        crop_mrz.assert_awaited_once_with(b"full-page")
        paddle_run.assert_not_awaited()
        mrz_run.assert_not_awaited()
        vlm_run.assert_not_awaited()

    def test_cropped_scanner_text_is_extracted_and_normalized(self) -> None:
        raw_result = {
            "items": [
                {"text": f"{MRZ[0][:12]} {MRZ[0][12:]}".lower()},
                {"text": f"{MRZ[1][:18]} {MRZ[1][18:]}".lower()},
            ],
            "image_width": 800,
            "image_height": 100,
        }
        with (
            patch(
                "app.modules.v1.passport.service.crop_mrz_bytes",
                new=AsyncMock(return_value=b"crop"),
            ),
            patch(
                "app.modules.v1.passport.service.mrz_scanner_runtime.run",
                new=AsyncMock(return_value=raw_result),
            ) as mrz_run,
        ):
            response = self.client.post(
                PATH,
                files={"image": ("passport.jpg", b"fake-image", "image/jpeg")},
            )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["data"]["mrz"], MRZ)
        self.assertEqual(body["data"]["normalization"]["corrections_count"], 0)
        mrz_run.assert_awaited_once_with(
            "mrz_crop_text",
            b"crop",
            timeout_seconds=settings.mrz_scanner_document_hard_timeout_seconds,
        )

    def test_cropped_scanner_without_passport_mrz_returns_ocr_failed(
        self,
    ) -> None:
        for result in (
            {"items": []},
            {"items": [{"text": "Обычный текст без MRZ"}]},
            {
                "items": [
                    {
                        "text": (
                            "I<UTO123456789012345678901234\n"
                            "SECOND123456789012345678901234"
                        )
                    }
                ]
            },
            {
                "items": [
                    {
                        "text": (
                            "P<XXXUNKNOWN<<PERSON<<<<<<<<<<<<<<<<<<<<<<<\n"
                            "7512345672XXX9001011M3501014<<<<<<<<<<<<<<02"
                        )
                    }
                ]
            },
            {"items": [{"text": 123}]},
            ["invalid"],
        ):
            with (
                self.subTest(result=result),
                patch(
                    "app.modules.v1.passport.service.crop_mrz_bytes",
                    new=AsyncMock(return_value=b"crop"),
                ),
                patch(
                    "app.modules.v1.passport.service.mrz_scanner_runtime.run",
                    new=AsyncMock(return_value=result),
                ) as mrz_run,
            ):
                response = self.client.post(
                    PATH,
                    files={
                        "image": ("passport.jpg", b"fake-image", "image/jpeg")
                    },
                )
            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.json()["error"]["code"], "OCR_FAILED")
            mrz_run.assert_awaited_once_with(
                "mrz_crop_text",
                b"crop",
                timeout_seconds=settings.mrz_scanner_document_hard_timeout_seconds,
            )

    def test_mrz_runtime_errors_preserve_public_error_envelope(self) -> None:
        for code, status, message in (
            (ErrorCode.OCR_ENGINE_UNAVAILABLE, 503, "OCR engine недоступен"),
            (ErrorCode.OCR_TIMEOUT, 504, "Превышен таймаут обработки"),
            (
                ErrorCode.OCR_FAILED,
                422,
                "OCR не смог получить пригодный результат",
            ),
        ):
            with self.subTest(code=code):
                with (
                    patch(
                        "app.modules.paddle.runtime.paddle_runtime.run",
                        new_callable=AsyncMock,
                    ) as paddle_run,
                    patch(
                        "app.modules.v1.passport.service.mrz_scanner_runtime.run",
                        new=AsyncMock(side_effect=AppError(code, message)),
                    ) as mrz_run,
                ):
                    response = self.client.post(
                        PATH,
                        data={"crop": "false"},
                        files={
                            "image": (
                                "passport.jpg",
                                b"fake-image",
                                "image/jpeg",
                            )
                        },
                    )

                self.assertEqual(response.status_code, status)
                self.assertEqual(
                    response.json(),
                    {
                        "success": False,
                        "data": None,
                        "error": {"code": code.value, "message": message},
                    },
                )
                mrz_run.assert_awaited_once_with(
                    "passport_v1",
                    b"fake-image",
                    timeout_seconds=settings.mrz_scanner_document_hard_timeout_seconds,
                )
                paddle_run.assert_not_awaited()

    def test_mrz_empty_result_is_reported_as_recognition_failure(self) -> None:
        with (
            patch(
                "app.modules.paddle.runtime.paddle_runtime.run",
                new_callable=AsyncMock,
            ) as paddle_run,
            patch(
                "app.modules.v1.passport.service.mrz_scanner_runtime.run",
                new=AsyncMock(return_value={"mrz": []}),
            ) as mrz_run,
        ):
            response = self.client.post(
                PATH,
                data={"crop": "false"},
                files={"image": ("passport.jpg", b"fake-image", "image/jpeg")},
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json(),
            {
                "success": False,
                "data": None,
                "error": {
                    "code": "OCR_FAILED",
                    "message": "Не удалось распознать MRZ паспорта",
                },
            },
        )
        mrz_run.assert_awaited_once()
        paddle_run.assert_not_awaited()

    def test_unknown_normalized_header_is_not_returned_as_success(
        self,
    ) -> None:
        with patch(
            "app.modules.v1.passport.service.mrz_scanner_runtime.run",
            new=AsyncMock(
                return_value={
                    "mrz": [
                        "P<XXXUNKNOWN<<PERSON".ljust(44, "<"),
                        "7512345672XXX9001011M3501014<<<<<<<<<<<<<<02",
                    ]
                }
            ),
        ):
            response = self.client.post(
                PATH,
                data={"crop": "false"},
                files={"image": ("passport.jpg", b"fake-image", "image/jpeg")},
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json()["error"],
            {
                "code": "OCR_FAILED",
                "message": "Неизвестный или неподдерживаемый формат паспорта",
            },
        )

    def test_response_keeps_explicit_report_without_corrections(self) -> None:
        normalization = {
            "corrections_count": 0,
            "replacements_count": 0,
            "insertions_count": 0,
            "corrections": [],
            "issues": [],
        }
        body = self.recognize(
            {"mrz": MRZ, "diagnostics": {"normalization": normalization}}
        )

        self.assertEqual(body["data"]["normalization"], normalization)

    def test_disabled_normalization_hides_report_but_keeps_processed_mrz(
        self,
    ) -> None:
        normalization = {
            "corrections_count": 1,
            "replacements_count": 1,
            "insertions_count": 0,
            "corrections": [
                {
                    "line": 2,
                    "position": 11,
                    "source_position": 11,
                    "field": "Гражданство",
                    "operation": "замена",
                    "source": "4",
                    "replacement": "A",
                    "reason": "Цифра заменена похожей буквой.",
                }
            ],
            "issues": [],
        }

        body = self.recognize(
            {"mrz": MRZ, "diagnostics": {"normalization": normalization}},
            normalization=False,
        )

        self.assertEqual(
            body["data"],
            {
                "mrz": MRZ,
                "normalization": None,
                **PARSED_MRZ,
                "meta": expected_meta(),
            },
        )

    def test_response_preserves_structure_issue_and_unchanged_lower_line(
        self,
    ) -> None:
        mrz = [
            "PNRUSIVANOV<<IVAN<IVANOVI3".ljust(44, "<"),
            "511234569RU59001158H<<<<<<<5150620770001<88",
        ]
        normalization = {
            "corrections_count": 0,
            "replacements_count": 0,
            "insertions_count": 0,
            "corrections": [],
            "issues": [
                {
                    "line": 2,
                    "code": "invalid_lower_line_structure",
                    "message": "Нарушена структура нижней строки MRZ",
                }
            ],
        }
        body = self.recognize(
            {
                "mrz": mrz,
                "raw_mrz": mrz,
                "diagnostics": {"normalization": normalization},
            },
        )

        self.assertTrue(body["success"])
        self.assertIsNone(body["error"])
        self.assertEqual(body["data"]["mrz"], mrz)
        self.assertEqual(
            body["data"]["normalization"],
            normalization,
        )
        self.assertEqual(body["data"]["passport_type"], "internal")
        self.assertEqual(body["data"]["issuing_country"], "RUS")
        self.assertEqual(body["data"]["surname"], "ИВАНОВ")
        self.assertIsNone(body["data"]["mrz_valid"])

    def test_response_reports_null_when_runtime_has_no_report(self) -> None:
        for extra in ({}, {"diagnostics": {}}, {"diagnostics": None}):
            with self.subTest(extra=extra):
                body = self.recognize({"mrz": MRZ, **extra})

                self.assertEqual(
                    body["data"],
                    {
                        "mrz": MRZ,
                        "normalization": None,
                        **PARSED_MRZ,
                        "meta": expected_meta(),
                    },
                )
