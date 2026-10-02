from __future__ import annotations

from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.errors import AppError, ErrorCode
from app.main import app

PATH = "/v1/ocr/mrz"
PASSPORT_MRZ = [
    "P<ARMHAYRAPETYAN<<ARAM".ljust(44, "<"),
    "AA12345678ARM9001158M3001156<<<<<<<<<<<<<<06",
]
INTERNAL_MRZ = [
    "PNRUSIVANOV<<IVAN<IVANOVI3".ljust(44, "<"),
    "511234569RU59001158H<<<<<<<5150620770001<88",
]
VALID_INTERNAL_MRZ = [
    "PNRUSTESTOV<<TEST<TESTOVI3".ljust(44, "<"),
    "4511234569RUS9001158M<<<<<<<5150620770001<88",
]
EMPTY_REPORT = {
    "corrections_count": 0,
    "replacements_count": 0,
    "insertions_count": 0,
    "corrections": [],
    "issues": [],
}


def _raw(lines: list[str]) -> dict[str, Any]:
    return {"items": [{"text": line} for line in lines]}


def _post(
    raw_result: Any,
    *,
    crop: bool = False,
    normalize: bool = True,
    normalization: bool = True,
    extra_data: dict[str, str] | None = None,
) -> tuple[Any, dict[str, AsyncMock]]:
    mocks = {
        "crop": AsyncMock(return_value=b"cropped-mrz"),
        "scanner": AsyncMock(return_value=raw_result),
    }
    with (
        patch("app.modules.v1.mrz_api.service.crop_mrz_bytes", mocks["crop"]),
        patch(
            "app.modules.v1.mrz_api.service.mrz_scanner_runtime.run",
            mocks["scanner"],
        ),
        patch(
            "app.modules.v1.mrz_api.service.settings",
            replace(settings, mrz_scanner_document_hard_timeout_seconds=17.0),
        ),
        patch(
            "app.modules.v1.mrz_api.service.perf_counter",
            side_effect=[10.0, 10.25],
        ),
    ):
        form = {
            "crop": str(crop).lower(),
            "normalize": str(normalize).lower(),
            "normalization": str(normalization).lower(),
            **(extra_data or {}),
        }
        response = TestClient(app).post(
            PATH,
            data=form,
            files={"image": ("passport.jpg", b"fake-image", "image/jpeg")},
        )
    return response, mocks


def _assert_scanner(mocks: dict[str, AsyncMock], *, crop: bool) -> None:
    if crop:
        mocks["crop"].assert_awaited_once_with(b"fake-image")
    else:
        mocks["crop"].assert_not_awaited()
    mocks["scanner"].assert_awaited_once_with(
        "mrz_crop_text",
        b"cropped-mrz" if crop else b"fake-image",
        timeout_seconds=17.0,
    )


@pytest.mark.parametrize("crop", (False, True))
def test_passport_crop_modes_use_the_same_scanner_and_unified_fields(crop):
    response, mocks = _post(_raw(PASSPORT_MRZ), crop=crop)

    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["mrz"] == PASSPORT_MRZ
    assert data["document_type"] == "international_passport"
    assert data["document_code"] == "P"
    assert data["document_series"] is None
    assert data["document_number"] == "AA1234567"
    assert data["normalization"] == EMPTY_REPORT
    assert data["meta"] == {
        "engine": "mrz-scanner",
        "model": "mrz",
        "duration_seconds": 0.25,
    }
    assert "passport_type" not in data
    assert "passport_series" not in data
    assert "passport_number" not in data
    _assert_scanner(mocks, crop=crop)


def test_internal_passport_mrz_uses_the_same_document_type_field():
    response, mocks = _post(_raw(INTERNAL_MRZ))

    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["document_type"] == "internal_passport"
    assert data["document_code"] == "PN"
    assert data["issuing_country"] == "RUS"
    assert data["surname"] == "ИВАНОВ"
    assert data["document_number"] is None
    assert data["normalization"]["issues"][0]["line"] == 2
    _assert_scanner(mocks, crop=False)


def test_valid_internal_passport_keeps_series_and_number():
    response, mocks = _post(_raw(VALID_INTERNAL_MRZ))

    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["document_type"] == "internal_passport"
    assert data["document_series"] == "45 15"
    assert data["document_number"] == "123456"
    assert data["mrz_valid"] is True
    _assert_scanner(mocks, crop=False)


@pytest.mark.parametrize("normalize", (False, True))
@pytest.mark.parametrize("normalization", (False, True))
def test_passport_normalize_and_report_flags_are_independent(
    normalize: bool, normalization: bool
):
    noisy = [
        "P<ARMHAYRAPETYAN<<AR4M<<",
        "44123456784RM9001158N3001156<<<<<<<<<<<<<<",
    ]
    response, mocks = _post(
        _raw(noisy), normalize=normalize, normalization=normalization
    )

    assert response.status_code == 200, response.text
    data = response.json()["data"]
    if normalize:
        assert data["mrz"] == PASSPORT_MRZ
        assert data["document_number"] == "AA1234567"
        if normalization:
            assert data["normalization"]["corrections_count"] > 0
    else:
        assert data["mrz"] == noisy
        assert data["document_number"] is None
        if normalization:
            assert data["normalization"] == EMPTY_REPORT
    if not normalization:
        assert data["normalization"] is None
    _assert_scanner(mocks, crop=False)


@pytest.mark.parametrize("crop", (False, True))
def test_obsolete_model_form_cannot_change_mrz_engine(crop):
    response, mocks = _post(
        _raw(PASSPORT_MRZ),
        crop=crop,
        extra_data={"model": "qwen3-vl-8b"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["data"]["meta"]["model"] == "mrz"
    _assert_scanner(mocks, crop=crop)


def test_crop_failure_stops_before_scanner():
    scanner = AsyncMock()
    with (
        patch(
            "app.modules.v1.mrz_api.service.crop_mrz_bytes",
            AsyncMock(
                side_effect=AppError(
                    ErrorCode.OCR_FAILED, "Не удалось найти MRZ на изображении"
                )
            ),
        ),
        patch(
            "app.modules.v1.mrz_api.service.mrz_scanner_runtime.run", scanner
        ),
    ):
        response = TestClient(app).post(
            PATH,
            files={"image": ("passport.jpg", b"fake-image", "image/jpeg")},
        )

    assert response.status_code == 422
    assert response.json()["error"] == {
        "code": "OCR_FAILED",
        "message": "Не удалось найти MRZ на изображении",
    }
    scanner.assert_not_awaited()


@pytest.mark.parametrize(
    "raw_result",
    (
        {"items": []},
        {"items": [{"text": "Обычный текст без MRZ"}]},
        {"items": [{"text": 123}]},
        ["invalid"],
        _raw(
            [
                "P<XXXUNKNOWN<<PERSON".ljust(44, "<"),
                "7512345672XXX9001011M3501014<<<<<<<<<<<<<<02",
            ]
        ),
    ),
)
def test_unrecognized_or_invalid_mrz_returns_ocr_failed(raw_result):
    response, mocks = _post(raw_result)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "OCR_FAILED"
    _assert_scanner(mocks, crop=False)


@pytest.mark.parametrize(
    ("code", "status", "message"),
    (
        (ErrorCode.OCR_ENGINE_UNAVAILABLE, 503, "OCR engine недоступен"),
        (ErrorCode.OCR_TIMEOUT, 504, "Превышен таймаут обработки"),
        (
            ErrorCode.OCR_FAILED,
            422,
            "OCR не смог получить пригодный результат",
        ),
    ),
)
def test_scanner_errors_keep_public_error_envelope(code, status, message):
    scanner = AsyncMock(side_effect=AppError(code, message))
    with patch(
        "app.modules.v1.mrz_api.service.mrz_scanner_runtime.run", scanner
    ):
        response = TestClient(app).post(
            PATH,
            data={"crop": "false"},
            files={"image": ("passport.jpg", b"fake-image", "image/jpeg")},
        )

    assert response.status_code == status
    assert response.json() == {
        "success": False,
        "data": None,
        "error": {"code": code.value, "message": message},
    }
    scanner.assert_awaited_once()
