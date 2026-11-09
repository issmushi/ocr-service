from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.field_selection import resolve_fields
from app.main import app
from app.modules.v1.crop.egrn_layout import FIELD_LABELS
from app.modules.v1.russia.egrn.field_selection import EgrnField
from app.modules.v1.russia.egrn.schemas import EgrnData
from app.modules.v1.russia.egrn.service import field_value


def test_partial_result_contains_only_document_values_and_meta():
    raw = {
        "fields": [
            {
                "name": "application_number",
                "ocr": {
                    "text": "Заявление №°00042",
                    "status": "success",
                    "error": None,
                },
            },
            {
                "name": "address",
                "ocr": {
                    "text": "ТЕСТОВЫЙ ГОРОД\nДОМ 123",
                    "status": "success",
                    "error": None,
                },
            },
            {
                "name": "actuality_date",
                "ocr": {
                    "text": "Данные актуальны на О5.11.2026",
                    "status": "success",
                    "error": None,
                },
            },
        ],
        "meta": {"duration_seconds": 0.25},
    }
    with patch(
        "app.modules.v1.russia.egrn.service.read_egrn_fields_upload",
        new_callable=AsyncMock,
        return_value=raw,
    ) as read:
        response = TestClient(app).post(
            "/v1/ocr/egrn", files={"image": ("example.png", b"fixture")}
        )
    assert response.status_code == 200
    assert read.await_args.kwargs == {
        "ocr_model": "paddle-cyrillic",
        "include_page": False,
        "fields": resolve_fields(None, EgrnField),
    }
    assert response.json()["data"] == {
        **dict.fromkeys(FIELD_LABELS),
        "application_number": "00042",
        "address": "ТЕСТОВЫЙ ГОРОД ДОМ 123",
        "actuality_date": None,
        "cadastral_number": None,
        "object_name": None,
        "meta": {
            "model": "paddle-cyrillic",
            "engine": "paddleocr",
            "duration_seconds": 0.25,
        },
    }


def test_all_crop_fields_are_selectable_in_structured_recognition():
    assert len(FIELD_LABELS) == 19
    assert set(resolve_fields(None, EgrnField)) == set(FIELD_LABELS)
    assert set(EgrnData.model_fields) == set(FIELD_LABELS) | {"meta"}


def test_engine_failure_remains_an_error_for_integrators():
    error = {"code": "OCR_ENGINE_UNAVAILABLE", "message": "Недоступен Paddle"}
    raw = {
        "fields": [{"name": "address", "ocr": {"error": error}}],
        "meta": {},
    }
    with patch(
        "app.modules.v1.russia.egrn.service.read_egrn_fields_upload",
        new_callable=AsyncMock,
        return_value=raw,
    ):
        response = TestClient(app).post(
            "/v1/ocr/egrn", files={"image": ("example.png", b"fixture")}
        )
    assert response.status_code == 503
    assert response.json()["error"] == error


def test_all_fields_still_fails_when_no_recognition_values_are_available():
    # Full empty recognition is an error; selected missing fields are null.
    raw = {"fields": [], "meta": {"duration_seconds": 0.1}}
    with patch(
        "app.modules.v1.russia.egrn.service.read_egrn_fields_upload",
        new=AsyncMock(return_value=raw),
    ):
        client = TestClient(app)
        files = {"image": ("example.png", b"fixture")}
        response = client.post("/v1/ocr/egrn", files=files)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "OCR_FAILED"
        partial = client.post(
            "/v1/ocr/egrn", files=files, data={"fields": "address"}
        )
        assert partial.status_code == 200
        assert set(partial.json()["data"]) == {"address", "meta"}
        assert partial.json()["data"]["address"] is None


@pytest.mark.parametrize(
    "name,text,expected",
    [
        ("actuality_date", "Данные актуальны на 05.11.2026", "05.11.2026"),
        ("application_number", "Заявление №123О", None),
        ("area", "40,50", "40,50"),
        ("cadastral_value", "1 000 000.00", "1 000 000.00"),
        ("claims", "Данные\nотсутствуют", "Данные отсутствуют"),
        (
            "parent_cadastral_numbers",
            "00:00:0000000:001\n00:00:0000000:002",
            "00:00:0000000:001 00:00:0000000:002",
        ),
        ("cadastral_number_assignment_date", "01.01.2020", "01.01.2020"),
    ],
)
def test_read_labels_without_guessing_digits(name, text, expected):
    assert field_value(name, text) == expected
