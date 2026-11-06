from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
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
    }
    assert response.json()["data"] == {
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


@pytest.mark.parametrize(
    "name,text,expected",
    [
        ("actuality_date", "Данные актуальны на 05.11.2026", "05.11.2026"),
        ("application_number", "Заявление №123О", None),
    ],
)
def test_read_labels_without_guessing_digits(name, text, expected):
    assert field_value(name, text) == expected
