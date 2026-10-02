"""Driving licence routes keep incomplete OCR out of typed fields."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import settings
from app.main import app
from app.modules.v1.russia.driver_license.parsing import (
    driver_license_fields,
    reconcile_date_reading,
)

PATH = "/v1/ocr/driver-license"
SERVICE = "app.modules.v1.russia.driver_license.service"


@pytest.fixture
def client():
    test_client = TestClient(app)
    yield test_client
    test_client.close()


@pytest.fixture
def image_bytes():
    output = BytesIO()
    Image.new("RGB", (120, 60), "white").save(output, format="PNG")
    return output.getvalue()


def _upload(content: bytes):
    return {"image": ("synthetic.png", content, "image/png")}


def _ocr(*, birth_date: str = "03.02.1987") -> dict:
    return {
        "doctype": "DL_2011",
        "ocr": {
            "Last_name_ru": "ПРИМЕРОВ",
            "First_name_ru": "ИВАН",
            "Middle_name_ru": "ПРИМЕРОВИЧ",
            "Last_name_en": "PRIMEROV",
            "First_name_en": "IVAN",
            "Middle_name_en": "PRIMEROVICH",
            "Birth_date": birth_date,
            "Birth_place_ru": "ПРИМЕРСК",
            "Issue_date": "04.05.2020",
            "Expiration_date": "04.05.2030",
            "Issue_organization_ru": "ГИБДД",
            "Issue_organisation_code": "1234",
            "Licence_number": "11 22 345678",
            "Living_region_ru": "ПРИМЕРСКАЯ ОБЛ.",
            "Driver_class": "B B1 C C1 M",
        },
        "crops": {},
    }


def test_driver_license_reads_typed_fields_without_unneeded_fallback(
    client, image_bytes
):
    result = _ocr()
    with (
        patch(
            f"{SERVICE}.russian_runtime.run",
            new=AsyncMock(return_value=result),
        ) as russian_run,
        patch(
            f"{SERVICE}.paddle_runtime.run", new_callable=AsyncMock
        ) as paddle_run,
    ):
        response = client.post(PATH, files=_upload(image_bytes))

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["license_series"] == "11 22"
    assert data["license_number"] == "345678"
    assert data["date_of_birth"] == "03.02.1987"
    assert data["issue_date"] == "04.05.2020"
    assert data["expiry_date"] == "04.05.2030"
    assert data["categories"] == ["B", "B1", "C", "C1", "M"]
    assert data["department_code"] == "1234"
    assert data["meta"]["model"] == "russian-ocr"
    assert data["meta"]["duration_seconds"] >= 0
    russian_run.assert_awaited_once()
    paddle_run.assert_not_awaited()


def test_driver_license_rereads_only_invalid_date_crop(client, image_bytes):
    result = _ocr(birth_date="03.02.198S")
    result["crops"] = {"Birth_date": b"birth-date-crop"}
    with (
        patch(
            f"{SERVICE}.russian_runtime.run",
            new=AsyncMock(return_value=result),
        ),
        patch.object(type(settings), "is_model_enabled", return_value=True),
        patch(
            f"{SERVICE}.paddle_runtime.run",
            new=AsyncMock(
                return_value={"text": "03.02.1985", "confidence": 0.91}
            ),
        ) as paddle_run,
    ):
        response = client.post(PATH, files=_upload(image_bytes))

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["date_of_birth"] == "03.02.1985"
    assert data["meta"]["engine"] == "russian-docs-ocr+paddleocr"
    paddle_run.assert_awaited_once()
    assert paddle_run.await_args.args == (
        "field_text_cyrillic",
        b"birth-date-crop",
    )


def test_driver_license_does_not_guess_low_confidence_date_or_number(
    client, image_bytes
):
    result = _ocr(birth_date="03.02.198S")
    result["ocr"]["Licence_number"] = "11 22 34567S"
    result["crops"] = {"Birth_date": b"birth-date-crop"}
    with (
        patch(
            f"{SERVICE}.russian_runtime.run",
            new=AsyncMock(return_value=result),
        ),
        patch.object(type(settings), "is_model_enabled", return_value=True),
        patch(
            f"{SERVICE}.paddle_runtime.run",
            new=AsyncMock(
                return_value={"text": "03.02.1985", "confidence": 0.42}
            ),
        ),
    ):
        response = client.post(PATH, files=_upload(image_bytes))

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["date_of_birth"] is None
    assert data["license_series"] is None
    assert data["license_number"] is None


def test_driver_license_rejects_other_document(client, image_bytes):
    result = _ocr()
    result["doctype"] = "INTPASSPORT_2011"
    with patch(
        f"{SERVICE}.russian_runtime.run",
        new=AsyncMock(return_value=result),
    ):
        response = client.post(PATH, files=_upload(image_bytes))

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "OCR_FAILED"


def test_date_recheck_rejects_conflicting_legible_digits():
    assert reconcile_date_reading("03.02.198S", "04.02.1955") is None
    assert reconcile_date_reading("03.02.198S", "03.02.1955") is None


def test_adjacent_dates_are_not_silently_accepted_as_one_field():
    fields = driver_license_fields(
        {"Expiration_date": "04.05.2020 04.05.2030"}
    )
    assert fields["expiry_date"] is None
    assert (
        reconcile_date_reading("04.05.202S", "04.05.2020 04.05.2030") is None
    )


def test_driver_license_openapi_has_one_image_and_russia_tag():
    operation = app.openapi()["paths"][PATH]["post"]
    assert operation["tags"] == ["Russia"]
    schema = operation["requestBody"]["content"]["multipart/form-data"]
    assert "schema" in schema
    assert "DriverLicenseResponseSchema" in str(operation["responses"]["200"])
