from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import settings
from app.core.errors import AppError, ErrorCode
from app.main import app

PATH = "/v1/ocr/passport/international"
INTERNAL_PATH = "/v1/ocr/passport/internal"
SERVICE = "app.modules.v1.passport.service"
MRZ = [
    "P<RUSTESTOV<<IVAN<IVANOVICH".ljust(44, "<"),
    "1234567890RUS9001158M3001156".ljust(44, "<"),
]


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


def files(content: bytes):
    return {"image": ("synthetic.png", content, "image/png")}


def test_international_endpoint_returns_visual_fields_and_original_mrz(
    client, image_bytes
):
    result = {
        "mrz": MRZ,
        "items": [{"text": "private diagnostic"}],
        "passport": {
            "surname": "ТЕСТОВ",
            "name": "ИВАН",
            "patronymic": "ИВАНОВИЧ",
            "surname_latin": "TESTOV",
            "name_latin": "IVAN",
            "patronymic_latin": "IVANOVICH",
            "sex": "M",
            "date_of_birth": "1990-01-15",
            "birth_place_ru": "Г. ПРИМЕРСК",
            "birth_place_en": "PRIMERSK",
            "passport_number": "12 3456789",
            "issue_date": "2020-04-05",
            "expiry_date": "2030-04-05",
            "Issue_organization_ru": "УПРАВЛЕНИЕ ПРИМЕРА",
            "issue_organization_code": "TEST001",
            "internal_only": "must not leak",
        },
    }
    with (
        patch(
            f"{SERVICE}.russian_runtime.run",
            new=AsyncMock(return_value=result),
        ) as russian_run,
        patch(f"{SERVICE}.perf_counter", side_effect=[10.0, 10.4]),
    ):
        response = client.post(PATH, files=files(image_bytes))

    assert response.status_code == 200, response.text
    assert response.json() == {
        "success": True,
        "data": {
            "mrz": MRZ,
            "passport_type": "international",
            "document_code": "P",
            "issuing_country": "RUS",
            "surname": "ТЕСТОВ",
            "name": "ИВАН",
            "patronymic": "ИВАНОВИЧ",
            "surname_latin": "TESTOV",
            "name_latin": "IVAN",
            "patronymic_latin": "IVANOVICH",
            "sex": "МУЖ.",
            "date_of_birth": "15.01.1990",
            "birth_place_ru": "Г. ПРИМЕРСК",
            "birth_place_en": "PRIMERSK",
            "passport_number": "123456789",
            "nationality": "RUS",
            "issue_date": "05.04.2020",
            "expiry_date": "05.04.2030",
            "Issue_organization_ru": "УПРАВЛЕНИЕ ПРИМЕРА",
            "issue_organization_code": "TEST001",
            "meta": {
                "engine": "russian-docs-ocr",
                "model": "russian-ocr",
                "duration_seconds": 0.40000000000000036,
            },
        },
        "error": None,
    }
    russian_run.assert_awaited_once_with(
        "passport_international",
        image_bytes,
        timeout_seconds=settings.russian_ocr_document_hard_timeout_seconds,
    )


def test_international_endpoint_keeps_unread_fields_null(client, image_bytes):
    with patch(
        f"{SERVICE}.russian_runtime.run",
        new=AsyncMock(return_value={"mrz": [MRZ[0]], "passport": {}}),
    ):
        response = client.post(PATH, files=files(image_bytes))

    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["mrz"] is None
    assert data["passport_type"] == "international"
    assert data["document_code"] == "P"
    assert data["issuing_country"] == data["nationality"] == "RUS"
    for field in (
        "surname",
        "name",
        "patronymic",
        "surname_latin",
        "name_latin",
        "patronymic_latin",
        "sex",
        "date_of_birth",
        "birth_place_ru",
        "birth_place_en",
        "passport_number",
        "issue_date",
        "expiry_date",
        "Issue_organization_ru",
        "issue_organization_code",
    ):
        assert data[field] is None, field


@pytest.mark.parametrize(
    ("upload", "status", "code"),
    [
        ({}, 400, "NO_IMAGE"),
        ({"image": ("scan.png", b"", "image/png")}, 400, "EMPTY_IMAGE"),
        (
            {"image": ("scan.pdf", b"file", "application/pdf")},
            400,
            "UNSUPPORTED_FORMAT",
        ),
        (
            {"image": ("scan.png", b"not-an-image", "image/png")},
            400,
            "IMAGE_READ_ERROR",
        ),
    ],
)
def test_invalid_international_upload_does_not_start_ocr(
    client, upload, status, code
):
    with patch(
        f"{SERVICE}.russian_runtime.run", new_callable=AsyncMock
    ) as run:
        response = client.post(PATH, files=upload)

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    run.assert_not_awaited()


def test_international_upload_size_limit_prevents_ocr(client, image_bytes):
    with (
        patch(
            f"{SERVICE}.settings",
            new=SimpleNamespace(ocr_max_image_size_bytes=3),
        ),
        patch(f"{SERVICE}.russian_runtime.run", new_callable=AsyncMock) as run,
    ):
        response = client.post(PATH, files=files(image_bytes))

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "IMAGE_TOO_LARGE"
    run.assert_not_awaited()


def test_international_runtime_error_keeps_api_envelope(client, image_bytes):
    with patch(
        f"{SERVICE}.russian_runtime.run",
        new=AsyncMock(
            side_effect=AppError(ErrorCode.OCR_TIMEOUT, "OCR timed out")
        ),
    ):
        response = client.post(PATH, files=files(image_bytes))

    assert response.status_code == 504
    assert response.json() == {
        "success": False,
        "data": None,
        "error": {"code": "OCR_TIMEOUT", "message": "OCR timed out"},
    }


def test_international_endpoint_does_not_change_internal_contract(
    client, image_bytes
):
    with patch(
        f"{SERVICE}.russian_runtime.run",
        new=AsyncMock(
            return_value={"passport": {"passport_number": "123456789"}}
        ),
    ) as run:
        response = client.post(INTERNAL_PATH, files=files(image_bytes))

    assert response.status_code == 200
    assert response.json()["data"]["passport_type"] == "internal"
    assert response.json()["data"]["passport_number"] is None
    run.assert_awaited_once_with(
        "passport_text",
        image_bytes,
        timeout_seconds=settings.russian_ocr_document_hard_timeout_seconds,
    )


def test_international_openapi_is_under_russia_with_image_only():
    schema = app.openapi()
    operation = schema["paths"][PATH]["post"]
    assert operation["tags"] == ["Russia"]
    request_schema = operation["requestBody"]["content"][
        "multipart/form-data"
    ]["schema"]
    body = schema["components"]["schemas"][
        request_schema["$ref"].rsplit("/", 1)[-1]
    ]
    assert set(body["properties"]) == {"image"}
    assert body["required"] == ["image"]
    assert (
        operation["responses"]["200"]["content"]["application/json"]["schema"][
            "$ref"
        ]
        == "#/components/schemas/PassportInternationalResponseSchema"
    )
