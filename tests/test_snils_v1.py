from __future__ import annotations

from io import BytesIO
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import settings
from app.main import app

PATH = "/v1/ocr/snils"
SERVICE = "app.modules.v1.russia.snils.service"


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


def test_snils_uses_paddle_on_detected_crops_and_recovers_number(
    client, image_bytes
):
    russian_result = {
        "doctype": "SNILS_1996",
        "ocr": {
            "Licence_number": "789-123 00",  # Word splitting lost the prefix.
            "Last_name_ru": "ПРИМЕРОВ",
            "First_name_ru": "ИВАН",
            "Middle_name_ru": "ПРИМЕРОВИЧ",
            "Birth_date": "неразборчиво",
            "Birth_place_ru": "ПРИМЕРСК",
            "Sex_ru": "М",
            "Issue_date": "неразборчиво",
        },
        "crops": {
            "Licence_number": b"number-crop",
            "Birth_date": b"birth-crop",
            "Issue_date": b"issue-crop",
        },
    }
    paddle_text = {
        b"number-crop": "456-789-123 00",
        b"birth-crop": "2 января 1990",
        b"issue-crop": "10 января 2011 года",
    }

    async def read_crop(operation, content, *, timeout_seconds):
        assert operation == "field_text_cyrillic"
        assert timeout_seconds == settings.paddle_ocr_full_hard_timeout_seconds
        return {"text": paddle_text[content], "confidence": 0.98}

    with (
        patch(
            f"{SERVICE}.russian_runtime.run",
            new=AsyncMock(return_value=russian_result),
        ) as russian_run,
        patch(
            f"{SERVICE}.paddle_runtime.run",
            new=AsyncMock(side_effect=read_crop),
        ) as paddle_run,
    ):
        response = client.post(PATH, files=files(image_bytes))

    assert response.status_code == 200, response.text
    assert response.json()["error"] is None
    data = response.json()["data"]
    assert data == {
        "snils_number": "456-789-123 00",
        "surname": "ПРИМЕРОВ",
        "name": "ИВАН",
        "patronymic": "ПРИМЕРОВИЧ",
        "date_of_birth": "02.01.1990",
        "birth_place": "ПРИМЕРСК",
        "sex": "МУЖ.",
        "registration_date": "10.01.2011",
        "meta": data["meta"],
    }
    assert data["meta"]["engine"] == "russian-docs-ocr+paddleocr"
    assert data["meta"]["model"] == "russian-ocr+cyrillic_PP-OCRv5_mobile_rec"
    assert data["meta"]["duration_seconds"] >= 0
    russian_run.assert_awaited_once_with(
        "snils",
        image_bytes,
        timeout_seconds=settings.russian_ocr_document_hard_timeout_seconds,
    )
    assert [call.args[1] for call in paddle_run.await_args_list] == [
        b"number-crop",
        b"birth-crop",
        b"issue-crop",
    ]


def test_snils_missing_date_crops_return_null_even_if_russian_ocr_has_dates(
    client, image_bytes
):
    russian_result = {
        "doctype": "SNILS_1996",
        "ocr": {
            "Licence_number": "456-789-123 00",
            "Birth_date": "2 ЯНВАРЯ 1990",
            "Issue_date": "10 ЯНВАРЯ 2011 ГОДА",
            "Sex_ru": "Ж",
        },
        "crops": {},
    }
    with (
        patch(
            f"{SERVICE}.russian_runtime.run",
            new=AsyncMock(return_value=russian_result),
        ),
        patch(
            f"{SERVICE}.paddle_runtime.run", new_callable=AsyncMock
        ) as paddle,
    ):
        response = client.post(PATH, files=files(image_bytes))

    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["snils_number"] == "456-789-123 00"
    assert data["date_of_birth"] is None
    assert data["registration_date"] is None
    assert data["sex"] == "ЖЕН."
    assert data["meta"]["engine"] == "russian-docs-ocr"
    assert data["meta"]["model"] == "russian-ocr"
    paddle.assert_not_awaited()


def test_snils_falls_back_to_complete_russian_number_if_paddle_misses_digits(
    client, image_bytes
):
    russian_result = {
        "doctype": "SNILS_1996",
        "ocr": {
            "Licence_number": "456-789-123 00",
            "Birth_date": "2 ЯНВАРЯ 1990",
        },
        "crops": {
            "Licence_number": b"number-crop",
            "Birth_date": b"birth-crop",
        },
    }
    readings = {
        b"number-crop": "789-123 00",
        b"birth-crop": "не удалось прочитать",
    }

    async def read_crop(_operation, content, **_kwargs):
        return {"text": readings[content]}

    with (
        patch(
            f"{SERVICE}.russian_runtime.run",
            new=AsyncMock(return_value=russian_result),
        ),
        patch(
            f"{SERVICE}.paddle_runtime.run",
            new=AsyncMock(side_effect=read_crop),
        ),
    ):
        response = client.post(PATH, files=files(image_bytes))

    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["snils_number"] == "456-789-123 00"
    assert data["date_of_birth"] is None
    assert data["registration_date"] is None


def test_snils_rejects_wrong_document_before_paddle(client, image_bytes):
    with (
        patch(
            f"{SERVICE}.russian_runtime.run",
            new=AsyncMock(
                return_value={
                    "doctype": "INTPassport_2011",
                    "ocr": {},
                    "crops": {},
                }
            ),
        ),
        patch(
            f"{SERVICE}.paddle_runtime.run", new_callable=AsyncMock
        ) as paddle,
    ):
        response = client.post(PATH, files=files(image_bytes))

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "OCR_FAILED"
    paddle.assert_not_awaited()


@pytest.mark.parametrize(
    ("upload", "status", "code"),
    [
        ({}, 400, "NO_IMAGE"),
        ({"image": ("synthetic.png", b"", "image/png")}, 400, "EMPTY_IMAGE"),
        (
            {"image": ("synthetic.pdf", b"not-an-image", "application/pdf")},
            400,
            "UNSUPPORTED_FORMAT",
        ),
        (
            {"image": ("synthetic.png", b"not-an-image", "image/png")},
            400,
            "IMAGE_READ_ERROR",
        ),
    ],
)
def test_snils_invalid_upload_does_not_start_ocr(client, upload, status, code):
    with (
        patch(
            f"{SERVICE}.russian_runtime.run", new_callable=AsyncMock
        ) as russian,
        patch(
            f"{SERVICE}.paddle_runtime.run", new_callable=AsyncMock
        ) as paddle,
    ):
        response = client.post(PATH, files=upload)

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    russian.assert_not_awaited()
    paddle.assert_not_awaited()


def test_snils_openapi_is_under_russia_and_accepts_one_image_only():
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
        == "#/components/schemas/SnilsResponseSchema"
    )
