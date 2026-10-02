from __future__ import annotations

import asyncio
from dataclasses import replace
from io import BytesIO
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import settings
from app.core.errors import AppError, ErrorCode
from app.main import app

PATH = "/v1/ocr/full_test"
SERVICE = "app.modules.v1.full.russian_service"
TEXT = "  Заголовок: Пример № 12\nMixed CASE  01.02.2026\n\tСтрока. \n"


@pytest.fixture
def client():
    # Keep real model processes and the database/queue lifecycle stopped.
    client = TestClient(app)
    yield client
    client.close()


@pytest.fixture
def image_content() -> bytes:
    output = BytesIO()
    Image.new("RGB", (64, 48), "white").save(output, format="PNG")
    return output.getvalue()


@pytest.fixture
def engines():
    # Patch the shared runtime objects where they are defined. The test
    # endpoint need not import either alternative engine to prove isolation.
    with (
        patch(
            "app.modules.russian.runtime.russian_runtime.run",
            new_callable=AsyncMock,
        ) as russian,
        patch(
            "app.modules.paddle.runtime.paddle_runtime.run",
            new_callable=AsyncMock,
        ) as paddle,
        patch(
            "app.modules.v1.full.service.recognize_vlm_image",
            new_callable=AsyncMock,
        ) as vlm,
    ):
        russian.return_value = {"text": TEXT}
        paddle.return_value = {"text": TEXT}
        vlm.return_value = {"items": [{"text": TEXT}]}
        yield {"russian": russian, "paddle": paddle, "vlm": vlm}


def upload(client, content, **data):
    return client.post(
        PATH,
        files={"image": ("scan.png", content, "image/png")},
        data=data,
    )


def test_returns_raw_text_and_fixed_hybrid_metadata(
    client, image_content, engines
):
    with (
        patch(f"{SERVICE}.perf_counter", side_effect=[10.0, 10.375]),
        patch(
            f"{SERVICE}.settings",
            replace(settings, russian_ocr_full_hard_timeout_seconds=23.0),
        ),
        patch(f"{SERVICE}.ensure_ocr_model_enabled") as enabled,
    ):
        response = upload(client, image_content)

    assert response.status_code == 200, response.text
    assert response.json() == {
        "success": True,
        "data": {
            "text": TEXT,
            "meta": {
                "engine": "russian-docs-ocr",
                "model": "russian-ocr",
                "detector": "PP-OCRv5_mobile_det",
                "duration_seconds": 0.375,
            },
        },
        "error": None,
    }
    enabled.assert_called_once_with("russian-ocr")
    engines["russian"].assert_awaited_once_with(
        "full_text", image_content, timeout_seconds=23.0
    )
    engines["paddle"].assert_not_awaited()
    engines["vlm"].assert_not_awaited()


def test_only_russian_model_needs_to_be_enabled(
    client, image_content, engines
):
    with patch(
        "app.core.model_availability.settings",
        replace(settings, ocr_models="russian-ocr"),
    ):
        response = upload(client, image_content)

    assert response.status_code == 200, response.text
    assert response.json()["data"]["text"] == TEXT
    assert response.json()["data"]["meta"]["model"] == "russian-ocr"
    assert response.json()["data"]["meta"]["detector"] == "PP-OCRv5_mobile_det"
    engines["russian"].assert_awaited_once_with(
        "full_text",
        image_content,
        timeout_seconds=settings.russian_ocr_full_hard_timeout_seconds,
    )
    engines["paddle"].assert_not_awaited()
    engines["vlm"].assert_not_awaited()


@pytest.mark.parametrize("model", ["paddle", "qwen3-vl-8b", "unknown"])
def test_extra_fields_cannot_change_fixed_hybrid_dispatch(
    client, image_content, engines, model
):
    response = upload(
        client,
        image_content,
        model=model,
        detector="PP-OCRv5_server_det",
        crop="true",
    )

    assert response.status_code == 200, response.text
    assert response.json()["data"]["meta"]["model"] == "russian-ocr"
    assert response.json()["data"]["meta"]["detector"] == "PP-OCRv5_mobile_det"
    engines["russian"].assert_awaited_once_with(
        "full_text",
        image_content,
        timeout_seconds=settings.russian_ocr_full_hard_timeout_seconds,
    )
    engines["paddle"].assert_not_awaited()
    engines["vlm"].assert_not_awaited()


@pytest.mark.parametrize(
    "result",
    [
        None,
        [],
        {},
        {"text": None},
        {"text": 123},
        {"text": ["Текст"]},
        {"text": ""},
        {"text": " \n\t "},
    ],
)
def test_empty_or_wrong_text_is_an_ocr_failure(
    client, image_content, engines, result
):
    engines["russian"].return_value = result

    response = upload(client, image_content)

    assert response.status_code == 422, response.text
    body = response.json()
    assert body["success"] is False
    assert body["data"] is None
    assert body["error"]["code"] == "OCR_FAILED"
    engines["russian"].assert_awaited_once()


@pytest.mark.parametrize(
    ("code", "status"),
    [
        (ErrorCode.OCR_ENGINE_UNAVAILABLE, 503),
        (ErrorCode.OCR_TIMEOUT, 504),
        (ErrorCode.OCR_FAILED, 422),
    ],
)
def test_runtime_errors_keep_the_public_envelope(
    client, image_content, engines, code, status
):
    engines["russian"].side_effect = AppError(code, "Ошибка тестового движка")

    response = upload(client, image_content)

    assert response.status_code == status
    assert response.json() == {
        "success": False,
        "data": None,
        "error": {"code": code.value, "message": "Ошибка тестового движка"},
    }
    engines["russian"].assert_awaited_once()
    engines["paddle"].assert_not_awaited()
    engines["vlm"].assert_not_awaited()


def test_disabled_russian_model_fails_before_reading_or_dispatch(
    client, engines
):
    with patch(
        "app.core.model_availability.settings",
        replace(settings, ocr_models="paddle,mrz"),
    ):
        response = upload(client, b"not-read")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "OCR_ENGINE_UNAVAILABLE"
    assert "russian-ocr" in response.json()["error"]["message"]
    for engine in engines.values():
        engine.assert_not_awaited()


@pytest.mark.parametrize(
    ("filename", "content", "code"),
    [
        ("scan.pdf", b"pdf", "UNSUPPORTED_FORMAT"),
        ("scan.png", b"", "EMPTY_IMAGE"),
        ("scan.png", b"not an image", "IMAGE_READ_ERROR"),
    ],
)
def test_invalid_upload_is_rejected_before_recognition(
    client, engines, filename, content, code
):
    response = client.post(
        PATH, files={"image": (filename, content, "image/png")}
    )

    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == code
    engines["russian"].assert_not_awaited()


def test_missing_image_uses_standard_no_image_error(client, engines):
    response = client.post(PATH)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "NO_IMAGE"
    engines["russian"].assert_not_awaited()


def test_upload_size_limit_applies_before_decoding(client, engines):
    with patch(
        f"{SERVICE}.settings", replace(settings, ocr_max_image_size_mb=1)
    ):
        response = upload(client, b"x" * (1024 * 1024 + 1))

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "IMAGE_TOO_LARGE"
    engines["russian"].assert_not_awaited()


def test_image_dimensions_are_validated(client, image_content, engines):
    with patch(
        "app.core.images.validation.settings",
        replace(settings, ocr_max_image_width=32),
    ):
        response = upload(client, image_content)

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "IMAGE_TOO_LARGE"
    engines["russian"].assert_not_awaited()


@pytest.mark.parametrize(
    "case", ["success", "extension", "empty", "decode", "timeout"]
)
def test_service_closes_upload_on_success_and_failure(
    image_content, engines, case
):
    from app.modules.v1.full.russian_service import (
        recognize_full_russian_upload,
    )

    content = {"empty": b"", "decode": b"broken"}.get(case, image_content)
    image = UploadFile(
        filename="scan.pdf" if case == "extension" else "scan.png",
        file=BytesIO(content),
    )
    if case == "timeout":
        engines["russian"].side_effect = AppError(
            ErrorCode.OCR_TIMEOUT, "Таймаут"
        )

    if case == "success":
        result = asyncio.run(recognize_full_russian_upload(image))
        assert result["data"]["text"] == TEXT
    else:
        with pytest.raises(AppError):
            asyncio.run(recognize_full_russian_upload(image))
    assert image.file.closed


def test_openapi_has_only_image_and_fixed_hybrid_response():
    schema = app.openapi()

    def dereference(value):
        return schema["components"]["schemas"][
            value["$ref"].rsplit("/", 1)[-1]
        ]

    operation = schema["paths"][PATH]["post"]
    description = operation["description"]
    assert "PP-OCRv5_mobile_det" in description
    assert "Russian OCR" in description
    assert "accurate" in description
    assert "Paddle не используется" not in description
    body = dereference(
        operation["requestBody"]["content"]["multipart/form-data"]["schema"]
    )
    assert set(body["properties"]) == {"image"}
    assert body["required"] == ["image"]
    assert body["properties"]["image"]["format"] == "binary"
    response_ref = operation["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    assert response_ref["$ref"].endswith("/RussianFullOcrResponseSchema")
    response = dereference(response_ref)
    data = dereference(
        next(
            item
            for item in response["properties"]["data"]["anyOf"]
            if "$ref" in item
        )
    )
    meta = dereference(data["properties"]["meta"])
    assert meta["properties"]["engine"]["const"] == "russian-docs-ocr"
    assert meta["properties"]["model"]["const"] == "russian-ocr"
    assert meta["properties"]["detector"]["const"] == "PP-OCRv5_mobile_det"
    assert "detector" in meta["required"]
    assert {"200", "400", "413", "422", "503", "504"} <= set(
        operation["responses"]
    )


@pytest.mark.parametrize("model", ["paddle", "qwen3-vl-4b", "qwen3-vl-8b"])
def test_existing_full_endpoint_keeps_its_selected_engine(
    client, image_content, engines, model
):
    response = client.post(
        "/v1/ocr/full",
        files={"image": ("scan.png", image_content, "image/png")},
        data={"model": model},
    )

    assert response.status_code == 200, response.text
    assert response.json()["data"]["meta"]["model"] == model
    engines["russian"].assert_not_awaited()
    engines["paddle" if model == "paddle" else "vlm"].assert_awaited_once()
    engines["vlm" if model == "paddle" else "paddle"].assert_not_awaited()
