from __future__ import annotations

from io import BytesIO
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import settings
from app.core.errors import AppError, ErrorCode
from app.main import app

PATH = "/v1/ocr/full"
SERVICE = "app.modules.v1.full.service"
MODELS = ("paddle", "qwen3-vl-4b", "qwen3-vl-8b")
TEXT = "Заголовок: Пример № 12\nA mixed Case — 01.02.2026\nP<RUSABCD<<NAME<<<<"


@pytest.fixture
def client():
    # Do not start real model processes or the database/queue lifecycle.
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
    with (
        patch(
            f"{SERVICE}.paddle_runtime.run", new_callable=AsyncMock
        ) as paddle,
        patch(f"{SERVICE}.recognize_vlm_image", new_callable=AsyncMock) as vlm,
    ):
        paddle.return_value = {"text": TEXT, "raw_text_count": 3}
        vlm.return_value = {
            "items": [{"text": TEXT}],
            "image_width": 64,
            "image_height": 48,
        }
        yield {"paddle": paddle, "vlm": vlm}


def upload(client: TestClient, content: bytes, model: str | None = "paddle"):
    return client.post(
        PATH,
        files={"image": ("scan.png", content, "image/png")},
        data={} if model is None else {"model": model},
    )


@pytest.mark.parametrize("model", MODELS)
def test_full_returns_all_text_synchronously_with_model_and_duration(
    client,
    image_content,
    engines,
    model,
):
    with patch(f"{SERVICE}.perf_counter", side_effect=[10.0, 10.375]):
        response = upload(client, image_content, model)

    assert response.status_code == 200, response.text
    assert response.json() == {
        "success": True,
        "data": {
            "text": TEXT,
            "meta": {
                "engine": "paddleocr" if model == "paddle" else "qwen3-vl",
                "model": model,
                "duration_seconds": 0.375,
            },
        },
        "error": None,
    }
    if model == "paddle":
        engines["paddle"].assert_awaited_once()
        args, kwargs = engines["paddle"].await_args
        assert args == ("full", image_content)
        assert (
            kwargs["timeout_seconds"]
            == settings.paddle_ocr_full_hard_timeout_seconds
        )
        engines["vlm"].assert_not_awaited()
    else:
        engines["vlm"].assert_awaited_once_with(
            image_content,
            model=model,
            filename="scan.png",
            content_type="image/png",
            task="ocr",
        )
        engines["paddle"].assert_not_awaited()


def test_default_model_is_paddle(client, image_content, engines):
    response = upload(client, image_content, model=None)

    assert response.status_code == 200, response.text
    assert response.json()["data"]["meta"]["model"] == "paddle"
    engines["paddle"].assert_awaited_once()
    engines["vlm"].assert_not_awaited()


@pytest.mark.parametrize("model", MODELS[1:])
def test_qwen_joins_every_text_block_preserving_order_case_and_line_breaks(
    client,
    image_content,
    engines,
    model,
):
    blocks = ["Коротко", "Mixed CASE 123\nВторая строка", "Коротко", "x"]
    engines["vlm"].return_value = {
        "items": [{"text": text, "confidence": None} for text in blocks],
    }

    response = upload(client, image_content, model)

    assert response.status_code == 200, response.text
    assert response.json()["data"]["text"] == "\n".join(blocks)
    assert set(response.json()["data"]) == {"text", "meta"}


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    ("code", "status"),
    [
        (ErrorCode.OCR_ENGINE_UNAVAILABLE, 503),
        (ErrorCode.OCR_TIMEOUT, 504),
        (ErrorCode.OCR_FAILED, 422),
    ],
)
def test_engine_errors_keep_standard_error_response(
    client,
    image_content,
    engines,
    model,
    code,
    status,
):
    engine = engines["paddle" if model == "paddle" else "vlm"]
    engine.side_effect = AppError(code, "Ошибка тестового движка")

    response = upload(client, image_content, model)

    assert response.status_code == status
    assert response.json() == {
        "success": False,
        "data": None,
        "error": {"code": code.value, "message": "Ошибка тестового движка"},
    }
    engine.assert_awaited_once()
    engines["vlm" if model == "paddle" else "paddle"].assert_not_awaited()


@pytest.mark.parametrize(
    "result",
    [
        None,
        [],
        {},
        {"text": None},
        {"text": 123},
        {"text": ""},
        {"text": " \n\t "},
    ],
)
def test_paddle_empty_or_malformed_result_is_ocr_failure(
    client,
    image_content,
    engines,
    result,
):
    engines["paddle"].return_value = result

    response = upload(client, image_content, "paddle")

    assert response.status_code == 422, response.text
    assert response.json()["success"] is False
    assert response.json()["data"] is None
    assert response.json()["error"]["code"] == "OCR_FAILED"


@pytest.mark.parametrize("model", MODELS[1:])
@pytest.mark.parametrize(
    "result",
    [
        None,
        {},
        {"items": "not-a-list"},
        {"items": []},
        {"items": [{}]},
        {"items": [{"text": None}]},
        {"items": [{"text": 123}]},
        {"items": [{"text": ""}, {"text": " \n\t "}]},
        {"items": [{"text": "Распознано"}, {"text": None}]},
    ],
)
def test_qwen_empty_or_malformed_result_is_ocr_failure(
    client,
    image_content,
    engines,
    model,
    result: Any,
):
    engines["vlm"].return_value = result

    response = upload(client, image_content, model)

    assert response.status_code == 422, response.text
    assert response.json()["success"] is False
    assert response.json()["data"] is None
    assert response.json()["error"]["code"] == "OCR_FAILED"


@pytest.mark.parametrize("model", ["mrz", "russian-ocr", "surya", "unknown"])
def test_invalid_model_is_rejected_before_recognition(
    client, image_content, engines, model
):
    response = upload(client, image_content, model)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    engines["paddle"].assert_not_awaited()
    engines["vlm"].assert_not_awaited()


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    ("content", "code"),
    [(b"", "EMPTY_IMAGE"), (b"not an image", "IMAGE_READ_ERROR")],
)
def test_invalid_upload_is_rejected_before_recognition(
    client, engines, model, content, code
):
    response = upload(client, content, model)

    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == code
    engines["paddle"].assert_not_awaited()
    engines["vlm"].assert_not_awaited()


def test_openapi_documents_one_image_and_exactly_three_models():
    schema = app.openapi()
    operation = schema["paths"][PATH]["post"]
    ref = operation["requestBody"]["content"]["multipart/form-data"]["schema"][
        "$ref"
    ]
    body = schema["components"]["schemas"][ref.rsplit("/", 1)[-1]]
    assert body["required"] == ["image"]
    assert set(body["properties"]) == {"image", "model"}
    assert body["properties"]["image"]["type"] == "string"
    assert body["properties"]["image"]["format"] == "binary"
    model = body["properties"]["model"]
    assert model["default"] == "paddle"
    choices = schema["components"]["schemas"][model["$ref"].rsplit("/", 1)[-1]]
    assert choices["enum"] == list(MODELS)
    assert {"200", "400", "413", "422", "503", "504"} <= set(
        operation["responses"]
    )
