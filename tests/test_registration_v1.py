from __future__ import annotations

from io import BytesIO
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import settings
from app.main import app

PATH = "/v1/ocr/passport/registration_test"
SERVICE = "app.modules.v1.russia.registration.service"


@pytest.fixture
def client():
    test_client = TestClient(app)
    yield test_client
    test_client.close()


@pytest.fixture
def image_bytes():
    stream = BytesIO()
    Image.new("RGB", (120, 80), "white").save(stream, "PNG")
    return stream.getvalue()


def registration_result():
    return {
        "doctype": "INTPASSPORTADDR_ALL",
        "address": "Г. ПРИМЕРСК\n[рукопись]",
        "has_handwritten": True,
        "address_lines": [
            {
                "kind": "printed",
                "text": "Г. ПРИМЕРСК",
                "p_handwritten": 0.01,
                "stamp": 0,
                "quad": [[10, 20], [90, 20], [90, 30], [10, 30]],
            },
            {
                "kind": "handwritten",
                "text": None,
                "p_handwritten": 0.98,
                "stamp": 0,
                "quad": None,
            },
        ],
        "registration_stamps": [
            {
                "kind": "registered",
                "confidence": 0.95,
                "quad": [[5, 5], [100, 5], [100, 60], [5, 60]],
                "lines": [0, 1],
                "address": "Г. ПРИМЕРСК\n[рукопись]",
                "cancels": None,
                "current": True,
            }
        ],
    }


def test_registration_returns_address_lines_and_stamps(client, image_bytes):
    with patch(
        f"{SERVICE}.russian_runtime.run",
        new_callable=AsyncMock,
        return_value=registration_result(),
    ) as recognize:
        response = client.post(
            PATH, files={"image": ("synthetic.png", image_bytes, "image/png")}
        )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    payload = response.json()
    assert payload["success"] is True
    assert payload["error"] is None
    data = payload["data"]
    assert data["document_type"] == "INTPASSPORTADDR_ALL"
    assert data["address"] == "Г. ПРИМЕРСК\n[рукопись]"
    assert data["has_handwritten"] is True
    assert data["address_lines"][1]["text"] is None
    assert data["address_lines"][1]["quad"] is None
    assert data["address_lines"][0]["stamp"] == 0
    assert data["registration_stamps"][0]["current"] is True
    assert data["meta"]["engine"] == "russian-docs-ocr"
    assert data["meta"]["model"] == "russian-ocr"
    assert data["meta"]["duration_seconds"] >= 0
    recognize.assert_awaited_once_with(
        "passport_registration",
        image_bytes,
        timeout_seconds=settings.russian_ocr_document_hard_timeout_seconds,
    )


@pytest.mark.parametrize("doctype", ["NONE", "INTPASSPORT", "SNILS"])
def test_registration_rejects_another_document(client, image_bytes, doctype):
    result = registration_result()
    result["doctype"] = doctype
    with patch(
        f"{SERVICE}.russian_runtime.run",
        new_callable=AsyncMock,
        return_value=result,
    ):
        response = client.post(
            PATH, files={"image": ("synthetic.png", image_bytes, "image/png")}
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "OCR_FAILED"
    assert response.json()["data"] is None


@pytest.mark.parametrize(
    "result",
    [
        None,
        {"doctype": "INTPASSPORTADDR_ALL"},
        {
            **registration_result(),
            "address_lines": [{"kind": "printed", "p_handwritten": 2}],
        },
    ],
)
def test_registration_rejects_malformed_result(client, image_bytes, result):
    with patch(
        f"{SERVICE}.russian_runtime.run",
        new_callable=AsyncMock,
        return_value=result,
    ):
        response = client.post(
            PATH, files={"image": ("synthetic.png", image_bytes, "image/png")}
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "OCR_FAILED"


def test_registration_validates_image_before_inference(client):
    with patch(
        f"{SERVICE}.russian_runtime.run", new_callable=AsyncMock
    ) as recognize:
        response = client.post(
            PATH, files={"image": ("broken.png", b"not an image", "image/png")}
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "IMAGE_READ_ERROR"
    recognize.assert_not_awaited()


def test_registration_openapi_requires_only_image(client):
    schema = client.get("/openapi.json").json()
    operation = schema["paths"][PATH]["post"]
    assert operation["tags"] == ["Russia"]
    body_ref = operation["requestBody"]["content"]["multipart/form-data"][
        "schema"
    ]["$ref"]
    body = schema["components"]["schemas"][body_ref.rsplit("/", 1)[1]]
    assert body["required"] == ["image"]
    assert set(body["properties"]) == {"image"}
