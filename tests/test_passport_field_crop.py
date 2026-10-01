from __future__ import annotations

import base64
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import settings
from app.core.errors import AppError, ErrorCode
from app.main import app
from app.modules.russian.fields import detect_passport_fields_bytes
from app.modules.russian.inference_worker import russian_inference_worker
from app.modules.v1.crop.passport_fields import crop_passport_fields_upload

PATH = "/v1/ocr/crop/passport-fields"
SERVICE = "app.modules.v1.crop.passport_fields"
WORKER = "app.modules.russian.fields"


def image_bytes(image: Image.Image | None = None, **kwargs) -> bytes:
    output = BytesIO()
    if image is None:
        image = Image.new("RGB", (80, 60), "white")
    image.save(output, format="PNG", **kwargs)
    return output.getvalue()


def detector_result() -> dict:
    image = np.full((60, 80, 3), (220, 15, 35), dtype=np.uint8)
    return {
        "document_image": image,
        "document_detected": True,
        "rotation_degrees": 0,
        "orientation_confidence": 0.98,
        "fields": [
            {
                "name": "Last_name_ru",
                "bbox": [6, 6, 54, 24],
                "detection_bbox": [10, 10, 50, 20],
                "confidence": 0.95,
                "rotation_degrees": 0,
                "image": image[6:24, 6:54],
            }
        ],
        "timings": {"orientation": 0.01, "document": 0.02, "fields": 0.03},
    }


@pytest.fixture
def client():
    # Real models and queue infrastructure are not started by this client.
    client = TestClient(app)
    yield client
    client.close()


@pytest.fixture
def engine():
    with patch(f"{WORKER}.detect_passport_fields") as detector:
        detector.return_value = detector_result()
        result = detect_passport_fields_bytes(image_bytes(), pipeline=Mock())
    with patch(
        f"{SERVICE}.russian_runtime.run", new_callable=AsyncMock
    ) as runtime:
        runtime.return_value = result
        yield runtime


def test_endpoint_returns_previews_coordinates_and_detector_metadata(
    client, engine
):
    content = image_bytes()
    response = client.post(
        PATH, files={"image": ("page.png", content, "image/png")}
    )

    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    data = response.json()["data"]
    assert data["document"]["width"] == 80
    assert data["document"]["height"] == 60
    field = data["fields"][0]
    assert field["label"] == "Фамилия"
    assert field["bbox"] == [6, 6, 54, 24]
    assert field["detection_bbox"] == [10, 10, 50, 20]
    assert (field["width"], field["height"]) == (48, 18)
    assert "text" not in field
    assert data["meta"]["engine"] == "russian-docs-ocr"
    assert data["meta"]["model"] == "text-fields-detector"
    assert data["meta"]["duration_seconds"] >= 0
    assert data["meta"]["timings"]["fields"] == 0.03
    engine.assert_awaited_once_with(
        "passport_fields",
        content,
        timeout_seconds=settings.russian_ocr_document_hard_timeout_seconds,
        crop_document=True,
        padding=4,
    )


def test_endpoint_passes_crop_and_padding_controls_and_accepts_no_fields(
    client, engine
):
    engine.return_value["fields"] = []
    engine.return_value["document"]["detected"] = False
    response = client.post(
        PATH,
        files={"image": ("page.png", image_bytes(), "image/png")},
        data={"crop_document": "false", "padding": "0"},
    )

    assert response.status_code == 200
    assert response.json()["data"]["fields"] == []
    assert response.json()["data"]["document"]["detected"] is False
    assert engine.await_args.kwargs["crop_document"] is False
    assert engine.await_args.kwargs["padding"] == 0


@pytest.mark.parametrize("padding", [-1, 65, "not-a-number"])
def test_endpoint_rejects_invalid_padding_before_inference(
    client, engine, padding
):
    response = client.post(
        PATH,
        files={"image": ("page.png", image_bytes(), "image/png")},
        data={"padding": str(padding)},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    engine.assert_not_awaited()


@pytest.mark.parametrize(
    ("filename", "content", "code"),
    [
        ("page.png", b"", "EMPTY_IMAGE"),
        ("page.png", b"broken", "IMAGE_READ_ERROR"),
        ("page.pdf", b"pdf", "UNSUPPORTED_FORMAT"),
    ],
)
def test_endpoint_rejects_bad_upload_before_inference(
    client, engine, filename, content, code
):
    response = client.post(
        PATH, files={"image": (filename, content, "image/png")}
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == code
    engine.assert_not_awaited()


@pytest.mark.parametrize(
    ("code", "status"),
    [
        (ErrorCode.OCR_ENGINE_UNAVAILABLE, 503),
        (ErrorCode.OCR_TIMEOUT, 504),
        (ErrorCode.OCR_FAILED, 422),
    ],
)
def test_endpoint_preserves_runtime_errors(client, engine, code, status):
    engine.side_effect = AppError(code, "Тестовая ошибка")
    response = client.post(
        PATH, files={"image": ("page.png", image_bytes(), "image/png")}
    )

    assert response.status_code == status
    assert response.json()["error"]["code"] == code.value
    assert response.json()["data"] is None


def test_disabled_model_closes_upload_without_running_detector():
    import asyncio

    upload = SimpleNamespace(close=AsyncMock())
    with patch(
        f"{SERVICE}.ensure_ocr_model_enabled",
        side_effect=AppError(ErrorCode.OCR_ENGINE_UNAVAILABLE, "Отключён"),
    ):
        with pytest.raises(AppError):
            asyncio.run(crop_passport_fields_upload(upload))
    upload.close.assert_awaited_once()


def test_preview_encoding_preserves_rgb_and_uses_exif_orientation():
    image = Image.new("RGB", (80, 60), "white")
    exif = Image.Exif()
    exif[274] = 6
    pipeline = Mock()
    with patch(
        f"{WORKER}.detect_passport_fields", return_value=detector_result()
    ) as detect:
        result = detect_passport_fields_bytes(
            image_bytes(image, exif=exif), pipeline, padding=8
        )

    assert detect.call_args.args[0].shape == (80, 60, 3)
    assert detect.call_args.kwargs["pipeline"] is pipeline
    assert detect.call_args.kwargs["padding"] == 8
    encoded = result["fields"][0]["image"].split(",", 1)[1]
    with Image.open(BytesIO(base64.b64decode(encoded))) as preview:
        assert preview.size == (48, 18)
        rgb = np.asarray(preview).mean(axis=(0, 1))
        assert np.abs(rgb - [220, 15, 35]).max() < 3


def test_worker_reuses_existing_pipeline_for_field_detection():
    with (
        patch(
            "app.modules.russian.inference_worker.serve_inference_worker"
        ) as serve,
        patch(f"{WORKER}.detect_passport_fields_bytes") as detect,
        patch(
            "app.modules.russian.passport.recognize_passport_text_bytes"
        ) as ocr,
    ):
        russian_inference_worker(Mock(), "cpu")
        dispatch = serve.call_args.args[2]
        pipeline = Mock()
        dispatch(
            pipeline,
            "passport_fields",
            (b"image",),
            {"crop_document": False, "padding": 8},
        )

    detect.assert_called_once_with(
        b"image", pipeline=pipeline, crop_document=False, padding=8
    )
    ocr.assert_not_called()
