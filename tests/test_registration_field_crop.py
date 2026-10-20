from __future__ import annotations

import base64
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.modules.russian.inference_worker import russian_inference_worker
from app.modules.russian.registration_fields import (
    detect_registration_fields_bytes,
)
from app.ocr_engine.russian.registration_fields import (
    detect_registration_fields,
)

PATH = "/v1/ocr/crop/registration-fields-test"
SERVICE = "app.modules.v1.crop.registration_fields"


def stub_pipeline(doctype="INTPASSPORTADDR_ALL"):
    image = np.full((160, 240, 3), (210, 140, 90), dtype=np.uint8)
    image[60:80, 60:150] = (20, 30, 40)
    pipeline = Mock()
    pipeline.process_img.return_value = SimpleNamespace(
        img_with_fixed_perspective=image,
        meta_results={
            "Angle90": {"confidence": 0.99},
            "DocDetector": {"segm": [object()]},
        },
        doctype=doctype,
        documents=[{}],
        angle=90,
    )
    for name, box in (
        ("address_lines", [110, 75, 100, 20, 0.2, 0.95, 0, "Address"]),
        (
            "registration_stamps",
            [110, 80, 150, 100, 0.2, 0.94, 0, "Stamp_registered"],
        ),
    ):
        detector = getattr(pipeline, name)
        detector.model_name = name
        detector.predict.return_value = {name: {"obbox": [box]}}
    return pipeline


def image_bytes():
    stream = BytesIO()
    Image.new("RGB", (240, 160), "white").save(stream, "PNG")
    return stream.getvalue()


def test_detector_returns_upright_crops_and_geometry_without_ocr():
    pipeline = stub_pipeline()
    result = detect_registration_fields(
        np.zeros((160, 240, 3), dtype=np.uint8), pipeline, padding=4
    )
    assert result["document_type"] == "INTPASSPORTADDR_ALL"
    line, stamp = result["fields"]
    assert line["image"].shape == (28, 108, 3)
    assert stamp["image"].shape == (108, 158, 3)
    assert line["quad"] != line["crop_quad"]
    assert np.asarray(line["quad"]).shape == (4, 2)
    assert line["rotation_degrees"] == pytest.approx(np.degrees(0.2))
    assert [line["kind"], stamp["kind"]] == [
        "address_line",
        "registration_stamp",
    ]
    assert stamp["label"] == "Штамп регистрации"
    assert pipeline.process_img.call_args.kwargs == {
        "ocr": False,
        "find_text_fields": False,
        "check_quality": False,
    }
    pipeline.words_detector.predict_transform.assert_not_called()
    pipeline.ocr_cyr.predict.assert_not_called()
    pipeline.address_textkind.predict.assert_not_called()


def test_other_document_does_not_load_registration_detectors():
    pipeline = stub_pipeline("INTPASSPORT_2011")
    result = detect_registration_fields(
        np.zeros((160, 240, 3), dtype=np.uint8), pipeline
    )
    assert result["fields"] == []
    pipeline._ensure_address_models.assert_not_called()
    pipeline.address_lines.predict.assert_not_called()


def test_adapter_returns_only_serializable_lossless_previews():
    pipeline = stub_pipeline()
    result = detect_registration_fields_bytes(image_bytes(), pipeline=pipeline)
    expected = detect_registration_fields(
        np.zeros((160, 240, 3), dtype=np.uint8), pipeline
    )
    for field, raw in zip(result["fields"], expected["fields"], strict=True):
        assert field["image"].startswith("data:image/png;base64,")
        with Image.open(
            BytesIO(base64.b64decode(field["image"].split(",", 1)[1]))
        ) as decoded:
            np.testing.assert_array_equal(np.asarray(decoded), raw["image"])
    assert result["document"]["image"].startswith("data:image/jpeg;base64,")


@pytest.fixture
def client():
    client = TestClient(app)
    yield client
    client.close()


@pytest.fixture
def runtime():
    result = detect_registration_fields_bytes(
        image_bytes(), pipeline=stub_pipeline()
    )
    with patch(
        f"{SERVICE}.russian_runtime.run",
        new_callable=AsyncMock,
        return_value=result,
    ) as runtime:
        yield runtime


def test_endpoint_defaults_to_crops_only(client, runtime):
    response = client.post(
        PATH, files={"image": ("registration.png", image_bytes(), "image/png")}
    )
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    data = response.json()["data"]
    assert len(data["fields"]) == 2
    assert data["meta"]["ocr"] is None
    assert all(field["ocr"] is None for field in data["fields"])
    assert runtime.await_args.args[0] == "registration_fields"
    assert runtime.await_args.kwargs["padding"] == 4


def test_selected_ocr_receives_exact_displayed_pngs(client, runtime):
    with patch(
        "app.modules.v1.crop.field_ocr.paddle_runtime.run",
        new_callable=AsyncMock,
        return_value={"text": "ТЕСТ", "confidence": 0.9, "device": "cpu"},
    ) as recognize:
        response = client.post(
            PATH,
            files={"image": ("registration.png", image_bytes(), "image/png")},
            data={"ocr_model": "paddle-cyrillic", "padding": "0"},
        )
    assert response.status_code == 200, response.text
    fields = response.json()["data"]["fields"]
    assert recognize.await_count == 2
    for field, call in zip(fields, recognize.await_args_list, strict=True):
        assert call.args == (
            "field_text_cyrillic",
            base64.b64decode(field["image"].split(",", 1)[1]),
        )
        assert field["ocr"]["text"] == "ТЕСТ"
    assert runtime.await_args.kwargs["padding"] == 0


@pytest.mark.parametrize(
    ("content", "padding", "status"),
    [(b"broken", "4", 400), (image_bytes(), "-1", 422)],
)
def test_invalid_input_is_rejected_before_detection(
    client, runtime, content, padding, status
):
    response = client.post(
        PATH,
        files={"image": ("image.png", content, "image/png")},
        data={"padding": padding},
    )
    assert response.status_code == status
    runtime.assert_not_awaited()


def test_registration_worker_reuses_pipeline():
    with (
        patch(
            "app.modules.russian.inference_worker.serve_inference_worker"
        ) as serve,
        patch(
            "app.modules.russian.registration_fields."
            "detect_registration_fields_bytes"
        ) as detect,
    ):
        russian_inference_worker(Mock(), "cpu")
        pipeline = Mock()
        dispatch = serve.call_args.args[2]
        dispatch(pipeline, "registration_fields", (b"png",), {"padding": 5})
        detect.assert_called_once_with(b"png", pipeline=pipeline, padding=5)


def test_registration_crop_has_separate_swagger_route(client):
    schema = client.get("/openapi.json").json()
    operation = schema["paths"][PATH]["post"]
    assert operation["tags"] == ["Crop"]
    assert "/v1/ocr/crop/document-fields" in schema["paths"]
    ref = operation["requestBody"]["content"]["multipart/form-data"]["schema"][
        "$ref"
    ]
    body = schema["components"]["schemas"][ref.rsplit("/", 1)[1]]
    assert body["required"] == ["image"]
    assert set(body["properties"]) == {"image", "padding", "ocr_model"}
