from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from app.modules.russian.inference_worker import russian_inference_worker
from app.modules.russian.registration import (
    _probability,
    recognize_registration_bytes,
)
from app.ocr_engine.russian.vendor.document_processing import Pipeline
from tests.russian.test_engine_adapter import image_bytes

MODEL_MODULE = (
    "app.ocr_engine.russian.vendor.document_processing.pipeline.pipeline"
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (-5.960464477539063e-08, 0.0),
        (1.0000001192092896, 1.0),
        (0.42, 0.42),
        (-0.1, -0.1),
        (1.1, 1.1),
    ],
)
def test_probability_only_corrects_float_roundoff(raw, expected):
    assert _probability(raw) == expected


def test_registration_adapter_excludes_images_and_serializes_geometry():
    result = SimpleNamespace(
        doctype="INTPASSPORTADDR_ALL",
        ocr={"Address": "Г. ПРИМЕРСК", "Address_has_handwritten": False},
        meta_results={
            "Address_lines": [
                {
                    "kind": "printed",
                    "text": "Г. ПРИМЕРСК",
                    "p_handwritten": np.float32(-5.960464477539063e-08),
                    "stamp": None,
                }
            ],
            "Image": np.zeros((100, 100, 3), dtype=np.uint8),
        },
        address_line_quads=[
            np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=np.float32)
        ],
        registration_stamps=[],
    )
    pipeline = Mock()
    pipeline.process_img.return_value = result
    data = recognize_registration_bytes(image_bytes(), pipeline=pipeline)
    assert data["address_lines"][0]["quad"] == [
        [0, 0],
        [10, 0],
        [10, 5],
        [0, 5],
    ]
    assert "Image" not in json.dumps(data)
    assert data["address_lines"][0]["p_handwritten"] == 0.0
    assert data["registration_stamps"] == []

    result.address_line_quads = None
    data = recognize_registration_bytes(image_bytes(), pipeline=pipeline)
    assert data["address_lines"][0]["quad"] is None


def test_registration_models_load_once_and_retry_after_failure():
    pipeline = object.__new__(Pipeline)
    pipeline.device = "cpu"
    pipeline._registration_verbose = False
    pipeline.address_lines = None
    pipeline.address_textkind = None
    pipeline.registration_stamps = None
    with (
        patch(f"{MODEL_MODULE}.AddressLinesDetector") as lines,
        patch(f"{MODEL_MODULE}.AddressTextKindClassifier") as textkind,
        patch(f"{MODEL_MODULE}.RegistrationStamps") as stamps,
    ):
        stamps.side_effect = RuntimeError("model unavailable")
        with pytest.raises(RuntimeError, match="model unavailable"):
            pipeline._ensure_address_models()
        assert pipeline.address_lines is None
        assert pipeline.address_textkind is None
        assert pipeline.registration_stamps is None

        stamps.side_effect = None
        pipeline._ensure_address_models()
        pipeline._ensure_address_models()
        assert (
            lines.call_count == textkind.call_count == stamps.call_count == 2
        )
        assert pipeline.address_lines is lines.return_value
        assert pipeline.address_textkind is textkind.return_value
        assert pipeline.registration_stamps is stamps.return_value
        stamps.assert_called_with(
            model_format="ONNX", device="cpu", verbose=False
        )


def test_registration_worker_reuses_resident_pipeline():
    with (
        patch(
            "app.modules.russian.inference_worker.serve_inference_worker"
        ) as serve,
        patch(
            "app.modules.russian.registration.recognize_registration_bytes"
        ) as recognize,
    ):
        russian_inference_worker(Mock(), "cpu")
        dispatch = serve.call_args.args[2]
        pipeline = Mock()
        for content in (b"first", b"second"):
            dispatch(pipeline, "passport_registration", (content,), {})
            recognize.assert_called_with(content, pipeline=pipeline)
