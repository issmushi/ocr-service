from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import cv2
import numpy as np
import pytest
from PIL import Image

from app.modules.paddle.fields import recognize_field_text_bytes
from app.modules.paddle.inference_worker import paddle_inference_worker


def image_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (41, 17), (220, 70, 20)).save(output, format="JPEG")
    return output.getvalue()


def recognizer(text: str, confidence: float):
    text_model = Mock()
    text_model.predict.return_value = [
        {"rec_text": text, "rec_score": confidence}
    ]
    pipeline = SimpleNamespace(
        text_rec_model=text_model,
        text_det_model=Mock(),
        textline_orientation_model=Mock(),
        doc_preprocessor=Mock(),
    )
    ocr = Mock(paddlex_pipeline=SimpleNamespace(_pipeline=pipeline))
    return ocr, pipeline


def test_field_reads_the_whole_crop_without_detection_or_orientation():
    ocr, pipeline = recognizer("ТЕСТОВ", 0.93)
    content = image_bytes()

    result = recognize_field_text_bytes(content, ocr=ocr)

    assert result == {"text": "ТЕСТОВ", "confidence": 0.93, "device": "cpu"}
    pipeline.text_rec_model.predict.assert_called_once()
    crops = pipeline.text_rec_model.predict.call_args.args[0]
    assert len(crops) == 1
    assert crops[0].shape == (17, 41, 3)
    expected = cv2.imdecode(
        np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    np.testing.assert_array_equal(crops[0], expected)
    # A red RGB upload reaches Paddle in OpenCV's BGR order.
    assert crops[0][0, 0, 2] > crops[0][0, 0, 0]
    pipeline.text_det_model.predict.assert_not_called()
    pipeline.textline_orientation_model.predict.assert_not_called()
    pipeline.doc_preprocessor.predict.assert_not_called()
    ocr.predict.assert_not_called()
    ocr.ocr.assert_not_called()


def test_empty_recognition_remains_empty():
    ocr, _ = recognizer("", 0.0)
    assert recognize_field_text_bytes(image_bytes(), ocr=ocr) == {
        "text": "",
        "confidence": 0.0,
        "device": "cpu",
    }


@pytest.mark.parametrize("content", [b"", b"not an image"])
def test_invalid_image_does_not_run_recognition(content):
    ocr, pipeline = recognizer("", 0.0)
    with pytest.raises(ValueError, match="Cannot read image"):
        recognize_field_text_bytes(content, ocr=ocr)
    pipeline.text_rec_model.predict.assert_not_called()


@pytest.mark.parametrize("device", ["cpu", "gpu:0"])
def test_worker_reuses_or_lazily_caches_the_cpu_field_engine(device):
    general, mrz, field_cpu = object(), object(), object()
    connection = object()
    expected = {"text": "ТЕСТОВ", "confidence": 0.93, "device": "cpu"}

    with (
        patch(
            "app.modules.paddle.inference_worker.serve_inference_worker"
        ) as serve,
        patch(
            "app.ocr_engine.paddle.factory.create_ocr",
            side_effect=[general, field_cpu],
        ) as create_general,
        patch(
            "app.ocr_engine.paddle.factory.create_mrz_ocr", return_value=mrz
        ) as create_mrz,
        patch(
            "app.modules.paddle.fields.recognize_field_text_bytes",
            return_value=expected,
        ) as read_field,
        patch(
            "app.modules.v1.full.paddle.recognize_full_bytes",
            return_value={"text": "full image", "raw_text_count": 1},
        ) as read_full,
    ):
        paddle_inference_worker(connection, device)
        actual_connection, initialize, dispatch = serve.call_args.args
        assert actual_connection is connection
        engine = initialize()
        assert engine == {"general": general, "mrz": mrz}
        create_general.assert_called_once_with(device=device)
        create_mrz.assert_called_once_with(device=device)

        content = image_bytes()
        for _ in range(2):
            assert dispatch(engine, "field_text", (content,), {}) == expected
        expected_engine = general if device == "cpu" else field_cpu
        assert read_field.call_args_list == [
            call(content, ocr=expected_engine),
            call(content, ocr=expected_engine),
        ]
        expected_calls = [call(device=device)]
        if device != "cpu":
            expected_calls.append(call(device="cpu"))
            assert engine["field_cpu"] is field_cpu
        assert create_general.call_args_list == expected_calls

        # A field request must not replace the engine for normal full OCR.
        dispatch(engine, "full", (content,), {})
        read_full.assert_called_once_with(content, ocr=general)
