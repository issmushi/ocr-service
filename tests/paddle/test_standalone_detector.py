from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from app.ocr_engine.paddle.detector import detect_text_model_regions
from app.ocr_engine.paddle.factory import create_cpu_text_detector


def test_factory_loads_only_mobile_text_detection_on_cpu():
    detection = Mock()
    recognition = Mock()
    pipeline = Mock()
    package = SimpleNamespace(
        TextDetection=detection,
        TextRecognition=recognition,
        PaddleOCR=pipeline,
    )
    with patch.dict("sys.modules", {"paddleocr": package}):
        assert create_cpu_text_detector() is detection.return_value
    assert detection.call_args.kwargs["model_name"] == "PP-OCRv5_mobile_det"
    assert detection.call_args.kwargs["device"] == "cpu"
    assert detection.call_args.kwargs["limit_type"] == "max"
    recognition.assert_not_called()
    pipeline.assert_not_called()


@pytest.mark.parametrize("as_iterator", [False, True])
def test_standalone_detector_accepts_wrapper_lists_and_predictor_iterators(
    as_iterator,
):
    image = np.zeros((40, 60, 3), np.uint8)
    polygon = [[2, 5], [40, 5], [40, 18], [2, 18]]
    records = [{"dt_polys": [polygon], "dt_scores": [0.95]}]
    model = Mock()
    model.predict.return_value = iter(records) if as_iterator else records
    result = detect_text_model_regions(model, image)
    assert result.image is image
    assert result.regions[0].box == polygon
    assert result.regions[0].confidence == pytest.approx(0.95)


def test_empty_detection_does_not_invent_text_regions():
    model = Mock()
    model.predict.return_value = []
    image = np.zeros((40, 60, 3), np.uint8)
    result = detect_text_model_regions(model, image)
    assert result.image is image
    assert result.regions == ()
