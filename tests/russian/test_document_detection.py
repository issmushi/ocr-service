from __future__ import annotations

from types import MethodType
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from app.ocr_engine.russian.vendor.document_processing import (
    Pipeline,
    pipeline_modules,
)
from app.ocr_engine.russian.vendor.document_processing.pipeline import options


def pipeline_with_detections(boxes: list) -> Pipeline:
    """Exercise grouping, cropping and geometry without model weights."""
    detector = object.__new__(pipeline_modules.DocumentDetector)
    detector.model_name = "DocumentDetector"
    detector.model = Mock()
    detector.model.predict.return_value = boxes
    pipeline = object.__new__(Pipeline)
    pipeline.document_detector = detector
    pipeline.probe = None

    def read(self, image, **_options):
        # This is the image that the real type classifier and OCR receive.
        self.read_image = image
        return self.results

    pipeline._process_document = MethodType(read, pipeline)
    return pipeline


def test_largest_document_is_cut_at_source_resolution_before_classification():
    image = np.zeros((3000, 4000, 3), dtype=np.uint8)
    image[800:1600, 1000:1800] = (90, 140, 210)
    pipeline = pipeline_with_detections(
        [
            [10, 10, 60, 60, 0.99, 0, "document"],
            [375, 300, 675, 600, 0.95, 0, "document"],
            [390, 310, 500, 580, 0.91, 1, "page"],
        ]
    )

    result = pipeline.process_img(image)

    # Detector sees a resized frame; OCR keeps all source pixels of the crop,
    # including the upstream 3% margin around the document.
    detector_input = pipeline.document_detector.model.predict.call_args.args[0]
    assert detector_input.shape == (1125, 1500, 3)
    np.testing.assert_array_equal(
        pipeline.read_image, image[776:1624, 976:1824]
    )
    assert result.document_index == 0
    assert len(result.documents) == 2
    assert len(result.documents[0]["pages"]) == 1
    np.testing.assert_allclose(
        result.to_input(np.array([[0, 0], [848, 848]], dtype=float)),
        [[976, 776], [1824, 1624]],
    )


@pytest.mark.parametrize("detector_enabled", [True, False])
def test_no_document_uses_whole_frame(detector_enabled):
    image = np.zeros((900, 1800, 3), dtype=np.uint8)
    pipeline = pipeline_with_detections([])
    if not detector_enabled:
        pipeline.document_detector = None

    result = pipeline.process_img(image)

    assert result.documents == []
    assert result.document_index is None
    np.testing.assert_array_equal(
        pipeline.read_image, cv2.resize(image, (1500, 750))
    )


@pytest.mark.parametrize("method", ["process_img", "process_frame"])
def test_excluded_quality_stages_are_rejected_before_inference(method):
    pipeline = pipeline_with_detections([])
    with pytest.raises(ValueError, match="not included"):
        getattr(pipeline, method)(np.zeros((10, 10, 3)), check_quality=True)
    pipeline.document_detector.model.predict.assert_not_called()


@pytest.mark.parametrize(
    "doc_type,field",
    [
        ("INTPASSPORT", "Last_name_ru"),
        ("EXTPASSPORT", "Last_name_en"),
        ("SNILS", "Birth_date"),
        ("DL", "Birth_place_ru"),
    ],
)
def test_split_pipeline_keeps_document_field_reading(doc_type, field):
    pipeline = object.__new__(Pipeline)
    pipeline.probe = None
    pipeline.ocr_device = "cpu"
    pipeline.ocr_options = options.OCROptionsClass.make_options(doc_type)
    pipeline._start_document()
    for attribute, key in (
        ("ocr_cyr", "OCRCyrillic"),
        ("ocr_lat", "OCRLatin"),
    ):
        engine = Mock()
        engine.model_name = key
        engine.predict.return_value = {key: {"ocr_output": "TEST"}}
        engine.fix_errors.side_effect = lambda *, field_type, text: text
        setattr(pipeline, attribute, engine)
    words = {field: {"patches": [np.zeros((32, 80, 3))], "ocr": []}}

    pipeline._ocr(words, doc_type)

    assert pipeline.results.ocr[field] == "TEST"
