from unittest.mock import Mock

import numpy as np
import pytest

from app.modules.russian.passport import recognize_passport_text_bytes
from app.ocr_engine.russian.vendor.document_processing import Pipeline
from app.ocr_engine.russian.vendor.document_processing.pipeline import options
from tests.russian.test_engine_adapter import image_bytes


def field_pipeline() -> Pipeline:
    """Exercise real filtering and OCR routing with fake networks."""
    pipeline = object.__new__(Pipeline)
    pipeline.document_detector = None
    pipeline.probe = None
    pipeline.ocr_device = "cpu"

    def classify(image):
        pipeline.results._meta_results.update(
            DocType="INTPASSPORT_2011",
            Angle90={"warped_img": image, "geometry": None, "angle": 0},
        )

    def detect(_image, **_kwargs):
        labels = ["Last_name_ru", "Birth_place_ru", "Birth_date"]
        pipeline.results._meta_results["TextFieldsDetector"] = {
            "bbox": [
                [0, i * 40, 80, i * 40 + 32, 0.99, i, label]
                for i, label in enumerate(labels)
            ],
            "warped_img": [
                np.full((32, 80, 3), i + 1, dtype=np.uint8)
                for i in range(len(labels))
            ],
        }

    pipeline._doctype_angle = classify
    pipeline._fields_detector = detect
    pipeline.words_detector = Mock(model_name="WordsDetector")
    pipeline.words_detector.predict_transform.side_effect = lambda patch: {
        "WordsDetector": {"bbox": [[0, 0, 80, 32]], "warped_img": [patch]}
    }
    for attribute, key in (
        ("ocr_cyr", "OCRCyrillic"),
        ("ocr_lat", "OCRLatin"),
    ):
        engine = Mock(model_name=key)
        engine.predict.return_value = {key: {"ocr_output": "ТЕСТ"}}
        engine.fix_errors.side_effect = lambda *, field_type, text: text
        setattr(pipeline, attribute, engine)
    return pipeline


def test_unrequested_fields_never_reach_splitter_or_recognizer_and_reset():
    pipeline = field_pipeline()
    image = np.zeros((160, 120, 3), dtype=np.uint8)
    first = pipeline.process_img(
        image, get_doc_borders=False, ocr_fields=["Last_name_ru"]
    )
    assert first.ocr == {"Last_name_ru": "ТЕСТ"}
    pipeline.words_detector.predict_transform.assert_not_called()
    pipeline.ocr_cyr.predict.assert_called_once()
    pipeline.ocr_lat.predict.assert_not_called()

    second = pipeline.process_img(
        image, get_doc_borders=False, ocr_fields=["Birth_place_ru"]
    )
    assert second.ocr == {"Birth_place_ru": "ТЕСТ"}
    pipeline.words_detector.predict_transform.assert_called_once()
    assert pipeline.ocr_cyr.predict.call_count == 2
    pipeline.ocr_lat.predict.assert_not_called()

    third = pipeline.process_img(image, get_doc_borders=False)
    assert set(third.ocr) == {"Last_name_ru", "Birth_place_ru", "Birth_date"}
    assert pipeline.words_detector.predict_transform.call_count == 2
    assert pipeline.ocr_cyr.predict.call_count == 4
    pipeline.ocr_lat.predict.assert_called_once()
    assert "Birth_place_ru" in options.OCROptionsINTPassport.ru_fields
    assert "Birth_date" in options.OCROptionsINTPassport.en_fields
    assert first.ocr == {"Last_name_ru": "ТЕСТ"}


def test_empty_engine_selection_does_not_read_any_text():
    pipeline = field_pipeline()
    result = pipeline.process_img(
        np.zeros((160, 120, 3), dtype=np.uint8),
        get_doc_borders=False,
        ocr_fields=[],
    )
    assert not result.ocr
    pipeline.words_detector.predict_transform.assert_not_called()
    pipeline.ocr_cyr.predict.assert_not_called()
    pipeline.ocr_lat.predict.assert_not_called()


@pytest.mark.parametrize(
    "fields,expected",
    [
        (("surname",), ("Last_name_ru",)),
        (("passport_number",), ("Licence_number",)),
        (("passport_series", "passport_number"), ("Licence_number",)),
        (("mrz",), ("MRZ",)),
        (("passport_type",), ()),
    ],
)
def test_public_selection_reaches_engine_with_shared_number_crop(
    fields, expected
):
    pipeline = Mock()
    pipeline.process_img.return_value.ocr = {}
    pipeline.process_img.return_value.doctype = "INTPASSPORT_2011"
    recognize_passport_text_bytes(image_bytes(), pipeline, fields=fields)
    assert pipeline.process_img.call_args.kwargs["ocr_fields"] == expected
