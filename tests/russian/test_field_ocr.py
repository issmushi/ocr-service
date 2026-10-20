from unittest.mock import Mock, patch

import numpy as np
import pytest

from app.modules.russian.inference_worker import russian_inference_worker
from app.ocr_engine.russian.field_recognizer import (
    create_cpu_field_reader,
    recognize_field_text,
)


def test_raw_cyrillic_output_does_not_run_field_corrections():
    image = np.full((20, 70, 3), (15, 25, 35), dtype=np.uint8)
    reader = Mock()
    reader.ocr_cyr.predict.return_value = {
        "OCRCyrillic": {"ocr_output": ".ПРИМЕР."}
    }
    result = recognize_field_text(image, reader, field_name="Last_name_ru")
    assert result == {"text": ".ПРИМЕР.", "confidence": None, "device": "cpu"}
    np.testing.assert_array_equal(
        reader.ocr_cyr.predict.call_args.args[0], image
    )
    reader.ocr_cyr.fix_errors.assert_not_called()
    reader.words_detector.predict_transform.assert_not_called()


@pytest.mark.parametrize(
    "field_name",
    ["Birth_place_ru", "Licence_number", "Address", "Registration_stamp"],
)
def test_multiword_fields_preserve_word_order_and_use_cyrillic(field_name):
    image = np.zeros((20, 60, 3), dtype=np.uint8)
    patches = [image[:, :20], image[:, 20:]]
    reader = Mock()
    reader.words_detector.predict_transform.return_value = {
        "WordsDetector": {"warped_img": patches}
    }
    reader.ocr_cyr.predict.side_effect = [
        {"OCRCyrillic": {"ocr_output": text}} for text in ["1234", "567890"]
    ]
    result = recognize_field_text(image, reader, field_name=field_name)
    assert result["text"] == "1234 567890"
    reader.ocr_lat.predict.assert_not_called()


def test_date_keeps_model_separators_without_normalization():
    reader = Mock()
    reader.ocr_lat.predict.return_value = {
        "OCRLatin": {"ocr_output": "02/03/1990"}
    }
    result = recognize_field_text(
        np.zeros((20, 60, 3), dtype=np.uint8), reader, field_name="Birth_date"
    )
    assert result["text"] == "02/03/1990"
    reader.ocr_lat.fix_errors.assert_not_called()


def test_cpu_reader_reuses_models_and_gpu_worker_gets_cpu_only_models():
    pipeline = Mock()
    assert create_cpu_field_reader(pipeline, device="cpu") is pipeline
    module = (
        "app.ocr_engine.russian.vendor.document_processing.pipeline_modules"
    )
    with (
        patch(f"{module}.OCRCyrillic") as cyr,
        patch(f"{module}.OCRLatin") as lat,
        patch(f"{module}.WordsDetector") as words,
    ):
        reader = create_cpu_field_reader(pipeline, device="gpu")
    cyr.assert_called_once_with(tier="accurate", device="cpu")
    lat.assert_called_once_with(tier="accurate", device="cpu")
    words.assert_called_once_with(model_format="ONNX", device="cpu")
    assert reader.ocr_cyr is cyr.return_value


def test_worker_lazily_reuses_accurate_cpu_field_reader():
    with (
        patch(
            "app.modules.russian.inference_worker.serve_inference_worker"
        ) as serve,
        patch(
            "app.ocr_engine.russian.field_recognizer.create_cpu_field_reader"
        ) as create,
        patch(
            "app.modules.russian.field_ocr.recognize_field_text_bytes"
        ) as recognize,
    ):
        russian_inference_worker(Mock(), "gpu")
        dispatch = serve.call_args.args[2]
        create.assert_not_called()
        pipeline = Mock()
        for _ in range(2):
            dispatch(
                pipeline,
                "field_text_accurate",
                (b"png",),
                {"field_name": "Sex_ru"},
            )
        create.assert_called_once_with(pipeline, device="gpu")
        assert recognize.call_count == 2
        recognize.assert_called_with(
            b"png", reader=create.return_value, field_name="Sex_ru"
        )
