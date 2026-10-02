from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from app.ocr_engine.russian.full_recognizer import recognize_full_text
from app.ocr_engine.russian.vendor.document_processing import (
    pipeline_modules,
)


def _reader(boxes: list[list], texts: list[str]) -> Mock:
    detector = object.__new__(pipeline_modules.WordsDetector)
    detector.model_name = "WordsDetector"
    detector.model = Mock()
    detector.model.predict.return_value = boxes
    recognizer = object.__new__(pipeline_modules.OCRCyrillic)
    recognizer.model_name = "OCRCyrillic"
    recognizer.model = Mock()
    recognizer.model.predict.side_effect = [[text] for text in texts]
    recognizer.fix_errors = Mock(side_effect=AssertionError("No corrections"))
    reader = Mock()
    reader.words_detector = detector
    reader.ocr_cyr = recognizer
    return reader


def test_whole_image_words_are_grouped_into_rows_then_read_left_to_right():
    image = np.arange(80 * 100 * 3, dtype=np.uint8).reshape(80, 100, 3)
    reader = _reader(
        [
            [55, 40, 85, 50, 0.95, 0, "Word"],
            [52, 9, 80, 19, 0.95, 0, "Word"],
            [5, 41, 35, 51, 0.95, 0, "Word"],
            [4, 10, 30, 20, 0.95, 0, "Word"],
        ],
        ["ПЕРВАЯ", "СТРОКА", "ВТОРАЯ", "СТРОКА"],
    )

    result = recognize_full_text(image, reader)

    assert result == {"text": "ПЕРВАЯ СТРОКА\nВТОРАЯ СТРОКА"}
    reader.words_detector.model.predict.assert_called_once()
    np.testing.assert_array_equal(
        reader.words_detector.model.predict.call_args.args[0], image
    )
    expected = [
        image[10:20, 4:30],
        image[9:19, 52:80],
        image[41:51, 5:35],
        image[40:50, 55:85],
    ]
    actual = reader.ocr_cyr.model.predict.call_args_list
    assert len(actual) == len(expected)
    for call, patch in zip(actual, expected):
        np.testing.assert_array_equal(call.args[0], patch)
    reader.assert_not_called()
    reader.process_img.assert_not_called()
    reader.doc_detector.predict_transform.assert_not_called()
    reader.text_fields.predict.assert_not_called()
    reader.ocr_cyr.fix_errors.assert_not_called()


def test_empty_detection_returns_empty_text_without_ocr():
    reader = _reader([], [])
    assert recognize_full_text(np.zeros((40, 60, 3), np.uint8), reader) == {
        "text": ""
    }
    reader.ocr_cyr.model.predict.assert_not_called()


def test_raw_text_is_not_normalized_corrected_or_uppercased():
    reader = _reader(
        [
            [1, 1, 20, 10, 0.9, 0, "Word"],
            [25, 1, 50, 10, 0.9, 0, "Word"],
        ],
        [".Тестов.", "22/06/2010"],
    )
    assert recognize_full_text(np.zeros((40, 60, 3), np.uint8), reader) == {
        "text": ".Тестов. 22/06/2010"
    }
    reader.ocr_cyr.fix_errors.assert_not_called()


def test_boxes_are_clipped_without_padding_and_invalid_boxes_are_skipped():
    image = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
    reader = _reader(
        [
            [-3, -2, 8, 7, 0.9, 0, "Word"],
            [25, 14, 40, 25, 0.9, 0, "Word"],
            [32, 14, 40, 25, 0.9, 0, "Word"],
            [-8, -8, -1, -1, 0.9, 0, "Word"],
            [10, 10, 10, 15, 0.9, 0, "Word"],
        ],
        ["СВЕРХУ", "СНИЗУ"],
    )
    assert recognize_full_text(image, reader) == {"text": "СВЕРХУ\nСНИЗУ"}
    calls = reader.ocr_cyr.model.predict.call_args_list
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0].args[0], image[:7, :8])
    np.testing.assert_array_equal(calls[1].args[0], image[14:, 25:])


def test_empty_recognized_words_do_not_add_blank_lines_or_extra_spaces():
    reader = _reader(
        [
            [1, 1, 10, 10, 0.9, 0, "Word"],
            [20, 1, 30, 10, 0.9, 0, "Word"],
            [40, 1, 50, 10, 0.9, 0, "Word"],
            [1, 21, 10, 30, 0.9, 0, "Word"],
        ],
        ["ТЕКСТ", "", "ТЕКСТ", ""],
    )
    assert recognize_full_text(np.zeros((40, 60, 3), np.uint8), reader) == {
        "text": "ТЕКСТ ТЕКСТ"
    }


@pytest.mark.parametrize(
    "image",
    [
        None,
        np.zeros((0, 5, 3), np.uint8),
        np.zeros((5, 5), np.uint8),
        np.zeros((5, 5, 4), np.uint8),
        np.zeros((5, 5, 3), np.float32),
    ],
)
def test_invalid_rgb_images_fail_before_detection(image):
    reader = _reader([], [])
    with pytest.raises(ValueError, match="RGB uint8"):
        recognize_full_text(image, reader)
    reader.words_detector.model.predict.assert_not_called()
