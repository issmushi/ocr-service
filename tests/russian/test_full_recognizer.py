from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest

from app.core.images.regions import crop_quadrilateral
from app.modules.russian.full_pipeline import recognize_full_text
from app.ocr_engine.contracts import DetectedTextRegion, TextDetectionResult

DETECT = "app.modules.russian.full_pipeline.detect_text_model_regions"


def _box(x1: int, y1: int, x2: int, y2: int) -> list[list[int]]:
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _reader(texts: list[str] | None = None) -> Mock:
    reader = Mock()
    reader.words_detector.predict_transform.side_effect = lambda image: {
        "WordsDetector": {"warped_img": [image]}
    }
    if texts is not None:
        reader.ocr_cyr.predict.side_effect = [
            {"OCRCyrillic": {"ocr_output": text}} for text in texts
        ]
    reader.ocr_cyr.fix_errors.side_effect = AssertionError("No corrections")
    return reader


def _detection(image: np.ndarray, boxes: list) -> TextDetectionResult:
    return TextDetectionResult(
        image=image,
        regions=tuple(DetectedTextRegion(box=box) for box in boxes),
    )


def test_regions_are_read_left_to_right_within_rows_then_top_to_bottom():
    image = np.zeros((90, 140, 3), dtype=np.uint8)
    boxes = [
        _box(70, 50, 130, 70),
        _box(70, 9, 130, 29),
        _box(5, 51, 60, 71),
        _box(5, 10, 60, 30),
    ]
    reader = _reader(["ПЕРВАЯ", "СТРОКА", "ВТОРАЯ", "СТРОКА"])
    with patch(DETECT, return_value=_detection(image, boxes)):
        result = recognize_full_text(image, reader, Mock())
    assert result == {"text": "ПЕРВАЯ СТРОКА\nВТОРАЯ СТРОКА"}
    calls = reader.words_detector.predict_transform.call_args_list
    for call, box in zip(calls, [boxes[3], boxes[1], boxes[2], boxes[0]]):
        np.testing.assert_array_equal(
            call.args[0], crop_quadrilateral(image, box)
        )
    reader.ocr_cyr.fix_errors.assert_not_called()
    reader.process_img.assert_not_called()


def test_dense_page_keeps_every_region_including_short_and_long_text():
    image = np.zeros((900, 240, 3), dtype=np.uint8)
    boxes = [
        _box(column * 70 + 5, row * 30 + 5, column * 70 + 60, row * 30 + 24)
        for row in range(28)
        for column in range(3)
    ]
    texts = ["Я", "PNRUS" + "<" * 60] + [f"СЛОВО{i}" for i in range(82)]
    reader = _reader(texts)
    with patch(DETECT, return_value=_detection(image, list(reversed(boxes)))):
        result = recognize_full_text(image, reader, Mock())
    expected = "\n".join(
        " ".join(texts[start : start + 3]) for start in range(0, len(texts), 3)
    )
    assert result == {"text": expected}
    assert reader.words_detector.predict_transform.call_count == 84
    assert reader.ocr_cyr.predict.call_count == 84


def test_detector_gets_bgr_but_perspective_crop_uses_original_rgb_pixels():
    image = np.arange(80 * 120 * 3, dtype=np.uint8).reshape(80, 120, 3)
    original = image.copy()
    polygon = [[11, 12], [100, 7], [105, 47], [16, 52]]
    expected = crop_quadrilateral(image, polygon)
    # Coordinates remain original-image coordinates even if another image is
    # returned as auxiliary detector metadata.
    detection = _detection(np.zeros((5, 5, 3), np.uint8), [polygon])
    detector = Mock()
    reader = _reader(["ТЕКСТ"])
    with patch(DETECT, return_value=detection) as detect:
        result = recognize_full_text(image, reader, detector)
    assert result == {"text": "ТЕКСТ"}
    assert detect.call_args.args[0] is detector
    np.testing.assert_array_equal(detect.call_args.args[1], image[:, :, ::-1])
    line_crop = reader.words_detector.predict_transform.call_args.args[0]
    np.testing.assert_array_equal(line_crop, expected)
    np.testing.assert_array_equal(
        reader.ocr_cyr.predict.call_args.args[0], expected
    )
    np.testing.assert_array_equal(image, original)


def test_word_patches_are_read_raw_without_field_specific_corrections():
    image = np.zeros((60, 120, 3), np.uint8)
    first = np.full((20, 30, 3), 25, np.uint8)
    second = np.full((20, 60, 3), 150, np.uint8)
    reader = _reader([".Тестов.", "22/06/2010"])
    reader.words_detector.predict_transform.side_effect = None
    reader.words_detector.predict_transform.return_value = {
        "WordsDetector": {"warped_img": [first, second]}
    }
    with patch(DETECT, return_value=_detection(image, [_box(2, 2, 110, 30)])):
        result = recognize_full_text(image, reader, Mock())
    assert result == {"text": ".Тестов. 22/06/2010"}
    calls = reader.ocr_cyr.predict.call_args_list
    assert calls[0].args[0] is first
    assert calls[1].args[0] is second
    reader.ocr_cyr.fix_errors.assert_not_called()
    reader.ocr_lat.predict.assert_not_called()


@pytest.mark.parametrize(
    "word_patches",
    [[], [np.zeros((0, 8, 3), np.uint8), np.zeros((8, 0, 3), np.uint8)]],
)
def test_missing_word_boxes_fall_back_to_the_whole_detected_line(word_patches):
    image = np.arange(60 * 120 * 3, dtype=np.uint8).reshape(60, 120, 3)
    box = _box(2, 2, 110, 30)
    reader = _reader(["СТРОКАНЕПРОПАЛА"])
    reader.words_detector.predict_transform.side_effect = None
    reader.words_detector.predict_transform.return_value = {
        "WordsDetector": {"warped_img": word_patches}
    }
    with patch(DETECT, return_value=_detection(image, [box])):
        result = recognize_full_text(image, reader, Mock())
    assert result == {"text": "СТРОКАНЕПРОПАЛА"}
    np.testing.assert_array_equal(
        reader.ocr_cyr.predict.call_args.args[0],
        crop_quadrilateral(image, box),
    )


def test_invalid_polygons_are_skipped_but_valid_regions_still_run():
    image = np.zeros((60, 120, 3), np.uint8)
    boxes = [
        [],
        [[1, 2], [2, 3]],
        _box(1, 1, 2, 2),
        _box(1, 1, 1, 20),
        _box(150, 1, 200, 20),
        [[float("nan"), 1], [3, 1], [3, 20], [1, 20]],
        _box(2, 2, 110, 30),
    ]
    reader = _reader(["ТЕКСТ"])
    with patch(DETECT, return_value=_detection(image, boxes)):
        assert recognize_full_text(image, reader, Mock()) == {"text": "ТЕКСТ"}
    reader.words_detector.predict_transform.assert_called_once()


def test_empty_line_detection_returns_empty_without_running_word_models():
    image = np.zeros((40, 60, 3), np.uint8)
    reader = _reader([])
    with patch(DETECT, return_value=_detection(image, [])):
        assert recognize_full_text(image, reader, Mock()) == {"text": ""}
    reader.words_detector.predict_transform.assert_not_called()
    reader.ocr_cyr.predict.assert_not_called()


@pytest.mark.parametrize("stage", ["line_detector", "word_detector", "ocr"])
def test_inference_errors_are_not_reported_as_successful_empty_text(stage):
    image = np.zeros((40, 60, 3), np.uint8)
    reader = _reader(["ТЕКСТ"])
    with patch(
        DETECT, return_value=_detection(image, [_box(2, 2, 50, 30)])
    ) as detect:
        failing_stage = {
            "line_detector": detect,
            "word_detector": reader.words_detector.predict_transform,
            "ocr": reader.ocr_cyr.predict,
        }[stage]
        failing_stage.side_effect = RuntimeError("inference failed")
        with pytest.raises(RuntimeError, match="inference failed"):
            recognize_full_text(image, reader, Mock())


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
    with patch(DETECT) as detect:
        with pytest.raises(ValueError, match="RGB uint8"):
            recognize_full_text(image, Mock(), Mock())
    detect.assert_not_called()
