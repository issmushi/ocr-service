from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.modules.paddle.text import recognize_all_text_regions
from app.ocr_engine.contracts import RecognizedText
from app.ocr_engine.paddle.recognizer import recognize_text_crops


class FakeModel:
    def __init__(self, results):
        self.results = results
        self.inputs = None

    def predict(self, inputs):
        self.inputs = inputs
        yield from self.results


class TextRecognizerTests(unittest.TestCase):
    def setUp(self):
        self.crops = [
            np.arange(36, dtype=np.uint8).reshape(3, 4, 3),
            np.arange(60, 96, dtype=np.uint8).reshape(3, 4, 3),
        ]
        self.recognition_results = [
            {"rec_text": "ИВАНОВ", "rec_score": 0.9},
            {"rec_text": "1234", "rec_score": 0.8},
        ]
        self.recognizer = FakeModel(self.recognition_results)
        self.orientation = FakeModel(
            [
                {"class_ids": np.array([0])},
                {"class_ids": [1]},
            ]
        )
        self.pipeline = SimpleNamespace(
            text_det_model=FakeModel([]),
            text_rec_model=self.recognizer,
            textline_orientation_model=self.orientation,
        )
        self.ocr = SimpleNamespace(
            paddlex_pipeline=SimpleNamespace(_pipeline=self.pipeline),
        )

    def test_orientation_is_disabled_by_default(self):
        result = recognize_text_crops(self.ocr, self.crops)
        self.assertIsNone(self.orientation.inputs)
        for actual, expected in zip(self.recognizer.inputs, self.crops):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual([item.text for item in result], ["ИВАНОВ", "1234"])

    def test_requested_orientation_rotates_only_class_one_before_recognition(
        self,
    ):
        originals = [crop.copy() for crop in self.crops]
        result = recognize_text_crops(
            self.ocr,
            self.crops,
            use_textline_orientation=True,
        )
        for actual, expected in zip(self.orientation.inputs, originals):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(self.recognizer.inputs[0], originals[0])
        np.testing.assert_array_equal(
            self.recognizer.inputs[1], originals[1][::-1, ::-1]
        )
        for actual, expected in zip(self.crops, originals):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual([item.text for item in result], ["ИВАНОВ", "1234"])
        self.assertEqual([item.confidence for item in result], [0.9, 0.8])

    def test_missing_orientation_model_fails_only_when_requested(self):
        del self.pipeline.textline_orientation_model
        self.assertEqual(len(recognize_text_crops(self.ocr, self.crops)), 2)
        self.recognizer.inputs = None
        with self.assertRaisesRegex(
            RuntimeError, "orientation model is unavailable"
        ):
            recognize_text_crops(
                self.ocr, self.crops, use_textline_orientation=True
            )
        self.assertIsNone(self.recognizer.inputs)

    def test_orientation_result_count_must_match_crops(self):
        for count in (0, 1, 3):
            with self.subTest(count=count):
                self.orientation.results = [{"class_ids": [0]}] * count
                with self.assertRaisesRegex(
                    RuntimeError, "orientation result count"
                ):
                    recognize_text_crops(
                        self.ocr, self.crops, use_textline_orientation=True
                    )
                self.assertIsNone(self.recognizer.inputs)

    def test_invalid_orientation_results_raise_runtime_error(self):
        for invalid in (
            {},
            {"class_ids": []},
            {"class_ids": [2]},
            {"class_ids": ["bad"]},
            None,
        ):
            with self.subTest(result=invalid):
                self.orientation.results = [{"class_ids": [0]}, invalid]
                with self.assertRaisesRegex(
                    RuntimeError, "invalid textline orientation"
                ):
                    recognize_text_crops(
                        self.ocr, self.crops, use_textline_orientation=True
                    )
                self.assertIsNone(self.recognizer.inputs)

    def test_recognition_result_count_must_match_crops(self):
        for count in (0, 1, 3):
            with self.subTest(count=count):
                self.recognizer.results = [
                    {"rec_text": "TEST", "rec_score": 0.9}
                ] * count
                with self.assertRaisesRegex(
                    RuntimeError, "recognition result count"
                ):
                    recognize_text_crops(self.ocr, self.crops)

    def test_empty_crops_do_not_run_models(self):
        self.assertEqual(recognize_text_crops(self.ocr, []), [])
        self.assertIsNone(self.recognizer.inputs)
        self.assertIsNone(self.orientation.inputs)

    def test_all_regions_forward_orientation_and_preserve_source_boxes(self):
        image = np.arange(80 * 100 * 3, dtype=np.uint8).reshape(80, 100, 3)
        boxes = [
            [[5, 35], [80, 35], [80, 55], [5, 55]],
            [[5, 5], [80, 5], [80, 25], [5, 25]],
        ]
        self.pipeline.text_det_model.results = [
            {
                "input_img": image,
                "dt_polys": boxes,
                "dt_scores": [0.7, 0.6],
            }
        ]
        result = recognize_all_text_regions(
            self.ocr, image, use_textline_orientation=True
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(
            [item["box"] for item in result], [boxes[1], boxes[0]]
        )
        self.assertEqual(
            [item["detection_confidence"] for item in result], [0.6, 0.7]
        )
        self.assertEqual([item["text"] for item in result], ["ИВАНОВ", "1234"])
        np.testing.assert_array_equal(
            self.recognizer.inputs[0], self.orientation.inputs[0]
        )
        np.testing.assert_array_equal(
            self.recognizer.inputs[1],
            self.orientation.inputs[1][::-1, ::-1],
        )

    def test_selected_region_mismatch_does_not_silently_drop_a_block(self):
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        boxes = [
            [[5, 5], [80, 5], [80, 25], [5, 25]],
            [[5, 35], [80, 35], [80, 55], [5, 55]],
        ]
        self.pipeline.text_det_model.results = [
            {"input_img": image, "dt_polys": boxes}
        ]
        with patch(
            "app.modules.paddle.text.recognize_text_crops",
            return_value=[RecognizedText(text="TEST", confidence=0.9)],
        ):
            with self.assertRaisesRegex(
                RuntimeError, "does not match selected regions"
            ):
                recognize_all_text_regions(self.ocr, image)


if __name__ == "__main__":
    unittest.main()
