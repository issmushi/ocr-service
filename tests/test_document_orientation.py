from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from app.core.images.orientation import rotate_document
from app.ocr_engine.contracts import DocumentOrientation
from app.ocr_engine.paddle.orientation import predict_document_orientation


class FakeOrientationModel:
    def __init__(self, results):
        self.results = results
        self.inputs = []

    def predict(self, image):
        self.inputs.append(image)
        yield from self.results


class DocumentRotationTests(unittest.TestCase):
    def test_four_clockwise_rotations_preserve_every_pixel_and_channel(self):
        # Unique pixels on a nonsquare image make direction and dimensions
        # observable; no symmetry can hide a swapped 90/270 mapping.
        source = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
        original = source.copy()
        for angle in (0, 90, 180, 270):
            with self.subTest(angle=angle):
                rotated = rotate_document(source, angle)
                expected = np.rot90(original, k=-(angle // 90))
                np.testing.assert_array_equal(rotated, expected)
                np.testing.assert_array_equal(source, original)
                self.assertEqual(rotated.dtype, source.dtype)
                self.assertEqual(rotated.shape, expected.shape)
                restored = rotate_document(rotated, (-angle) % 360)
                np.testing.assert_array_equal(restored, original)
        self.assertIs(rotate_document(source, 0), source)

    def test_grayscale_and_noncontiguous_images_keep_pixels(self):
        source = np.arange(35, dtype=np.uint8).reshape(5, 7)[:, ::2]
        for angle in (0, 90, 180, 270):
            with self.subTest(angle=angle):
                np.testing.assert_array_equal(
                    rotate_document(source, np.int64(angle)),
                    np.rot90(source, k=-(angle // 90)),
                )

    def test_invalid_rotation_values_are_not_coerced_or_normalized(self):
        image = np.zeros((3, 5, 3), dtype=np.uint8)
        for invalid in (
            -90,
            45,
            360,
            450,
            "90",
            90.0,
            None,
            True,
            False,
            float("nan"),
        ):
            with self.subTest(rotation=invalid), self.assertRaises(ValueError):
                rotate_document(image, invalid)

    def test_invalid_or_empty_image_is_rejected(self):
        for image in (
            None,
            [],
            np.zeros((0, 5, 3)),
            np.zeros((3, 0, 3)),
            np.zeros(5),
        ):
            with self.subTest(image=image), self.assertRaises(ValueError):
                rotate_document(image, 0)


class DocumentOrientationTests(unittest.TestCase):
    def setUp(self):
        self.image = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
        self.model = FakeOrientationModel(
            [{"label_names": ["0"], "scores": [0.9]}]
        )
        self.preprocessor = SimpleNamespace(doc_ori_classify_model=self.model)
        self.detector = Mock()
        self.recognizer = Mock()
        self.unwarping = Mock()
        self.preprocessor.doc_unwarping_model = self.unwarping
        self.pipeline = SimpleNamespace(
            text_det_model=self.detector,
            text_rec_model=self.recognizer,
            doc_preprocessor_pipeline=self.preprocessor,
            predict=Mock(),
        )
        self.ocr = SimpleNamespace(
            paddlex_pipeline=SimpleNamespace(_pipeline=self.pipeline),
            predict=Mock(),
            ocr=Mock(),
        )

    def assert_only_classifier_ran(self):
        self.detector.assert_not_called()
        self.detector.predict.assert_not_called()
        self.recognizer.assert_not_called()
        self.recognizer.predict.assert_not_called()
        self.unwarping.assert_not_called()
        self.unwarping.predict.assert_not_called()
        self.pipeline.predict.assert_not_called()
        self.ocr.predict.assert_not_called()
        self.ocr.ocr.assert_not_called()

    def test_all_paddle_labels_become_clockwise_corrections_that_restore_image(
        self,
    ):
        for label, correction in (
            ("0", 0),
            ("90", 270),
            ("180", 180),
            ("270", 90),
        ):
            with self.subTest(label=label):
                turned = np.rot90(self.image, k=-(int(label) // 90)).copy()
                self.model.results = [
                    {"label_names": [label], "scores": np.array([0.91])}
                ]
                orientation = predict_document_orientation(
                    turned, ocr=self.ocr
                )
                self.assertEqual(
                    orientation, DocumentOrientation(correction, 0.91)
                )
                self.assertIs(self.model.inputs[-1], turned)
                np.testing.assert_array_equal(
                    rotate_document(turned, orientation.rotation_degrees),
                    self.image,
                )
        self.assertEqual(len(self.model.inputs), 4)
        self.assert_only_classifier_ran()

    def test_wrapper_reuses_the_existing_document_classifier(self):
        self.pipeline.doc_preprocessor_pipeline = SimpleNamespace(
            _pipeline=self.preprocessor
        )
        first = predict_document_orientation(self.image, ocr=self.ocr)
        second = predict_document_orientation(self.image, ocr=self.ocr)
        self.assertEqual(first, DocumentOrientation(0, 0.9))
        self.assertEqual(second, first)
        self.assertEqual(len(self.model.inputs), 2)
        self.assertIs(self.model.inputs[0], self.image)
        self.assertIs(self.model.inputs[1], self.image)
        self.assert_only_classifier_ran()

    def test_top_prediction_and_low_confidence_are_preserved_for_caller(self):
        self.model.results = [
            {
                "label_names": np.array([90, 0, 270, 180]),
                "scores": np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32),
            }
        ]
        orientation = predict_document_orientation(self.image, ocr=self.ocr)
        self.assertEqual(orientation.rotation_degrees, 270)
        self.assertAlmostEqual(orientation.confidence, 0.4, places=6)
        self.assert_only_classifier_ran()

    def test_probability_endpoints_are_valid(self):
        for score in (0.0, 1.0):
            with self.subTest(score=score):
                self.model.results = [
                    {"label_names": ["180"], "scores": [score]}
                ]
                self.assertEqual(
                    predict_document_orientation(self.image, ocr=self.ocr),
                    DocumentOrientation(180, score),
                )

    def test_missing_or_noncallable_classifier_does_not_start_other_models(
        self,
    ):
        for preprocessor in (
            None,
            SimpleNamespace(),
            SimpleNamespace(_pipeline=None),
            SimpleNamespace(doc_ori_classify_model=object()),
        ):
            with self.subTest(preprocessor=preprocessor):
                self.pipeline.doc_preprocessor_pipeline = preprocessor
                with self.assertRaisesRegex(
                    RuntimeError, "orientation model is unavailable"
                ):
                    predict_document_orientation(self.image, ocr=self.ocr)
        self.assertEqual(self.model.inputs, [])
        self.assert_only_classifier_ran()

    def test_no_prediction_or_multiple_predictions_is_an_error(self):
        for count in (0, 2):
            with self.subTest(count=count):
                self.model.results = [
                    {"label_names": ["0"], "scores": [0.9]}
                ] * count
                with self.assertRaisesRegex(
                    RuntimeError, "orientation result count"
                ):
                    predict_document_orientation(self.image, ocr=self.ocr)
        self.assert_only_classifier_ran()

    def test_invalid_labels_and_scores_do_not_produce_an_orientation(self):
        invalid_results = [
            None,
            {},
            {"class_ids": [1], "scores": [0.9]},
            {"label_names": [], "scores": []},
            {"label_names": ["0"], "scores": []},
            {"label_names": ["0", "90"], "scores": [0.9]},
            {"label_names": "90", "scores": [0.9]},
            {"label_names": ["90"], "scores": 0.9},
            {"label_names": [["90"]], "scores": [[0.9]]},
        ]
        invalid_results.extend(
            {"label_names": [label], "scores": [0.9]}
            for label in (
                None,
                True,
                "",
                "bad",
                "45",
                "360",
                "90.0",
                " 90 ",
                -90,
                45,
                90.5,
                90.0,
            )
        )
        invalid_results.extend(
            {"label_names": ["90"], "scores": [score]}
            for score in (
                None,
                True,
                "0.9",
                "bad",
                -0.01,
                1.01,
                float("nan"),
                float("inf"),
                float("-inf"),
            )
        )
        invalid_results.extend(
            [
                {"label_names": ["0", "bad"], "scores": [0.9, 0.1]},
                {"label_names": ["0", "90"], "scores": [0.9, float("nan")]},
            ]
        )
        for raw in invalid_results:
            with self.subTest(result=raw):
                self.model.results = [raw]
                with self.assertRaisesRegex(
                    RuntimeError, "invalid document orientation"
                ):
                    predict_document_orientation(self.image, ocr=self.ocr)
        self.assert_only_classifier_ran()


if __name__ == "__main__":
    unittest.main()
