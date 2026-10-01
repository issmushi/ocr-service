from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from app.ocr_engine.russian.fields import detect_passport_fields
from app.ocr_engine.russian.vendor.document_processing import (
    pipeline_modules,
)


def stub_pipeline(
    boxes: list[list] | None = None,
    *,
    angle: int = 0,
    segments: list | None = None,
) -> Mock:
    """Use real stage transformations with mocked model inference only."""
    angle_module = object.__new__(pipeline_modules.Angle90)
    angle_module.model_name = "Angle90"
    angle_module.model = Mock()
    angle_module.model.predict.return_value = (angle, np.float32(0.98))
    document_module = object.__new__(pipeline_modules.DocDetector)
    document_module.model_name = "DocDetector"
    document_module.model = Mock()
    document_module.model.predict.return_value = (
        None if not segments else np.array([[0, 0, 8, 8, 0.95, 0]]),
        None,
        segments,
    )
    fields_module = object.__new__(pipeline_modules.TextFieldsDetector)
    fields_module.model_name = "TextFieldsDetector"
    fields_module.model = Mock()
    fields_module.model.predict.return_value = boxes or []
    engine = Mock()
    engine.angle90 = angle_module
    engine.doc_detector = document_module
    engine.text_fields = fields_module
    return engine


class FieldDetectorTests(unittest.TestCase):
    def test_orients_once_and_preserves_full_resolution_without_ocr(self):
        image = np.zeros((1602, 1900, 3), dtype=np.uint8)
        image[:2, :2] = (10, 20, 30)
        engine = stub_pipeline(
            [[1, 2, 5, 7, 0.91, 0, "Last_name_ru"]], angle=90
        )

        result = detect_passport_fields(image, engine, padding=0)

        expected = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        np.testing.assert_array_equal(result["document_image"], expected)
        np.testing.assert_array_equal(
            result["fields"][0]["image"], expected[2:7, 1:5]
        )
        self.assertEqual(result["original_width"], 1900)
        self.assertEqual(result["original_height"], 1602)
        self.assertEqual(result["rotation_degrees"], 90)
        self.assertAlmostEqual(
            result["orientation_confidence"], 0.98, places=6
        )
        self.assertFalse(result["document_detected"])
        engine.angle90.model.predict.assert_called_once()
        engine.doc_detector.model.predict.assert_called_once()
        engine.text_fields.model.predict.assert_called_once()
        np.testing.assert_array_equal(
            engine.text_fields.model.predict.call_args.args[0], expected
        )
        engine.assert_not_called()
        engine.doctype.assert_not_called()
        engine.doctype.predict.assert_not_called()
        engine.words_detector.predict_transform.assert_not_called()
        engine.ocr_ru.predict.assert_not_called()
        engine.ocr_en.predict.assert_not_called()

    def test_crops_document_once_then_uses_document_coordinates(self):
        image = np.arange(12 * 14 * 3, dtype=np.uint8).reshape(12, 14, 3)
        document = image[2:10, 3:11].copy()
        engine = stub_pipeline(
            [[2, 1, 6, 3, np.float32(0.87), 2, "First_name_ru"]],
            segments=[np.array([[3, 2], [3, 10], [11, 10], [11, 2]])],
        )
        module_path = (
            "app.ocr_engine.russian.vendor.document_processing."
            "pipeline_modules.doc_detector.doc_detector.fix_perspective"
        )
        with patch(module_path, return_value=(document, image)) as transform:
            result = detect_passport_fields(image, engine, padding=1)

        transform.assert_called_once()
        engine.doc_detector.model.predict.assert_called_once()
        self.assertIs(result["document_image"], document)
        self.assertTrue(result["document_detected"])
        np.testing.assert_array_equal(
            engine.text_fields.model.predict.call_args.args[0], document
        )
        field = result["fields"][0]
        self.assertEqual(field["bbox"], [1, 0, 7, 4])
        self.assertEqual(field["detection_bbox"], [2, 1, 6, 3])
        self.assertAlmostEqual(field["confidence"], 0.87, places=6)
        np.testing.assert_array_equal(field["image"], document[:4, 1:7])

    def test_keeps_duplicates_and_rotates_only_vertical_number_crops(self):
        image = np.arange(12 * 14 * 3, dtype=np.uint8).reshape(12, 14, 3)
        engine = stub_pipeline(
            [
                [1, 2, 3, 9, 0.99, 0, "Licence_number"],
                [4, 2, 12, 4, 0.72, 0, "Licence_number"],
                [1, 2, 3, 9, 0.82, 1, "Birth_place_ru"],
            ]
        )

        result = detect_passport_fields(
            image, engine, padding=0, crop_document=False
        )

        self.assertEqual(len(result["fields"]), 3)
        vertical, horizontal, birthplace = result["fields"]
        self.assertEqual(vertical["rotation_degrees"], 90)
        self.assertEqual(vertical["bbox"], [1, 2, 3, 9])
        np.testing.assert_array_equal(
            vertical["image"],
            cv2.rotate(image[2:9, 1:3], cv2.ROTATE_90_COUNTERCLOCKWISE),
        )
        self.assertEqual(horizontal["rotation_degrees"], 0)
        np.testing.assert_array_equal(horizontal["image"], image[2:4, 4:12])
        self.assertEqual(birthplace["rotation_degrees"], 0)
        np.testing.assert_array_equal(birthplace["image"], image[2:9, 1:3])
        engine.doc_detector.model.predict.assert_not_called()
        self.assertFalse(result["document_detected"])
        self.assertNotIn("document", result["timings"])

    def test_clips_padding_and_discards_empty_or_off_image_detections(self):
        image = np.arange(12 * 14 * 3, dtype=np.uint8).reshape(12, 14, 3)
        engine = stub_pipeline(
            [
                [-3, -2, 3, 4, 0.9, 0, "Last_name_ru"],
                [11, 9, 18, 20, 0.8, 1, "First_name_ru"],
                [20, 20, 22, 22, 0.9, 2, "Middle_name_ru"],
                [4, 4, 4, 7, 0.9, 2, "Middle_name_ru"],
            ]
        )

        result = detect_passport_fields(image, engine, padding=2)

        first, second = result["fields"]
        self.assertEqual(first["detection_bbox"], [0, 0, 3, 4])
        self.assertEqual(first["bbox"], [0, 0, 5, 6])
        np.testing.assert_array_equal(first["image"], image[:6, :5])
        self.assertEqual(second["detection_bbox"], [11, 9, 14, 12])
        self.assertEqual(second["bbox"], [9, 7, 14, 12])
        np.testing.assert_array_equal(second["image"], image[7:, 9:])

    def test_number_orientation_uses_detection_before_padding(self):
        image = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)
        engine = stub_pipeline([[80, 0, 90, 40, 0.99, 0, "Licence_number"]])

        result = detect_passport_fields(image, engine, padding=32)

        field = result["fields"][0]
        self.assertEqual(field["detection_bbox"], [80, 0, 90, 40])
        self.assertEqual(field["bbox"], [48, 0, 122, 72])
        self.assertEqual(field["rotation_degrees"], 90)
        np.testing.assert_array_equal(
            field["image"],
            cv2.rotate(image[:72, 48:122], cv2.ROTATE_90_COUNTERCLOCKWISE),
        )

    def test_no_detections_returns_oriented_image_and_stage_timings(self):
        image = np.zeros((12, 14, 3), dtype=np.uint8)
        result = detect_passport_fields(image, stub_pipeline())
        self.assertEqual(result["fields"], [])
        self.assertIs(result["document_image"], image)
        self.assertEqual(
            set(result["timings"]),
            {"orientation", "document", "fields", "total"},
        )
        self.assertTrue(
            all(value >= 0 for value in result["timings"].values())
        )

    def test_invalid_input_or_padding_fails_before_model_loading(self):
        invalid_images = [
            None,
            np.zeros((0, 4, 3), dtype=np.uint8),
            np.zeros((4, 4), dtype=np.uint8),
            np.zeros((4, 4, 4), dtype=np.uint8),
            np.zeros((4, 4, 3), dtype=np.float32),
        ]
        with patch("app.ocr_engine.russian.fields.create_pipeline") as factory:
            for image in invalid_images:
                with self.subTest(image=image):
                    with self.assertRaises(ValueError):
                        detect_passport_fields(image)
            for padding in (-1, 65, 0.5, True):
                with self.subTest(padding=padding):
                    with self.assertRaises(ValueError):
                        detect_passport_fields(
                            np.zeros((4, 4, 3), dtype=np.uint8),
                            padding=padding,
                        )
        factory.assert_not_called()

    def test_path_input_decodes_rgb_and_factory_forwards_device(self):
        image = np.zeros((12, 14, 3), dtype=np.uint8)
        image[:] = (12, 34, 56)
        success, encoded = cv2.imencode(".png", image)
        self.assertTrue(success)
        engine = stub_pipeline()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "паспорт.png"
            path.write_bytes(encoded.tobytes())
            with patch(
                "app.ocr_engine.russian.fields.create_pipeline",
                return_value=engine,
            ) as factory:
                result = detect_passport_fields(path, device="gpu")
        factory.assert_called_once_with(device="gpu")
        np.testing.assert_array_equal(
            result["document_image"], image[:, :, ::-1]
        )


if __name__ == "__main__":
    unittest.main()
