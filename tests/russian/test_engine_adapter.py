from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from app.ocr_engine.russian.assets import models_dir
from app.ocr_engine.russian.factory import create_pipeline
from app.ocr_engine.russian.recognizer import (
    decode_image_bytes,
    recognize_image,
)
from app.ocr_engine.russian.vendor.document_processing import Pipeline
from app.ocr_engine.russian.vendor.document_processing.config import (
    DEFAULT_CFG,
)
from app.ocr_engine.russian.vendor.document_processing.pipeline import (
    pipeline,
)


def image_bytes(extension: str = ".png") -> bytes:
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    image[:, :24] = (25, 60, 190)
    image[:, 24:] = (200, 150, 10)
    success, encoded = cv2.imencode(extension, image)
    assert success
    return encoded.tobytes()


class EngineAdapterTests(unittest.TestCase):
    def test_factory_uses_embedded_pipeline_and_forwards_options(self) -> None:
        with (
            patch("app.ocr_engine.russian.factory.verify_models") as verify,
            patch(
                "app.ocr_engine.russian.vendor.document_processing.Pipeline",
            ) as pipeline,
            patch(
                "app.ocr_engine.russian.factory.import_module"
            ) as import_module,
        ):
            self.assertIs(create_pipeline("gpu"), pipeline.return_value)
        pipeline.assert_called_once_with(device="gpu")
        verify.assert_called_once_with()
        import_module.assert_called_once_with("sklearn.neighbors")

    def test_missing_classifier_dependency_fails_before_pipeline_start(
        self,
    ) -> None:
        with (
            patch("app.ocr_engine.russian.factory.verify_models"),
            patch(
                "app.ocr_engine.russian.vendor.document_processing.Pipeline",
            ) as pipeline,
            patch(
                "app.ocr_engine.russian.factory.import_module",
                side_effect=ModuleNotFoundError("sklearn.neighbors"),
            ),
        ):
            with self.assertRaises(ModuleNotFoundError):
                create_pipeline()
        pipeline.assert_not_called()

    def test_decoding_is_rgb_and_returns_raw_pipeline_result(self) -> None:
        content = image_bytes()
        decoded = decode_image_bytes(content)
        np.testing.assert_array_equal(decoded[0, 0], (190, 60, 25))
        result = SimpleNamespace(
            ocr={"Last_name_ru": "ПРИМЕР"}, doctype="NONE"
        )
        pipeline = Mock(return_value=result)
        self.assertIs(recognize_image(decoded, pipeline, img_size=640), result)
        np.testing.assert_array_equal(pipeline.call_args.args[0], decoded)
        self.assertEqual(pipeline.call_args.kwargs, {"img_size": 640})

    def test_malformed_input_is_rejected_before_model_loading(self) -> None:
        for content in (b"", b"not an image"):
            with (
                self.subTest(content=content),
                patch(
                    "app.ocr_engine.russian.recognizer.create_pipeline",
                ) as factory,
            ):
                with self.assertRaises(ValueError):
                    decode_image_bytes(content)
                factory.assert_not_called()

    def test_memory_and_legacy_file_preprocessing_have_identical_pixels(
        self,
    ) -> None:
        # Instantiate the Python container without an engine or model.
        processor = object.__new__(Pipeline)
        for extension in (".png", ".jpg", ".bmp"):
            with (
                self.subTest(extension=extension),
                tempfile.TemporaryDirectory() as tmp,
            ):
                content = image_bytes(extension)
                path = Path(tmp) / f"изображение{extension}"
                path.write_bytes(content)
                processor.results = pipeline.PipelineResults()
                from_file = processor._prepare_image(path, img_size=24)
                processor.results = pipeline.PipelineResults()
                from_memory = processor._prepare_image(
                    decode_image_bytes(content), img_size=24
                )
                np.testing.assert_array_equal(from_memory, from_file)

    def test_model_paths_resolve_to_external_model_directory(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                self.assertEqual(len(DEFAULT_CFG), 7)
                expected = models_dir()
                for model, directory in DEFAULT_CFG.items():
                    with self.subTest(model=model):
                        path = Path(directory)
                        self.assertTrue(path.is_absolute())
                        self.assertTrue(path.is_relative_to(expected))
                        self.assertNotIn(
                            "ocr_engine/russian/vendor", path.as_posix()
                        )
            finally:
                os.chdir(previous)
