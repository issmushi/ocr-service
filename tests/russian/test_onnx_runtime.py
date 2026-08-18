from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.ocr_engine.russian.vendor.document_processing.processing import (  # noqa: E501
    inference,
)

ModelInference = inference.ModelInference


class FakeInferenceSession:
    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="image", shape=[None, 1])]

    def run(self, _outputs: object, _inputs: object) -> list[object]:
        return []


class OnnxRuntimeTests(unittest.TestCase):
    def test_rejects_non_onnx_models_before_loading_runtime(self) -> None:
        for filename in ("model.ir", "model.mlmodel"):
            with (
                self.subTest(filename=filename),
                patch(
                    "app.ocr_engine.russian.vendor.document_processing.processing."
                    "inference.import_module"
                ) as import_module,
            ):
                with self.assertRaisesRegex(ValueError, "Unsupported"):
                    ModelInference(Path(filename), device="cpu")
                import_module.assert_not_called()

    def test_preloads_nvidia_libraries_for_gpu(self) -> None:
        runtime = SimpleNamespace(
            preload_dlls=Mock(),
            get_available_providers=lambda: ["CUDAExecutionProvider"],
            InferenceSession=Mock(return_value=FakeInferenceSession()),
        )

        with patch(
            "app.ocr_engine.russian.vendor.document_processing.processing.inference."
            "import_module",
            return_value=runtime,
        ):
            ModelInference(Path("model.onnx"), device="gpu")

        runtime.preload_dlls.assert_called_once_with(directory="")

    def test_does_not_preload_nvidia_libraries_for_cpu(self) -> None:
        runtime = SimpleNamespace(
            preload_dlls=Mock(),
            get_available_providers=lambda: ["CPUExecutionProvider"],
            InferenceSession=Mock(return_value=FakeInferenceSession()),
        )

        with patch(
            "app.ocr_engine.russian.vendor.document_processing.processing.inference."
            "import_module",
            return_value=runtime,
        ):
            ModelInference(Path("model.onnx"), device="cpu")

        runtime.preload_dlls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
