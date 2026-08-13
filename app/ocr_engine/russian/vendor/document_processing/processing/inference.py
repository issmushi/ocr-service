from importlib import import_module
from pathlib import Path

import numpy as np


class ModelInference:
    """Run inference through the embedded ONNX Runtime backend.

    Attributes:
        model: The loaded model object.
        device (str): Device to use for inference - 'cpu' or 'gpu'.
    """

    def __init__(
        self,
        model_path: Path,
        device: str = "gpu",
        verbose=False,
    ):
        """
        Initializes the ModelInference class.

        Args:
          model_path (Path): Path to model file or directory.
          device (str): Device to use for inference - 'cpu' or 'gpu'.
          verbose (bool): Whether to print debug statements.
        """

        self.device = device

        if model_path.suffix != ".onnx":
            raise ValueError(f"Unsupported RussianDocsOCR model: {model_path}")
        self.ort = import_module("onnxruntime")
        if self.device == "gpu":
            self.__preload_onnx_gpu_libraries()
        self.__load_onnx(model_path)
        self.predict = self.__predict_onnx
        if verbose:
            print("[+] ONNX inference loaded")

    def predict(self, tensor: np.ndarray):
        """Makes a prediction on the input tensor.

        Runs inference on the loaded model.

        Args:
           tensor (numpy.ndarray): Input tensor for model.

        Returns:
           numpy.ndarray: Output prediction

        """
        print("[!] Not yet generated")

    def __load_onnx(self, model_path: Path):
        onnx_model_path = model_path.as_posix()
        if self.device == "gpu":
            if "CUDAExecutionProvider" not in self.ort.get_available_providers():
                print(f"[!] {self.device} not found, using cpu")
                providers = [
                    "CPUExecutionProvider",
                ]
            else:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = [
                "CPUExecutionProvider",
            ]
        self.model = self.ort.InferenceSession(onnx_model_path, providers=providers)
        inp_shape = self.model.get_inputs()[0].shape[1:]
        ort_inputs = {
            self.model.get_inputs()[0]
            .name: np.random.rand(*np.append(1, inp_shape))
            .astype(np.float32)
        }
        self.model.run(None, ort_inputs)

    def __preload_onnx_gpu_libraries(self):
        preload_dlls = getattr(self.ort, "preload_dlls", None)
        if preload_dlls is not None:
            preload_dlls(directory="")

    def __predict_onnx(self, tensor: np.ndarray):
        tensor = tensor.astype(np.float32)
        ort_inputs = {self.model.get_inputs()[0].name: tensor}

        result = self.model.run(None, ort_inputs)
        pred = result[0] if len(result) == 1 else result
        return pred
