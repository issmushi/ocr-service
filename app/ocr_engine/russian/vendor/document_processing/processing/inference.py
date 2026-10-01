import numpy as np
import threading
from pathlib import Path
from importlib import import_module
from typing import List


# ONNX tensor element type -> numpy dtype. Anything not listed falls back to
# float32 (the historical default), so existing float models are unaffected while
# uint8-input models (the v2 OCR nets) get the correct dtype instead of a cast
# that onnxruntime would reject.
_ORT_TYPE_TO_NP = {
    'tensor(float)': np.float32,
    'tensor(float16)': np.float16,
    'tensor(double)': np.float64,
    'tensor(uint8)': np.uint8,
    'tensor(int8)': np.int8,
    'tensor(int32)': np.int32,
    'tensor(int64)': np.int64,
    'tensor(bool)': np.bool_,
}


def _np_dtype_for(ort_input) -> np.dtype:
    return np.dtype(_ORT_TYPE_TO_NP.get(ort_input.type, np.float32))


class ModelInference:
    """Class for making inferences from different model types.

    This class handles loading models in various formats such as ONNX,
    OpenVINO etc. and making predictions from them.

    Attributes:
        model: The loaded model object.
        device (str): Device to use for inference - 'cpu' or 'gpu'.
    """
    def __init__(self,
                 model_path: Path,
                 device: str = 'gpu',
                 verbose=False,
                 runtime: str = None,
                 ):
        """
        Initializes the ModelInference class.

        Args:
          model_path (Path): Path to model file or directory.
          device (str): Device to use for inference - 'cpu' or 'gpu'.
          verbose (bool): Whether to print debug statements.
          runtime (str): Which runtime to load the artifact with ('ONNX',
            'OpenVINO', 'CoreML'). Defaults to picking it from the file
            extension. OpenVINO reads .onnx directly, so its model.json points
            at the ONNX artifact and names the runtime explicitly instead of
            shipping a second copy of the weights under a .ir extension.
        """

        self.device = device
        self.verbose = verbose
        runtime = (runtime or '').lower()

        # Concurrent Run() on ONE CUDA session wedges the GPU: reproduced by
        # hammering the words detector from 8 threads, which hangs at ~200 calls
        # with the GPU pinned at 100%, or dies with
        # "CUDA error cudaErrorIllegalAddress" mid-graph. Serialised through this
        # lock the same 2400 calls pass in 22s. The lock is PER SESSION, so
        # stages that run different models concurrently
        # (Pipeline._quality_and_borders_parallel) keep their speedup; only
        # repeated calls into the same model (Pipeline._split_words) queue up.
        # CPU sessions are left unlocked - the failure is CUDA-specific and a
        # lock there would only cost throughput.
        self._run_lock = threading.Lock() if device == 'gpu' else None

        if runtime not in ('onnx', '') or model_path.suffix != '.onnx':
            raise ValueError(f"Unsupported RussianDocsOCR runtime/model: {runtime!r}, {model_path}")
        self.ort = import_module('onnxruntime')
        if device == 'gpu':
            preload = getattr(self.ort, 'preload_dlls', None)
            if preload is not None:
                preload(directory='')
        self.__load_onnx(model_path)
        self.predict = self.__predict_onnx


    def predict(self, tensors: List[np.ndarray]):
        """Makes a prediction on the input tensor.

        Runs inference on the loaded model.

        Args:
           tensor (numpy.ndarray): Input tensor for model.

        Returns:
           numpy.ndarray: Output prediction

        """
        raise NotImplemented("Need to implement this method")

    def __load_onnx(self, model_path: Path):
        onnx_model_path = model_path.as_posix()
        if self.device == 'gpu':
            available = self.ort.get_available_providers()
            providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                         if 'CUDAExecutionProvider' in available else ['CPUExecutionProvider'])
        else:
            providers = ['CPUExecutionProvider', ]
        self.model = self.ort.InferenceSession(onnx_model_path, providers=providers)
        inputs = self.model.get_inputs()
        ort_inputs = {}
        for inp in inputs:
            # batch=1, and replace dynamic axes (None / str names) with a safe size.
            # 1 is too small for strided OCR backbones (a 1px width collapses to an
            # empty feature map), so dynamic spatial dims use 320.
            shape = [1] + [d if isinstance(d, int) else 320 for d in inp.shape[1:]]
            dtype = _np_dtype_for(inp)
            if np.issubdtype(dtype, np.integer):
                ort_inputs[inp.name] = np.random.randint(0, 256, size=shape).astype(dtype)
            else:
                ort_inputs[inp.name] = np.random.rand(*shape).astype(dtype)
        # Warmup is best-effort priming (primes the CUDA graph/kernel-selection
        # so the first REAL call isn't the one paying that cost) - never let
        # it block model loading. Always surface a failure, not just under
        # verbose: a silently-skipped warmup would otherwise look identical to
        # a slow first real call, which is exactly the kind of thing this is
        # meant to prevent from going unnoticed.
        try:
            self.model.run(None, ort_inputs)
        except Exception as e:
            print(f"[!] ONNX warmup skipped for {onnx_model_path}: {e}")




    def __predict_onnx(self, tensors:  List[np.ndarray]):
        inputs = self.model.get_inputs()
        # cast each tensor to the dtype the model actually declares (float32 for
        # the detectors, uint8 for the OCR nets) rather than always float32.
        ort_inputs = {inp.name: tensors[i].astype(_np_dtype_for(inp)) for i, inp in enumerate(inputs)}

        if self._run_lock is None:
            return self.model.run(None, ort_inputs)
        with self._run_lock:                      # CUDA only - see __init__
            return self.model.run(None, ort_inputs)
