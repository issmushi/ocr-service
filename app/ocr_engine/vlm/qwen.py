from __future__ import annotations

import base64
import io
import os
import re
import shutil
import socket
import subprocess
from pathlib import Path
from time import monotonic, sleep
from typing import Any, BinaryIO

import httpx
from PIL import Image

# Official Qwen GGUF repositories; prefetch these revisions before startup.
MODEL_REVISIONS = {
    "4b": "1cd86afb9a95c410a6038ab3b40d8b578c892266",
    "8b": "f982a07559d4a2f6c8744d840bf6fccab30eea96",
}
OCR_PROMPT = (
    "Transcribe all visible text in this image. "
    "Preserve the original scripts, numbers, and line breaks. "
    "Do not translate, extract fields, or add explanations. "
    "Return only the transcription."
)
PASSPORT_FIELDS_PROMPT = (
    "The image is expected to show an internal passport\n"
    "of the Russian Federation. Extract its visible passport "
    "fields.\n"
    "Return only one valid JSON object with exactly these keys:\n"
    "passport_type, document_code, issuing_country, surname, "
    "name, patronymic, sex,\n"
    "date_of_birth, passport_series, passport_number, "
    "nationality, issue_date,\n"
    "expiry_date, department_code, personal_number.\n"
    "\n"
    "Use a JSON string for every non-null value. Do not use JSON "
    "numbers, booleans,\n"
    "arrays, or nested objects for field values.\n"
    "\n"
    "Use ISO alpha-3 country codes for issuing_country and "
    "nationality. For this\n"
    'scenario set passport_type to "internal", document_code to '
    '"PN", issuing_country and nationality to\n'
    '"RUS", and expiry_date and personal_number to null. Use '
    '"МУЖ." or "ЖЕН." for\n'
    "sex. Format dates as DD.MM.YYYY. Keep names in Cyrillic as "
    "printed on the\n"
    "document. A Russian internal passport has a four-digit "
    "series and a six-digit\n"
    "number. They may be printed vertically as one ten-digit "
    "sequence: put its first\n"
    "four digits in passport_series and its last six digits in "
    "passport_number. Use\n"
    "null for every other field that is absent, unreadable, or "
    "uncertain. Do not\n"
    "calculate missing dates and do not add explanations, "
    "Markdown, MRZ strings, or\n"
    "any keys outside this list."
)
TASK_PROMPTS = {
    "ocr": OCR_PROMPT,
    "passport_fields": PASSPORT_FIELDS_PROMPT,
}


def _positive_env(name: str, default: str, value_type: type = float) -> Any:
    value = value_type(os.getenv(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _device_from_startup_log(path: Path, offset: int) -> str:
    """Use allocated model tensors, not merely a detected GPU, as evidence."""
    try:
        with path.open("rb") as log:
            log.seek(offset)
            text = log.read().decode("utf-8", errors="replace")
    except OSError:
        return "unknown"
    buffers = re.findall(
        r"load_tensors:\s+(\S+) model buffer size\s*=\s*([0-9.]+) MiB", text
    )
    for backend, size in buffers:
        if float(size) <= 0:
            continue
        if backend.startswith(("MTL", "Metal")):
            return "metal"
        if backend.startswith("CUDA"):
            return "cuda"
    return "unknown"


class QwenEngine:
    """One locally owned llama-server running Qwen3-VL Instruct."""

    def __init__(self, model_size: str) -> None:
        size = model_size.lower().removeprefix("qwen3-vl-")
        if size not in MODEL_REVISIONS:
            raise ValueError("Qwen model size must be 4b or 8b")
        self.model_size = size
        self.model_id = f"qwen3-vl-{size}"
        self.repo_id = f"Qwen/Qwen3-VL-{size.upper()}-Instruct-GGUF"
        self._device = os.getenv("QWEN_DEVICE", "auto").strip().lower()
        if self._device not in {"auto", "cpu"}:
            raise ValueError("QWEN_DEVICE must be auto or cpu")
        self._process: subprocess.Popen | None = None
        self._client: httpx.Client | None = None
        self._log_file: BinaryIO | None = None
        self._log_path: Path | None = None
        self._ready = False
        self._runtime_device = "unknown"
        self._startup_timeout = _positive_env(
            "VLM_LLAMA_STARTUP_TIMEOUT_SECONDS", "300"
        )
        self._shutdown_timeout = _positive_env(
            "VLM_LLAMA_SHUTDOWN_TIMEOUT_SECONDS", "5"
        )
        self._request_timeout = _positive_env(
            "VLM_OCR_REQUEST_TIMEOUT_SECONDS", "600"
        )
        self._max_new_tokens = _positive_env(
            "VLM_OCR_MAX_NEW_TOKENS", "4096", int
        )

    @property
    def runtime_info(self) -> dict[str, str]:
        device = "cpu" if self._device == "cpu" else self._runtime_device
        return {
            "backend": "llamacpp",
            "device": device,
            "model": self.model_id,
        }

    def _model_path(self, *, projector: bool) -> Path:
        suffix = "MMPROJ_PATH" if projector else "MODEL_PATH"
        env_name = f"QWEN3_VL_{self.model_size.upper()}_{suffix}"
        override = os.getenv(env_name)
        if override:
            path = Path(override).expanduser()
        else:
            from huggingface_hub import hf_hub_download

            size = self.model_size.upper()
            filename = (
                f"mmproj-Qwen3VL-{size}-Instruct-F16.gguf"
                if projector
                else f"Qwen3VL-{size}-Instruct-Q4_K_M.gguf"
            )
            try:
                path = Path(
                    hf_hub_download(
                        repo_id=self.repo_id,
                        filename=filename,
                        revision=MODEL_REVISIONS[self.model_size],
                        local_files_only=True,
                    )
                )
            except Exception as exc:
                raise RuntimeError(
                    f"{self.model_id} weights are not cached; "
                    f"prefetch them or set {env_name}"
                ) from exc
        if not path.is_file():
            raise RuntimeError(
                f"{env_name} does not point to an existing model file"
            )
        return path

    def start(self) -> None:
        if (
            self._ready
            and self._process is not None
            and self._process.poll() is None
        ):
            return
        self.stop()
        model_path = self._model_path(projector=False)
        projector_path = self._model_path(projector=True)
        configured_binary = os.getenv("VLM_LLAMA_SERVER_BIN", "llama-server")
        binary = shutil.which(configured_binary) or str(
            Path(configured_binary).expanduser()
        )
        if not Path(binary).is_file():
            raise RuntimeError(
                "llama-server is unavailable; set VLM_LLAMA_SERVER_BIN"
            )
        port = _free_port()
        cache_home = Path(
            os.getenv("HF_HOME", "~/.cache/huggingface")
        ).expanduser()
        log_dir = Path(
            os.getenv("VLM_LLAMA_LOG_DIR", str(cache_home.parent / "logs"))
        ).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = log_dir / f"{self.model_id}.log"
        device_args = (
            [
                "-ngl",
                "0",
                "--device",
                "none",
                "--no-mmproj-offload",
                "--no-op-offload",
            ]
            if self._device == "cpu"
            else ["-ngl", "99"]
        )
        try:
            self._log_file = self._log_path.open("ab")
            startup_offset = self._log_file.tell()
            self._process = subprocess.Popen(
                [
                    binary,
                    "-m",
                    str(model_path),
                    "--mmproj",
                    str(projector_path),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--parallel",
                    "1",
                    "--ctx-size",
                    "16384",
                    *device_args,
                    "--image-min-tokens",
                    "1024",
                    "--alias",
                    self.model_id,
                    "--jinja",
                    "--offline",
                    "--no-webui",
                    "--log-verbosity",
                    "4",
                ],
                stdin=subprocess.DEVNULL,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self._client = httpx.Client(
                base_url=f"http://127.0.0.1:{port}",
                trust_env=False,
                timeout=self._request_timeout,
            )
            self._wait_until_ready()
            self._runtime_device = _device_from_startup_log(
                self._log_path, startup_offset
            )
            self._ready = True
        except BaseException:
            self.stop()
            raise

    def _wait_until_ready(self) -> None:
        deadline = monotonic() + self._startup_timeout
        while True:
            exit_code = self._process.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"{self.model_id} llama-server exited during startup "
                    f"with code {exit_code}; "
                    f"see {self._log_path}"
                )
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"{self.model_id} llama-server startup timed out"
                )
            try:
                response = self._client.get(
                    "/health", timeout=min(1.0, remaining)
                )
                if (
                    response.status_code == 200
                    and self._process.poll() is None
                ):
                    return
            except httpx.TransportError:
                pass
            sleep(min(0.1, max(0.0, deadline - monotonic())))

    def stop(self) -> None:
        self._ready = False
        self._runtime_device = "unknown"
        try:
            if self._process is not None:
                process = self._process
                if process.poll() is None:
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=self._shutdown_timeout)
                    except subprocess.TimeoutExpired:
                        try:
                            process.kill()
                        except ProcessLookupError:
                            pass
                        process.wait(timeout=self._shutdown_timeout)
                else:
                    process.wait()
                # Keep the handle if termination fails, so it can be retried.
                self._process = None
        finally:
            client, self._client = self._client, None
            try:
                if client is not None:
                    client.close()
            finally:
                if self._log_file is not None:
                    self._log_file.close()
                    self._log_file = None

    def recognize(
        self,
        image: Image.Image,
        *,
        task: str = "ocr",
    ) -> dict[str, Any]:
        if (
            not self._ready
            or self._process is None
            or self._process.poll() is not None
            or self._client is None
        ):
            raise RuntimeError(f"{self.model_id} is not started")
        try:
            prompt = TASK_PROMPTS[task]
        except KeyError as exc:
            raise ValueError("Unsupported Qwen task") from exc
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        try:
            response = self._client.post(
                "/v1/chat/completions",
                json={
                    "model": self.model_id,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": (
                                            f"data:image/png;base64,{image_b64}"
                                        )
                                    },
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "presence_penalty": 1.5,
                    "repeat_penalty": 1.0,
                    "seed": 3407,
                    "max_tokens": self._max_new_tokens,
                    "stream": False,
                },
            )
            response.raise_for_status()
            output = response.json()
        except httpx.TimeoutException as exc:
            raise TimeoutError("Qwen OCR request timed out") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise RuntimeError("Qwen OCR request failed") from exc
        choices = output.get("choices") if isinstance(output, dict) else None
        if (
            not isinstance(choices, list)
            or len(choices) != 1
            or not isinstance(choices[0], dict)
        ):
            raise RuntimeError("Qwen returned an invalid completion")
        message = choices[0].get("message")
        raw = message.get("content") if isinstance(message, dict) else None
        if not isinstance(raw, str):
            raise RuntimeError("Qwen returned invalid OCR text")
        return {
            "items": [
                {
                    "text": raw,
                    "confidence": None,
                    "box": None,
                    "detection_confidence": None,
                }
            ],
            "image_width": image.width,
            "image_height": image.height,
        }
