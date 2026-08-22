from __future__ import annotations

import base64
import io
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import httpx
import pytest
from PIL import Image

from app.ocr_engine.vlm import qwen
from app.ocr_engine.vlm.qwen import (
    MODEL_REVISIONS,
    OCR_PROMPT,
    PASSPORT_FIELDS_PROMPT,
    QwenEngine,
)

REAL_POPEN = subprocess.Popen


class Process:
    def __init__(self, *, exit_code=None, stubborn=False):
        self.returncode = exit_code
        self.stubborn = stubborn
        self.terminate_count = 0
        self.kill_count = 0
        self.wait_timeouts = []

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminate_count += 1
        if not self.stubborn:
            self.returncode = -15

    def kill(self):
        self.kill_count += 1
        self.returncode = -9

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        if self.returncode is None:
            raise subprocess.TimeoutExpired("llama-server", timeout)
        return self.returncode


@pytest.fixture
def dependencies(tmp_path, monkeypatch):
    binary = tmp_path / "llama-server"
    binary.write_text("stub executable")
    binary.chmod(0o755)
    monkeypatch.setenv("VLM_LLAMA_SERVER_BIN", str(binary))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "huggingface"))
    monkeypatch.setenv("VLM_LLAMA_LOG_DIR", str(tmp_path / "logs"))
    for env_name in (
        "VLM_OCR_MAX_NEW_TOKENS",
        "VLM_LLAMA_STARTUP_TIMEOUT_SECONDS",
        "VLM_LLAMA_SHUTDOWN_TIMEOUT_SECONDS",
        "VLM_OCR_REQUEST_TIMEOUT_SECONDS",
        "QWEN_DEVICE",
    ):
        monkeypatch.delenv(env_name, raising=False)
    files = {}
    for size in ("4B", "8B"):
        for name in (
            f"Qwen3VL-{size}-Instruct-Q4_K_M.gguf",
            f"mmproj-Qwen3VL-{size}-Instruct-F16.gguf",
        ):
            path = tmp_path / name
            path.write_bytes(b"stub weights")
            files[name] = path
        for suffix in ("MODEL_PATH", "MMPROJ_PATH"):
            monkeypatch.delenv(f"QWEN3_VL_{size}_{suffix}", raising=False)
    cached_file = Mock(
        side_effect=lambda **kwargs: str(files[kwargs["filename"]])
    )
    processes = []

    def spawn(*args, **kwargs):
        process = Process()
        processes.append(process)
        return process

    popen = Mock(side_effect=spawn)
    client = Mock()
    client.get.return_value = SimpleNamespace(status_code=200)
    client.post.return_value = Mock()
    client.post.return_value.json.return_value = {
        "choices": [{"message": {"content": "ИВАНОВ\n1234567890"}}]
    }
    client_factory = Mock(return_value=client)
    monkeypatch.setattr(qwen.subprocess, "Popen", popen)
    monkeypatch.setattr(qwen.httpx, "Client", client_factory)
    monkeypatch.setattr(qwen, "_free_port", Mock(return_value=18765))
    monkeypatch.setattr(qwen, "sleep", Mock())
    with patch.dict(
        sys.modules,
        {"huggingface_hub": SimpleNamespace(hf_hub_download=cached_file)},
    ):
        yield SimpleNamespace(
            binary=binary,
            files=files,
            cached_file=cached_file,
            popen=popen,
            processes=processes,
            client=client,
            client_factory=client_factory,
            tmp_path=tmp_path,
        )


@pytest.mark.parametrize("size", ["4b", "8b", "4B", "qwen3-vl-8b"])
def test_official_pinned_weights_are_cache_only_and_process_is_local(
    dependencies, size
):
    engine = QwenEngine(size)
    engine.start()
    engine.start()
    normalized = size.lower().removeprefix("qwen3-vl-")
    repo = f"Qwen/Qwen3-VL-{normalized.upper()}-Instruct-GGUF"
    assert dependencies.cached_file.call_args_list == [
        call(
            repo_id=repo,
            filename=f"Qwen3VL-{normalized.upper()}-Instruct-Q4_K_M.gguf",
            revision=MODEL_REVISIONS[normalized],
            local_files_only=True,
        ),
        call(
            repo_id=repo,
            filename=f"mmproj-Qwen3VL-{normalized.upper()}-Instruct-F16.gguf",
            revision=MODEL_REVISIONS[normalized],
            local_files_only=True,
        ),
    ]
    assert len(MODEL_REVISIONS[normalized]) == 40
    dependencies.popen.assert_called_once()
    command = dependencies.popen.call_args.args[0]
    assert command[0] == str(dependencies.binary)
    for flag, expected in (
        ("--host", "127.0.0.1"),
        ("--port", "18765"),
        ("--parallel", "1"),
        ("--ctx-size", "16384"),
        ("--alias", f"qwen3-vl-{normalized}"),
        ("--image-min-tokens", "1024"),
        ("-ngl", "99"),
    ):
        assert command[command.index(flag) + 1] == expected
    assert not {
        "--device",
        "--no-mmproj-offload",
        "--no-op-offload",
    }.intersection(command)
    assert "--offline" in command and "--jinja" in command
    assert dependencies.popen.call_args.kwargs["start_new_session"] is True
    assert dependencies.client_factory.call_args.kwargs == {
        "base_url": "http://127.0.0.1:18765",
        "trust_env": False,
        "timeout": 600.0,
    }
    assert engine.runtime_info == {
        "backend": "llamacpp",
        "device": "unknown",
        "model": f"qwen3-vl-{normalized}",
    }
    engine.stop()


@pytest.mark.parametrize("size", ["4b", "8b"])
@pytest.mark.parametrize("device", ["auto", "cpu"])
def test_device_selection_controls_all_gpu_offloading(
    dependencies, monkeypatch, size, device
):
    monkeypatch.setenv("QWEN_DEVICE", device)
    engine = QwenEngine(size)
    engine.start()
    command = dependencies.popen.call_args.args[0]
    assert command[command.index("-ngl") + 1] == (
        "0" if device == "cpu" else "99"
    )
    if device == "cpu":
        assert command[command.index("--device") + 1] == "none"
        assert "--no-mmproj-offload" in command
        assert "--no-op-offload" in command
        assert engine.runtime_info["device"] == "cpu"
    else:
        assert not {
            "--device",
            "--no-mmproj-offload",
            "--no-op-offload",
        }.intersection(command)
        assert engine.runtime_info["device"] == "unknown"
    engine.stop()


def test_auto_reports_current_model_buffer_and_does_not_reuse_previous_start(
    dependencies,
):
    log_path = dependencies.tmp_path / "logs" / "qwen3-vl-4b.log"
    log_path.parent.mkdir()
    log_path.write_text("load_tensors: CUDA0 model buffer size = 900.00 MiB\n")
    native_spawn = dependencies.popen.side_effect
    starts = iter(
        (
            b"load_tensors: MTL0_Mapped model buffer size = 1200.00 MiB\n",
            b"load_tensors: CPU_Mapped model buffer size = 1200.00 MiB\n",
        )
    )

    def spawn(*args, **kwargs):
        kwargs["stdout"].write(next(starts))
        kwargs["stdout"].flush()
        return native_spawn(*args, **kwargs)

    dependencies.popen.side_effect = spawn
    engine = QwenEngine("4b")
    engine.start()
    assert engine.runtime_info["device"] == "metal"
    engine.stop()
    assert engine.runtime_info["device"] == "unknown"
    engine.start()
    assert engine.runtime_info["device"] == "unknown"
    engine.stop()


@pytest.mark.parametrize("device", ["cuda", "mps", ""])
def test_unsupported_device_is_rejected_before_loading(
    dependencies, monkeypatch, device
):
    monkeypatch.setenv("QWEN_DEVICE", device)
    with pytest.raises(ValueError, match="QWEN_DEVICE must be auto or cpu"):
        QwenEngine("4b")
    dependencies.cached_file.assert_not_called()
    dependencies.popen.assert_not_called()


def test_explicit_model_paths_bypass_hf_and_keep_spaces_as_single_arguments(
    dependencies, monkeypatch
):
    model = dependencies.tmp_path / "model with spaces.gguf"
    model.write_bytes(b"weights")
    projector = dependencies.files["mmproj-Qwen3VL-4B-Instruct-F16.gguf"]
    monkeypatch.setenv("QWEN3_VL_4B_MODEL_PATH", str(model))
    monkeypatch.setenv("QWEN3_VL_4B_MMPROJ_PATH", str(projector))
    engine = QwenEngine("4b")
    engine.start()
    dependencies.cached_file.assert_not_called()
    command = dependencies.popen.call_args.args[0]
    assert command[command.index("-m") + 1] == str(model)
    assert command[command.index("--mmproj") + 1] == str(projector)
    engine.stop()


def test_uncached_weights_do_not_start_or_download_a_model(dependencies):
    dependencies.cached_file.side_effect = RuntimeError("not cached")
    with pytest.raises(RuntimeError, match="prefetch"):
        QwenEngine("4b").start()
    assert (
        dependencies.cached_file.call_args.kwargs["local_files_only"] is True
    )
    dependencies.popen.assert_not_called()


def test_missing_override_file_fails_before_spawn(dependencies, monkeypatch):
    monkeypatch.setenv(
        "QWEN3_VL_4B_MODEL_PATH", str(dependencies.tmp_path / "missing.gguf")
    )
    with pytest.raises(RuntimeError, match="existing model file"):
        QwenEngine("4b").start()
    dependencies.popen.assert_not_called()


def test_one_instruct_request_returns_unmodified_text_without_coordinates(
    dependencies, monkeypatch
):
    monkeypatch.setenv("VLM_OCR_MAX_NEW_TOKENS", "8192")
    engine = QwenEngine("8b")
    image = Image.new("RGB", (57, 39), "red")
    with pytest.raises(RuntimeError, match="not started"):
        engine.recognize(image)
    engine.start()
    raw = "  ФАМИЛИЯ\nИВАНОВ\nABC 00123 & < >\n"
    dependencies.client.post.return_value.json.return_value = {
        "choices": [{"message": {"content": raw}}]
    }
    output = engine.recognize(image)
    dependencies.client.post.assert_called_once()
    call_args = dependencies.client.post.call_args
    assert call_args.args == ("/v1/chat/completions",)
    payload = call_args.kwargs["json"]
    assert payload["model"] == "qwen3-vl-8b"
    assert {
        key: value
        for key, value in payload.items()
        if key not in {"model", "messages"}
    } == {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": 1.5,
        "repeat_penalty": 1.0,
        "seed": 3407,
        "max_tokens": 8192,
        "stream": False,
    }
    content = payload["messages"][0]["content"]
    assert content[1] == {"type": "text", "text": OCR_PROMPT}
    uri = content[0]["image_url"]["url"]
    assert uri.startswith("data:image/png;base64,")
    decoded = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
    assert decoded.size == image.size and decoded.getpixel((0, 0)) == (
        255,
        0,
        0,
    )
    assert output == {
        "items": [
            {
                "text": raw,
                "confidence": None,
                "box": None,
                "detection_confidence": None,
            }
        ],
        "image_width": 57,
        "image_height": 39,
    }
    engine.stop()


def test_passport_task_uses_structured_extraction_prompt(dependencies):
    engine = QwenEngine("4b")
    engine.start()
    dependencies.client.post.return_value.json.return_value = {
        "choices": [{"message": {"content": '{"surname":"ИВАНОВ"}'}}],
    }

    output = engine.recognize(
        Image.new("RGB", (30, 30)),
        task="passport_fields",
    )

    payload = dependencies.client.post.call_args.kwargs["json"]
    assert payload["messages"][0]["content"][1] == {
        "type": "text",
        "text": PASSPORT_FIELDS_PROMPT,
    }
    assert output["items"][0]["text"] == '{"surname":"ИВАНОВ"}'
    engine.stop()


@pytest.mark.parametrize(
    "output",
    [
        None,
        {},
        {"choices": []},
        {"choices": [{}, {}]},
        {"choices": ["bad"]},
        {"choices": [{"message": None}]},
        {"choices": [{"message": {"content": ["not text"]}}]},
    ],
)
def test_malformed_completions_are_rejected_without_invented_text(
    dependencies, output
):
    engine = QwenEngine("4b")
    engine.start()
    dependencies.client.post.return_value.json.return_value = output
    with pytest.raises(RuntimeError, match="Qwen returned"):
        engine.recognize(Image.new("RGB", (30, 30)))
    dependencies.client.post.assert_called_once()
    engine.stop()


def test_request_timeout_has_no_retry_and_next_request_remains_usable(
    dependencies,
):
    engine = QwenEngine("4b")
    engine.start()
    dependencies.client.post.side_effect = httpx.ReadTimeout("timeout")
    with pytest.raises(TimeoutError, match="timed out"):
        engine.recognize(Image.new("RGB", (30, 30)))
    dependencies.client.post.assert_called_once()
    dependencies.client.post.side_effect = None
    assert (
        engine.recognize(Image.new("RGB", (30, 30)))["items"][0]["text"]
        == "ИВАНОВ\n1234567890"
    )
    engine.stop()


def test_startup_waits_for_health_and_cleans_up_a_failed_child(dependencies):
    dependencies.client.get.side_effect = [
        httpx.ConnectError("not ready"),
        SimpleNamespace(status_code=503),
        SimpleNamespace(status_code=200),
    ]
    engine = QwenEngine("4b")
    engine.start()
    assert dependencies.client.get.call_count == 3
    process = dependencies.processes[0]
    process.returncode = 7
    with pytest.raises(RuntimeError, match="not started"):
        engine.recognize(Image.new("RGB", (30, 30)))
    engine.stop()
    assert process.terminate_count == 0 and process.wait_timeouts == [None]
    dependencies.client.close.assert_called_once()


def test_startup_timeout_terminates_child_and_closes_log_and_client(
    dependencies, monkeypatch
):
    monkeypatch.setattr(qwen, "monotonic", Mock(side_effect=[0.0, 301.0]))
    engine = QwenEngine("4b")
    with pytest.raises(TimeoutError, match="startup timed out"):
        engine.start()
    assert dependencies.processes[0].returncode == -15
    assert dependencies.popen.call_args.kwargs["stdout"].closed
    assert (
        engine._process is None
        and engine._client is None
        and not engine._ready
    )
    dependencies.client.close.assert_called_once()


def test_spawn_failure_closes_opened_log(dependencies):
    dependencies.popen.side_effect = OSError("cannot execute")
    engine = QwenEngine("4b")
    with pytest.raises(OSError, match="cannot execute"):
        engine.start()
    assert dependencies.popen.call_args.kwargs["stdout"].closed
    dependencies.client_factory.assert_not_called()
    assert engine._process is None


def test_stop_escalates_to_kill_waits_and_allows_restart(dependencies):
    engine = QwenEngine("4b")
    engine.start()
    process = dependencies.processes[0]
    process.stubborn = True
    engine.stop()
    assert process.terminate_count == process.kill_count == 1
    assert process.wait_timeouts == [5.0, 5.0]
    assert process.returncode == -9 and engine._process is None
    engine.stop()
    assert process.kill_count == 1
    engine.start()
    assert (
        len(dependencies.processes) == 2
        and dependencies.processes[1].poll() is None
    )
    engine.stop()


def test_stop_retains_a_child_handle_if_kill_does_not_finish(dependencies):
    engine = QwenEngine("4b")
    process = Process(stubborn=True)
    process.kill = Mock()  # Simulate a process that still cannot be reaped.
    engine._process = process
    engine._client = dependencies.client
    with pytest.raises(subprocess.TimeoutExpired):
        engine.stop()
    assert engine._process is process and not engine._ready
    dependencies.client.close.assert_called_once()
    process.returncode = -9
    engine.stop()
    assert engine._process is None


def test_stop_reaps_a_real_child_without_loading_a_model(dependencies):
    process = REAL_POPEN(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    engine = QwenEngine("4b")
    engine._process = process
    try:
        engine.stop()
        assert process.poll() is not None and engine._process is None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.parametrize("size", ["2b", "qwen3-vl-32b", "unknown"])
def test_unknown_model_is_rejected(size):
    with pytest.raises(ValueError, match="4b or 8b"):
        QwenEngine(size)


@pytest.mark.parametrize(
    "name",
    [
        "VLM_OCR_MAX_NEW_TOKENS",
        "VLM_LLAMA_STARTUP_TIMEOUT_SECONDS",
        "VLM_LLAMA_SHUTDOWN_TIMEOUT_SECONDS",
        "VLM_OCR_REQUEST_TIMEOUT_SECONDS",
    ],
)
def test_invalid_environment_budgets_are_rejected(monkeypatch, name):
    monkeypatch.setenv(name, "0")
    with pytest.raises(ValueError, match="positive"):
        QwenEngine("4b")
