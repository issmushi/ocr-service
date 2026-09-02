from __future__ import annotations

import asyncio
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app.modules.vlm.client import recognize_vlm_image
from app.modules.vlm.server import VlmBackendRuntime
from app.modules.vlm.server import create_app as vlm_app
from app.ocr_engine.vlm.qwen import _device_from_startup_log as qwen_log_device
from tests.test_vlm_backend import FakeEngine as QwenFake
from tests.test_vlm_backend import image_bytes


@pytest.mark.parametrize(
    "cpu_only,report_runtime",
    ((False, False), (True, False), (False, True), (True, True)),
)
def test_optional_client_flags_are_sent_only_when_requested(
    cpu_only, report_runtime
):
    raw = {"items": [{"text": "  RAW<0I>\n"}], "device": "metal"}

    def respond(request):
        for flag, enabled in (
            ("cpu_only", cpu_only),
            ("report_runtime", report_runtime),
        ):
            marker = b'name="' + flag.encode() + b'"'
            assert (marker in request.content) is enabled
            if enabled:
                assert marker + b"\r\n\r\ntrue\r\n" in request.content
        return httpx.Response(200, json=raw)

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    with patch(
        "app.modules.vlm.client.httpx.AsyncClient", return_value=client
    ):
        kwargs = {"cpu_only": cpu_only, "report_runtime": report_runtime}
        actual = asyncio.run(
            recognize_vlm_image(b"mrz-bytes", model="qwen3-vl-8b", **kwargs)
        )
    assert actual == raw


@pytest.mark.parametrize("device", ("metal", "mps", "cuda", "cpu", "unknown"))
def test_reporting_device_is_opt_in_and_does_not_require_cpu(device):
    engine = QwenFake("qwen3-vl-4b", [])
    app = vlm_app(VlmBackendRuntime(lambda _: engine))
    form = {"model": "qwen3-vl-4b"}
    engine.runtime_info = {"backend": "fake", "device": device}
    with TestClient(app) as client:
        files = {"image": ("mrz.png", image_bytes(), "image/png")}
        plain = client.post("/recognize", files=files, data=form)
        reported = client.post(
            "/recognize", files=files, data={**form, "report_runtime": "true"}
        )
        assert plain.status_code == reported.status_code == 200
        assert "device" not in plain.json()
        assert reported.json() == {**plain.json(), "device": device}
        restricted = client.post(
            "/recognize",
            files=files,
            data={
                **form,
                "report_runtime": "true",
                "cpu_only": "true",
            },
        )
        assert restricted.status_code == (200 if device == "cpu" else 503)


@pytest.mark.parametrize(
    "current,expected",
    (
        (
            "load_tensors: MTL0_Mapped model buffer size = 1206.59 MiB\n",
            "metal",
        ),
        ("load_tensors: Metal model buffer size = 1420.10 MiB\n", "metal"),
        ("load_tensors: CUDA0 model buffer size = 2300.00 MiB\n", "cuda"),
        (
            "load_tensors: CPU_Mapped model buffer size = 2500.00 MiB\n",
            "unknown",
        ),
        ("load_tensors: MTL0 model buffer size = 0.00 MiB\n", "unknown"),
        ("ggml_metal_init: found device: Apple M3 Max\n", "unknown"),
        ("", "unknown"),
    ),
)
def test_startup_evidence_ignores_old_model_buffers_and_gpu_enumeration(
    tmp_path, current, expected
):
    path = tmp_path / "llama.log"
    old = b"load_tensors: MTL0_Mapped model buffer size = 3333.33 MiB\n"
    path.write_bytes(old + current.encode())
    assert qwen_log_device(path, len(old)) == expected
    assert qwen_log_device(tmp_path / "missing.log", 0) == "unknown"
