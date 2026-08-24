from __future__ import annotations

import asyncio
import io
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.modules.vlm.server import (
    MODELS,
    VlmBackendRuntime,
    VlmUnavailableError,
    create_app,
    create_engine,
)


def image_bytes() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (12, 20), "white").save(stream, format="PNG")
    return stream.getvalue()


class FakeEngine:
    runtime_info = {"backend": "fake", "device": "cpu"}

    def __init__(
        self,
        model,
        events,
        *,
        start_error=None,
        recognize_error=None,
        stop_error=None,
    ):
        self.model = model
        self.events = events
        self.start_error = start_error
        self.recognize_error = recognize_error
        self.stop_error = stop_error

    def record(self, operation):
        self.events.append((self.model, operation, threading.get_ident()))

    def start(self):
        self.record("start")
        if self.start_error:
            raise self.start_error

    def stop(self):
        self.record("stop")
        if self.stop_error:
            raise self.stop_error

    def recognize(self, image, *, task="ocr"):
        self.record("recognize")
        if self.recognize_error:
            raise self.recognize_error
        assert image.mode == "RGB"
        return {
            "items": [
                {"text": "  ПАСПОРТ\n", "confidence": None, "box": None}
            ],
            "image_width": image.width,
            "image_height": image.height,
        }


def calls(events):
    return [(model, operation) for model, operation, _ in events]


def test_models_load_lazily_reuse_and_switch_on_one_thread():
    events = []
    runtime = VlmBackendRuntime(lambda model: FakeEngine(model, events))
    app = create_app(runtime)
    with TestClient(app) as client:
        assert client.get("/health").json() == {
            "status": "ready",
            "models": list(MODELS),
            "active_model": None,
        }
        assert not events
        for model in ("qwen3-vl-4b", "qwen3-vl-4b", "qwen3-vl-8b"):
            response = client.post(
                "/recognize",
                files={"image": ("scan.png", image_bytes(), "image/png")},
                data={"model": model},
            )
            assert response.status_code == 200
            assert response.json()["items"][0]["text"] == "  ПАСПОРТ\n"
            assert response.json()["image_width"] == 12
            assert client.get("/health").json()["active_model"] == model

    assert calls(events) == [
        ("qwen3-vl-4b", "start"),
        ("qwen3-vl-4b", "recognize"),
        ("qwen3-vl-4b", "recognize"),
        ("qwen3-vl-4b", "stop"),
        ("qwen3-vl-8b", "start"),
        ("qwen3-vl-8b", "recognize"),
        ("qwen3-vl-8b", "stop"),
    ]
    assert len({thread for _, _, thread in events}) == 1
    assert events[0][2] != threading.get_ident()
    assert runtime.status == "stopped"
    assert runtime.active_model is None
    # A stopped service can restart without retaining the previous model.
    with TestClient(app) as client:
        assert client.get("/health").json()["active_model"] is None


def test_passport_task_is_forwarded_to_the_loaded_engine():
    events, tasks = [], []

    class TaskEngine(FakeEngine):
        def recognize(self, image, *, task="ocr"):
            tasks.append(task)
            return super().recognize(image, task=task)

    with TestClient(
        create_app(VlmBackendRuntime(lambda model: TaskEngine(model, events)))
    ) as client:
        response = client.post(
            "/recognize",
            files={"image": ("scan.png", image_bytes(), "image/png")},
            data={"model": "qwen3-vl-4b", "task": "passport_fields"},
        )
        invalid = client.post(
            "/recognize",
            files={"image": ("scan.png", image_bytes(), "image/png")},
            data={"model": "qwen3-vl-4b", "task": "unknown"},
        )

    assert response.status_code == 200
    assert invalid.status_code == 422
    assert tasks == ["passport_fields"]


@pytest.mark.parametrize("size", ("4b", "8b"))
def test_qwen_factory_constructs_adapter_without_loading_weights(size):
    adapter = Mock()
    engine = object()
    adapter.return_value = engine
    with patch.dict(
        "sys.modules",
        {
            "app.ocr_engine.vlm.qwen": SimpleNamespace(QwenEngine=adapter),
        },
    ):
        assert create_engine(f"qwen3-vl-{size}") is engine
    adapter.assert_called_once_with(size)


def test_bad_image_and_unknown_model_do_not_load_any_engine():
    events = []
    runtime = VlmBackendRuntime(lambda model: FakeEngine(model, events))
    with TestClient(create_app(runtime)) as client:
        bad = client.post(
            "/recognize",
            files={"image": ("scan.png", b"invalid", "image/png")},
            data={"model": "qwen3-vl-4b"},
        )
        assert bad.status_code == 400
        for model in (
            "paddle",
            "removed-model",
            "paddleocr-vl-1.6",
            "chandra-ocr-2",
            "unknown",
        ):
            invalid = client.post(
                "/recognize",
                files={"image": ("scan.png", image_bytes(), "image/png")},
                data={"model": model},
            )
            assert invalid.status_code == 422
            with pytest.raises(ValueError, match="Unsupported VLM model"):
                create_engine(model)
        assert not events
        assert runtime.active_model is None

    asyncio.run(runtime.stop())
    with pytest.raises(VlmUnavailableError):
        asyncio.run(runtime.recognize(image_bytes(), "qwen3-vl-4b"))


def test_failed_load_cleans_up_and_next_model_can_run_without_leaking_details(
    caplog,
):
    events = []

    def factory(model):
        return FakeEngine(
            model,
            events,
            start_error=RuntimeError("private model details")
            if model == "qwen3-vl-8b"
            else None,
        )

    runtime = VlmBackendRuntime(factory)
    with TestClient(create_app(runtime)) as client:
        failed = client.post(
            "/recognize",
            files={"image": ("scan.png", image_bytes())},
            data={"model": "qwen3-vl-8b"},
        )
        assert failed.status_code == 503
        assert "private" not in failed.text
        assert runtime.active_model is None
        assert calls(events) == [
            ("qwen3-vl-8b", "start"),
            ("qwen3-vl-8b", "stop"),
        ]
        assert (
            client.post(
                "/recognize",
                files={"image": ("scan.png", image_bytes())},
                data={"model": "qwen3-vl-4b"},
            ).status_code
            == 200
        )
    assert "private" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
    assert calls(events)[-1] == ("qwen3-vl-4b", "stop")


def test_native_timeout_and_inference_error_leave_loaded_model_usable(caplog):
    events = []
    engine = FakeEngine("qwen3-vl-4b", events)
    runtime = VlmBackendRuntime(lambda model: engine)
    with TestClient(create_app(runtime)) as client:
        for error, status in (
            (TimeoutError("private OCR text"), 504),
            (ValueError("private OCR text"), 422),
        ):
            engine.recognize_error = error
            response = client.post(
                "/recognize",
                files={"image": ("scan.png", image_bytes())},
                data={"model": "qwen3-vl-4b"},
            )
            assert response.status_code == status
            assert "private" not in response.text
        engine.recognize_error = None
        assert (
            client.post(
                "/recognize",
                files={"image": ("scan.png", image_bytes())},
                data={"model": "qwen3-vl-4b"},
            ).status_code
            == 200
        )
    assert calls(events).count(("qwen3-vl-4b", "start")) == 1
    assert "private" not in caplog.text


def test_missing_adapter_does_not_block_another_model():
    events = []

    def factory(model):
        if model == "qwen3-vl-8b":
            raise ImportError("private dependency path")
        return FakeEngine(model, events)

    runtime = VlmBackendRuntime(factory)
    with TestClient(create_app(runtime)) as client:
        failed = client.post(
            "/recognize",
            files={"image": ("scan.png", image_bytes())},
            data={"model": "qwen3-vl-8b"},
        )
        assert failed.status_code == 503
        assert "private" not in failed.text
        assert runtime.active_model is None
        assert not events
        assert (
            client.post(
                "/recognize",
                files={"image": ("scan.png", image_bytes())},
                data={"model": "qwen3-vl-4b"},
            ).status_code
            == 200
        )


def test_failed_unload_keeps_old_model_owner_and_prevents_loading_another():
    events, created = [], []
    first_engine = FakeEngine("qwen3-vl-4b", events)

    def factory(model):
        created.append(model)
        return (
            first_engine
            if model == "qwen3-vl-4b"
            else FakeEngine(model, events)
        )

    runtime = VlmBackendRuntime(factory)
    with TestClient(create_app(runtime)) as client:
        assert (
            client.post(
                "/recognize",
                files={"image": ("scan.png", image_bytes())},
                data={"model": "qwen3-vl-4b"},
            ).status_code
            == 200
        )
        first_engine.stop_error = RuntimeError(
            "native process is still running"
        )
        failed = client.post(
            "/recognize",
            files={"image": ("scan.png", image_bytes())},
            data={"model": "qwen3-vl-8b"},
        )
        assert failed.status_code == 503
        assert created == ["qwen3-vl-4b"]
        assert runtime.active_model == "qwen3-vl-4b"
        first_engine.stop_error = None
        assert (
            client.post(
                "/recognize",
                files={"image": ("scan.png", image_bytes())},
                data={"model": "qwen3-vl-8b"},
            ).status_code
            == 200
        )
        assert calls(events)[-3:] == [
            ("qwen3-vl-4b", "stop"),
            ("qwen3-vl-8b", "start"),
            ("qwen3-vl-8b", "recognize"),
        ]


def test_failed_start_and_failed_cleanup_retain_owner_until_cleanup_succeeds():
    events, created = [], []
    failed_engine = FakeEngine(
        "qwen3-vl-8b",
        events,
        start_error=RuntimeError("load failed"),
        stop_error=RuntimeError("cleanup failed"),
    )

    def factory(model):
        created.append(model)
        return (
            failed_engine
            if model == "qwen3-vl-8b"
            else FakeEngine(model, events)
        )

    runtime = VlmBackendRuntime(factory)
    with TestClient(create_app(runtime)) as client:
        for model in ("qwen3-vl-8b", "qwen3-vl-4b"):
            assert (
                client.post(
                    "/recognize",
                    files={"image": ("scan.png", image_bytes())},
                    data={"model": model},
                ).status_code
                == 503
            )
        assert created == ["qwen3-vl-8b"]
        assert runtime.active_model is None
        failed_engine.stop_error = None
        assert (
            client.post(
                "/recognize",
                files={"image": ("scan.png", image_bytes())},
                data={"model": "qwen3-vl-4b"},
            ).status_code
            == 200
        )
        assert calls(events)[-3:] == [
            ("qwen3-vl-8b", "stop"),
            ("qwen3-vl-4b", "start"),
            ("qwen3-vl-4b", "recognize"),
        ]


def test_cancelled_request_blocks_model_switch_during_inference():
    async def check():
        events = []
        entered, release = threading.Event(), threading.Event()
        first_engine = FakeEngine("qwen3-vl-4b", events)
        recognize = first_engine.recognize

        def blocked(image, **kwargs):
            first_engine.record("inference_enter")
            entered.set()
            assert release.wait(5)
            return recognize(image, **kwargs)

        first_engine.recognize = blocked
        runtime = VlmBackendRuntime(
            lambda model: (
                first_engine
                if model == "qwen3-vl-4b"
                else FakeEngine(model, events)
            )
        )
        await runtime.start()
        try:
            first = asyncio.create_task(
                runtime.recognize(image_bytes(), "qwen3-vl-4b")
            )
            assert await asyncio.to_thread(entered.wait, 5)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            second = asyncio.create_task(
                runtime.recognize(image_bytes(), "qwen3-vl-8b")
            )
            await asyncio.sleep(0)
            assert not second.done()
            assert calls(events) == [
                ("qwen3-vl-4b", "start"),
                ("qwen3-vl-4b", "inference_enter"),
            ]
            release.set()
            assert (await second)["image_height"] == 20
            assert calls(events)[2:] == [
                ("qwen3-vl-4b", "recognize"),
                ("qwen3-vl-4b", "stop"),
                ("qwen3-vl-8b", "start"),
                ("qwen3-vl-8b", "recognize"),
            ]
        finally:
            release.set()
            await runtime.stop()

    asyncio.run(check())


def test_cancelled_shutdown_waits_for_inference_before_unloading():
    async def check():
        events = []
        entered, release = threading.Event(), threading.Event()
        engine = FakeEngine("qwen3-vl-4b", events)
        recognize = engine.recognize

        def blocked(image, **kwargs):
            entered.set()
            assert release.wait(5)
            return recognize(image, **kwargs)

        engine.recognize = blocked
        runtime = VlmBackendRuntime(lambda model: engine)
        await runtime.start()
        try:
            request = asyncio.create_task(
                runtime.recognize(image_bytes(), "qwen3-vl-4b")
            )
            assert await asyncio.to_thread(entered.wait, 5)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            stopping = asyncio.create_task(runtime.stop())
            await asyncio.sleep(0)
            stopping.cancel()
            await asyncio.sleep(0)
            assert runtime.status == "stopping"
            assert not stopping.done()
            assert calls(events) == [("qwen3-vl-4b", "start")]
            with pytest.raises(VlmUnavailableError):
                await runtime.recognize(image_bytes(), "qwen3-vl-8b")
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await stopping
            assert calls(events) == [
                ("qwen3-vl-4b", "start"),
                ("qwen3-vl-4b", "recognize"),
                ("qwen3-vl-4b", "stop"),
            ]
            assert runtime.status == "stopped"
            assert runtime.active_model is None
        finally:
            release.set()
            await runtime.stop()

    asyncio.run(check())
