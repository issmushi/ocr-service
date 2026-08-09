from __future__ import annotations

import asyncio
import threading
import unittest
from contextlib import contextmanager
from dataclasses import replace
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from app.core.config import settings
from app.core.errors import AppError, ErrorCode
from app.core.infrastructure.inference import (
    InferenceInputError,
    InferenceTimeoutError,
    InferenceUnavailableError,
    RemoteInferenceError,
)
from app.main import app
from app.modules.v1.passport.inference_worker import (
    mrz_scanner_inference_worker,
)
from app.modules.v1.passport.runtime import MrzScannerRuntime


class FakeExecutor:
    def __init__(self) -> None:
        self.status = "stopped"
        self.worker_pid = None
        self.restart_calls = 0
        self.stop_calls = 0
        self.run_calls = []
        self.state_after_restart = "ready"
        self.restart_entered = threading.Event()
        self.restart_gate = None
        self.stop_entered = threading.Event()
        self.stop_gate = None
        self.closed = False
        self.run_entered = asyncio.Event()
        self.run_gate = None
        self.error = None
        self.result = {"mrz": ["upper", "lower"]}
        self.active_calls = 0
        self.max_active_calls = 0

    @property
    def ready(self):
        return self.status == "ready"

    @property
    def busy(self):
        return self.status == "starting" or self.active_calls > 0

    def restart(self):
        if self.closed:
            return
        self.restart_calls += 1
        self.restart_entered.set()
        if self.restart_gate is not None:
            self.restart_gate.wait(timeout=2)
        self.status = self.state_after_restart
        self.worker_pid = 1234

    def stop(self):
        self.stop_calls += 1
        self.stop_entered.set()
        if self.stop_gate is not None:
            self.stop_gate.wait(timeout=2)
        self.closed = True
        self.status = "stopped"
        self.worker_pid = None

    async def run(self, operation, content, **kwargs):
        self.run_calls.append((operation, content, kwargs))
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        self.run_entered.set()
        try:
            if self.run_gate is not None:
                await self.run_gate.wait()
            if self.error is not None:
                raise self.error
            return self.result
        finally:
            self.active_calls -= 1


class MrzScannerRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.executors = []

        def create_executor(*args, **kwargs):
            executor = FakeExecutor()
            self.executors.append(executor)
            return executor

        self.factory_patch = patch(
            "app.modules.v1.passport.runtime.InferenceExecutor",
            side_effect=create_executor,
        )
        self.factory = self.factory_patch.start()
        self.addCleanup(self.factory_patch.stop)
        self.runtime = MrzScannerRuntime()
        self.executor = self.executors[0]

    async def asyncTearDown(self) -> None:
        await self.runtime.stop()

    async def run_document(self):
        return await self.runtime.run(
            "passport_v1", b"image", timeout_seconds=18
        )

    async def wait_for_spawn(self):
        self.assertTrue(
            await asyncio.to_thread(self.executor.restart_entered.wait, 1)
        )

    @contextmanager
    def mocked_api_lifecycle(self):
        from app.bootstrap import lifecycle

        async def start_worker():
            return asyncio.create_task(asyncio.sleep(3600))

        with (
            patch.object(lifecycle, "mrz_scanner_runtime", self.runtime),
            patch.object(lifecycle.paddle_runtime, "start", AsyncMock()),
            patch.object(lifecycle.paddle_runtime, "stop", AsyncMock()),
            patch.object(lifecycle.russian_runtime, "start", AsyncMock()),
            patch.object(lifecycle.russian_runtime, "stop", AsyncMock()),
            patch.object(
                lifecycle, "start_embedded_worker", side_effect=start_worker
            ),
            patch.object(lifecycle, "cleanup_expired_jobs", AsyncMock()),
            patch.object(lifecycle, "recover_expired_leases", AsyncMock()),
            patch.object(lifecycle, "dispatch_outbox_events", AsyncMock()),
            patch.object(
                lifecycle, "async_engine", SimpleNamespace(dispose=AsyncMock())
            ),
        ):
            yield lifecycle

    async def test_initializes_lazily_and_reuses_the_independent_executor(
        self,
    ):
        self.assertEqual(self.runtime.status, "stopped")
        self.assertFalse(self.runtime.ready)
        self.assertEqual(self.executor.restart_calls, 0)
        self.assertEqual(await self.run_document(), self.executor.result)
        self.assertEqual(await self.run_document(), self.executor.result)
        self.assertEqual(self.executor.restart_calls, 1)
        self.assertEqual(self.runtime.worker_pid, 1234)
        self.assertEqual(
            self.executor.run_calls,
            [
                ("passport_v1", b"image", {"timeout_seconds": 18}),
                ("passport_v1", b"image", {"timeout_seconds": 18}),
            ],
        )
        self.assertEqual(self.factory.call_args.args[0], "mrz-scanner")
        self.assertIs(
            self.factory.call_args.kwargs["worker_target"],
            mrz_scanner_inference_worker,
        )
        self.assertEqual(
            self.factory.call_args.kwargs["startup_timeout_seconds"],
            settings.mrz_scanner_startup_timeout_seconds,
        )

    async def test_serializes_requests_including_the_first_lazy_start(self):
        self.executor.run_gate = asyncio.Event()
        first = asyncio.create_task(self.run_document())
        await self.executor.run_entered.wait()
        second = asyncio.create_task(self.run_document())
        await asyncio.sleep(0)
        self.assertTrue(self.runtime.busy)
        self.assertEqual(len(self.executor.run_calls), 1)
        self.executor.run_gate.set()
        self.assertEqual(
            await asyncio.gather(first, second), [self.executor.result] * 2
        )
        self.assertEqual(self.executor.max_active_calls, 1)
        self.assertEqual(self.executor.restart_calls, 1)
        self.assertFalse(self.runtime.busy)

    async def test_cancelled_lazy_initialization_stops_process_and_can_retry(
        self,
    ):
        self.executor.state_after_restart = "starting"
        request = asyncio.create_task(self.run_document())
        await self.wait_for_spawn()
        request.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(request, timeout=1)
        self.assertEqual(self.executor.stop_calls, 1)
        self.assertEqual(self.executor.run_calls, [])
        self.assertFalse(self.runtime.busy)
        self.assertEqual(await self.run_document(), self.executors[1].result)
        self.assertEqual(self.executor.restart_calls, 1)
        self.assertEqual(self.executors[1].restart_calls, 1)

    async def test_cancellation_during_spawn_waits_before_stopping_process(
        self,
    ):
        self.executor.restart_gate = threading.Event()
        request = asyncio.create_task(self.run_document())
        try:
            await self.wait_for_spawn()
            request.cancel()
            await asyncio.sleep(0)
            self.assertFalse(request.done())
            self.assertEqual(self.executor.stop_calls, 0)
        finally:
            self.executor.restart_gate.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(request, timeout=1)
        self.assertEqual(self.executor.stop_calls, 1)
        self.assertEqual(self.runtime.status, "stopped")

    async def test_shutdown_during_startup_rejects_active_and_queued_requests(
        self,
    ):
        self.executor.state_after_restart = "starting"
        active = asyncio.create_task(self.run_document())
        await self.wait_for_spawn()
        queued = asyncio.create_task(self.run_document())
        await self.runtime.stop()
        for request in (active, queued):
            with self.assertRaises(AppError) as caught:
                await asyncio.wait_for(request, timeout=1)
            self.assertEqual(
                caught.exception.code, ErrorCode.OCR_ENGINE_UNAVAILABLE
            )
        self.assertEqual(self.executor.run_calls, [])
        self.assertEqual(self.runtime.status, "stopped")
        self.assertEqual(self.executor.restart_calls, 1)

    async def test_explicit_start_after_stop_uses_a_fresh_executor(self):
        await self.runtime.start()
        await self.runtime.stop()
        await self.runtime.start()
        self.assertEqual(len(self.executors), 2)
        self.assertEqual(self.executor.stop_calls, 1)
        self.assertEqual(self.executors[1].restart_calls, 1)
        self.assertTrue(self.runtime.ready)

    async def test_lazy_start_reopens_without_spawning_until_the_next_request(
        self,
    ):
        await self.runtime.start(lazy=True)
        self.assertFalse(self.runtime.ready)
        self.assertEqual(self.executor.restart_calls, 0)
        await self.run_document()
        await self.runtime.stop()
        await self.runtime.start(lazy=True)
        self.assertEqual(len(self.executors), 1)
        self.assertIsNone(self.runtime.worker_pid)
        self.assertEqual(self.runtime.status, "stopped")
        await self.run_document()
        self.assertEqual(len(self.executors), 2)
        self.assertEqual(self.executors[1].restart_calls, 1)

    async def test_repeated_api_lifespans_keep_mrz_lazy_and_available(self):
        with self.mocked_api_lifecycle() as lifecycle:
            for cycle in (1, 2):
                with self.subTest(cycle=cycle):
                    async with lifecycle.lifespan(app):
                        self.assertEqual(self.runtime.status, "stopped")
                        self.assertIsNone(self.runtime.worker_pid)
                        self.assertEqual(
                            await self.run_document(), self.executor.result
                        )
                    self.assertEqual(self.runtime.status, "stopped")
                    self.assertIsNone(self.runtime.worker_pid)
        self.assertEqual(len(self.executors), 2)
        self.assertTrue(
            all(executor.restart_calls == 1 for executor in self.executors)
        )

    async def test_failed_subsequent_api_startup_closes_the_reopened_runtime(
        self,
    ):
        with self.mocked_api_lifecycle() as lifecycle:
            async with lifecycle.lifespan(app):
                await self.run_document()
            lifecycle.start_embedded_worker.side_effect = RuntimeError(
                "worker startup failed"
            )
            with self.assertRaisesRegex(RuntimeError, "worker startup failed"):
                await lifecycle.startup()
            self.assertEqual(self.runtime.status, "stopped")
            self.assertIsNone(self.runtime.worker_pid)
            self.assertEqual(lifecycle.paddle_runtime.stop.await_count, 2)
            with self.assertRaises(AppError) as caught:
                await self.run_document()
            self.assertEqual(
                caught.exception.code, ErrorCode.OCR_ENGINE_UNAVAILABLE
            )
            self.assertEqual(len(self.executors), 1)

    async def test_cancelled_shutdown_still_stops_a_concurrent_spawn(self):
        self.executor.restart_gate = threading.Event()
        request = asyncio.create_task(self.run_document())
        await self.wait_for_spawn()
        stopping = asyncio.create_task(self.runtime.stop())
        try:
            await asyncio.sleep(0)
            stopping.cancel()
            await asyncio.sleep(0)
            self.assertFalse(stopping.done())
        finally:
            self.executor.restart_gate.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(stopping, timeout=1)
        with self.assertRaises(AppError) as caught:
            await asyncio.wait_for(request, timeout=1)
        self.assertEqual(
            caught.exception.code, ErrorCode.OCR_ENGINE_UNAVAILABLE
        )
        self.assertEqual(self.runtime.status, "stopped")
        self.assertEqual(self.executor.stop_calls, 1)

    async def test_start_during_stop_selects_executor_after_shutdown_finishes(
        self,
    ):
        await self.runtime.start()
        self.executor.stop_gate = threading.Event()
        stopping = asyncio.create_task(self.runtime.stop())
        self.assertTrue(
            await asyncio.to_thread(self.executor.stop_entered.wait, 1)
        )
        starting = asyncio.create_task(self.runtime.start())
        try:
            await asyncio.sleep(0)
        finally:
            self.executor.stop_gate.set()
        await asyncio.wait_for(asyncio.gather(stopping, starting), timeout=1)
        self.assertEqual(len(self.executors), 2)
        self.assertTrue(self.runtime.ready)
        self.assertEqual(self.executors[1].restart_calls, 1)

    async def test_startup_failure_is_reported_once_and_next_request_can_retry(
        self,
    ):
        self.executor.state_after_restart = "unavailable"
        with self.assertRaises(AppError) as caught:
            await self.run_document()
        self.assertEqual(
            caught.exception.code, ErrorCode.OCR_ENGINE_UNAVAILABLE
        )
        self.assertEqual(self.executor.restart_calls, 1)
        self.executor.state_after_restart = "ready"
        self.assertEqual(await self.run_document(), self.executor.result)
        self.assertEqual(self.executor.restart_calls, 2)

    async def test_waits_for_existing_recovery_without_restarting_again(self):
        self.executor.status = "starting"
        request = asyncio.create_task(self.run_document())
        await asyncio.sleep(0)
        self.executor.status = "ready"
        self.assertEqual(
            await asyncio.wait_for(request, timeout=1), self.executor.result
        )
        self.assertEqual(self.executor.restart_calls, 0)

    async def test_maps_executor_errors_to_existing_api_codes(self):
        for error, expected_code in (
            (InferenceTimeoutError("deadline"), ErrorCode.OCR_TIMEOUT),
            (
                InferenceUnavailableError("exited"),
                ErrorCode.OCR_ENGINE_UNAVAILABLE,
            ),
            (InferenceInputError("bad image"), ErrorCode.IMAGE_READ_ERROR),
            (
                RemoteInferenceError("RuntimeError", "failed"),
                ErrorCode.OCR_FAILED,
            ),
        ):
            with self.subTest(error=type(error).__name__):
                self.executor.error = error
                with self.assertRaises(AppError) as caught:
                    await self.run_document()
                self.assertEqual(caught.exception.code, expected_code)
                self.assertFalse(self.runtime.busy)

    async def test_closed_process_during_shutdown_maps_to_unavailable(self):
        self.executor.run_gate = asyncio.Event()
        self.executor.error = ValueError("process object is closed")
        request = asyncio.create_task(self.run_document())
        await self.executor.run_entered.wait()
        await self.runtime.stop()
        self.executor.run_gate.set()
        with self.assertRaises(AppError) as caught:
            await request
        self.assertEqual(
            caught.exception.code, ErrorCode.OCR_ENGINE_UNAVAILABLE
        )

    async def test_rejects_invalid_result_and_invalid_timeout(self):
        with self.assertRaises(ValueError):
            await self.runtime.run("passport_v1", b"image", timeout_seconds=0)
        self.assertEqual(self.executor.restart_calls, 0)
        self.executor.result = "invalid"
        with self.assertRaises(AppError) as caught:
            await self.run_document()
        self.assertEqual(caught.exception.code, ErrorCode.OCR_FAILED)


class MrzScannerWorkerTests(unittest.TestCase):
    def test_initialization_and_dispatch_use_only_the_mrz_factory_and_pipeline(
        self,
    ):
        engine = object()
        factory = ModuleType("app.ocr_engine.mrz.factory")
        factory.create_mrz_scanner = Mock(return_value=engine)
        pipeline = ModuleType("app.modules.v1.passport.pipeline")
        pipeline.process_passport_bytes_with_mrz_scanner = Mock(
            return_value={"mrz": []}
        )

        def serve(connection, initialize, dispatch):
            self.assertIs(connection, sentinel_connection)
            self.assertIs(initialize(), engine)
            self.assertEqual(
                dispatch(engine, "passport_v1", (b"image",), {}), {"mrz": []}
            )
            with self.assertRaisesRegex(
                ValueError, "Unsupported MRZ Scanner operation"
            ):
                dispatch(engine, "other", (), {})

        sentinel_connection = object()
        with (
            patch.dict(
                "sys.modules",
                {
                    "app.ocr_engine.mrz.factory": factory,
                    "app.modules.v1.passport.pipeline": pipeline,
                },
            ),
            patch(
                "app.modules.v1.passport.inference_worker.serve_inference_worker",
                side_effect=serve,
            ),
        ):
            mrz_scanner_inference_worker(sentinel_connection)
        factory.create_mrz_scanner.assert_called_once_with()
        pipeline.process_passport_bytes_with_mrz_scanner.assert_called_once_with(
            b"image", scanner=engine
        )

    def test_scanner_timeouts_must_be_positive(self):
        for name in (
            "mrz_scanner_startup_timeout_seconds",
            "mrz_scanner_shutdown_timeout_seconds",
            "mrz_scanner_document_hard_timeout_seconds",
        ):
            for value in (0, -1):
                with self.subTest(setting=name, value=value):
                    with self.assertRaisesRegex(ValueError, name.upper()):
                        replace(settings, **{name: value})


if __name__ == "__main__":
    unittest.main()
