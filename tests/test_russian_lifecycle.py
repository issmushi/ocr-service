from __future__ import annotations

import asyncio
import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.bootstrap import lifecycle
from app.main import app


@contextmanager
def mocked_lifecycle():
    events = []
    tasks = []

    def runtime(name):
        return SimpleNamespace(
            start=AsyncMock(
                side_effect=lambda **kwargs: events.append(
                    (name, "start", kwargs)
                )
            ),
            stop=AsyncMock(side_effect=lambda: events.append((name, "stop"))),
        )

    async def wait_forever():
        await asyncio.Event().wait()

    async def start_worker():
        task = asyncio.create_task(wait_forever())
        tasks.append(task)
        return task

    mocks = SimpleNamespace(
        paddle=runtime("paddle"),
        russian=runtime("russian"),
        mrz=runtime("mrz"),
        engine=SimpleNamespace(dispose=AsyncMock()),
        start_worker=AsyncMock(side_effect=start_worker),
        tasks=tasks,
        events=events,
    )
    replacements = {
        "paddle_runtime": mocks.paddle,
        "russian_runtime": mocks.russian,
        "mrz_scanner_runtime": mocks.mrz,
        "async_engine": mocks.engine,
        "start_embedded_worker": mocks.start_worker,
        "cleanup_expired_jobs": wait_forever,
        "recover_expired_leases": wait_forever,
        "dispatch_outbox_events": wait_forever,
        "worker_task": None,
        "cleanup_task": None,
        "lease_recovery_task": None,
        "outbox_dispatcher_task": None,
    }
    with ExitStack() as stack:
        for name, replacement in replacements.items():
            stack.enter_context(patch.object(lifecycle, name, replacement))
        yield mocks


class RussianLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_lifespan_starts_russian_eagerly_and_mrz_lazily_then_stops(
        self,
    ):
        with mocked_lifecycle() as mocks:
            async with lifecycle.lifespan(app):
                self.assertEqual(
                    mocks.events[:3],
                    [
                        ("paddle", "start", {}),
                        ("russian", "start", {}),
                        ("mrz", "start", {"lazy": True}),
                    ],
                )
                self.assertIs(lifecycle.worker_task, mocks.tasks[0])
                backgrounds = [
                    lifecycle.worker_task,
                    lifecycle.cleanup_task,
                    lifecycle.lease_recovery_task,
                    lifecycle.outbox_dispatcher_task,
                ]
            for runtime in (mocks.paddle, mocks.russian, mocks.mrz):
                runtime.stop.assert_awaited_once_with()
            mocks.engine.dispose.assert_awaited_once_with()
            self.assertTrue(all(task.done() for task in backgrounds))
            self.assertTrue(
                all(
                    getattr(lifecycle, name) is None
                    for name in (
                        "worker_task",
                        "cleanup_task",
                        "lease_recovery_task",
                        "outbox_dispatcher_task",
                    )
                )
            )

    async def test_failure_at_any_startup_stage_cleans_all_workers(self):
        for stage in ("paddle", "russian", "mrz", "start_worker"):
            with self.subTest(stage=stage), mocked_lifecycle() as mocks:
                operation = (
                    mocks.start_worker
                    if stage == "start_worker"
                    else getattr(mocks, stage).start
                )
                operation.side_effect = RuntimeError(f"{stage} failed")
                with self.assertRaisesRegex(RuntimeError, f"{stage} failed"):
                    await lifecycle.startup()
                for runtime in (mocks.paddle, mocks.russian, mocks.mrz):
                    runtime.stop.assert_awaited_once_with()
                mocks.engine.dispose.assert_awaited_once_with()

    async def test_cleanup_error_does_not_mask_the_original_startup_failure(
        self,
    ):
        with mocked_lifecycle() as mocks:
            mocks.russian.start.side_effect = RuntimeError(
                "model initialization failed"
            )
            mocks.russian.stop.side_effect = RuntimeError("stop failed")
            with self.assertLogs("app.bootstrap.lifecycle", level="ERROR"):
                with self.assertRaisesRegex(
                    RuntimeError, "model initialization failed"
                ):
                    await lifecycle.startup()
            mocks.paddle.stop.assert_awaited_once_with()
            mocks.mrz.stop.assert_awaited_once_with()
            mocks.engine.dispose.assert_awaited_once_with()

    async def test_cancelled_startup_cleans_already_started_processes(self):
        entered = asyncio.Event()

        async def starting_russian():
            entered.set()
            await asyncio.Event().wait()

        with mocked_lifecycle() as mocks:
            mocks.russian.start.side_effect = starting_russian
            task = asyncio.create_task(lifecycle.startup())
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
            for runtime in (mocks.paddle, mocks.russian, mocks.mrz):
                runtime.stop.assert_awaited_once_with()
            mocks.engine.dispose.assert_awaited_once_with()

    async def test_one_stop_failure_still_cleans_other_workers_and_database(
        self,
    ):
        with mocked_lifecycle() as mocks:
            await lifecycle.startup()
            mocks.mrz.stop.side_effect = RuntimeError("MRZ stop failed")
            with self.assertRaisesRegex(RuntimeError, "MRZ stop failed"):
                await lifecycle.shutdown()
            mocks.russian.stop.assert_awaited_once_with()
            mocks.paddle.stop.assert_awaited_once_with()
            mocks.engine.dispose.assert_awaited_once_with()
            self.assertTrue(mocks.tasks[0].done())

    async def test_cancelled_shutdown_waits_until_worker_cleanup_finishes(
        self,
    ):
        entered = asyncio.Event()
        finish = asyncio.Event()

        async def stopping_russian():
            entered.set()
            await finish.wait()

        with mocked_lifecycle() as mocks:
            await lifecycle.startup()
            mocks.russian.stop.side_effect = stopping_russian
            task = asyncio.create_task(lifecycle.shutdown())
            await entered.wait()
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            task.cancel()
            finish.set()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
            mocks.russian.stop.assert_awaited_once_with()
            mocks.paddle.stop.assert_awaited_once_with()
            mocks.mrz.stop.assert_awaited_once_with()
            mocks.engine.dispose.assert_awaited_once_with()

    async def test_worker_startup_error_preserves_the_original_cause(self):
        failure = ValueError("queue unavailable")

        async def failing_worker(ready_event):
            raise failure

        with patch.object(lifecycle, "worker", side_effect=failing_worker):
            with self.assertRaisesRegex(
                RuntimeError, "OCR worker failed to start"
            ) as caught:
                await lifecycle.start_embedded_worker()

        self.assertIs(caught.exception.__cause__, failure)

    async def test_worker_stopping_before_readiness_fails_startup(self):
        async def stopped_worker(ready_event):
            return

        with patch.object(lifecycle, "worker", side_effect=stopped_worker):
            with self.assertRaisesRegex(
                RuntimeError, "OCR worker stopped during startup"
            ):
                await lifecycle.start_embedded_worker()

    async def test_worker_startup_timeout_waits_for_worker_cleanup(self):
        stopped = asyncio.Event()

        async def stuck_worker(ready_event):
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        with (
            patch.object(lifecycle, "worker", side_effect=stuck_worker),
            patch.object(
                lifecycle,
                "settings",
                SimpleNamespace(ocr_worker_startup_timeout_seconds=0.01),
            ),
        ):
            with self.assertRaisesRegex(
                TimeoutError, "OCR worker startup timed out"
            ):
                await lifecycle.start_embedded_worker()

        self.assertTrue(stopped.is_set())

    async def test_cancelled_embedded_worker_start_does_not_orphan_its_task(
        self,
    ):
        entered = asyncio.Event()
        stopped = asyncio.Event()

        async def starting_worker(ready_event):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        with patch.object(lifecycle, "worker", side_effect=starting_worker):
            task = asyncio.create_task(lifecycle.start_embedded_worker())
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
            self.assertTrue(stopped.is_set())

    async def test_cancel_after_worker_is_ready_still_stops_unreturned_worker(
        self,
    ):
        stopped = asyncio.Event()
        original_gather = asyncio.gather

        async def ready_worker(ready_event):
            ready_event.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        def cancel_during_ready_cleanup(*tasks, **kwargs):
            if (
                len(tasks) == 1
                and tasks[0].get_coro().__qualname__ == "Event.wait"
            ):
                asyncio.current_task().cancel()
            return original_gather(*tasks, **kwargs)

        with (
            patch.object(lifecycle, "worker", side_effect=ready_worker),
            patch.object(
                lifecycle.asyncio,
                "gather",
                side_effect=cancel_during_ready_cleanup,
            ),
        ):
            task = asyncio.create_task(lifecycle.start_embedded_worker())
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
            self.assertTrue(stopped.is_set())


if __name__ == "__main__":
    unittest.main()
