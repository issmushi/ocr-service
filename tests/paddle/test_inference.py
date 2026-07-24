from __future__ import annotations

import asyncio
import logging
import os
import time
import unittest
from multiprocessing.connection import Connection
from typing import Any

from app.core.infrastructure.inference import (
    InferenceBusyError,
    InferenceExecutor,
    InferenceInputError,
    InferenceTimeoutError,
    InferenceUnavailableError,
    serve_inference_worker,
)


def inference_worker_fixture(connection: Connection) -> None:
    def initialize() -> None:
        return None

    def dispatch(
        _engine: Any,
        operation: str,
        args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> Any:
        if operation == "sleep":
            time.sleep(float(args[0]))
            return "done"
        if operation == "input_error":
            raise ValueError("invalid image")
        if operation == "crash":
            os._exit(17)
        return args[0] if args else "ok"

    serve_inference_worker(connection, initialize, dispatch)


class InferenceExecutorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.executor = InferenceExecutor(
            "test-engine",
            logging.getLogger(__name__),
            worker_target=inference_worker_fixture,
            startup_timeout_seconds=5,
            shutdown_timeout_seconds=0.2,
        )
        self.executor.start()

    def tearDown(self) -> None:
        self.executor.stop()

    async def wait_until_ready(self) -> None:
        for _ in range(200):
            if self.executor.ready:
                return
            await asyncio.sleep(0.01)
        self.fail("Inference worker did not become ready")

    async def test_rejects_parallel_inference_without_blocking_event_loop(
        self,
    ) -> None:
        first = asyncio.create_task(
            self.executor.run("sleep", 0.2, timeout_seconds=1)
        )
        await asyncio.sleep(0.02)

        self.assertTrue(self.executor.busy)
        with self.assertRaises(InferenceBusyError):
            await self.executor.run("echo", "second", timeout_seconds=1)

        self.assertEqual(await first, "done")
        self.assertFalse(self.executor.busy)

    async def test_hard_timeout_kills_and_restarts_worker(self) -> None:
        original_pid = self.executor.worker_pid

        with self.assertRaises(InferenceTimeoutError):
            await self.executor.run("sleep", 1, timeout_seconds=0.05)

        await self.wait_until_ready()
        self.assertNotEqual(self.executor.worker_pid, original_pid)
        self.assertEqual(
            await self.executor.run("echo", "ready", timeout_seconds=1),
            "ready",
        )

    async def test_cancelled_request_kills_and_restarts_worker(self) -> None:
        original_pid = self.executor.worker_pid
        request = asyncio.create_task(
            self.executor.run("sleep", 1, timeout_seconds=2)
        )
        await asyncio.sleep(0.02)
        request.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await request

        await self.wait_until_ready()
        self.assertNotEqual(self.executor.worker_pid, original_pid)

    async def test_remote_value_error_preserves_input_category(self) -> None:
        with self.assertRaises(InferenceInputError):
            await self.executor.run("input_error", timeout_seconds=1)
        self.assertTrue(self.executor.ready)

    async def test_crashed_worker_is_replaced(self) -> None:
        original_pid = self.executor.worker_pid

        with self.assertRaises(InferenceUnavailableError):
            await self.executor.run("crash", timeout_seconds=1)

        await self.wait_until_ready()
        self.assertNotEqual(self.executor.worker_pid, original_pid)


if __name__ == "__main__":
    unittest.main()
