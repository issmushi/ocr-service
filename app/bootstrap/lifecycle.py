"""Start and stop the process-wide OCR runtimes and background tasks."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.bootstrap.background import (
    cleanup_expired_jobs,
    dispatch_outbox_events,
    recover_expired_leases,
)
from app.core.config import settings
from app.core.db.session import async_engine
from app.modules.paddle import paddle_runtime
from app.modules.russian.runtime import russian_runtime
from app.modules.v1.passport.runtime import mrz_scanner_runtime
from app.workers.ocr_worker import worker

logger = logging.getLogger(__name__)

# Routes use process-wide runtime instances; their maintenance tasks have the
# same ownership and are released together during application shutdown.
cleanup_task: asyncio.Task[None] | None = None
outbox_dispatcher_task: asyncio.Task[None] | None = None
lease_recovery_task: asyncio.Task[None] | None = None
worker_task: asyncio.Task[None] | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await startup()
    try:
        yield
    finally:
        await shutdown()


async def start_embedded_worker() -> asyncio.Task[None]:
    ready_event = asyncio.Event()
    task = asyncio.create_task(
        worker(ready_event),
        name="ocr-worker",
    )
    ready_task = asyncio.create_task(ready_event.wait())
    try:
        try:
            done, _pending = await asyncio.wait(
                {task, ready_task},
                timeout=settings.ocr_worker_startup_timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if task in done:
                error = task.exception()
                if error is not None:
                    raise RuntimeError("OCR worker failed to start") from error
                raise RuntimeError("OCR worker stopped during startup")
            if ready_task not in done:
                raise TimeoutError("OCR worker startup timed out")
            return task
        finally:
            # Cancellation here still belongs to startup: the caller has not
            # received the worker task and cannot include it in shutdown yet.
            ready_task.cancel()
            await asyncio.gather(ready_task, return_exceptions=True)
    except BaseException:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise


async def startup() -> None:
    global cleanup_task, lease_recovery_task
    global outbox_dispatcher_task, worker_task
    logger.info(
        "API startup host=%s port=%s queue=%s storage=%s log_dir=%s",
        settings.host,
        settings.port,
        settings.ocr_queue_name,
        settings.storage_path,
        settings.log_path,
    )
    try:
        if settings.is_model_enabled("paddle"):
            await paddle_runtime.start()
        if settings.is_model_enabled("russian-ocr"):
            await russian_runtime.start()
        if settings.is_model_enabled("mrz"):
            await mrz_scanner_runtime.start(lazy=True)
        worker_task = await start_embedded_worker()
        cleanup_task = asyncio.create_task(cleanup_expired_jobs())
        lease_recovery_task = asyncio.create_task(recover_expired_leases())
        outbox_dispatcher_task = asyncio.create_task(dispatch_outbox_events())
    except BaseException:
        # Startup can fail after a process has spawned or a background task
        # has started. Use the same complete cleanup as a normal shutdown.
        try:
            await shutdown()
        except BaseException:
            logger.exception("API startup cleanup failed")
        raise


async def _shutdown_resources() -> None:
    global cleanup_task, lease_recovery_task
    global outbox_dispatcher_task, worker_task
    background_tasks = (
        worker_task,
        cleanup_task,
        lease_recovery_task,
        outbox_dispatcher_task,
    )
    for task in background_tasks:
        if task:
            task.cancel()
    await asyncio.gather(
        *(task for task in background_tasks if task is not None),
        return_exceptions=True,
    )
    worker_task = None
    cleanup_task = None
    lease_recovery_task = None
    outbox_dispatcher_task = None
    # An individual cleanup failure must not leave the other native workers
    # running. Their independent processes can stop concurrently.
    results = await asyncio.gather(
        mrz_scanner_runtime.stop(),
        russian_runtime.stop(),
        paddle_runtime.stop(),
        async_engine.dispose(),
        return_exceptions=True,
    )
    errors = [
        result for result in results if isinstance(result, BaseException)
    ]
    if errors:
        raise errors[0]


async def shutdown() -> None:
    logger.info("API shutdown started")
    cleanup = asyncio.create_task(_shutdown_resources())
    cancelled = False
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            break
    if cancelled:
        if not cleanup.cancelled():
            cleanup.exception()
        raise asyncio.CancelledError
    cleanup.result()
    logger.info("API shutdown complete")
