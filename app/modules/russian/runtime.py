from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from app.core.config import settings
from app.core.errors import AppError, ErrorCode
from app.core.infrastructure.inference import (
    InferenceExecutor,
    InferenceInputError,
    InferenceTimeoutError,
    InferenceUnavailableError,
    RemoteInferenceError,
)
from app.modules.russian.inference_worker import russian_inference_worker

logger = logging.getLogger(__name__)


async def _complete_lifecycle_task(task: asyncio.Task[None]) -> None:
    """Finish lifecycle work before propagating any caller cancellation."""
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            if not cancelled:
                raise
    if cancelled:
        # Retrieve a possible exception before propagating cancellation.
        if not task.cancelled():
            task.exception()
        raise asyncio.CancelledError
    task.result()


async def _complete_lifecycle_call(operation: Callable[[], None]) -> None:
    await _complete_lifecycle_task(
        asyncio.create_task(asyncio.to_thread(operation))
    )


class RussianRuntime:
    """Own a RussianDocsOCR process and serialize inference requests."""

    def __init__(self) -> None:
        self._execution_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._closing = False
        self._replace_executor = False
        self._executor = self._create_executor()

    @staticmethod
    def _create_executor() -> InferenceExecutor:
        return InferenceExecutor(
            "russian-docs-ocr",
            logger,
            worker_target=russian_inference_worker,
            worker_args=(settings.russian_ocr_device,),
            startup_timeout_seconds=settings.russian_ocr_startup_timeout_seconds,
            shutdown_timeout_seconds=settings.russian_ocr_shutdown_timeout_seconds,
        )

    @property
    def ready(self) -> bool:
        return self._executor.ready

    @property
    def busy(self) -> bool:
        return self._execution_lock.locked() or self._executor.busy

    @property
    def worker_pid(self) -> int | None:
        return self._executor.worker_pid

    @property
    def status(self) -> str:
        return self._executor.status

    async def _stop_executor(self, executor: InferenceExecutor) -> None:
        async def cleanup() -> None:
            async with self._lifecycle_lock:
                try:
                    await asyncio.to_thread(executor.stop)
                finally:
                    if executor is self._executor:
                        self._replace_executor = True

        # Protect lock acquisition too: a cancelled shutdown may be queued
        # behind spawn, and must still stop the process when spawn finishes.
        await _complete_lifecycle_task(asyncio.create_task(cleanup()))

    async def _ensure_ready(self) -> InferenceExecutor:
        if self._closing:
            raise InferenceUnavailableError(
                "RussianDocsOCR runtime is stopping"
            )
        executor: InferenceExecutor | None = None
        try:
            async with self._lifecycle_lock:
                if self._closing:
                    raise InferenceUnavailableError(
                        "RussianDocsOCR runtime is stopping"
                    )
                if self._replace_executor:
                    self._executor = self._create_executor()
                    self._replace_executor = False
                executor = self._executor
                status = await asyncio.to_thread(lambda: executor.status)
                if status in {"stopped", "unavailable"}:
                    logger.info("Starting RussianDocsOCR runtime")
                    # restart() only spawns; unlike start(), it does not
                    # leave a thread waiting for initialization after cancel.
                    await _complete_lifecycle_call(executor.restart)

            while not self._closing:
                status = await asyncio.to_thread(lambda: executor.status)
                if status == "ready":
                    return executor
                if status != "starting":
                    raise InferenceUnavailableError(
                        "RussianDocsOCR worker failed to start"
                    )
                await asyncio.sleep(0.05)
            raise InferenceUnavailableError(
                "RussianDocsOCR runtime is stopping"
            )
        except asyncio.CancelledError:
            if executor is not None:
                await self._stop_executor(executor)
            raise
        except OSError as error:
            raise InferenceUnavailableError(
                "RussianDocsOCR worker failed to start"
            ) from error

    async def start(self, *, lazy: bool = False) -> None:
        async with self._execution_lock:
            self._closing = False
            if not lazy:
                await self._ensure_ready()

    async def stop(self) -> None:
        logger.info("Stopping RussianDocsOCR runtime")
        self._closing = True
        await self._stop_executor(self._executor)

    async def run(
        self,
        operation: str,
        content: bytes,
        *,
        timeout_seconds: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        try:
            async with self._execution_lock:
                executor = await self._ensure_ready()
                result = await executor.run(
                    operation,
                    content,
                    timeout_seconds=timeout_seconds,
                    **kwargs,
                )
        except InferenceTimeoutError as error:
            raise AppError(
                ErrorCode.OCR_TIMEOUT, "Превышен таймаут обработки"
            ) from error
        except InferenceUnavailableError as error:
            raise AppError(
                ErrorCode.OCR_ENGINE_UNAVAILABLE, "OCR engine недоступен"
            ) from error
        except InferenceInputError as error:
            raise AppError(
                ErrorCode.IMAGE_READ_ERROR, "Изображение невозможно прочитать"
            ) from error
        except RemoteInferenceError as error:
            logger.error(
                "RussianDocsOCR inference failed operation=%s "
                "error_type=%s remote_traceback=%s",
                operation,
                error.error_type,
                error.remote_traceback,
            )
            raise AppError(
                ErrorCode.OCR_FAILED,
                "OCR не смог получить пригодный результат",
            ) from error
        except ValueError as error:
            # Closing a multiprocessing.Process during shutdown can interrupt
            # the executor's is_alive() check with 'process object is closed'.
            if not self._closing:
                raise
            raise AppError(
                ErrorCode.OCR_ENGINE_UNAVAILABLE, "OCR engine недоступен"
            ) from error

        if not isinstance(result, dict):
            logger.error(
                "RussianDocsOCR returned invalid result operation=%s type=%s",
                operation,
                type(result).__name__,
            )
            raise AppError(
                ErrorCode.OCR_FAILED,
                "OCR не смог получить пригодный результат",
            )
        return result


russian_runtime = RussianRuntime()
