from __future__ import annotations

import asyncio
import logging
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
from app.modules.paddle.inference_worker import paddle_inference_worker

logger = logging.getLogger(__name__)


class PaddleRuntime:
    """Own one PaddleOCR model process and serialize every inference call."""

    def __init__(self) -> None:
        self._execution_lock = asyncio.Lock()
        self._executor = InferenceExecutor(
            "paddleocr",
            logger,
            worker_target=paddle_inference_worker,
            worker_args=(settings.ocr_device,),
            startup_timeout_seconds=(
                settings.paddle_ocr_startup_timeout_seconds
            ),
            shutdown_timeout_seconds=(
                settings.paddle_ocr_shutdown_timeout_seconds
            ),
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

    async def start(self) -> None:
        logger.info(
            "Starting embedded PaddleOCR runtime device=%s",
            settings.ocr_device,
        )
        await asyncio.to_thread(self._executor.start)

    async def stop(self) -> None:
        logger.info("Stopping embedded PaddleOCR runtime")
        await asyncio.to_thread(self._executor.stop)

    async def run(
        self,
        operation: str,
        content: bytes,
        *,
        timeout_seconds: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            async with self._execution_lock:
                result = await self._executor.run(
                    operation,
                    content,
                    timeout_seconds=timeout_seconds,
                    **kwargs,
                )
        except InferenceTimeoutError as error:
            raise AppError(
                ErrorCode.OCR_TIMEOUT,
                "Превышен таймаут обработки",
            ) from error
        except InferenceUnavailableError as error:
            raise AppError(
                ErrorCode.OCR_ENGINE_UNAVAILABLE,
                "OCR engine недоступен",
            ) from error
        except InferenceInputError as error:
            raise AppError(
                ErrorCode.IMAGE_READ_ERROR,
                "Изображение невозможно прочитать",
            ) from error
        except RemoteInferenceError as error:
            logger.error(
                "PaddleOCR inference failed operation=%s error_type=%s "
                "remote_traceback=%s",
                operation,
                error.error_type,
                error.remote_traceback,
            )
            raise AppError(
                ErrorCode.OCR_FAILED,
                "OCR не смог получить пригодный результат",
            ) from error

        if not isinstance(result, dict):
            logger.error(
                "PaddleOCR returned invalid result operation=%s type=%s",
                operation,
                type(result).__name__,
            )
            raise AppError(
                ErrorCode.OCR_FAILED,
                "OCR не смог получить пригодный результат",
            )
        return result


paddle_runtime = PaddleRuntime()
