from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
from collections.abc import Coroutine
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, TypeVar
from uuid import uuid4

import aio_pika
import httpx

from app.core.config import settings
from app.core.db.session import async_session_factory
from app.core.errors import AppError, ErrorCode, encode_error
from app.core.infrastructure.queue import publish_dead_letter
from app.core.infrastructure.storage import delete_upload
from app.modules.tasks.enums import JobStatus
from app.modules.tasks.exceptions import JobLeaseBusy, JobLeaseLost
from app.modules.tasks.execution import execute_task_operation
from app.modules.tasks.service import (
    claim_job,
    claim_stale_published_job,
    get_job,
    mark_done,
    mark_failed,
    mark_job_retry_exhausted,
    mark_queued,
    mark_timeout,
    renew_job_lease,
)

logger = logging.getLogger(__name__)
WORKER_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex[:8]}"
ResultT = TypeVar("ResultT")
TERMINAL_STATUSES = {
    JobStatus.DONE.value,
    JobStatus.FAILED.value,
    JobStatus.TIMEOUT.value,
    JobStatus.CANCELLED.value,
}


def validate_lease_settings() -> None:
    lease_seconds = settings.ocr_worker_lease_seconds
    heartbeat_seconds = settings.ocr_worker_heartbeat_seconds
    if lease_seconds <= 0 or heartbeat_seconds <= 0:
        raise RuntimeError("Worker lease and heartbeat must be positive")
    if heartbeat_seconds >= lease_seconds:
        raise RuntimeError(
            "OCR_WORKER_HEARTBEAT_SECONDS must be less than "
            "OCR_WORKER_LEASE_SECONDS"
        )
    if settings.ocr_queued_watchdog_interval_seconds <= 0:
        raise RuntimeError(
            "OCR_QUEUED_WATCHDOG_INTERVAL_SECONDS must be positive"
        )
    if settings.ocr_queued_watchdog_stale_seconds <= 0:
        raise RuntimeError(
            "OCR_QUEUED_WATCHDOG_STALE_SECONDS must be positive"
        )


def parse_job_message(body: bytes) -> str:
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Queue message must contain a JSON object")

    job_id = payload.get("job_id")
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("Queue message must contain a non-empty job_id")
    return job_id.strip()


def lease_retry_delay(job: dict[str, Any]) -> float:
    default_delay = max(0.1, settings.ocr_retry_delay_seconds)
    lease_expires_at = job.get("lease_expires_at")
    if not isinstance(lease_expires_at, datetime):
        return default_delay
    if lease_expires_at.tzinfo is None:
        lease_expires_at = lease_expires_at.replace(tzinfo=UTC)

    remaining = (lease_expires_at - datetime.now(UTC)).total_seconds()
    return max(0.1, min(default_delay, max(0.1, remaining)))


async def maintain_job_lease(job_id: str, worker_id: str) -> None:
    interval = max(0.1, settings.ocr_worker_heartbeat_seconds)
    while True:
        await asyncio.sleep(interval)
        renewed = await renew_job_lease(
            async_session_factory,
            job_id,
            worker_id,
        )
        if not renewed:
            raise JobLeaseLost(job_id)
        logger.debug(
            "Worker lease renewed job_id=%s worker_id=%s",
            job_id,
            worker_id,
        )


async def run_with_job_lease(
    job_id: str,
    worker_id: str,
    operation: Coroutine[Any, Any, ResultT],
) -> ResultT:
    operation_task = asyncio.create_task(operation)
    heartbeat_task = asyncio.create_task(maintain_job_lease(job_id, worker_id))
    try:
        done, _pending = await asyncio.wait(
            {operation_task, heartbeat_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if heartbeat_task in done:
            error = heartbeat_task.exception()
            if error is not None:
                raise error
            raise JobLeaseLost(job_id)
        return await operation_task
    finally:
        heartbeat_task.cancel()
        if not operation_task.done():
            operation_task.cancel()
        await asyncio.gather(
            operation_task,
            heartbeat_task,
            return_exceptions=True,
        )


async def execute_ocr_job(
    job_id: str,
    job: dict[str, Any],
    image_path: Path,
) -> dict[str, Any]:
    content = await asyncio.to_thread(image_path.read_bytes)
    logger.debug(
        "Worker loaded image bytes job_id=%s pid=%s size=%s",
        job_id,
        os.getpid(),
        len(content),
    )
    return await execute_task_operation(job, content)


async def process_job(
    job_id: str,
    worker_id: str = WORKER_ID,
    *,
    claimed_job: dict[str, Any] | None = None,
) -> None:
    started_at = perf_counter()
    logger.debug(
        "Worker process_job entered job_id=%s worker_id=%s pid=%s",
        job_id,
        worker_id,
        os.getpid(),
    )
    if claimed_job is None:
        attempts = await claim_job(
            async_session_factory,
            job_id,
            worker_id,
        )
        job = await get_job(async_session_factory, job_id)
    else:
        job = claimed_job
        attempts = int(job["attempts"])
    if job is None:
        logger.warning(
            "Worker received missing job job_id=%s",
            job_id,
        )
        return
    logger.debug(
        "Worker loaded job from db job_id=%s pid=%s status=%s filename=%s "
        "content_type=%s attempts=%s max_attempts=%s",
        job_id,
        os.getpid(),
        job["status"],
        job["filename"],
        job["content_type"],
        job["attempts"],
        job["max_attempts"],
    )
    if job["status"] in TERMINAL_STATUSES:
        logger.info(
            "Worker skipped terminal job job_id=%s status=%s",
            job_id,
            job["status"],
        )
        return

    if attempts == 0:
        if int(job["attempts"]) >= int(job["max_attempts"]):
            failed = await mark_job_retry_exhausted(
                async_session_factory,
                job_id,
                encode_error(
                    ErrorCode.INTERNAL_ERROR,
                    ("Worker остановился после исчерпания попыток"),
                ),
            )
            if failed:
                logger.error(
                    "Worker marked abandoned job failed job_id=%s "
                    "attempts=%s max_attempts=%s",
                    job_id,
                    job["attempts"],
                    job["max_attempts"],
                )
                if not settings.ocr_save_uploads:
                    delete_upload(job["image_path"])
                return

            job = await get_job(async_session_factory, job_id)
            if job is None or job["status"] in TERMINAL_STATUSES:
                return

        retry_after_seconds = lease_retry_delay(job)
        logger.info(
            "Worker deferred job owned by another worker job_id=%s "
            "owner=%s lease_expires_at=%s retry_after_seconds=%s",
            job_id,
            job.get("worker_id"),
            job.get("lease_expires_at"),
            retry_after_seconds,
        )
        raise JobLeaseBusy(
            job_id,
            job.get("lease_expires_at"),
            retry_after_seconds,
        )

    if job.get("worker_id") != worker_id:
        raise JobLeaseBusy(
            job_id,
            job.get("lease_expires_at"),
            lease_retry_delay(job),
        )

    image_path = Path(job["image_path"])
    logger.info(
        "Worker started job job_id=%s pid=%s attempt=%s "
        "filename=%s content_type=%s",
        job_id,
        os.getpid(),
        attempts,
        job["filename"],
        job["content_type"],
    )
    try:
        payload = await run_with_job_lease(
            job_id,
            worker_id,
            execute_ocr_job(job_id, job, image_path),
        )
    except JobLeaseLost:
        logger.warning(
            "Worker stopped OCR after losing lease job_id=%s worker_id=%s",
            job_id,
            worker_id,
        )
        raise
    except AppError as error:
        encoded_error = encode_error(error.code, error.message)
        if error.code == ErrorCode.OCR_TIMEOUT:
            duration_ms = int((perf_counter() - started_at) * 1000)
            logger.error(
                "Worker timed out job_id=%s attempt=%s duration_ms=%s "
                "code=%s message=%s",
                job_id,
                attempts,
                duration_ms,
                error.code.value,
                error.message,
            )
            transitioned = await mark_timeout(
                async_session_factory,
                job_id,
                encoded_error,
                worker_id,
            )
        elif attempts >= int(job["max_attempts"]):
            duration_ms = int((perf_counter() - started_at) * 1000)
            logger.error(
                "Worker failed job job_id=%s attempt=%s max_attempts=%s "
                "duration_ms=%s code=%s message=%s",
                job_id,
                attempts,
                job["max_attempts"],
                duration_ms,
                error.code.value,
                error.message,
            )
            transitioned = await mark_failed(
                async_session_factory,
                job_id,
                encoded_error,
                worker_id,
            )
        else:
            logger.warning(
                "Worker requeued job after AppError job_id=%s attempt=%s "
                "code=%s message=%s",
                job_id,
                attempts,
                error.code.value,
                error.message,
            )
            transitioned = await mark_queued(
                async_session_factory,
                job_id,
                encoded_error,
                worker_id,
            )
            if not transitioned:
                raise JobLeaseLost(job_id)
            raise

        if not transitioned:
            raise JobLeaseLost(job_id)
        if not settings.ocr_save_uploads:
            delete_upload(str(image_path))
    except (
        asyncio.TimeoutError,
        TimeoutError,
        httpx.TimeoutException,
    ) as error:
        duration_ms = int((perf_counter() - started_at) * 1000)
        logger.error(
            "Worker timeout exception job_id=%s attempt=%s "
            "duration_ms=%s error=%s",
            job_id,
            attempts,
            duration_ms,
            error,
        )
        transitioned = await mark_timeout(
            async_session_factory,
            job_id,
            encode_error(
                ErrorCode.OCR_TIMEOUT,
                "Превышен таймаут обработки",
            ),
            worker_id,
        )
        if not transitioned:
            raise JobLeaseLost(job_id)
        if not settings.ocr_save_uploads:
            delete_upload(str(image_path))
    except Exception as error:
        encoded_error = encode_error(
            ErrorCode.INTERNAL_ERROR,
            "Непредвиденная ошибка",
        )
        if attempts >= int(job["max_attempts"]):
            logger.exception(
                "Worker failed job with unexpected error job_id=%s attempt=%s "
                "max_attempts=%s",
                job_id,
                attempts,
                job["max_attempts"],
            )
            transitioned = await mark_failed(
                async_session_factory,
                job_id,
                encoded_error,
                worker_id,
            )
            if not transitioned:
                raise JobLeaseLost(job_id)
            if not settings.ocr_save_uploads:
                delete_upload(str(image_path))
        else:
            logger.warning(
                "Worker requeued job after unexpected error job_id=%s "
                "attempt=%s error=%s",
                job_id,
                attempts,
                error,
            )
            transitioned = await mark_queued(
                async_session_factory,
                job_id,
                encoded_error,
                worker_id,
            )
            if not transitioned:
                raise JobLeaseLost(job_id)
            raise
    else:
        completed = await mark_done(
            async_session_factory,
            job_id,
            payload,
            worker_id,
        )
        if not completed:
            logger.warning(
                "Worker discarded OCR result after losing ownership "
                "job_id=%s worker_id=%s",
                job_id,
                worker_id,
            )
            return
        duration_ms = int((perf_counter() - started_at) * 1000)
        logger.info(
            "Worker completed job job_id=%s engine=%s duration_ms=%s",
            job_id,
            payload.get("engine"),
            duration_ms,
        )
        if not settings.ocr_save_uploads:
            delete_upload(str(image_path))


async def handle_message(message: aio_pika.IncomingMessage) -> None:
    try:
        job_id = parse_job_message(message.body)
        logger.info(
            "Worker queue message received job_id=%s pid=%s delivery_tag=%s "
            "redelivered=%s",
            job_id,
            os.getpid(),
            message.delivery_tag,
            message.redelivered,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        reason = f"{type(error).__name__}: {error}"
        logger.warning(
            "Worker received malformed queue message pid=%s body_size=%s "
            "delivery_tag=%s redelivered=%s reason=%s",
            os.getpid(),
            len(message.body),
            message.delivery_tag,
            message.redelivered,
            reason,
        )
        try:
            await publish_dead_letter(
                message.body,
                reason=reason,
                message_id=(
                    message.message_id
                    if isinstance(message.message_id, str)
                    else None
                ),
                correlation_id=(
                    message.correlation_id
                    if isinstance(message.correlation_id, str)
                    else None
                ),
                content_type=(
                    message.content_type
                    if isinstance(message.content_type, str)
                    else None
                ),
            )
        except Exception:
            logger.exception(
                "Worker failed to publish malformed message to DLQ "
                "body_size=%s delivery_tag=%s",
                len(message.body),
                message.delivery_tag,
            )
            await asyncio.sleep(settings.ocr_retry_delay_seconds)
            await message.nack(requeue=True)
        else:
            logger.warning(
                "Worker moved malformed message to DLQ queue=%s "
                "delivery_tag=%s",
                settings.ocr_dead_letter_queue_name,
                message.delivery_tag,
            )
            await message.ack()
        return

    try:
        await process_job(job_id)
    except JobLeaseBusy as error:
        logger.info(
            "Worker requeued message while job lease is active "
            "job_id=%s lease_expires_at=%s retry_after_seconds=%s",
            job_id,
            error.lease_expires_at,
            error.retry_after_seconds,
        )
        await asyncio.sleep(error.retry_after_seconds)
        await message.nack(requeue=True)
    except JobLeaseLost:
        logger.warning(
            "Worker nacked job after losing lease job_id=%s",
            job_id,
        )
        await asyncio.sleep(settings.ocr_retry_delay_seconds)
        await message.nack(requeue=True)
    except Exception:
        logger.warning(
            "Worker nacked job for retry job_id=%s",
            job_id,
            exc_info=True,
        )
        await asyncio.sleep(settings.ocr_retry_delay_seconds)
        await message.nack(requeue=True)
    else:
        await message.ack()


async def reconcile_stale_queued_job(
    execution_lock: asyncio.Lock,
) -> bool:
    async with execution_lock:
        job = await claim_stale_published_job(
            async_session_factory,
            WORKER_ID,
        )
        if job is None:
            return False

        job_id = str(job["id"])
        if job["status"] == JobStatus.FAILED.value:
            logger.error(
                "Queued watchdog marked exhausted job failed job_id=%s "
                "attempts=%s max_attempts=%s",
                job_id,
                job["attempts"],
                job["max_attempts"],
            )
            if not settings.ocr_save_uploads:
                delete_upload(job["image_path"])
            return True

        logger.warning(
            "Queued watchdog claimed stale published job job_id=%s "
            "attempt=%s stale_seconds=%s",
            job_id,
            job["attempts"],
            settings.ocr_queued_watchdog_stale_seconds,
        )
        try:
            await process_job(
                job_id,
                WORKER_ID,
                claimed_job=job,
            )
        except JobLeaseLost:
            logger.warning(
                "Queued watchdog lost job lease job_id=%s",
                job_id,
            )
        except Exception:
            logger.warning(
                "Queued watchdog left job for retry job_id=%s",
                job_id,
                exc_info=True,
            )
        return True


async def queued_job_watchdog(execution_lock: asyncio.Lock) -> None:
    interval = settings.ocr_queued_watchdog_interval_seconds
    logger.info(
        "Queued watchdog started interval_seconds=%s stale_seconds=%s",
        interval,
        settings.ocr_queued_watchdog_stale_seconds,
    )
    while True:
        await asyncio.sleep(interval)
        try:
            await reconcile_stale_queued_job(execution_lock)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Queued watchdog iteration failed")


async def worker(
    ready_event: asyncio.Event | None = None,
) -> None:
    validate_lease_settings()
    logger.info(
        "Worker connecting RabbitMQ worker_id=%s host=%s port=%s queue=%s "
        "dlq=%s lease_seconds=%s heartbeat_seconds=%s",
        WORKER_ID,
        settings.rabbitmq_host,
        settings.rabbitmq_port,
        settings.ocr_queue_name,
        settings.ocr_dead_letter_queue_name,
        settings.ocr_worker_lease_seconds,
        settings.ocr_worker_heartbeat_seconds,
    )
    connection = await aio_pika.connect_robust(settings.rabbitmq_url)
    watchdog_task: asyncio.Task[None] | None = None
    try:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)
        queue = await channel.declare_queue(
            settings.ocr_queue_name,
            durable=True,
        )
        await channel.declare_queue(
            settings.ocr_dead_letter_queue_name,
            durable=True,
        )
        execution_lock = asyncio.Lock()
        async with queue.iterator(exclusive=True) as iterator:
            watchdog_task = asyncio.create_task(
                queued_job_watchdog(execution_lock)
            )
            logger.info(
                "Worker ready worker_id=%s queue=%s "
                "prefetch_count=1 exclusive=true",
                WORKER_ID,
                settings.ocr_queue_name,
            )
            if ready_event is not None:
                ready_event.set()

            async for message in iterator:
                async with execution_lock:
                    await handle_message(message)
    finally:
        logger.info("Worker shutdown started")
        if watchdog_task is not None:
            watchdog_task.cancel()
            await asyncio.gather(
                watchdog_task,
                return_exceptions=True,
            )
        await connection.close()
        logger.info("Worker shutdown complete")
