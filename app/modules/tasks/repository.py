from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, delete, func, or_, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import settings
from app.core.errors import ErrorCode, encode_error
from app.modules.tasks.enums import (
    ACTIVE_STATUSES,
    TERMINAL_STATUSES,
    JobStatus,
)
from app.modules.tasks.exceptions import QueueCapacityExceeded
from app.modules.tasks.models import OCRJob, OCRJobOutbox

JOB_ADMISSION_LOCK_ID = 1_348_675_421


@dataclass(frozen=True)
class LeaseRecoveryResult:
    requeued: int
    failed: int


def serialize_job(job: OCRJob) -> dict[str, Any]:
    return {
        "id": job.id,
        "mode": job.mode,
        "status": job.status,
        "filename": job.filename,
        "content_type": job.content_type,
        "image_path": job.image_path,
        "options": job.options or {},
        "result": job.result,
        "error": job.error,
        "attempts": job.attempts,
        "max_attempts": job.max_attempts,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "worker_id": job.worker_id,
        "lease_expires_at": job.lease_expires_at,
        "heartbeat_at": job.heartbeat_at,
    }


async def create_job(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    job_id: str,
    mode: str,
    filename: str,
    content_type: str,
    image_path: str,
    options: dict[str, Any] | None = None,
) -> None:
    async with session_factory() as session:
        async with session.begin():
            await session.execute(
                text("SELECT pg_advisory_xact_lock(:lock_id)"),
                {"lock_id": JOB_ADMISSION_LOCK_ID},
            )
            active_jobs_result = await session.execute(
                select(func.count())
                .select_from(OCRJob)
                .where(OCRJob.status.in_(ACTIVE_STATUSES))
            )
            active_jobs = int(active_jobs_result.scalar_one())
            if active_jobs >= settings.ocr_queue_max_size:
                raise QueueCapacityExceeded(
                    active_jobs,
                    settings.ocr_queue_max_size,
                )

            session.add_all(
                [
                    OCRJob(
                        id=job_id,
                        mode=mode,
                        status=JobStatus.QUEUED.value,
                        filename=filename,
                        content_type=content_type,
                        image_path=image_path,
                        options=dict(options or {}),
                        max_attempts=settings.ocr_max_attempts,
                    ),
                    OCRJobOutbox(
                        job_id=job_id,
                        payload={"job_id": job_id},
                    ),
                ]
            )


async def get_job(
    session_factory: async_sessionmaker[AsyncSession],
    job_id: str,
) -> dict[str, Any] | None:
    async with session_factory() as session:
        job = await session.get(OCRJob, job_id)
        return serialize_job(job) if job else None


async def claim_job(
    session_factory: async_sessionmaker[AsyncSession],
    job_id: str,
    worker_id: str,
) -> int:
    now = datetime.now(UTC)
    lease_expires_at = now + timedelta(
        seconds=max(1.0, settings.ocr_worker_lease_seconds)
    )
    async with session_factory() as session:
        result = await session.execute(
            update(OCRJob)
            .where(
                OCRJob.id == job_id,
                OCRJob.attempts < OCRJob.max_attempts,
                or_(
                    OCRJob.status == JobStatus.QUEUED.value,
                    and_(
                        OCRJob.status == JobStatus.PROCESSING.value,
                        or_(
                            OCRJob.lease_expires_at.is_(None),
                            OCRJob.lease_expires_at <= now,
                        ),
                    ),
                ),
            )
            .values(
                status=JobStatus.PROCESSING.value,
                attempts=OCRJob.attempts + 1,
                error=None,
                worker_id=worker_id,
                lease_expires_at=lease_expires_at,
                heartbeat_at=now,
                started_at=func.coalesce(OCRJob.started_at, func.now()),
                finished_at=None,
                updated_at=func.now(),
            )
            .returning(OCRJob.attempts)
        )
        attempts = result.scalar_one_or_none() or 0
        await session.commit()
        return int(attempts)


async def claim_stale_published_job(
    session_factory: async_sessionmaker[AsyncSession],
    worker_id: str,
) -> dict[str, Any] | None:
    """Atomically claim a queued job whose RabbitMQ delivery may be lost."""
    now = datetime.now(UTC)
    stale_before = now - timedelta(
        seconds=max(1.0, settings.ocr_queued_watchdog_stale_seconds)
    )
    lease_expires_at = now + timedelta(
        seconds=max(1.0, settings.ocr_worker_lease_seconds)
    )

    async with session_factory() as session:
        async with session.begin():
            result = await session.execute(
                select(OCRJob)
                .join(
                    OCRJobOutbox,
                    OCRJobOutbox.job_id == OCRJob.id,
                )
                .where(
                    OCRJob.status == JobStatus.QUEUED.value,
                    OCRJob.updated_at <= stale_before,
                    OCRJobOutbox.published_at.is_not(None),
                )
                .order_by(OCRJob.updated_at, OCRJob.created_at)
                .with_for_update(of=OCRJob, skip_locked=True)
                .limit(1)
            )
            job = result.scalar_one_or_none()
            if job is None:
                return None

            if job.attempts >= job.max_attempts:
                job.status = JobStatus.FAILED.value
                job.error = encode_error(
                    ErrorCode.INTERNAL_ERROR,
                    "Задача не доставлена после исчерпания попыток",
                )
                job.worker_id = None
                job.lease_expires_at = None
                job.heartbeat_at = None
                job.finished_at = now
                job.updated_at = now
                return serialize_job(job)

            job.status = JobStatus.PROCESSING.value
            job.attempts += 1
            job.error = None
            job.worker_id = worker_id
            job.lease_expires_at = lease_expires_at
            job.heartbeat_at = now
            job.started_at = job.started_at or now
            job.finished_at = None
            job.updated_at = now
            return serialize_job(job)


async def renew_job_lease(
    session_factory: async_sessionmaker[AsyncSession],
    job_id: str,
    worker_id: str,
) -> bool:
    now = datetime.now(UTC)
    lease_expires_at = now + timedelta(
        seconds=max(1.0, settings.ocr_worker_lease_seconds)
    )
    async with session_factory() as session:
        result = await session.execute(
            update(OCRJob)
            .where(
                OCRJob.id == job_id,
                OCRJob.status == JobStatus.PROCESSING.value,
                OCRJob.worker_id == worker_id,
            )
            .values(
                lease_expires_at=lease_expires_at,
                heartbeat_at=now,
                updated_at=func.now(),
            )
            .returning(OCRJob.id)
        )
        renewed_job_id = result.scalar_one_or_none()
        await session.commit()
        return renewed_job_id is not None


async def mark_queued(
    session_factory: async_sessionmaker[AsyncSession],
    job_id: str,
    error: str,
    worker_id: str,
) -> bool:
    async with session_factory() as session:
        result = await session.execute(
            update(OCRJob)
            .where(
                OCRJob.id == job_id,
                OCRJob.status == JobStatus.PROCESSING.value,
                OCRJob.worker_id == worker_id,
            )
            .values(
                status=JobStatus.QUEUED.value,
                error=error,
                worker_id=None,
                lease_expires_at=None,
                heartbeat_at=None,
                updated_at=func.now(),
            )
            .returning(OCRJob.id)
        )
        queued_job_id = result.scalar_one_or_none()
        await session.commit()
        return queued_job_id is not None


async def mark_done(
    session_factory: async_sessionmaker[AsyncSession],
    job_id: str,
    payload: dict[str, Any],
    worker_id: str,
) -> bool:
    async with session_factory() as session:
        result = await session.execute(
            update(OCRJob)
            .where(
                OCRJob.id == job_id,
                OCRJob.status == JobStatus.PROCESSING.value,
                OCRJob.worker_id == worker_id,
            )
            .values(
                status=JobStatus.DONE.value,
                result=payload,
                error=None,
                worker_id=None,
                lease_expires_at=None,
                heartbeat_at=None,
                finished_at=func.now(),
                updated_at=func.now(),
            )
            .returning(OCRJob.id)
        )
        done_job_id = result.scalar_one_or_none()
        await session.commit()
        return done_job_id is not None


async def mark_failed(
    session_factory: async_sessionmaker[AsyncSession],
    job_id: str,
    error: str,
    worker_id: str,
) -> bool:
    async with session_factory() as session:
        result = await session.execute(
            update(OCRJob)
            .where(
                OCRJob.id == job_id,
                OCRJob.status == JobStatus.PROCESSING.value,
                OCRJob.worker_id == worker_id,
            )
            .values(
                status=JobStatus.FAILED.value,
                error=error,
                worker_id=None,
                lease_expires_at=None,
                heartbeat_at=None,
                finished_at=func.now(),
                updated_at=func.now(),
            )
            .returning(OCRJob.id)
        )
        failed_job_id = result.scalar_one_or_none()
        await session.commit()
        return failed_job_id is not None


async def mark_timeout(
    session_factory: async_sessionmaker[AsyncSession],
    job_id: str,
    error: str,
    worker_id: str,
) -> bool:
    async with session_factory() as session:
        result = await session.execute(
            update(OCRJob)
            .where(
                OCRJob.id == job_id,
                OCRJob.status == JobStatus.PROCESSING.value,
                OCRJob.worker_id == worker_id,
            )
            .values(
                status=JobStatus.TIMEOUT.value,
                error=error,
                worker_id=None,
                lease_expires_at=None,
                heartbeat_at=None,
                finished_at=func.now(),
                updated_at=func.now(),
            )
            .returning(OCRJob.id)
        )
        timeout_job_id = result.scalar_one_or_none()
        await session.commit()
        return timeout_job_id is not None


async def mark_job_retry_exhausted(
    session_factory: async_sessionmaker[AsyncSession],
    job_id: str,
    error: str,
) -> bool:
    now = datetime.now(UTC)
    async with session_factory() as session:
        result = await session.execute(
            update(OCRJob)
            .where(
                OCRJob.id == job_id,
                OCRJob.attempts >= OCRJob.max_attempts,
                or_(
                    OCRJob.status == JobStatus.QUEUED.value,
                    and_(
                        OCRJob.status == JobStatus.PROCESSING.value,
                        or_(
                            OCRJob.lease_expires_at.is_(None),
                            OCRJob.lease_expires_at <= now,
                        ),
                    ),
                ),
            )
            .values(
                status=JobStatus.FAILED.value,
                error=error,
                worker_id=None,
                lease_expires_at=None,
                heartbeat_at=None,
                finished_at=func.now(),
                updated_at=func.now(),
            )
            .returning(OCRJob.id)
        )
        failed_job_id = result.scalar_one_or_none()
        await session.commit()
        return failed_job_id is not None


async def recover_expired_job_leases(
    session_factory: async_sessionmaker[AsyncSession],
) -> LeaseRecoveryResult:
    now = datetime.now(UTC)
    requeued = 0
    failed = 0
    async with session_factory() as session:
        async with session.begin():
            result = await session.execute(
                select(OCRJob)
                .where(
                    OCRJob.status == JobStatus.PROCESSING.value,
                    or_(
                        OCRJob.lease_expires_at.is_(None),
                        OCRJob.lease_expires_at <= now,
                    ),
                )
                .with_for_update(skip_locked=True)
            )
            jobs = list(result.scalars())
            for job in jobs:
                job.worker_id = None
                job.lease_expires_at = None
                job.heartbeat_at = None
                job.updated_at = now
                if job.attempts >= job.max_attempts:
                    job.status = JobStatus.FAILED.value
                    job.error = encode_error(
                        ErrorCode.INTERNAL_ERROR,
                        ("Worker остановился после исчерпания попыток"),
                    )
                    job.finished_at = now
                    failed += 1
                    continue

                job.status = JobStatus.QUEUED.value
                job.error = encode_error(
                    ErrorCode.INTERNAL_ERROR,
                    ("Worker lease истек, задача поставлена повторно"),
                )
                job.finished_at = None
                requeued += 1

    return LeaseRecoveryResult(requeued=requeued, failed=failed)


async def mark_cancelled(
    session_factory: async_sessionmaker[AsyncSession],
    job_id: str,
) -> bool:
    async with session_factory() as session:
        result = await session.execute(
            update(OCRJob)
            .where(OCRJob.id == job_id, OCRJob.status.in_(ACTIVE_STATUSES))
            .values(
                status=JobStatus.CANCELLED.value,
                error=None,
                worker_id=None,
                lease_expires_at=None,
                heartbeat_at=None,
                finished_at=func.now(),
                updated_at=func.now(),
            )
            .returning(OCRJob.id)
        )
        cancelled_job_id = result.scalar_one_or_none()
        await session.commit()
        return cancelled_job_id is not None


async def get_expired_jobs(
    session_factory: async_sessionmaker[AsyncSession],
) -> list[dict[str, Any]]:
    expires_before = datetime.now(UTC) - timedelta(
        seconds=settings.ocr_result_ttl_seconds
    )
    async with session_factory() as session:
        result = await session.execute(
            select(OCRJob).where(
                OCRJob.status.in_(TERMINAL_STATUSES),
                OCRJob.finished_at.is_not(None),
                OCRJob.finished_at < expires_before,
            )
        )
        return [serialize_job(job) for job in result.scalars()]


async def delete_job(
    session_factory: async_sessionmaker[AsyncSession],
    job_id: str,
) -> None:
    async with session_factory() as session:
        await session.execute(delete(OCRJob).where(OCRJob.id == job_id))
        await session.commit()
