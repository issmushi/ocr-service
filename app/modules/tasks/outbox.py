from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import settings
from app.core.db.session import async_session_factory
from app.core.infrastructure.queue import publish_job
from app.modules.tasks.models import OCRJobOutbox

logger = logging.getLogger(__name__)
PublishJob = Callable[[dict[str, Any]], Awaitable[None]]


class DispatchOutcome(StrEnum):
    EMPTY = "empty"
    PUBLISHED = "published"
    FAILED = "failed"


@dataclass(frozen=True)
class OutboxDispatchResult:
    published: int
    failed: int


def retry_delay_seconds(attempts: int) -> float:
    multiplier = 2 ** min(max(attempts - 1, 0), 10)
    return min(
        settings.ocr_retry_delay_seconds * multiplier,
        settings.ocr_outbox_max_retry_delay_seconds,
    )


async def dispatch_next_outbox_event(
    session_factory: async_sessionmaker[AsyncSession],
    publisher: PublishJob,
) -> DispatchOutcome:
    async with session_factory() as session:
        async with session.begin():
            result = await session.execute(
                select(OCRJobOutbox)
                .where(
                    OCRJobOutbox.published_at.is_(None),
                    OCRJobOutbox.next_attempt_at <= func.now(),
                )
                .order_by(OCRJobOutbox.created_at)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            event = result.scalar_one_or_none()
            if event is None:
                return DispatchOutcome.EMPTY

            try:
                await asyncio.wait_for(
                    publisher(event.payload),
                    timeout=settings.ocr_outbox_publish_timeout_seconds,
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                event.attempts += 1
                delay = retry_delay_seconds(event.attempts)
                event.next_attempt_at = datetime.now(UTC) + timedelta(
                    seconds=delay
                )
                event.last_error = (f"{type(error).__name__}: {error}")[:2000]
                logger.warning(
                    "Outbox publish failed job_id=%s attempt=%s "
                    "retry_delay_seconds=%s error=%s",
                    event.job_id,
                    event.attempts,
                    delay,
                    error,
                )
                return DispatchOutcome.FAILED

            event.attempts += 1
            event.published_at = datetime.now(UTC)
            event.last_error = None
            logger.info(
                "Outbox event published job_id=%s attempt=%s queue=%s",
                event.job_id,
                event.attempts,
                settings.ocr_queue_name,
            )
            return DispatchOutcome.PUBLISHED


async def dispatch_pending_outbox_events(
    *,
    limit: int | None = None,
    session_factory: async_sessionmaker[AsyncSession] = async_session_factory,
    publisher: PublishJob = publish_job,
) -> OutboxDispatchResult:
    published = 0
    failed = 0
    batch_size = max(1, limit or settings.ocr_outbox_batch_size)

    for _ in range(batch_size):
        outcome = await dispatch_next_outbox_event(
            session_factory,
            publisher,
        )
        if outcome == DispatchOutcome.PUBLISHED:
            published += 1
            continue
        if outcome == DispatchOutcome.FAILED:
            failed += 1
        break

    return OutboxDispatchResult(published=published, failed=failed)
