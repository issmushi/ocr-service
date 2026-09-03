"""Periodic maintenance for OCR tasks and their stored uploads."""

from __future__ import annotations

import asyncio
import logging

from app.core.config import settings
from app.core.db.session import async_session_factory
from app.core.infrastructure.storage import delete_upload
from app.modules.tasks.outbox import dispatch_pending_outbox_events
from app.modules.tasks.service import (
    delete_job,
    get_expired_jobs,
    recover_expired_job_leases,
)

logger = logging.getLogger(__name__)


async def cleanup_expired_jobs() -> None:
    logger.info(
        "Cleanup task started interval_seconds=%s",
        settings.ocr_cleanup_interval_seconds,
    )
    while True:
        await asyncio.sleep(settings.ocr_cleanup_interval_seconds)
        try:
            expired_jobs = await get_expired_jobs(async_session_factory)
            if expired_jobs:
                logger.info(
                    "Deleting expired OCR jobs count=%s",
                    len(expired_jobs),
                )
            for job in expired_jobs:
                delete_upload(job["image_path"])
                await delete_job(async_session_factory, job["id"])
        except Exception:
            logger.exception("OCR cleanup failed")


async def recover_expired_leases() -> None:
    logger.info(
        "Worker lease recovery started interval_seconds=%s",
        settings.ocr_lease_recovery_interval_seconds,
    )
    while True:
        try:
            result = await recover_expired_job_leases(async_session_factory)
            if result.requeued or result.failed:
                logger.warning(
                    "Expired worker leases recovered requeued=%s failed=%s",
                    result.requeued,
                    result.failed,
                )
        except Exception:
            logger.exception("Worker lease recovery failed")
        await asyncio.sleep(settings.ocr_lease_recovery_interval_seconds)


async def dispatch_outbox_events() -> None:
    logger.info(
        "Outbox dispatcher started interval_seconds=%s batch_size=%s "
        "publish_timeout_seconds=%s",
        settings.ocr_outbox_poll_interval_seconds,
        settings.ocr_outbox_batch_size,
        settings.ocr_outbox_publish_timeout_seconds,
    )
    while True:
        try:
            result = await dispatch_pending_outbox_events()
            if result.published:
                logger.info(
                    "Outbox batch completed published=%s failed=%s",
                    result.published,
                    result.failed,
                )
        except Exception:
            logger.exception("Outbox dispatcher iteration failed")
        await asyncio.sleep(settings.ocr_outbox_poll_interval_seconds)
