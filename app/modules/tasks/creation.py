from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import UploadFile

from app.core.db.session import async_session_factory
from app.core.errors import AppError, ErrorCode, api_success_response
from app.core.images.uploads import store_image_upload
from app.core.infrastructure.storage import delete_upload
from app.modules.tasks.enums import JobStatus, TaskOperation
from app.modules.tasks.exceptions import QueueCapacityExceeded
from app.modules.tasks.service import create_job


async def create_image_task(
    image: UploadFile,
    *,
    operation: TaskOperation,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save a validated image, then atomically create the task and outbox."""
    task_id = str(uuid4())
    try:
        upload = await store_image_upload(image, task_id, "image.jpg")
    finally:
        await image.close()

    try:
        await create_job(
            async_session_factory,
            job_id=task_id,
            mode=operation.value,
            options=options or {},
            filename=upload.filename,
            content_type=upload.content_type,
            image_path=str(upload.path),
        )
    except QueueCapacityExceeded as error:
        delete_upload(str(upload.path))
        raise AppError(
            ErrorCode.QUEUE_FULL,
            f"Очередь переполнена: {error.active_jobs}/{error.limit} "
            "активных задач",
        ) from error
    except Exception:
        delete_upload(str(upload.path))
        raise

    return api_success_response(
        {"task_id": task_id, "status": JobStatus.QUEUED.value}
    )
