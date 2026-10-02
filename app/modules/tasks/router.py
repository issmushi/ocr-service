from __future__ import annotations

import base64
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, File, Form, Path, Response, UploadFile
from fastapi.responses import JSONResponse

from app.core.api_schemas import (
    COMMON_ERROR_RESPONSES,
    OCR_ENGINE_UNAVAILABLE_RESPONSE,
    ApiResponseSchema,
)
from app.core.config import settings
from app.core.db.session import async_session_factory
from app.core.errors import (
    AppError,
    ErrorCode,
    api_success_response,
    decode_error,
)
from app.core.infrastructure.storage import delete_upload
from app.core.model_availability import ensure_ocr_model_enabled
from app.modules.tasks.api_schemas import (
    TaskCancelledResponseSchema,
    TaskCreatedResponseSchema,
    TaskStatusResponseSchema,
)
from app.modules.tasks.creation import create_image_task
from app.modules.tasks.enums import TERMINAL_STATUSES, JobStatus, TaskOperation
from app.modules.tasks.service import get_job, mark_cancelled
from app.modules.v1.full.schemas import FullOcrModel

router = APIRouter(prefix="/v1/tasks", tags=["Tasks"])


def isoformat_z(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    return str(value)


CREATION_ERRORS = {
    400: COMMON_ERROR_RESPONSES[400],
    413: COMMON_ERROR_RESPONSES[413],
    422: {
        "model": ApiResponseSchema,
        "description": "Некорректные параметры задачи.",
    },
    429: {
        "model": ApiResponseSchema,
        "description": "QUEUE_FULL: очередь задач заполнена.",
    },
    503: OCR_ENGINE_UNAVAILABLE_RESPONSE,
    500: COMMON_ERROR_RESPONSES[500],
}
TASK_ERRORS = {
    404: {
        "model": ApiResponseSchema,
        "description": "TASK_NOT_FOUND: задача не найдена.",
    },
    422: {
        "model": ApiResponseSchema,
        "description": "Некорректный UUID задачи.",
    },
    500: COMMON_ERROR_RESPONSES[500],
}


@router.post(
    "/mrz",
    status_code=202,
    response_model=TaskCreatedResponseSchema,
    summary="Создать задачу распознавания MRZ документа",
    responses=CREATION_ERRORS,
)
async def create_mrz_task(
    image: UploadFile = File(...),
    crop: bool = Form(True),
    normalize: bool = Form(True),
    normalization: bool = Form(True),
) -> dict[str, Any]:
    ensure_ocr_model_enabled("mrz")
    return await create_image_task(
        image,
        operation=TaskOperation.MRZ,
        options={
            "crop": crop,
            "normalize": normalize,
            "normalization": normalization,
        },
    )


@router.post(
    "/passport/internal",
    status_code=202,
    response_model=TaskCreatedResponseSchema,
    summary="Создать задачу распознавания внутреннего паспорта РФ",
    responses=CREATION_ERRORS,
)
async def create_passport_internal_task(
    image: UploadFile = File(...),
) -> dict[str, Any]:
    ensure_ocr_model_enabled("russian-ocr")
    return await create_image_task(
        image,
        operation=TaskOperation.PASSPORT_PAGE,
        options={},
    )


@router.post(
    "/passport/international",
    status_code=202,
    response_model=TaskCreatedResponseSchema,
    summary="Создать задачу распознавания загранпаспорта РФ",
    responses=CREATION_ERRORS,
)
async def create_passport_international_task(
    image: UploadFile = File(...),
) -> dict[str, Any]:
    ensure_ocr_model_enabled("russian-ocr")
    return await create_image_task(
        image,
        operation=TaskOperation.PASSPORT_INTERNATIONAL,
        options={},
    )


@router.post(
    "/snils",
    status_code=202,
    response_model=TaskCreatedResponseSchema,
    summary="Создать задачу распознавания СНИЛС",
    responses=CREATION_ERRORS,
)
async def create_snils_task(
    image: UploadFile = File(...),
) -> dict[str, Any]:
    ensure_ocr_model_enabled("russian-ocr")
    ensure_ocr_model_enabled("paddle")
    return await create_image_task(
        image,
        operation=TaskOperation.SNILS,
        options={},
    )


@router.post(
    "/full",
    status_code=202,
    response_model=TaskCreatedResponseSchema,
    summary="Создать задачу полного распознавания текста",
    responses=CREATION_ERRORS,
)
async def create_full_task(
    image: UploadFile = File(...),
    model: FullOcrModel = Form(FullOcrModel.PADDLE),
) -> dict[str, Any]:
    ensure_ocr_model_enabled(model.value)
    return await create_image_task(
        image, operation=TaskOperation.FULL_V1, options={"model": model.value}
    )


@router.post(
    "/crop/document",
    status_code=202,
    response_model=TaskCreatedResponseSchema,
    summary="Создать задачу обрезки документа",
    responses=CREATION_ERRORS,
)
async def create_document_crop_task(
    image: UploadFile = File(...),
) -> dict[str, Any]:
    return await create_image_task(
        image, operation=TaskOperation.CROP_DOCUMENT
    )


@router.post(
    "/crop/mrz",
    status_code=202,
    response_model=TaskCreatedResponseSchema,
    summary="Создать задачу обрезки MRZ",
    responses=CREATION_ERRORS,
)
async def create_mrz_crop_task(
    image: UploadFile = File(...),
) -> dict[str, Any]:
    return await create_image_task(image, operation=TaskOperation.CROP_MRZ)


async def _get_task(task_id: str) -> dict[str, Any]:
    job = await get_job(async_session_factory, task_id)
    if job is None:
        raise AppError(ErrorCode.TASK_NOT_FOUND, "Задача не найдена")
    return job


def _task_error(job: dict[str, Any]) -> dict[str, str] | None:
    error = decode_error(job.get("error"))
    if error is None and job["status"] in {
        JobStatus.FAILED.value,
        JobStatus.TIMEOUT.value,
    }:
        if job["status"] == JobStatus.TIMEOUT.value:
            return {
                "code": ErrorCode.OCR_TIMEOUT.value,
                "message": "Превышен таймаут обработки",
            }
        return {
            "code": ErrorCode.INTERNAL_ERROR.value,
            "message": "Непредвиденная ошибка",
        }
    return error


@router.get(
    "/{task_id}",
    response_model=TaskStatusResponseSchema,
    summary="Получить состояние задачи",
    responses=TASK_ERRORS,
    description=(
        "Возвращает состояние задачи. Ошибка обработки находится в "
        "data.error; результат доступен через /result."
    ),
)
async def get_task_status(
    task_id: UUID = Path(..., description="UUID задачи."),
) -> dict[str, Any]:
    job = await _get_task(str(task_id))
    return api_success_response(
        {
            "task_id": str(task_id),
            "status": job["status"],
            "operation": job["mode"],
            "created_at": isoformat_z(job.get("created_at")),
            "started_at": isoformat_z(job.get("started_at")),
            "finished_at": isoformat_z(job.get("finished_at")),
            "error": _task_error(job),
        }
    )


@router.get(
    "/{task_id}/result",
    response_class=Response,
    summary="Получить готовый результат задачи",
    description=(
        "Возвращает тот же JSON или JPEG, что синхронный endpoint. "
        "Для незавершённой или отменённой задачи возвращает 409."
    ),
    responses={
        200: {
            "description": "Готовый результат распознавания или обрезки.",
            "content": {"application/json": {}, "image/jpeg": {}},
        },
        **TASK_ERRORS,
        409: {
            "model": ApiResponseSchema,
            "description": "TASK_NOT_READY или TASK_CANCELLED.",
        },
        400: COMMON_ERROR_RESPONSES[400],
        413: COMMON_ERROR_RESPONSES[413],
        503: {
            "model": ApiResponseSchema,
            "description": "Ошибка доступности движка при выполнении задачи.",
        },
        504: {
            "model": ApiResponseSchema,
            "description": "OCR_TIMEOUT: обработка превысила таймаут.",
        },
    },
)
async def get_task_result(
    task_id: UUID = Path(..., description="UUID задачи."),
) -> Response:
    job = await _get_task(str(task_id))
    status = job["status"]
    if status == JobStatus.CANCELLED.value:
        raise AppError(ErrorCode.TASK_CANCELLED, "Задача отменена")
    if status in {JobStatus.FAILED.value, JobStatus.TIMEOUT.value}:
        error = _task_error(job)
        assert error is not None
        raise AppError(ErrorCode(error["code"]), error["message"])
    if status != JobStatus.DONE.value:
        raise AppError(
            ErrorCode.TASK_NOT_READY, "Результат задачи ещё не готов"
        )

    result = job.get("result")
    if isinstance(result, dict):
        if result.get("format") == "json" and isinstance(
            result.get("response"), dict
        ):
            return JSONResponse(
                result["response"], headers={"Cache-Control": "no-store"}
            )
        if result.get("format") == "jpeg" and isinstance(
            result.get("content_base64"), str
        ):
            try:
                content = base64.b64decode(
                    result["content_base64"], validate=True
                )
            except ValueError:
                content = b""
            if content:
                return Response(
                    content,
                    media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"},
                )
    raise AppError(
        ErrorCode.INTERNAL_ERROR,
        "Не удалось прочитать сохранённый результат задачи",
    )


@router.post(
    "/{task_id}/cancel",
    response_model=TaskCancelledResponseSchema,
    summary="Отменить задачу",
    responses=TASK_ERRORS,
)
async def cancel_task(
    task_id: UUID = Path(..., description="UUID задачи."),
) -> dict[str, Any]:
    task_id_value = str(task_id)
    job = await _get_task(task_id_value)
    if job["status"] in TERMINAL_STATUSES:
        return api_success_response(
            {
                "task_id": task_id_value,
                "status": job["status"],
                "cancelled": False,
            }
        )

    cancelled = await mark_cancelled(async_session_factory, task_id_value)
    latest = await _get_task(task_id_value)
    # The worker can claim a queued task between the read and atomic cancel.
    # started_at survives cancellation and proves whether processing began.
    if (
        cancelled
        and not settings.ocr_save_uploads
        and job["status"] == JobStatus.QUEUED.value
        and latest["status"] == JobStatus.CANCELLED.value
        and latest.get("started_at") is None
    ):
        delete_upload(latest["image_path"])
    return api_success_response(
        {
            "task_id": task_id_value,
            "status": latest["status"],
            "cancelled": cancelled,
        }
    )
