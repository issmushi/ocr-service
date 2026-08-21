from __future__ import annotations

from typing import Any

import httpx

from app.core.config import settings
from app.core.errors import AppError, ErrorCode


async def recognize_vlm_image(
    content: bytes,
    *,
    model: str,
    filename: str = "image.jpg",
    content_type: str = "image/jpeg",
    task: str | None = None,
    cpu_only: bool = False,
    report_runtime: bool = False,
) -> dict[str, Any]:
    """Send the selected model and image to the local VLM service."""
    try:
        async with httpx.AsyncClient(
            timeout=settings.vlm_ocr_timeout_seconds,
            trust_env=False,
        ) as client:
            response = await client.post(
                f"{settings.vlm_ocr_url.rstrip('/')}/recognize",
                files={"image": (filename, content, content_type)},
                data={
                    "model": model,
                    **({"task": task} if task else {}),
                    **({"cpu_only": "true"} if cpu_only else {}),
                    **({"report_runtime": "true"} if report_runtime else {}),
                },
            )
    except httpx.TimeoutException as error:
        raise AppError(
            ErrorCode.OCR_TIMEOUT, "Превышено время ожидания распознавания VLM"
        ) from error
    except httpx.RequestError as error:
        raise AppError(
            ErrorCode.OCR_ENGINE_UNAVAILABLE, "Локальный движок VLM недоступен"
        ) from error

    if response.status_code == 400:
        raise AppError(
            ErrorCode.IMAGE_READ_ERROR, "Не удалось прочитать изображение"
        )
    if response.status_code in {502, 503}:
        raise AppError(
            ErrorCode.OCR_ENGINE_UNAVAILABLE, "Выбранная модель VLM недоступна"
        )
    if response.status_code == 504:
        raise AppError(
            ErrorCode.OCR_TIMEOUT, "Превышено время ожидания распознавания VLM"
        )
    if not response.is_success:
        raise AppError(
            ErrorCode.OCR_FAILED, "VLM не смогла распознать изображение"
        )
    try:
        result = response.json()
    except ValueError as error:
        raise AppError(
            ErrorCode.OCR_FAILED, "VLM не смогла получить пригодный результат"
        ) from error
    if not isinstance(result, dict):
        raise AppError(
            ErrorCode.OCR_FAILED, "VLM не смогла получить пригодный результат"
        )
    return result
