from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Any

from fastapi import UploadFile

from app.core.config import settings
from app.core.errors import AppError, ErrorCode, api_success_response
from app.core.images.uploads import prepare_upload_stream
from app.core.images.validation import (
    validate_image_bytes,
    validate_image_extension,
)
from app.core.model_availability import ensure_ocr_model_enabled
from app.modules.paddle.runtime import paddle_runtime
from app.modules.v1.full.schemas import FullOcrModel
from app.modules.vlm.client import recognize_vlm_image

ENGINE_BY_MODEL = {
    FullOcrModel.PADDLE: "paddleocr",
    FullOcrModel.QWEN3_VL_4B: "qwen3-vl",
    FullOcrModel.QWEN3_VL_8B: "qwen3-vl",
}


def _recognized_text(result: Any, model: FullOcrModel) -> str:
    if not isinstance(result, dict):
        raise AppError(
            ErrorCode.OCR_FAILED, "Модель вернула некорректный ответ OCR"
        )
    if model == FullOcrModel.PADDLE:
        text = result.get("text")
        if not isinstance(text, str):
            raise AppError(
                ErrorCode.OCR_FAILED, "Модель вернула некорректный ответ OCR"
            )
    else:
        items = result.get("items")
        if not isinstance(items, list) or any(
            not isinstance(item, dict) or not isinstance(item.get("text"), str)
            for item in items
        ):
            raise AppError(
                ErrorCode.OCR_FAILED, "Модель вернула некорректный ответ OCR"
            )
        text = "\n".join(item["text"] for item in items)
    if not text.strip():
        raise AppError(ErrorCode.OCR_FAILED, "Не удалось получить текст")
    return text


async def recognize_full_upload(
    image: UploadFile,
    *,
    model: FullOcrModel = FullOcrModel.PADDLE,
) -> dict[str, Any]:
    """Return all OCR text directly without creating a queued task."""
    ensure_ocr_model_enabled(model.value)
    filename = image.filename or "image.jpg"
    content_type = image.content_type or "application/octet-stream"
    try:
        suffix = validate_image_extension(filename)
        await prepare_upload_stream(
            image, max_bytes=settings.ocr_max_image_size_bytes
        )
        content = await image.read()
    finally:
        await image.close()
    await asyncio.to_thread(validate_image_bytes, content)

    started = perf_counter()
    if model == FullOcrModel.PADDLE:
        result = await paddle_runtime.run(
            "full",
            content,
            timeout_seconds=settings.paddle_ocr_full_hard_timeout_seconds,
            suffix=suffix,
            max_side=1600,
        )
    elif model in {FullOcrModel.QWEN3_VL_4B, FullOcrModel.QWEN3_VL_8B}:
        result = await recognize_vlm_image(
            content,
            model=model.value,
            filename=filename,
            content_type=content_type,
            task="ocr",
        )
    else:  # pragma: no cover - FastAPI validates the enum before dispatch.
        raise AppError(ErrorCode.VALIDATION_ERROR, "Неизвестная модель OCR")
    duration_seconds = perf_counter() - started

    return api_success_response(
        {
            "text": _recognized_text(result, model),
            "meta": {
                "engine": ENGINE_BY_MODEL[model],
                "model": model.value,
                "duration_seconds": duration_seconds,
            },
        }
    )
