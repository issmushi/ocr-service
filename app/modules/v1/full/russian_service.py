"""Experimental whole-image OCR using the existing Russian OCR worker."""

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
from app.modules.russian.runtime import russian_runtime
from app.ocr_engine.paddle.factory import FULL_TEXT_DETECTOR_MODEL


async def recognize_full_russian_upload(image: UploadFile) -> dict[str, Any]:
    ensure_ocr_model_enabled("russian-ocr")
    try:
        validate_image_extension(image.filename or "image.jpg")
        await prepare_upload_stream(
            image, max_bytes=settings.ocr_max_image_size_bytes
        )
        content = await image.read()
    finally:
        await image.close()
    await asyncio.to_thread(validate_image_bytes, content)

    started = perf_counter()
    result = await russian_runtime.run(
        "full_text",
        content,
        timeout_seconds=settings.russian_ocr_full_hard_timeout_seconds,
    )
    duration_seconds = perf_counter() - started
    if not isinstance(result, dict) or not isinstance(result.get("text"), str):
        raise AppError(
            ErrorCode.OCR_FAILED, "Модель вернула некорректный ответ OCR"
        )
    text = result["text"]
    if not text.strip():
        raise AppError(ErrorCode.OCR_FAILED, "Не удалось получить текст")
    return api_success_response(
        {
            "text": text,
            "meta": {
                "engine": "russian-docs-ocr",
                "model": "russian-ocr",
                "detector": FULL_TEXT_DETECTOR_MODEL,
                "duration_seconds": duration_seconds,
            },
        }
    )
