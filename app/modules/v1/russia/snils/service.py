"""RussianDocsOCR plus targeted Cyrillic Paddle recognition for SNILS."""

from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Any

from fastapi import UploadFile
from pydantic import ValidationError

from app.core.config import settings
from app.core.errors import AppError, ErrorCode, api_success_response
from app.core.images.uploads import prepare_upload_stream
from app.core.images.validation import (
    validate_image_bytes,
    validate_image_extension,
)
from app.core.model_availability import ensure_ocr_model_enabled
from app.modules.paddle.runtime import paddle_runtime
from app.modules.russian.runtime import russian_runtime
from app.modules.v1.russia.snils.parsing import snils_fields
from app.modules.v1.russia.snils.schemas import SnilsDataSchema

DATE_FIELDS = ("Birth_date", "Issue_date")


async def _read_cyrillic_crop(content: bytes) -> str | None:
    result = await paddle_runtime.run(
        "field_text_cyrillic",
        content,
        timeout_seconds=settings.paddle_ocr_full_hard_timeout_seconds,
    )
    text = result.get("text")
    if not isinstance(text, str):
        raise AppError(ErrorCode.OCR_FAILED, "Некорректный ответ OCR поля")
    return text.strip() or None


async def recognize_snils_upload(image: UploadFile) -> dict[str, Any]:
    """Read a SNILS card and re-read its exact date/number rectangles."""
    ensure_ocr_model_enabled("russian-ocr")
    ensure_ocr_model_enabled("paddle")
    validate_image_extension(image.filename or "image.jpg")
    await prepare_upload_stream(
        image, max_bytes=settings.ocr_max_image_size_bytes
    )
    content = await image.read()
    await asyncio.to_thread(validate_image_bytes, content)

    started = perf_counter()
    result = await russian_runtime.run(
        "snils",
        content,
        timeout_seconds=settings.russian_ocr_document_hard_timeout_seconds,
    )
    if not isinstance(result, dict) or not str(
        result.get("doctype") or ""
    ).upper().startswith("SNILS"):
        raise AppError(ErrorCode.OCR_FAILED, "На изображении не найден СНИЛС")
    raw_ocr = result.get("ocr")
    if not isinstance(raw_ocr, dict):
        raise AppError(ErrorCode.OCR_FAILED, "Некорректный ответ OCR СНИЛС")
    crops = result.get("crops")
    if not isinstance(crops, dict):
        crops = {}

    readings: dict[str, str | None] = {}
    paddle_used = False
    for name in ("Licence_number", *DATE_FIELDS):
        crop = crops.get(name)
        if not isinstance(crop, bytes) or not crop:
            continue
        paddle_used = True
        readings[name] = await _read_cyrillic_crop(crop)

    try:
        data = SnilsDataSchema.model_validate(
            {
                **snils_fields(
                    raw_ocr,
                    number_text=readings.get("Licence_number"),
                    birth_date_text=readings.get("Birth_date"),
                    issue_date_text=readings.get("Issue_date"),
                ),
                "meta": {
                    "engine": (
                        "russian-docs-ocr+paddleocr"
                        if paddle_used
                        else "russian-docs-ocr"
                    ),
                    "model": (
                        "russian-ocr+cyrillic_PP-OCRv5_mobile_rec"
                        if paddle_used
                        else "russian-ocr"
                    ),
                    "duration_seconds": perf_counter() - started,
                },
            }
        )
    except ValidationError as error:
        raise AppError(
            ErrorCode.OCR_FAILED, "OCR не смог получить пригодный результат"
        ) from error
    return api_success_response(data.model_dump(mode="json"))
