"""Read Russian driving licences with the resident RussianDocsOCR worker."""

from __future__ import annotations

import asyncio
import logging
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
from app.modules.v1.russia.driver_license.parsing import (
    driver_license_fields,
    parse_driver_license_date,
    reconcile_date_reading,
)
from app.modules.v1.russia.driver_license.schemas import (
    DriverLicenseDataSchema,
)

logger = logging.getLogger(__name__)
DATE_FIELDS = ("Birth_date", "Issue_date", "Expiration_date")
MIN_FALLBACK_CONFIDENCE = 0.8


async def _recover_invalid_dates(
    raw_ocr: dict[str, Any], crops: Any
) -> tuple[dict[str, str], bool]:
    """Re-read only damaged date patches with the optional Cyrillic model."""
    if not isinstance(crops, dict) or not settings.is_model_enabled("paddle"):
        return {}, False
    replacements: dict[str, str] = {}
    paddle_used = False
    for name in DATE_FIELDS:
        if parse_driver_license_date(raw_ocr.get(name)) is not None:
            continue
        crop = crops.get(name)
        if not isinstance(crop, bytes) or not crop:
            continue
        try:
            result = await paddle_runtime.run(
                "field_text_cyrillic",
                crop,
                timeout_seconds=settings.paddle_ocr_full_hard_timeout_seconds,
            )
        except AppError as error:
            logger.warning(
                "Date OCR fallback failed for %s: %s", name, error.code
            )
            continue
        paddle_used = True
        if not isinstance(result, dict):
            continue
        text = result.get("text")
        confidence = result.get("confidence")
        try:
            score = float(confidence)
        except (TypeError, ValueError):
            continue
        if isinstance(text, str) and score >= MIN_FALLBACK_CONFIDENCE:
            recovered = reconcile_date_reading(raw_ocr.get(name), text)
            if recovered is not None:
                replacements[name] = recovered
    return replacements, paddle_used


async def recognize_driver_license_upload(image: UploadFile) -> dict[str, Any]:
    """Find, orient and read one Russian driving licence photo."""
    ensure_ocr_model_enabled("russian-ocr")
    validate_image_extension(image.filename or "image.jpg")
    await prepare_upload_stream(
        image, max_bytes=settings.ocr_max_image_size_bytes
    )
    content = await image.read()
    await asyncio.to_thread(validate_image_bytes, content)

    started = perf_counter()
    result = await russian_runtime.run(
        "driver_license",
        content,
        timeout_seconds=settings.russian_ocr_document_hard_timeout_seconds,
    )
    if not isinstance(result, dict) or not str(
        result.get("doctype") or ""
    ).upper().startswith("DL_"):
        raise AppError(
            ErrorCode.OCR_FAILED,
            "На изображении не найдено водительское удостоверение РФ",
        )
    raw_ocr = result.get("ocr")
    if not isinstance(raw_ocr, dict):
        raise AppError(ErrorCode.OCR_FAILED, "Некорректный ответ OCR прав")
    replacements, paddle_used = await _recover_invalid_dates(
        raw_ocr, result.get("crops")
    )

    try:
        data = DriverLicenseDataSchema.model_validate(
            {
                **driver_license_fields({**raw_ocr, **replacements}),
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
