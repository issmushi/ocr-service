"""Registration-page experiment using the shared Russian OCR worker."""

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
from app.modules.russian.runtime import russian_runtime
from app.modules.v1.russia.registration.schemas import RegistrationDataSchema


async def recognize_registration_upload(image: UploadFile) -> dict[str, Any]:
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
        "passport_registration",
        content,
        timeout_seconds=settings.russian_ocr_document_hard_timeout_seconds,
    )
    if not isinstance(result, dict) or not str(
        result.get("doctype") or ""
    ).startswith("INTPASSPORTADDR"):
        raise AppError(
            ErrorCode.OCR_FAILED,
            "На изображении не найдена страница прописки с адресными строками",
        )
    try:
        data = RegistrationDataSchema.model_validate(
            {
                "document_type": result["doctype"],
                "address": result.get("address"),
                "has_handwritten": result["has_handwritten"],
                "address_lines": result["address_lines"],
                "registration_stamps": result["registration_stamps"],
                "meta": {
                    "engine": "russian-docs-ocr",
                    "model": "russian-ocr",
                    "duration_seconds": perf_counter() - started,
                },
            }
        )
    except (KeyError, ValidationError) as error:
        raise AppError(
            ErrorCode.OCR_FAILED, "Некорректный ответ OCR страницы прописки"
        ) from error
    return api_success_response(data.model_dump(mode="json"))
