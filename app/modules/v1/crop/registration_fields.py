"""Test registration cropping through the shared Russian OCR worker."""

from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Any

from fastapi import UploadFile

from app.core.config import settings
from app.core.errors import api_success_response
from app.core.images.uploads import prepare_upload_stream
from app.core.images.validation import (
    validate_image_bytes,
    validate_image_extension,
)
from app.core.model_availability import ensure_ocr_model_enabled
from app.modules.russian.runtime import russian_runtime
from app.modules.v1.crop.field_ocr import recognize_field_previews
from app.modules.v1.crop.field_schemas import FieldOcrModel


async def crop_registration_fields_upload(
    image: UploadFile,
    *,
    padding: int = 4,
    ocr_model: FieldOcrModel = FieldOcrModel.NONE,
) -> dict[str, Any]:
    try:
        ensure_ocr_model_enabled("russian-ocr")
        if ocr_model != FieldOcrModel.NONE:
            installed_model = {
                FieldOcrModel.PADDLE_CYRILLIC: "paddle",
                FieldOcrModel.RUSSIAN_OCR_ACCURATE: "russian-ocr",
            }.get(ocr_model, ocr_model.value)
            ensure_ocr_model_enabled(installed_model)
        validate_image_extension(image.filename or "image.jpg")
        await prepare_upload_stream(
            image, max_bytes=settings.ocr_max_image_size_bytes
        )
        content = await image.read()
    finally:
        await image.close()

    started = perf_counter()
    await asyncio.to_thread(validate_image_bytes, content)
    result = await russian_runtime.run(
        "registration_fields",
        content,
        padding=padding,
        timeout_seconds=settings.russian_ocr_document_hard_timeout_seconds,
    )
    ocr_meta = await recognize_field_previews(result["fields"], ocr_model)
    return api_success_response(
        {
            "document_type": result["document_type"],
            "document": result["document"],
            "fields": result["fields"],
            "meta": {
                "engine": "russian-docs-ocr",
                "model": "address-lines-and-registration-stamps",
                "duration_seconds": perf_counter() - started,
                "timings": result["timings"],
                "ocr": ocr_meta,
            },
        }
    )
