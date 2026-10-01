"""Experimental field cropping without passport text recognition."""

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


async def crop_passport_fields_upload(
    image: UploadFile,
    *,
    crop_document: bool = True,
    padding: int = 4,
) -> dict[str, Any]:
    try:
        ensure_ocr_model_enabled("russian-ocr")
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
        "passport_fields",
        content,
        timeout_seconds=settings.russian_ocr_document_hard_timeout_seconds,
        crop_document=crop_document,
        padding=padding,
    )
    return api_success_response(
        {
            "document": result["document"],
            "fields": result["fields"],
            "meta": {
                "engine": "russian-docs-ocr",
                "model": "text-fields-detector",
                "duration_seconds": perf_counter() - started,
                "timings": result["timings"],
            },
        }
    )
