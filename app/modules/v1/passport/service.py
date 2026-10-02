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
from app.modules.v1.passport.parsing import (
    normalize_visual_passport_data,
)
from app.modules.v1.passport.schemas import (
    PassportInternalDataSchema,
)

PASSPORT_OCR_ENGINE_BY_MODEL = {
    "russian-ocr": "russian-docs-ocr",
}


def _passport_ocr_meta(
    model: str,
    duration_seconds: float,
) -> dict[str, Any]:
    return {
        "engine": PASSPORT_OCR_ENGINE_BY_MODEL[model],
        "model": model,
        "duration_seconds": duration_seconds,
    }


async def recognize_passport_internal_upload(
    image: UploadFile,
) -> dict[str, Any]:
    """Read an internal Russian passport through RussianDocsOCR."""
    ensure_ocr_model_enabled("russian-ocr")
    validate_image_extension(image.filename or "image.jpg")
    await prepare_upload_stream(
        image,
        max_bytes=settings.ocr_max_image_size_bytes,
    )
    content = await image.read()
    await asyncio.to_thread(validate_image_bytes, content)
    started = perf_counter()
    result = await russian_runtime.run(
        "passport_text",
        content,
        timeout_seconds=settings.russian_ocr_document_hard_timeout_seconds,
    )
    duration_seconds = perf_counter() - started

    try:
        if not isinstance(result, dict):
            raise AppError(
                ErrorCode.OCR_FAILED,
                "OCR не смог получить пригодный результат",
            )
        passport_data = normalize_visual_passport_data(result.get("passport"))
        passport_data.update(
            {
                "passport_type": "internal",
                "document_code": "PN",
                "issuing_country": "RUS",
                "nationality": "RUS",
                "expiry_date": None,
                "personal_number": None,
                "mrz_valid": None,
            }
        )
        data = PassportInternalDataSchema.model_validate(
            {
                "mrz": None,
                "normalization": None,
                **passport_data,
                "meta": _passport_ocr_meta("russian-ocr", duration_seconds),
            }
        )
    except ValidationError as error:
        raise AppError(
            ErrorCode.OCR_FAILED,
            "OCR не смог получить пригодный результат",
        ) from error
    return api_success_response(data.model_dump(mode="json"))
