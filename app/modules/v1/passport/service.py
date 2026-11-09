from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Any

from fastapi import UploadFile
from pydantic import ValidationError

from app.core.config import settings
from app.core.errors import AppError, ErrorCode, api_success_response
from app.core.field_selection import resolve_fields
from app.core.images.uploads import prepare_upload_stream
from app.core.images.validation import (
    validate_image_bytes,
    validate_image_extension,
)
from app.core.model_availability import ensure_ocr_model_enabled
from app.modules.russian.passport import passport_mrz_lines
from app.modules.russian.runtime import russian_runtime
from app.modules.v1.passport.field_selection import resolve_internal_fields
from app.modules.v1.passport.international_fields import (
    PassportInternationalField,
)
from app.modules.v1.passport.parsing import (
    normalize_visual_international_passport_data,
    normalize_visual_passport_data,
)
from app.modules.v1.passport.schemas import (
    PassportInternalDataSchema,
    PassportInternationalDataSchema,
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


async def _recognize_russian_passport_upload(
    image: UploadFile,
    *,
    operation: str,
    fields: tuple[str, ...] | None = None,
) -> tuple[dict[str, Any], float]:
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
        operation,
        content,
        timeout_seconds=settings.russian_ocr_document_hard_timeout_seconds,
        **({"fields": fields} if fields is not None else {}),
    )
    duration_seconds = perf_counter() - started
    if not isinstance(result, dict):
        raise AppError(
            ErrorCode.OCR_FAILED,
            "OCR не смог получить пригодный результат",
        )
    return result, duration_seconds


async def recognize_passport_internal_upload(
    image: UploadFile,
    *,
    fields: list[str] | None = None,
) -> dict[str, Any]:
    """Read an internal Russian passport through RussianDocsOCR."""
    selected = resolve_internal_fields(fields)
    result, duration_seconds = await _recognize_russian_passport_upload(
        image, operation="passport_text", fields=selected
    )
    try:
        passport_data = normalize_visual_passport_data(result.get("passport"))
        available = {
            **passport_data,
            "mrz": passport_mrz_lines(result.get("mrz")),
            "passport_type": "internal",
        }
        data = PassportInternalDataSchema.model_validate(
            {
                **{field: available[field] for field in selected},
                "meta": _passport_ocr_meta("russian-ocr", duration_seconds),
            }
        )
    except ValidationError as error:
        raise AppError(
            ErrorCode.OCR_FAILED,
            "OCR не смог получить пригодный результат",
        ) from error
    return api_success_response(
        data.model_dump(mode="json", exclude_unset=True)
    )


async def recognize_passport_international_upload(
    image: UploadFile,
    *,
    fields: list[str] | None = None,
) -> dict[str, Any]:
    """Read visual fields of a Russian international passport."""
    selected = resolve_fields(fields, PassportInternationalField)
    result, duration_seconds = await _recognize_russian_passport_upload(
        image, operation="passport_international", fields=selected
    )
    try:
        passport_data = normalize_visual_international_passport_data(
            result.get("passport")
        )
        passport_data["mrz"] = passport_mrz_lines(result.get("mrz"))
        data = PassportInternationalDataSchema.model_validate(
            {
                **{field: passport_data[field] for field in selected},
                "meta": _passport_ocr_meta("russian-ocr", duration_seconds),
            }
        )
    except ValidationError as error:
        raise AppError(
            ErrorCode.OCR_FAILED,
            "OCR не смог получить пригодный результат",
        ) from error
    return api_success_response(
        data.model_dump(mode="json", exclude_unset=True)
    )
