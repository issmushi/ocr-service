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
from app.modules.v1.crop.service import crop_mrz_bytes
from app.modules.v1.passport.parsing import (
    normalize_visual_passport_data,
    parse_normalized_passport_mrz,
)
from app.modules.v1.passport.pipeline import (
    normalize_passport_mrz_from_text_result,
)
from app.modules.v1.passport.runtime import mrz_scanner_runtime
from app.modules.v1.passport.schemas import (
    PassportInternalDataSchema,
)

PASSPORT_MRZ_LINE_COUNT = 2
PASSPORT_OCR_ENGINE_BY_MODEL = {
    "mrz": "mrz-scanner",
    "qwen3-vl-4b": "qwen3-vl",
    "qwen3-vl-8b": "qwen3-vl",
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


async def recognize_passport_upload(
    image: UploadFile,
    *,
    crop: bool = True,
    normalize: bool = True,
    normalization: bool = True,
) -> dict[str, Any]:
    ensure_ocr_model_enabled("mrz")
    validate_image_extension(image.filename or "image.jpg")
    await prepare_upload_stream(
        image,
        max_bytes=settings.ocr_max_image_size_bytes,
    )
    content = await image.read()
    if crop:
        content = await crop_mrz_bytes(content)
    started = perf_counter()
    normalization_options = {} if normalize else {"normalize": False}
    if crop:
        raw_result = await mrz_scanner_runtime.run(
            "mrz_crop_text",
            content,
            timeout_seconds=settings.mrz_scanner_document_hard_timeout_seconds,
        )
        result = normalize_passport_mrz_from_text_result(
            raw_result, engine="mrz", **normalization_options
        )
    else:
        result = await mrz_scanner_runtime.run(
            "passport_v1",
            content,
            timeout_seconds=settings.mrz_scanner_document_hard_timeout_seconds,
            **normalization_options,
        )
    duration_seconds = perf_counter() - started
    mrz = result.get("mrz")
    if (
        not isinstance(mrz, list)
        or len(mrz) != PASSPORT_MRZ_LINE_COUNT
        or not all(isinstance(line, str) and line.strip() for line in mrz)
    ):
        raise AppError(
            ErrorCode.OCR_FAILED,
            "Не удалось распознать MRZ паспорта",
        )

    diagnostics = result.get("diagnostics")
    normalization_report = None
    if normalization and not normalize:
        normalization_report = {
            "corrections_count": 0,
            "replacements_count": 0,
            "insertions_count": 0,
            "corrections": [],
            "issues": [],
        }
    elif normalization and isinstance(diagnostics, dict):
        normalization_report = diagnostics.get("normalization")
    passport_data = parse_normalized_passport_mrz(mrz)
    if passport_data["passport_type"] is None:
        raise AppError(
            ErrorCode.OCR_FAILED,
            "Неизвестный или неподдерживаемый формат паспорта",
        )
    return api_success_response(
        {
            "mrz": mrz,
            "normalization": normalization_report,
            **passport_data,
            "meta": _passport_ocr_meta("mrz", duration_seconds),
        }
    )


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
        passport_data = normalize_visual_passport_data(
            result.get("passport")
        )
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
