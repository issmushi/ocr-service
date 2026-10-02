from __future__ import annotations

from time import perf_counter
from typing import Any

from fastapi import UploadFile
from pydantic import ValidationError

from app.core.config import settings
from app.core.errors import AppError, ErrorCode, api_success_response
from app.core.images.uploads import prepare_upload_stream
from app.core.images.validation import validate_image_extension
from app.core.model_availability import ensure_ocr_model_enabled
from app.modules.v1.crop.service import crop_mrz_bytes
from app.modules.v1.idcard.parsing import parse_idcard_mrz
from app.modules.v1.idcard.pipeline import normalize_idcard_mrz
from app.modules.v1.idcard.profiles import IdcardMrzFormatError
from app.modules.v1.mrz_api.schemas import UnifiedMrzDataSchema
from app.modules.v1.passport.normalization import (
    PassportHeaderNormalizationError,
)
from app.modules.v1.passport.parsing import parse_normalized_passport_mrz
from app.modules.v1.passport.pipeline import normalize_passport_mrz
from app.modules.v1.passport.runtime import mrz_scanner_runtime


def _passport_result(
    lines: list[str], *, normalize: bool
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    try:
        result = normalize_passport_mrz(lines, normalize=normalize)
    except PassportHeaderNormalizationError:
        return None
    mrz = result["mrz"]
    if len(mrz) != 2:
        return None
    parsed = parse_normalized_passport_mrz(mrz)
    passport_type = parsed.pop("passport_type")
    if passport_type not in {"internal", "international"}:
        return None
    parsed["document_type"] = f"{passport_type}_passport"
    parsed["document_series"] = parsed.pop("passport_series")
    parsed["document_number"] = parsed.pop("passport_number")
    return result, parsed


def _idcard_result(
    lines: list[str], *, normalize: bool
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    try:
        result = normalize_idcard_mrz(lines, normalize=normalize)
        parsed = parse_idcard_mrz(result["mrz"])
    except IdcardMrzFormatError:
        return None
    return result, parsed


def _raw_lines(result: Any) -> list[str]:
    if not isinstance(result, dict) or not isinstance(
        result.get("items"), list
    ):
        raise AppError(
            ErrorCode.OCR_FAILED, "Модель вернула некорректный ответ OCR"
        )
    items = result["items"]
    if any(
        not isinstance(item, dict) or not isinstance(item.get("text"), str)
        for item in items
    ):
        raise AppError(
            ErrorCode.OCR_FAILED, "Модель вернула некорректный ответ OCR"
        )
    return [item["text"] for item in items]


async def recognize_mrz_upload(
    image: UploadFile,
    *,
    crop: bool = True,
    normalize: bool = True,
    normalization: bool = True,
) -> dict[str, Any]:
    """Scan once, then dispatch by the supported document's MRZ header."""
    ensure_ocr_model_enabled("mrz")
    validate_image_extension(image.filename or "image.jpg")
    await prepare_upload_stream(
        image, max_bytes=settings.ocr_max_image_size_bytes
    )
    content = await image.read()
    if crop:
        content = await crop_mrz_bytes(content)

    started = perf_counter()
    # This operation returns text rows for both a cropped zone and a full
    # image; the scanner can locate MRZ itself when crop=false.
    raw_result = await mrz_scanner_runtime.run(
        "mrz_crop_text",
        content,
        timeout_seconds=settings.mrz_scanner_document_hard_timeout_seconds,
    )
    duration_seconds = perf_counter() - started
    lines = _raw_lines(raw_result)

    candidates = [
        candidate
        for candidate in (
            _passport_result(lines, normalize=normalize),
            _idcard_result(lines, normalize=normalize),
        )
        if candidate is not None
    ]
    if len(candidates) != 1:
        raise AppError(
            ErrorCode.OCR_FAILED,
            "Неизвестный или неоднозначный формат MRZ документа",
        )
    result, document = candidates[0]
    try:
        data = UnifiedMrzDataSchema.model_validate(
            {
                "mrz": result["mrz"],
                "normalization": (
                    result["diagnostics"]["normalization"]
                    if normalization
                    else None
                ),
                **document,
                "meta": {
                    "engine": "mrz-scanner",
                    "model": "mrz",
                    "duration_seconds": duration_seconds,
                },
            }
        )
    except ValidationError as error:
        raise AppError(
            ErrorCode.OCR_FAILED, "OCR не смог получить пригодный результат"
        ) from error
    return api_success_response(data.model_dump(mode="json"))
