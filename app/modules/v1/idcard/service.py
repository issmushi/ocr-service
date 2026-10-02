from __future__ import annotations

from time import perf_counter
from typing import Any

from fastapi import UploadFile

from app.core.config import settings
from app.core.errors import AppError, ErrorCode, api_success_response
from app.core.images.uploads import prepare_upload_stream
from app.core.images.validation import validate_image_extension
from app.core.model_availability import ensure_ocr_model_enabled
from app.modules.v1.crop.service import crop_mrz_bytes
from app.modules.v1.idcard.parsing import parse_idcard_mrz
from app.modules.v1.idcard.pipeline import normalize_idcard_mrz
from app.modules.v1.idcard.profiles import IdcardMrzFormatError
from app.modules.v1.passport.runtime import mrz_scanner_runtime


async def recognize_idcard_upload(
    image: UploadFile,
    *,
    crop: bool = True,
    normalize: bool = True,
    normalization: bool = True,
) -> dict[str, Any]:
    ensure_ocr_model_enabled("mrz")
    validate_image_extension(image.filename or "image.jpg")
    await prepare_upload_stream(
        image, max_bytes=settings.ocr_max_image_size_bytes
    )
    content = await image.read()
    if crop:
        content = await crop_mrz_bytes(content)

    started = perf_counter()
    # MRZ Scanner also locates MRZ itself when crop=false supplies a full card.
    raw_result = await mrz_scanner_runtime.run(
        "mrz_crop_text",
        content,
        timeout_seconds=settings.mrz_scanner_document_hard_timeout_seconds,
    )
    duration_seconds = perf_counter() - started
    if not isinstance(raw_result, dict) or not isinstance(
        raw_result.get("items"), list
    ):
        raise AppError(
            ErrorCode.OCR_FAILED, "Модель вернула некорректный ответ OCR"
        )
    items = raw_result["items"]
    if any(
        not isinstance(item, dict) or not isinstance(item.get("text"), str)
        for item in items
    ):
        raise AppError(
            ErrorCode.OCR_FAILED, "Модель вернула некорректный ответ OCR"
        )
    try:
        result = normalize_idcard_mrz(
            [item["text"] for item in items], normalize=normalize
        )
        document = parse_idcard_mrz(result["mrz"])
    except IdcardMrzFormatError as error:
        raise AppError(
            ErrorCode.OCR_FAILED,
            "Неизвестный или неподдерживаемый формат MRZ ID-карты",
        ) from error
    return api_success_response(
        {
            "mrz": result["mrz"],
            "normalization": result["diagnostics"]["normalization"]
            if normalization
            else None,
            **document,
            "meta": {
                "engine": "mrz-scanner",
                "model": "mrz",
                "duration_seconds": duration_seconds,
            },
        }
    )
