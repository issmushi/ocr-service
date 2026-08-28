from __future__ import annotations

from time import perf_counter
from typing import Any

from fastapi import UploadFile

from app.core.config import settings
from app.core.errors import AppError, ErrorCode, api_success_response
from app.core.images.uploads import prepare_upload_stream
from app.core.images.validation import validate_image_extension
from app.core.model_availability import ensure_ocr_model_enabled
from app.modules.paddle.runtime import paddle_runtime
from app.modules.v1.crop.service import crop_mrz_bytes
from app.modules.v1.idcard.parsing import parse_idcard_mrz
from app.modules.v1.idcard.pipeline import normalize_idcard_mrz
from app.modules.v1.idcard.profiles import IdcardMrzFormatError
from app.modules.v1.idcard.schemas import IdcardOcrModel
from app.modules.v1.passport.runtime import mrz_scanner_runtime
from app.modules.vlm.client import recognize_vlm_image

ENGINE_BY_MODEL = {
    "paddle": "paddleocr",
    "mrz": "mrz-scanner",
    "qwen3-vl-4b": "qwen3-vl",
    "qwen3-vl-8b": "qwen3-vl",
}


async def recognize_idcard_upload(
    image: UploadFile,
    *,
    model: IdcardOcrModel = IdcardOcrModel.PADDLE,
    crop: bool = True,
    normalize: bool = True,
    normalization: bool = True,
) -> dict[str, Any]:
    ensure_ocr_model_enabled(model.value)
    validate_image_extension(image.filename or "image.jpg")
    await prepare_upload_stream(
        image, max_bytes=settings.ocr_max_image_size_bytes
    )
    content = await image.read()
    if crop:
        content = await crop_mrz_bytes(content)

    started = perf_counter()
    raw_result = await _recognize(
        content,
        model,
        filename="mrz.jpg" if crop else image.filename or "image.jpg",
        content_type="image/jpeg"
        if crop
        else image.content_type or "application/octet-stream",
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
                "engine": ENGINE_BY_MODEL[model.value],
                "model": model.value,
                "duration_seconds": duration_seconds,
            },
        }
    )


async def _recognize(
    content: bytes,
    model: IdcardOcrModel,
    *,
    filename: str,
    content_type: str,
) -> dict[str, Any]:
    if model == IdcardOcrModel.PADDLE:
        return await paddle_runtime.run(
            "idcard_v1_text",
            content,
            timeout_seconds=settings.paddle_ocr_document_hard_timeout_seconds,
        )
    if model == IdcardOcrModel.MRZ:
        # The scanner locates MRZ itself, also when crop=false supplies the
        # full card.
        return await mrz_scanner_runtime.run(
            "mrz_crop_text",
            content,
            timeout_seconds=settings.mrz_scanner_document_hard_timeout_seconds,
        )
    if model in {IdcardOcrModel.QWEN3_VL_4B, IdcardOcrModel.QWEN3_VL_8B}:
        return await recognize_vlm_image(
            content,
            model=model.value,
            filename=filename,
            content_type=content_type,
        )
    raise AppError(ErrorCode.VALIDATION_ERROR, "Неизвестная модель OCR")
