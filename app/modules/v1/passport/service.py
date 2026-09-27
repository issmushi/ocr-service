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
from app.modules.paddle.runtime import paddle_runtime
from app.modules.russian.runtime import russian_runtime
from app.modules.v1.crop.service import (
    crop_document_image_bytes,
    crop_mrz_bytes,
)
from app.modules.v1.passport.parsing import (
    PassportVlmResultError,
    normalize_visual_passport_data,
    parse_normalized_passport_mrz,
    passport_data_from_vlm_result,
)
from app.modules.v1.passport.pipeline import (
    normalize_passport_mrz_from_text_result,
)
from app.modules.v1.passport.runtime import mrz_scanner_runtime
from app.modules.v1.passport.schemas import (
    PassportOcrModel,
    PassportPage,
    PassportPageDataSchema,
    PassportPageOcrModel,
)
from app.modules.vlm.client import recognize_vlm_image

PASSPORT_MRZ_LINE_COUNT = 2
PASSPORT_OCR_ENGINE_BY_MODEL = {
    PassportOcrModel.PADDLE.value: "paddleocr",
    PassportOcrModel.MRZ.value: "mrz-scanner",
    PassportOcrModel.QWEN3_VL_4B.value: "qwen3-vl",
    PassportOcrModel.QWEN3_VL_8B.value: "qwen3-vl",
    PassportPageOcrModel.RUSSIAN_OCR.value: "russian-docs-ocr",
}


def _passport_ocr_meta(
    model: PassportOcrModel | PassportPageOcrModel,
    duration_seconds: float,
) -> dict[str, Any]:
    return {
        "engine": PASSPORT_OCR_ENGINE_BY_MODEL[model.value],
        "model": model.value,
        "duration_seconds": duration_seconds,
    }


async def recognize_passport_upload(
    image: UploadFile,
    *,
    model: PassportOcrModel = PassportOcrModel.MRZ,
    crop: bool = True,
    normalize: bool = True,
    normalization: bool = True,
) -> dict[str, Any]:
    ensure_ocr_model_enabled(model.value)
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
        result = await _recognize_cropped_mrz(
            content,
            model,
            **normalization_options,
        )
    else:
        result = await _recognize_original_image(
            content,
            model,
            filename=image.filename or "image.jpg",
            content_type=image.content_type or "application/octet-stream",
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
            "meta": _passport_ocr_meta(model, duration_seconds),
        }
    )


async def _recognize_cropped_mrz(
    content: bytes,
    model: PassportOcrModel,
    *,
    normalize: bool = True,
) -> dict[str, Any]:
    normalization_options = {} if normalize else {"normalize": False}
    if model == PassportOcrModel.PADDLE:
        # The passport pipeline tries its MRZ-specific preprocessing variants.
        # Direct text detection can split two low-contrast MRZ rows into short
        # fragments even when the input is already a correct MRZ crop.
        return await paddle_runtime.run(
            "passport_v1",
            content,
            timeout_seconds=settings.paddle_ocr_document_hard_timeout_seconds,
            **normalization_options,
        )
    if model == PassportOcrModel.MRZ:
        raw_result = await mrz_scanner_runtime.run(
            "mrz_crop_text",
            content,
            timeout_seconds=settings.mrz_scanner_document_hard_timeout_seconds,
        )
    elif model in {
        PassportOcrModel.QWEN3_VL_4B,
        PassportOcrModel.QWEN3_VL_8B,
    }:
        raw_result = await recognize_vlm_image(
            content,
            model=model.value,
            filename="mrz.jpg",
            content_type="image/jpeg",
        )
    else:  # pragma: no cover - FastAPI validates the enum before dispatch.
        raise AppError(ErrorCode.VALIDATION_ERROR, "Неизвестная модель OCR")
    return normalize_passport_mrz_from_text_result(
        raw_result,
        engine=model.value,
        **normalization_options,
    )


async def _recognize_original_image(
    content: bytes,
    model: PassportOcrModel,
    *,
    filename: str,
    content_type: str,
    normalize: bool = True,
) -> dict[str, Any]:
    normalization_options = {} if normalize else {"normalize": False}
    if model == PassportOcrModel.PADDLE:
        result = await paddle_runtime.run(
            "passport_v1",
            content,
            timeout_seconds=settings.paddle_ocr_document_hard_timeout_seconds,
            **normalization_options,
        )
    elif model == PassportOcrModel.MRZ:
        result = await mrz_scanner_runtime.run(
            "passport_v1",
            content,
            timeout_seconds=settings.mrz_scanner_document_hard_timeout_seconds,
            **normalization_options,
        )
    elif model in {
        PassportOcrModel.QWEN3_VL_4B,
        PassportOcrModel.QWEN3_VL_8B,
    }:
        raw_result = await recognize_vlm_image(
            content,
            model=model.value,
            filename=filename,
            content_type=content_type,
        )
        result = normalize_passport_mrz_from_text_result(
            raw_result,
            engine=model.value,
            **normalization_options,
        )
    else:  # pragma: no cover - FastAPI validates the enum before dispatch.
        raise AppError(ErrorCode.VALIDATION_ERROR, "Неизвестная модель OCR")
    return result


async def recognize_passport_page_upload(
    image: UploadFile,
    *,
    page: PassportPage,
    model: PassportPageOcrModel = PassportPageOcrModel.RUSSIAN_OCR,
    crop: bool = True,
) -> dict[str, Any]:
    """Return normalized fields from one supported Russian passport page."""
    ensure_ocr_model_enabled(model.value)
    validate_image_extension(image.filename or "image.jpg")
    await prepare_upload_stream(
        image,
        max_bytes=settings.ocr_max_image_size_bytes,
    )
    content = await image.read()
    filename = image.filename or "image.jpg"
    content_type = image.content_type or "application/octet-stream"
    if crop:
        content = await crop_document_image_bytes(content)
        filename = "passport-cropped.jpg"
        content_type = "image/jpeg"
    if page != PassportPage.PAGE_3:  # pragma: no cover - validated by FastAPI.
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            "Поддерживается только третья страница внутреннего паспорта РФ",
        )
    if model == PassportPageOcrModel.RUSSIAN_OCR:
        await asyncio.to_thread(validate_image_bytes, content)
        started = perf_counter()
        result = await russian_runtime.run(
            "passport_text",
            content,
            timeout_seconds=settings.russian_ocr_document_hard_timeout_seconds,
        )
    else:
        started = perf_counter()
        result = await recognize_vlm_image(
            content,
            model=model.value,
            filename=filename,
            content_type=content_type,
            task="passport_fields",
        )
    duration_seconds = perf_counter() - started

    try:
        if not isinstance(result, dict):
            raise AppError(
                ErrorCode.OCR_FAILED,
                "OCR не смог получить пригодный результат",
            )
        if model == PassportPageOcrModel.RUSSIAN_OCR:
            passport_data = normalize_visual_passport_data(
                result.get("passport")
            )
        else:
            try:
                passport_data = passport_data_from_vlm_result(result)
            except PassportVlmResultError as error:
                raise AppError(
                    ErrorCode.OCR_FAILED,
                    "Qwen вернул некорректный формат паспортных данных",
                ) from error
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
        data = PassportPageDataSchema.model_validate(
            {
                "mrz": None,
                "normalization": None,
                **passport_data,
                "meta": _passport_ocr_meta(model, duration_seconds),
            }
        )
    except ValidationError as error:
        raise AppError(
            ErrorCode.OCR_FAILED,
            "OCR не смог получить пригодный результат",
        ) from error
    return api_success_response(data.model_dump(mode="json"))
