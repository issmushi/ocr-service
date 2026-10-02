from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, UploadFile

from app.core.api_schemas import (
    COMMON_ERROR_RESPONSES,
    OCR_ENGINE_UNAVAILABLE_RESPONSE,
    OCR_FAILED_RESPONSE,
    OCR_TIMEOUT_RESPONSE,
)
from app.modules.v1.mrz_api.schemas import UnifiedMrzResponseSchema
from app.modules.v1.mrz_api.service import recognize_mrz_upload

router = APIRouter(tags=["MRZ"])


@router.post(
    "/mrz",
    response_model=UnifiedMrzResponseSchema,
    summary="Распознать MRZ паспорта или ID-карты",
    description=(
        "Распознаёт MRZ через MRZ Scanner и автоматически определяет вид "
        "поддерживаемого документа: внутренний или заграничный паспорт, "
        "ID-карта. При crop=true сначала вырезает MRZ; при crop=false "
        "передаёт сканеру исходное изображение. normalize=true применяет "
        "правила исправления для распознанного вида документа. "
        "normalization=true возвращает журнал исправлений; false — null. "
        "Ответ содержит две или три MRZ-строки и общие поля документа."
    ),
    responses={
        400: COMMON_ERROR_RESPONSES[400],
        413: COMMON_ERROR_RESPONSES[413],
        422: OCR_FAILED_RESPONSE,
        503: OCR_ENGINE_UNAVAILABLE_RESPONSE,
        504: OCR_TIMEOUT_RESPONSE,
        500: COMMON_ERROR_RESPONSES[500],
    },
)
async def recognize_mrz(
    image: UploadFile = File(
        ..., description="Изображение паспорта или ID-карты с MRZ."
    ),
    crop: bool = Form(True, description="Вырезать MRZ перед распознаванием."),
    normalize: bool = Form(
        True, description="Исправлять и дополнять символы MRZ."
    ),
    normalization: bool = Form(
        True,
        description=(
            "Вернуть журнал исправлений; false — data.normalization=null."
        ),
    ),
) -> dict[str, Any]:
    return await recognize_mrz_upload(
        image,
        crop=crop,
        normalize=normalize,
        normalization=normalization,
    )
