from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, UploadFile

from app.core.api_schemas import (
    COMMON_ERROR_RESPONSES,
    OCR_ENGINE_UNAVAILABLE_RESPONSE,
    OCR_FAILED_RESPONSE,
    OCR_TIMEOUT_RESPONSE,
)
from app.modules.v1.idcard.schemas import (
    IdcardMrzResponseSchema,
    IdcardOcrModel,
)
from app.modules.v1.idcard.service import recognize_idcard_upload

router = APIRouter(tags=["OCR V1: ID Card"])


@router.post(
    "/idcard/mrz",
    response_model=IdcardMrzResponseSchema,
    summary="Распознать MRZ-строки ID-карты",
    description=(
        "Распознаёт ID-карты Казахстана с двумя "
        "или тремя строками MRZ и Кыргызстана "
        "с тремя строками. Модели: paddle, mrz, qwen3-vl-4b, qwen3-vl-8b. "
        "При crop=true выравнивает документ и "
        "передаёт OCR только найденную MRZ; "
        "при crop=false движок получает исходное изображение. "
        "normalize=true включает исправление "
        "заголовка, похожих букв и цифр, поля пола "
        "и достройку хвоста. При "
        "normalize=false остаются только удаление пробелов, "
        "uppercase и выбор строк: символы и хвосты не исправляются. "
        "normalization=true возвращает журнал "
        "исправлений (пустой при normalize=false), "
        "normalization=false возвращает data.normalization=null. "
        "Все три флага по умолчанию true. Из "
        "итоговых строк извлекает поля документа; "
        "meta содержит движок, модель и время OCR."
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
async def recognize_idcard(
    image: UploadFile = File(
        ..., description="Изображение стороны ID-карты с MRZ."
    ),
    model: IdcardOcrModel = Form(
        IdcardOcrModel.MRZ,
        description="Модель распознавания. По умолчанию MRZ Scanner (mrz).",
    ),
    crop: bool = Form(True, description="Вырезать MRZ перед OCR."),
    normalize: bool = Form(
        True, description="Применять исправления и достройку MRZ."
    ),
    normalization: bool = Form(
        True,
        description=(
            "Возвращать журнал коррекций; false — data.normalization=null."
        ),
    ),
) -> dict[str, Any]:
    return await recognize_idcard_upload(
        image,
        model=model,
        crop=crop,
        normalize=normalize,
        normalization=normalization,
    )
