"""Direct SNILS OCR route in the Russia Swagger group."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, UploadFile

from app.core.api_schemas import (
    COMMON_ERROR_RESPONSES,
    OCR_ENGINE_UNAVAILABLE_RESPONSE,
    OCR_FAILED_RESPONSE,
    OCR_TIMEOUT_RESPONSE,
)
from app.modules.v1.russia.snils.schemas import SnilsResponseSchema
from app.modules.v1.russia.snils.service import recognize_snils_upload

router = APIRouter(tags=["Russia"])


@router.post(
    "/snils",
    response_model=SnilsResponseSchema,
    summary="Распознать СНИЛС",
    description=(
        "Принимает фото СНИЛС. RussianDocsOCR находит документ и его поля. "
        "Вырезки даты рождения, даты регистрации и номера повторно читает "
        "кириллическая модель PaddleOCR. Нераспознанные поля равны null."
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
async def recognize_snils(
    image: UploadFile = File(..., description="Фото СНИЛС."),
) -> dict[str, Any]:
    return await recognize_snils_upload(image)
