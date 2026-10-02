from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, UploadFile

from app.core.api_schemas import (
    COMMON_ERROR_RESPONSES,
    OCR_ENGINE_UNAVAILABLE_RESPONSE,
    OCR_FAILED_RESPONSE,
    OCR_TIMEOUT_RESPONSE,
)
from app.modules.v1.passport.schemas import (
    PassportInternalResponseSchema,
)
from app.modules.v1.passport.service import recognize_passport_internal_upload

router = APIRouter(tags=["Russia"])


@router.post(
    "/passport/internal",
    response_model=PassportInternalResponseSchema,
    summary="Распознать внутренний паспорт РФ",
    description=(
        "Принимает фото внутреннего паспорта РФ и распознаёт его через "
        "RussianDocsOCR. Движок сам находит и выравнивает документ. "
        "Ответ содержит визуальные поля паспорта, включая «Кем выдан» "
        "(Issue_organization_ru), и meta с временем распознавания. "
        "Поля mrz, normalization и mrz_valid всегда равны null. "
        "Отсутствующие поля также равны null."
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
async def recognize_passport_internal(
    image: UploadFile = File(
        ...,
        description="Фото внутреннего паспорта РФ.",
    ),
) -> dict[str, Any]:
    return await recognize_passport_internal_upload(image)
