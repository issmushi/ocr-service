"""Swagger route for testing Russian passport registration pages."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Response, UploadFile

from app.core.api_schemas import (
    COMMON_ERROR_RESPONSES,
    OCR_ENGINE_UNAVAILABLE_RESPONSE,
    OCR_FAILED_RESPONSE,
    OCR_TIMEOUT_RESPONSE,
)
from app.modules.v1.russia.registration.schemas import (
    RegistrationResponseSchema,
)
from app.modules.v1.russia.registration.service import (
    recognize_registration_upload,
)

router = APIRouter(tags=["Russia"])


@router.post(
    "/passport/registration_test",
    response_model=RegistrationResponseSchema,
    summary="Тест: распознать страницу прописки",
    description=(
        "Загрузите фото страницы или разворота паспорта с пропиской. "
        "RussianDocsOCR читает печатные адреса и находит штампы регистрации "
        "и снятия с учёта. Рукописные строки помечает, но не расшифровывает. "
        "Даты штампов пока не читаются. Признак current определяется "
        "по расположению видимых штампов. Изображение не сохраняется."
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
async def recognize_registration_test(
    response: Response,
    image: UploadFile = File(..., description="Фото прописки в паспорте РФ."),
) -> dict[str, Any]:
    response.headers["Cache-Control"] = "no-store"
    return await recognize_registration_upload(image)
