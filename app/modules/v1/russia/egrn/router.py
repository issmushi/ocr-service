from typing import Any

from fastapi import APIRouter, File, Form, Response, UploadFile

from app.core.api_schemas import (
    COMMON_ERROR_RESPONSES,
    OCR_ENGINE_UNAVAILABLE_RESPONSE,
    OCR_FAILED_RESPONSE,
    OCR_TIMEOUT_RESPONSE,
)
from app.modules.v1.russia.egrn.schemas import EgrnModel, EgrnResponse
from app.modules.v1.russia.egrn.service import recognize_egrn_upload

router = APIRouter(tags=["Russia"])


@router.post(
    "/egrn",
    response_model=EgrnResponse,
    summary="Распознать выписку ЕГРН",
    description=(
        "Читает пять полей первой страницы выписки Госуслуг по "
        "поддерживаемому макету. Отдельные непрочитанные поля равны null."
    ),
    responses={
        **COMMON_ERROR_RESPONSES,
        422: OCR_FAILED_RESPONSE,
        503: OCR_ENGINE_UNAVAILABLE_RESPONSE,
        504: OCR_TIMEOUT_RESPONSE,
    },
)
async def recognize_egrn(
    response: Response,
    image: UploadFile = File(..., description="Скан первой страницы выписки."),
    ocr_model: EgrnModel = Form(EgrnModel.CYRILLIC),
) -> dict[str, Any]:
    response.headers["Cache-Control"] = "no-store"
    return await recognize_egrn_upload(image, ocr_model=ocr_model)
