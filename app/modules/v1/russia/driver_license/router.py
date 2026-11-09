"""Direct Russian driving licence OCR route."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, File, Form, UploadFile

from app.core.api_schemas import (
    COMMON_ERROR_RESPONSES,
    OCR_ENGINE_UNAVAILABLE_RESPONSE,
    OCR_FAILED_RESPONSE,
    OCR_TIMEOUT_RESPONSE,
)
from app.core.field_selection import VISUAL_FIELDS_DESCRIPTION
from app.modules.v1.russia.driver_license.field_selection import (
    DriverLicenseFields,
)
from app.modules.v1.russia.driver_license.schemas import (
    DriverLicenseResponseSchema,
)
from app.modules.v1.russia.driver_license.service import (
    recognize_driver_license_upload,
)

router = APIRouter(tags=["Russia"])


@router.post(
    "/driver-license",
    response_model=DriverLicenseResponseSchema,
    response_model_exclude_unset=True,
    summary="Распознать водительское удостоверение РФ",
    description=(
        "Принимает фото лицевой стороны водительского удостоверения РФ. "
        "RussianDocsOCR определяет ориентацию и находит поля документа. "
        "Некорректно прочитанные даты дополнительно проверяет кириллический "
        "PaddleOCR, если он включён. Непрочитанные поля равны null."
    ),
    responses={
        400: COMMON_ERROR_RESPONSES[400],
        413: COMMON_ERROR_RESPONSES[413],
        422: {
            **OCR_FAILED_RESPONSE,
            "description": (
                "OCR_FAILED или VALIDATION_ERROR: неверный список fields."
            ),
        },
        503: OCR_ENGINE_UNAVAILABLE_RESPONSE,
        504: OCR_TIMEOUT_RESPONSE,
        500: COMMON_ERROR_RESPONSES[500],
    },
)
async def recognize_driver_license(
    image: UploadFile = File(..., description="Фото водительских прав РФ."),
    fields: Annotated[
        DriverLicenseFields, Form(description=VISUAL_FIELDS_DESCRIPTION)
    ] = ["all"],
) -> dict[str, Any]:
    return await recognize_driver_license_upload(image, fields=fields)
