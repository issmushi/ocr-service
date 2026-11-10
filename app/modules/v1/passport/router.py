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
from app.modules.v1.passport.field_selection import (
    FIELDS_DESCRIPTION,
    PassportInternalFields,
)
from app.modules.v1.passport.international_fields import (
    PassportInternationalFields,
)
from app.modules.v1.passport.schemas import (
    PassportInternalResponseSchema,
    PassportInternationalResponseSchema,
)
from app.modules.v1.passport.service import (
    recognize_passport_internal_upload,
    recognize_passport_international_upload,
)

router = APIRouter(tags=["Russia Sync"])


@router.post(
    "/passport/internal",
    response_model=PassportInternalResponseSchema,
    response_model_exclude_unset=True,
    summary="Распознать внутренний паспорт РФ",
    description=(
        "Принимает фото внутреннего паспорта РФ и распознаёт его через "
        "RussianDocsOCR. Движок сам находит и выравнивает документ. "
        "Доступны текстовые поля основного разворота: ФИО, пол, "
        "дату и место рождения (birth_place_ru), серию и номер, дату выдачи, "
        "код подразделения и «Кем выдан» (Issue_organization_ru). "
        "В meta указано время распознавания. "
        "Поле mrz содержит две строки, если RussianDocsOCR распознал их, "
        "иначе null. Нераспознанные визуальные поля также равны null. "
        "fields ограничивает распознавание и ответ выбранными полями; "
        "по умолчанию all. meta возвращается всегда."
    ),
    responses={
        400: COMMON_ERROR_RESPONSES[400],
        413: COMMON_ERROR_RESPONSES[413],
        422: {
            **OCR_FAILED_RESPONSE,
            "description": (
                "OCR_FAILED: непригодный результат распознавания; "
                "VALIDATION_ERROR: некорректный список fields."
            ),
        },
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
    fields: Annotated[
        PassportInternalFields, Form(description=FIELDS_DESCRIPTION)
    ] = ["all"],
) -> dict[str, Any]:
    return await recognize_passport_internal_upload(image, fields=fields)


@router.post(
    "/passport/international",
    response_model=PassportInternationalResponseSchema,
    response_model_exclude_unset=True,
    summary="Распознать загранпаспорт РФ",
    description=(
        "Принимает фото загранпаспорта РФ и распознаёт визуальные поля "
        "через RussianDocsOCR. Возвращает ФИО на русском и латинице, "
        "полный номер паспорта, даты, место рождения и орган выдачи. "
        "MRZ возвращается как две строки, если движок их прочитал; "
        "иначе mrz равен null."
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
async def recognize_passport_international(
    image: UploadFile = File(
        ...,
        description="Фото загранпаспорта РФ.",
    ),
    fields: Annotated[
        PassportInternationalFields,
        Form(description=VISUAL_FIELDS_DESCRIPTION),
    ] = ["all"],
) -> dict[str, Any]:
    return await recognize_passport_international_upload(image, fields=fields)
