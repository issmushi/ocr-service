from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, UploadFile

from app.core.api_schemas import (
    COMMON_ERROR_RESPONSES,
    OCR_ENGINE_UNAVAILABLE_RESPONSE,
    OCR_FAILED_RESPONSE,
    OCR_TIMEOUT_RESPONSE,
)
from app.modules.v1.passport.schemas import (
    PassportInternalResponseSchema,
    PassportMrzResponseSchema,
)
from app.modules.v1.passport.service import (
    recognize_passport_internal_upload,
    recognize_passport_upload,
)

router = APIRouter(tags=["OCR V1: Passport"])


@router.post(
    "/passport/mrz",
    response_model=PassportMrzResponseSchema,
    summary="Распознать MRZ-строки паспорта",
    description=(
        "Распознаёт изображение паспорта через MRZ Scanner. "
        "При crop=true (по умолчанию) "
        "сначала выравнивает документ, находит MRZ "
        "и передаёт сканеру только этот фрагмент. При crop=false сканер "
        "находит MRZ на исходном изображении. "
        "При normalize=true (по умолчанию) нормализует тип документа, страну "
        "выдачи и применяет правила страны. Возвращает обе MRZ-строки "
        "и при normalization=true — журнал исправлений каждого символа "
        "с названием поля и позицией. При normalization=false поле "
        "data.normalization равно null. При normalize=false сохраняет "
        "распознанные символы, удаляет пробелы, переводит текст в верхний "
        "регистр и выбирает две самые длинные строки; журнал исправлений "
        "при normalization=true пустой. Из "
        "итоговых строк также извлекает тип и код "
        "паспорта, страну, ФИО, пол, даты, номер и доступные страновые поля."
        " Ответ содержит meta с движком, моделью и временем распознавания."
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
async def recognize_passport(
    image: UploadFile = File(
        ...,
        description="Изображение страницы паспорта с MRZ-зоной.",
    ),
    crop: bool = Form(
        True,
        description=(
            "true — найти и вырезать MRZ перед OCR; false — передать движку "
            "исходное изображение."
        ),
    ),
    normalize: bool = Form(
        True,
        description=(
            "true — применять исправления и дополнения MRZ; false — только "
            "удалить пробелы, перевести текст в верхний регистр и выбрать "
            "две самые длинные строки без исправления распознанных символов."
        ),
    ),
    normalization: bool = Form(
        True,
        description=(
            "true — вернуть журнал исправлений MRZ; false — вернуть "
            "data.normalization=null. На обработку самих MRZ-строк флаг "
            "не влияет."
        ),
    ),
) -> dict[str, Any]:
    return await recognize_passport_upload(
        image,
        crop=crop,
        normalization=normalization,
        **({} if normalize else {"normalize": False}),
    )


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
