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
    PassportMrzResponseSchema,
    PassportOcrModel,
    PassportPage,
    PassportPageOcrModel,
    PassportPageResponseSchema,
)
from app.modules.v1.passport.service import (
    recognize_passport_page_upload,
    recognize_passport_upload,
)

router = APIRouter(tags=["OCR V1: Passport"])


@router.post(
    "/passport/mrz",
    response_model=PassportMrzResponseSchema,
    summary="Распознать MRZ-строки паспорта",
    description=(
        "Принимает изображение паспорта и выбранный движок: paddle, mrz, "
        "qwen3-vl-4b или qwen3-vl-8b. "
        "При crop=true (по умолчанию) "
        "сначала выравнивает документ, находит MRZ "
        "и передаёт движку только этот фрагмент. При crop=false каждый движок "
        "обрабатывает исходное изображение по собственной логике. "
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
    model: PassportOcrModel = Form(
        PassportOcrModel.MRZ,
        description=(
            "Движок распознавания: mrz (по умолчанию), paddle, "
            "qwen3-vl-4b или qwen3-vl-8b."
        ),
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
        model=model,
        crop=crop,
        normalization=normalization,
        **({} if normalize else {"normalize": False}),
    )


@router.post(
    "/passport/page",
    response_model=PassportPageResponseSchema,
    summary="Распознать страницу внутреннего паспорта РФ",
    description=(
        "Распознаёт выбранную страницу внутреннего паспорта РФ через "
        "RussianDocsOCR (по умолчанию), Qwen3-VL-4B "
        "или Qwen3-VL-8B. RussianDocsOCR читает только "
        "поддерживаемые поля российского паспорта; Qwen получает отдельную "
        "инструкцию извлечь паспортные поля. При crop=true API находит, "
        "выравнивает и вырезает весь документ, не запуская поиск MRZ; при "
        "crop=false движок получает исходное "
        "изображение. Ответ содержит те же паспортные "
        "поля, что и /passport/mrz, а также одинаковый объект meta с движком, "
        "моделью и временем распознавания. "
        "Поля mrz, normalization и "
        "mrz_valid всегда равны null, поскольку MRZ не используется. "
        "passport_type, document_code, issuing_country и nationality равны "
        "internal, PN, RUS и RUS соответственно. "
        "Неуверенно распознанные или отсутствующие поля также равны null."
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
async def recognize_passport_page(
    image: UploadFile = File(
        ...,
        description="Изображение страницы внутреннего паспорта РФ.",
    ),
    page: PassportPage = Form(
        ...,
        description="Номер страницы. Сейчас поддерживается только страница 3.",
    ),
    model: PassportPageOcrModel = Form(
        PassportPageOcrModel.RUSSIAN_OCR,
        description=(
            "Модель: russian-ocr, qwen3-vl-4b или qwen3-vl-8b. "
            "Обычный PaddleOCR доступен только в эндпоинте MRZ."
        ),
    ),
    crop: bool = Form(
        True,
        description=(
            "true — найти, выровнять и вырезать весь документ перед OCR; "
            "false — передать исходное изображение. MRZ не ищется."
        ),
    ),
) -> dict[str, Any]:
    return await recognize_passport_page_upload(
        image,
        page=page,
        model=model,
        crop=crop,
    )
