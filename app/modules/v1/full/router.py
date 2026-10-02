from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, UploadFile

from app.core.api_schemas import (
    COMMON_ERROR_RESPONSES,
    OCR_ENGINE_UNAVAILABLE_RESPONSE,
    OCR_FAILED_RESPONSE,
    OCR_TIMEOUT_RESPONSE,
)
from app.modules.v1.full.russian_service import recognize_full_russian_upload
from app.modules.v1.full.schemas import (
    FullOcrModel,
    FullOcrResponseSchema,
    RussianFullOcrResponseSchema,
)
from app.modules.v1.full.service import recognize_full_upload

router = APIRouter(tags=["OCR V1: Full"])


@router.post(
    "/full",
    response_model=FullOcrResponseSchema,
    summary="Распознать весь текст изображения",
    description=(
        "Принимает одно изображение и возвращает весь распознанный текст "
        "в этом же запросе. Доступные модели: "
        "paddle, qwen3-vl-4b и qwen3-vl-8b. "
        "Сохраняет регистр и переводы строк в ответе модели. "
        "meta содержит движок, выбранную модель и время OCR."
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
async def recognize_full(
    image: UploadFile = File(
        ..., description="Изображение для полного распознавания текста."
    ),
    model: FullOcrModel = Form(
        FullOcrModel.PADDLE, description="Модель распознавания."
    ),
) -> dict[str, Any]:
    return await recognize_full_upload(image, model=model)


@router.post(
    "/full_test",
    response_model=RussianFullOcrResponseSchema,
    summary="Тест: весь текст изображения через Russian OCR",
    description=(
        "Принимает одно изображение целиком. Russian OCR находит слова "
        "на всём изображении и читает их кириллической моделью accurate. "
        "Возвращает сырой текст и время OCR. "
        "Слова объединяются слева направо, строки — сверху вниз. "
        "Без выделения паспортных полей, "
        "обрезки документа, исправлений и нормализации. "
        "Модель фиксирована: russian-ocr; Paddle не используется. "
        "Это эксперимент: детектор слов рассчитан на текст внутри полей "
        "и может пропускать или объединять текст на целой странице. "
        "Для теста загружайте изображение с текстом в правильной ориентации. "
        "Сложная вёрстка, таблицы и латиница могут распознаваться неточно."
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
async def recognize_full_test(
    image: UploadFile = File(
        ..., description="Изображение с русским печатным текстом."
    ),
) -> dict[str, Any]:
    return await recognize_full_russian_upload(image)
