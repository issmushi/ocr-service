from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from app.core.api_schemas import ErrorSchema


class FullOcrModel(str, Enum):
    PADDLE = "paddle"
    QWEN3_VL_4B = "qwen3-vl-4b"
    QWEN3_VL_8B = "qwen3-vl-8b"


class FullOcrMetaSchema(BaseModel):
    engine: Literal["paddleocr", "qwen3-vl"] = Field(
        ...,
        description="Движок, выполнивший распознавание.",
    )
    model: FullOcrModel = Field(
        ..., description="Выбранная модель распознавания."
    )
    duration_seconds: float = Field(
        ...,
        ge=0,
        description=(
            "Длительность вызова OCR в секундах, включая ожидание движка; "
            "чтение и проверка изображения в это время не входят."
        ),
    )


class FullOcrDataSchema(BaseModel):
    text: str = Field(
        ...,
        description=(
            "Весь распознанный текст с сохранением регистра и переводов строк."
        ),
    )
    meta: FullOcrMetaSchema = Field(
        ...,
        description="Движок, выбранная модель и время OCR.",
    )


class FullOcrResponseSchema(BaseModel):
    success: bool
    data: FullOcrDataSchema | None
    error: ErrorSchema | None


class RussianFullOcrMetaSchema(FullOcrMetaSchema):
    engine: Literal["russian-docs-ocr"] = Field(
        ..., description="Движок Russian OCR."
    )
    model: Literal["russian-ocr"] = Field(
        ..., description="Russian OCR accurate, кириллическая модель."
    )


class RussianFullOcrDataSchema(FullOcrDataSchema):
    text: str = Field(
        ...,
        description=(
            "Сырой текст Russian OCR. Слова объединяются пробелами, "
            "найденные строки — переводами строк. "
            "Паспортные исправления и нормализация не применяются."
        ),
    )
    meta: RussianFullOcrMetaSchema


class RussianFullOcrResponseSchema(BaseModel):
    success: bool
    data: RussianFullOcrDataSchema | None
    error: ErrorSchema | None
