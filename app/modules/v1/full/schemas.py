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
