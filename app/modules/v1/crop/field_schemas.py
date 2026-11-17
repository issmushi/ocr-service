"""Shared image previews and per-field OCR results."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from app.core.api_schemas import ErrorSchema

Coordinate = Annotated[int, Field(ge=0)]
BoundingBox = tuple[Coordinate, Coordinate, Coordinate, Coordinate]


class FieldOcrModel(str, Enum):
    NONE = "none"
    PADDLE = "paddle"
    PADDLE_CYRILLIC = "paddle-cyrillic"
    RUSSIAN_OCR_ACCURATE = "russian-ocr-accurate"
    QWEN3_VL_4B = "qwen3-vl-4b"
    QWEN3_VL_8B = "qwen3-vl-8b"


class FieldOcrResult(BaseModel):
    model: FieldOcrModel
    engine: Literal["paddleocr", "russian-docs-ocr", "qwen3-vl"]
    device: str | None = None
    status: Literal["success", "error", "skipped"]
    text: str | None = Field(
        None, description="Прочитанный текст без нормализации."
    )
    confidence: float | None = Field(None, ge=0, le=1)
    duration_seconds: float = Field(..., ge=0)
    error: ErrorSchema | None = None


class FieldOcrMeta(BaseModel):
    model: FieldOcrModel
    engine: Literal["paddleocr", "russian-docs-ocr", "qwen3-vl"]
    duration_seconds: float = Field(..., ge=0)
    processed_fields: int = Field(..., ge=0)
    failed_fields: int = Field(..., ge=0)


class ImagePreview(BaseModel):
    image: str = Field(
        ..., description="Изображение PNG или JPEG в виде data URL."
    )
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
