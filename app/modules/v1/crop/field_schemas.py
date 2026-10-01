"""Response for the experimental passport field detector."""

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
    QWEN3_VL_4B = "qwen3-vl-4b"
    QWEN3_VL_8B = "qwen3-vl-8b"


class FieldOcrResult(BaseModel):
    model: FieldOcrModel
    engine: Literal["paddleocr", "qwen3-vl"]
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
    engine: Literal["paddleocr", "qwen3-vl"]
    duration_seconds: float = Field(..., ge=0)
    processed_fields: int = Field(..., ge=0)
    failed_fields: int = Field(..., ge=0)


class ImagePreview(BaseModel):
    image: str = Field(
        ..., description="Изображение PNG или JPEG в виде data URL."
    )
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)


class DocumentPreview(ImagePreview):
    detected: bool = Field(
        ...,
        description=(
            "Найдены границы документа. При false показан полный кадр "
            "после исправления ориентации."
        ),
    )
    rotation_degrees: Literal[0, 90, 180, 270] = Field(
        ..., description="Поворот документа против часовой стрелки."
    )
    orientation_confidence: float | None = Field(None, ge=0, le=1)


class PassportFieldPreview(ImagePreview):
    image: str = Field(
        ...,
        description=(
            "Вырезка PNG без потерь в виде data URL; "
            "эти же байты получает OCR."
        ),
    )
    ocr: FieldOcrResult | None = Field(
        None,
        description=(
            "Результат OCR этого фрагмента. Null при ocr_model=none. "
            "Фото и подпись пропускаются (status=skipped). "
            "Ошибка отдельного поля не удаляет остальные вырезки."
        ),
    )
    name: str = Field(..., description="Название класса детектора.")
    label: str = Field(..., description="Название поля на русском.")
    bbox: BoundingBox = Field(
        ...,
        description=(
            "[x1, y1, x2, y2] области с отступом на возвращённом документе; "
            "правая и нижняя границы не включаются."
        ),
    )
    detection_bbox: BoundingBox = Field(
        ..., description="Границы поля до добавления отступов."
    )
    confidence: float = Field(
        ..., ge=0, le=1, description="Уверенность детектора поля."
    )
    rotation_degrees: Literal[0, 90, 180, 270] = Field(
        ...,
        description="Дополнительный поворот фрагмента против часовой стрелки.",
    )


class FieldCropMeta(BaseModel):
    engine: Literal["russian-docs-ocr"] = "russian-docs-ocr"
    model: Literal["text-fields-detector"] = "text-fields-detector"
    duration_seconds: float = Field(
        ...,
        ge=0,
        description=(
            "Обработка после чтения файла, включая проверку изображения, "
            "ожидание движка и подготовку фрагментов."
        ),
    )
    timings: dict[str, float] = Field(
        ..., description="Время отдельных этапов обработки в секундах."
    )
    ocr: FieldOcrMeta | None = None


class PassportFieldCropData(BaseModel):
    document: DocumentPreview
    fields: list[PassportFieldPreview]
    meta: FieldCropMeta


class PassportFieldCropResponse(BaseModel):
    success: bool
    data: PassportFieldCropData | None
    error: ErrorSchema | None
