from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from app.core.api_schemas import ErrorSchema
from app.modules.v1.crop.field_schemas import (
    BoundingBox,
    FieldOcrMeta,
    FieldOcrResult,
    ImagePreview,
)

EgrnFieldName = Literal[
    "application_number",
    "object_type",
    "cadastral_number",
    "cadastral_quarter_number",
    "cadastral_number_assignment_date",
    "previous_registration_number",
    "address",
    "floor",
    "area",
    "purpose",
    "object_name",
    "residential_premises_type",
    "cadastral_value",
    "parent_cadastral_numbers",
    "cadastral_engineer",
    "cultural_heritage",
    "claims",
    "specialized_housing",
    "actuality_date",
]


class EgrnOcrModel(str, Enum):
    NONE = "none"
    PADDLE = "paddle"
    PADDLE_CYRILLIC = "paddle-cyrillic"


class EgrnFieldOcrResult(FieldOcrResult):
    model: EgrnOcrModel
    engine: Literal["paddleocr"]


class EgrnFieldOcrMeta(FieldOcrMeta):
    model: EgrnOcrModel
    engine: Literal["paddleocr"]


class EgrnFieldPreview(ImagePreview):
    name: EgrnFieldName
    label: str
    ocr: EgrnFieldOcrResult | None = Field(
        None, description="Сырой текст вырезки. Null при ocr_model=none."
    )
    bbox: BoundingBox = Field(
        ...,
        description=(
            "[x1, y1, x2, y2] на выровненной странице data.document. "
            "Правая и нижняя границы не включаются. Вырезка — PNG data URL."
        ),
    )


class EgrnFieldMeta(BaseModel):
    engine: Literal["opencv"] = "opencv"
    model: Literal["egrn-gosuslugi-first-page"] = "egrn-gosuslugi-first-page"
    duration_seconds: float = Field(..., ge=0)
    ocr: EgrnFieldOcrMeta | None = None


class EgrnFieldCropData(BaseModel):
    document: ImagePreview
    fields: list[EgrnFieldPreview]
    missing_fields: list[EgrnFieldName] = Field(
        ..., description="Поля, для которых не найдена подходящая область."
    )
    meta: EgrnFieldMeta


class EgrnFieldCropResponse(BaseModel):
    success: bool
    data: EgrnFieldCropData | None
    error: ErrorSchema | None
