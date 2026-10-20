"""Registration crop previews and optional text recognition."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from app.core.api_schemas import ErrorSchema
from app.modules.v1.crop.field_schemas import (
    DocumentPreview,
    FieldOcrMeta,
    FieldOcrResult,
    ImagePreview,
)

Quad = Annotated[list[tuple[float, float]], Field(min_length=4, max_length=4)]


class RegistrationFieldPreview(ImagePreview):
    kind: Literal["address_line", "registration_stamp"]
    name: str
    label: str
    detector_label: str
    confidence: float = Field(..., ge=0, le=1)
    quad: Quad = Field(
        ...,
        description="Углы найденной области на возвращённом документе.",
    )
    crop_quad: Quad = Field(
        ...,
        description=(
            "Углы вырезки с отступом на возвращённом документе. "
            "За границей изображения пиксели заполняются OpenCV."
        ),
    )
    rotation_degrees: float = Field(
        ..., description="Угол выравнивания фрагмента против часовой стрелки."
    )
    ocr: FieldOcrResult | None = Field(
        None, description="OCR этой PNG-вырезки; null при ocr_model=none."
    )


class RegistrationCropMeta(BaseModel):
    engine: Literal["russian-docs-ocr"]
    model: Literal["address-lines-and-registration-stamps"]
    duration_seconds: float = Field(..., ge=0)
    timings: dict[str, float]
    ocr: FieldOcrMeta | None = None


class RegistrationCropData(BaseModel):
    document_type: str
    document: DocumentPreview
    fields: list[RegistrationFieldPreview]
    meta: RegistrationCropMeta


class RegistrationCropResponse(BaseModel):
    success: bool
    data: RegistrationCropData | None
    error: ErrorSchema | None
