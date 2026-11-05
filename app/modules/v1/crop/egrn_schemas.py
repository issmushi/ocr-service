from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.core.api_schemas import ErrorSchema
from app.modules.v1.crop.field_schemas import BoundingBox, ImagePreview

EgrnFieldName = Literal[
    "application_number",
    "cadastral_number",
    "address",
    "object_name",
    "actuality_date",
]


class EgrnFieldPreview(ImagePreview):
    name: EgrnFieldName
    label: str
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
