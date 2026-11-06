from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from app.core.api_schemas import ErrorSchema


class EgrnModel(StrEnum):
    PADDLE = "paddle"
    CYRILLIC = "paddle-cyrillic"


class EgrnMeta(BaseModel):
    engine: Literal["paddleocr"] = "paddleocr"
    model: EgrnModel
    duration_seconds: float = Field(ge=0)


class EgrnData(BaseModel):
    application_number: str | None
    cadastral_number: str | None
    address: str | None
    object_name: str | None
    actuality_date: str | None
    meta: EgrnMeta


class EgrnResponse(BaseModel):
    success: bool
    data: EgrnData | None
    error: ErrorSchema | None
