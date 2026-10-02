"""Public SNILS OCR response."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.api_schemas import ErrorSchema


class SnilsMetaSchema(BaseModel):
    engine: Literal["russian-docs-ocr", "russian-docs-ocr+paddleocr"] = Field(
        ..., description="Движки, использованные для распознавания."
    )
    model: Literal[
        "russian-ocr",
        "russian-ocr+cyrillic_PP-OCRv5_mobile_rec",
    ] = Field(..., description="Модели, использованные для распознавания.")
    duration_seconds: float = Field(
        ..., ge=0, description="Время OCR без чтения и проверки файла."
    )


class SnilsDataSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snils_number: str | None = Field(
        ..., description="Номер СНИЛС в формате XXX-XXX-XXX YY."
    )
    surname: str | None = Field(..., description="Фамилия.")
    name: str | None = Field(..., description="Имя.")
    patronymic: str | None = Field(..., description="Отчество.")
    date_of_birth: str | None = Field(
        ..., description="Дата рождения в формате ДД.ММ.ГГГГ."
    )
    birth_place: str | None = Field(..., description="Место рождения.")
    sex: Literal["МУЖ.", "ЖЕН."] | None = Field(..., description="Пол.")
    registration_date: str | None = Field(
        ..., description="Дата регистрации в формате ДД.ММ.ГГГГ."
    )
    meta: SnilsMetaSchema = Field(
        ..., description="Использованные OCR-модели и время обработки."
    )


class SnilsResponseSchema(BaseModel):
    success: bool
    data: SnilsDataSchema | None
    error: ErrorSchema | None
