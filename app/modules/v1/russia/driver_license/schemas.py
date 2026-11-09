"""Public response contract for Russian driving licence OCR."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.api_schemas import ErrorSchema


class DriverLicenseMetaSchema(BaseModel):
    engine: Literal["russian-docs-ocr", "russian-docs-ocr+paddleocr"] = Field(
        ..., description="Движок распознавания."
    )
    model: Literal[
        "russian-ocr", "russian-ocr+cyrillic_PP-OCRv5_mobile_rec"
    ] = Field(..., description="Модель OCR.")
    duration_seconds: float = Field(
        ..., ge=0, description="Время обработки OCR в секундах."
    )


class DriverLicenseDataSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    license_series: str | None = Field(
        None, description="Первые четыре цифры номера: серия в формате XX XX."
    )
    license_number: str | None = Field(
        None, description="Последние шесть цифр номера удостоверения."
    )
    surname: str | None = Field(None, description="Фамилия на русском.")
    name: str | None = Field(None, description="Имя на русском.")
    patronymic: str | None = Field(None, description="Отчество на русском.")
    surname_latin: str | None = Field(None, description="Фамилия латиницей.")
    name_latin: str | None = Field(None, description="Имя латиницей.")
    patronymic_latin: str | None = Field(
        None, description="Отчество латиницей."
    )
    date_of_birth: str | None = Field(
        None, description="Дата рождения ДД.ММ.ГГГГ."
    )
    birth_place_ru: str | None = Field(
        None, description="Место рождения на русском."
    )
    birth_place_en: str | None = Field(
        None, description="Место рождения латиницей."
    )
    issue_date: str | None = Field(None, description="Дата выдачи ДД.ММ.ГГГГ.")
    expiry_date: str | None = Field(
        None, description="Действительно до ДД.ММ.ГГГГ."
    )
    issue_organization_ru: str | None = Field(
        None, description="Орган выдачи на русском."
    )
    issue_organization_en: str | None = Field(
        None, description="Орган выдачи латиницей."
    )
    department_code: str | None = Field(None, description="Код органа выдачи.")
    residence_region_ru: str | None = Field(
        None, description="Регион проживания на русском."
    )
    residence_region_en: str | None = Field(
        None, description="Регион проживания латиницей."
    )
    categories: list[str] = Field(
        default_factory=list, description="Распознанные категории прав."
    )
    meta: DriverLicenseMetaSchema = Field(
        ..., description="Модель и время обработки."
    )


class DriverLicenseResponseSchema(BaseModel):
    success: bool
    data: DriverLicenseDataSchema | None
    error: ErrorSchema | None
