"""Text and stamp geometry returned by the registration experiment."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.api_schemas import ErrorSchema

Point = tuple[float, float]
Quad = Annotated[list[Point], Field(min_length=4, max_length=4)]


class RegistrationLineSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    kind: Literal["printed", "handwritten"] = Field(
        ..., description="Печатная или рукописная строка."
    )
    text: str | None = Field(
        ..., description="Прочитанный текст; для рукописной строки — null."
    )
    p_handwritten: float = Field(
        ..., ge=0, le=1, description="Оценка вероятности рукописного текста."
    )
    stamp: int | None = Field(
        ..., ge=0, description="Индекс штампа в registration_stamps, с нуля."
    )
    quad: Quad | None = Field(
        ..., description="Четыре угла строки на исходном фото, в пикселях."
    )


class RegistrationStampSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    kind: Literal["registered", "deregistered", "unclear"] = Field(
        ..., description="Регистрация, снятие с учёта или неясный штамп."
    )
    confidence: float = Field(
        ..., ge=0, le=1, description="Уверенность детектора штампов."
    )
    quad: Quad | None = Field(
        ..., description="Четыре угла штампа на исходном фото, в пикселях."
    )
    lines: list[Annotated[int, Field(ge=0)]] = Field(
        ..., description="Индексы строк этого штампа в address_lines, с нуля."
    )
    address: str = Field(
        ..., description="Только строки этого штампа, разделённые переносами."
    )
    cancels: int | None = Field(
        ..., ge=0, description="Индекс регистрации, которую снимает штамп."
    )
    current: bool = Field(
        ...,
        description=(
            "Предположение библиотеки о действующей регистрации по видимым "
            "штампам и их расположению. Даты штампов пока не читаются."
        ),
    )


class RegistrationMetaSchema(BaseModel):
    engine: Literal["russian-docs-ocr"]
    model: Literal["russian-ocr"]
    duration_seconds: float = Field(
        ..., ge=0, description="Время OCR, включая первую загрузку моделей."
    )


class RegistrationDataSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_type: str = Field(
        ..., description="Тип библиотеки, например INTPASSPORTADDR_ALL."
    )
    address: str | None = Field(
        ...,
        description=(
            "Все адресные строки в порядке чтения страницы. Для отдельных "
            "регистраций используйте address внутри registration_stamps."
        ),
    )
    has_handwritten: bool = Field(
        ..., description="Найдены рукописные строки, которые OCR не читает."
    )
    address_lines: list[RegistrationLineSchema]
    registration_stamps: list[RegistrationStampSchema]
    meta: RegistrationMetaSchema


class RegistrationResponseSchema(BaseModel):
    success: bool
    data: RegistrationDataSchema | None
    error: ErrorSchema | None
