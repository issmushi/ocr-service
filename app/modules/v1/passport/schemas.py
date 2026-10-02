from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.api_schemas import ErrorSchema


class PassportOcrMetaSchema(BaseModel):
    engine: Literal[
        "mrz-scanner",
        "qwen3-vl",
        "russian-docs-ocr",
    ] = Field(
        ...,
        description="Движок, выполнивший распознавание.",
    )
    model: Literal[
        "mrz",
        "russian-ocr",
        "qwen3-vl-4b",
        "qwen3-vl-8b",
    ] = Field(
        ...,
        description="Модель, выполнившая распознавание.",
    )
    duration_seconds: float = Field(
        ...,
        ge=0,
        description=(
            "Длительность вызова OCR в секундах, включая ожидание движка; "
            "чтение, проверка и обрезка изображения в это время не входят."
        ),
    )


class PassportMrzOcrMetaSchema(PassportOcrMetaSchema):
    engine: Literal["mrz-scanner"] = Field(
        ..., description="MRZ Scanner — движок распознавания MRZ."
    )
    model: Literal["mrz"] = Field(
        ..., description="Фиксированная модель распознавания MRZ."
    )


class PassportInternalOcrMetaSchema(PassportOcrMetaSchema):
    engine: Literal["russian-docs-ocr"] = Field(
        ..., description="RussianDocsOCR — движок распознавания паспорта."
    )
    model: Literal["russian-ocr"] = Field(
        ..., description="Фиксированная модель распознавания паспорта."
    )


class PassportMrzCorrectionSchema(BaseModel):
    line: Literal[1, 2] = Field(
        ...,
        description="Номер строки MRZ: 1 — верхняя, 2 — нижняя.",
    )
    position: int = Field(
        ...,
        ge=1,
        description="Позиция символа в итоговой строке MRZ, начиная с 1.",
    )
    source_position: int | None = Field(
        ...,
        ge=1,
        description=(
            "Позиция символа в исходной строке OCR, начиная с 1; "
            "null, если символ добавлен."
        ),
    )
    field: str = Field(
        ...,
        description=(
            "Название поля MRZ: например, Тип документа, Страна выдачи, "
            "Фамилия и имена, Номер паспорта или Гражданство."
        ),
    )
    operation: Literal["замена", "вставка"] = Field(
        ...,
        description="Операция над символом: замена или вставка.",
    )
    source: str = Field(
        ...,
        max_length=1,
        description="Исходный символ; пустая строка, если символ добавлен.",
    )
    replacement: str = Field(
        ...,
        min_length=1,
        max_length=1,
        description="Символ после исправления или добавленный символ.",
    )
    reason: str = Field(
        ...,
        description="Причина исправления на русском языке.",
    )


class PassportMrzNormalizationIssueSchema(BaseModel):
    line: Literal[1, 2] = Field(
        ...,
        description="Номер строки MRZ, в которой обнаружена проблема.",
    )
    code: str = Field(..., description="Код обнаруженной проблемы.")
    message: str = Field(
        ..., description="Описание проблемы на русском языке."
    )


class PassportMrzNormalizationSchema(BaseModel):
    corrections_count: int = Field(
        ...,
        ge=0,
        description=(
            "Общее количество исправлений символов: замен и добавлений."
        ),
    )
    replacements_count: int = Field(
        ...,
        ge=0,
        description="Количество заменённых символов.",
    )
    insertions_count: int = Field(
        ...,
        ge=0,
        description="Количество добавленных символов.",
    )
    corrections: list[PassportMrzCorrectionSchema] = Field(
        ...,
        description=(
            "Журнал исправлений, отдельная запись для каждого символа."
        ),
    )
    issues: list[PassportMrzNormalizationIssueSchema] = Field(
        default_factory=list,
        description=(
            "Обнаруженные проблемы, из-за "
            "которых часть нормализации не выполнена."
        ),
    )


class PassportInternalDataSchema(BaseModel):
    """Fields read from the visual page of an internal Russian passport."""

    model_config = ConfigDict(extra="forbid")

    mrz: list[str] | None = Field(
        ...,
        min_length=2,
        max_length=2,
        description=(
            "Две MRZ-строки, если RussianDocsOCR распознал их; иначе null."
        ),
    )
    passport_type: Literal["internal"] = Field(
        ..., description="Внутренний паспорт РФ."
    )
    surname: str | None = Field(..., description="Фамилия владельца.")
    name: str | None = Field(..., description="Имя владельца.")
    patronymic: str | None = Field(..., description="Отчество владельца.")
    sex: Literal["МУЖ.", "ЖЕН."] | None = Field(
        ..., description="Пол владельца."
    )
    date_of_birth: str | None = Field(
        ..., description="Дата рождения в формате ДД.ММ.ГГГГ."
    )
    passport_series: str | None = Field(
        ..., description="Четырёхзначная серия паспорта."
    )
    passport_number: str | None = Field(
        ..., description="Шестизначный номер паспорта."
    )
    issue_date: str | None = Field(
        ..., description="Дата выдачи в формате ДД.ММ.ГГГГ."
    )
    department_code: str | None = Field(
        ..., description="Код подразделения в формате XXX-XXX."
    )
    Issue_organization_ru: str | None = Field(
        ..., description="Кем выдан внутренний паспорт РФ."
    )
    meta: PassportInternalOcrMetaSchema = Field(
        ..., description="Движок RussianDocsOCR и время распознавания."
    )


class PassportInternalResponseSchema(BaseModel):
    success: bool
    data: PassportInternalDataSchema | None
    error: ErrorSchema | None
