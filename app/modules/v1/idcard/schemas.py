from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.core.api_schemas import ErrorSchema
from app.modules.v1.passport.schemas import (
    PassportMrzCorrectionSchema,
    PassportMrzNormalizationIssueSchema,
    PassportMrzNormalizationSchema,
    PassportMrzOcrMetaSchema,
)


class IdcardMrzCorrectionSchema(PassportMrzCorrectionSchema):
    line: Literal[1, 2, 3] = Field(
        ..., description="Номер строки MRZ, начиная с 1."
    )
    field: str = Field(
        ...,
        description=(
            "Поле MRZ: Тип документа, Номер "
            "документа, ИИН, ПИН, Фамилия и имена и другие."
        ),
    )


class IdcardMrzNormalizationIssueSchema(PassportMrzNormalizationIssueSchema):
    line: Literal[1, 2, 3] = Field(
        ..., description="Номер строки MRZ с проблемой."
    )


class IdcardMrzNormalizationSchema(PassportMrzNormalizationSchema):
    corrections: list[IdcardMrzCorrectionSchema]
    issues: list[IdcardMrzNormalizationIssueSchema] = Field(
        default_factory=list
    )


class IdcardMrzDataSchema(BaseModel):
    mrz: list[str] = Field(
        ...,
        min_length=2,
        max_length=3,
        description=(
            "Две или три выбранные строки MRZ; "
            "исправления выполняются только при normalize=true."
        ),
    )
    normalization: IdcardMrzNormalizationSchema | None = Field(
        default=None,
        description=(
            "Журнал коррекций; null при "
            "normalization=false. При normalize=false журнал пустой."
        ),
    )
    document_type: Literal["identity_card"] = Field(
        ..., description="Тип документа: ID-карта."
    )
    document_code: str = Field(..., description="Код документа: ID или I.")
    issuing_country: str = Field(
        ..., description="Трёхбуквенный код страны выдачи: KAZ или KGZ."
    )
    surname: str | None = Field(..., description="Фамилия владельца.")
    name: str | None = Field(..., description="Имя владельца.")
    patronymic: str | None = Field(
        ..., description="Отчество или дополнительные имена."
    )
    sex: Literal["МУЖ.", "ЖЕН."] | None = Field(
        ..., description="Пол владельца."
    )
    date_of_birth: str | None = Field(
        ..., description="Дата рождения в формате ДД.ММ.ГГГГ."
    )
    document_series: str | None = Field(
        ..., description="Отдельная серия; у поддерживаемых ID-карт null."
    )
    document_number: str | None = Field(
        ...,
        description=(
            "Номер ID-карты, включая буквенную часть и начальные нули."
        ),
    )
    nationality: str | None = Field(
        ..., description="Трёхбуквенный код гражданства."
    )
    issue_date: str | None = Field(
        ..., description="Дата выдачи; в поддерживаемых MRZ отсутствует, null."
    )
    expiry_date: str | None = Field(
        ..., description="Дата окончания срока действия в формате ДД.ММ.ГГГГ."
    )
    department_code: str | None = Field(
        ...,
        description=(
            "Код подразделения; в поддерживаемых MRZ отсутствует, null."
        ),
    )
    personal_number: str | None = Field(
        ..., description="ИИН Казахстана или ПИН Кыргызстана."
    )
    mrz_valid: bool | None = Field(
        ..., description="Проверка контрольных цифр; null для неполной MRZ."
    )
    meta: PassportMrzOcrMetaSchema = Field(
        ..., description="MRZ Scanner и время OCR."
    )


class IdcardMrzResponseSchema(BaseModel):
    success: bool
    data: IdcardMrzDataSchema | None
    error: ErrorSchema | None
