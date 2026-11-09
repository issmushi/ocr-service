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


class UnifiedMrzCorrectionSchema(PassportMrzCorrectionSchema):
    line: Literal[1, 2, 3] = Field(
        ..., description="Номер строки MRZ, начиная с 1."
    )
    field: str = Field(
        ..., description="Поле MRZ, в котором исправлен символ."
    )


class UnifiedMrzNormalizationIssueSchema(PassportMrzNormalizationIssueSchema):
    line: Literal[1, 2, 3] = Field(
        ..., description="Номер строки MRZ с обнаруженной проблемой."
    )


class UnifiedMrzNormalizationSchema(PassportMrzNormalizationSchema):
    corrections: list[UnifiedMrzCorrectionSchema]
    issues: list[UnifiedMrzNormalizationIssueSchema] = Field(
        default_factory=list
    )


class UnifiedMrzDataSchema(BaseModel):
    mrz: list[str] | None = Field(
        None,
        min_length=2,
        max_length=3,
        description="Две строки паспорта или две/три строки ID-карты.",
    )
    normalization: UnifiedMrzNormalizationSchema | None = Field(
        default=None,
        description=(
            "Журнал исправлений; null при normalization=false. "
            "При normalize=false журнал пустой."
        ),
    )
    document_type: (
        Literal["internal_passport", "international_passport", "identity_card"]
        | None
    ) = Field(None, description="Распознанный вид документа.")
    document_code: str | None = Field(None, description="Код документа в MRZ.")
    issuing_country: str | None = Field(
        None, description="Трёхбуквенный код страны выдачи."
    )
    surname: str | None = Field(None, description="Фамилия владельца.")
    name: str | None = Field(None, description="Имя владельца.")
    patronymic: str | None = Field(
        None, description="Отчество или дополнительные имена."
    )
    sex: Literal["МУЖ.", "ЖЕН."] | None = Field(
        None, description="Пол владельца."
    )
    date_of_birth: str | None = Field(
        None, description="Дата рождения в формате ДД.ММ.ГГГГ."
    )
    document_series: str | None = Field(
        None, description="Серия, если она выделяется для документа."
    )
    document_number: str | None = Field(
        None, description="Номер документа без пробелов."
    )
    nationality: str | None = Field(
        None, description="Трёхбуквенный код гражданства."
    )
    issue_date: str | None = Field(
        None, description="Дата выдачи в формате ДД.ММ.ГГГГ."
    )
    expiry_date: str | None = Field(
        None, description="Дата окончания действия в формате ДД.ММ.ГГГГ."
    )
    department_code: str | None = Field(
        None, description="Код подразделения, если он доступен."
    )
    personal_number: str | None = Field(
        None, description="Персональный номер, ИИН или ПИН."
    )
    mrz_valid: bool | None = Field(
        None,
        description=(
            "Результат проверки контрольных цифр; null для неполной MRZ."
        ),
    )
    meta: PassportMrzOcrMetaSchema = Field(
        ..., description="Движок, модель и время OCR."
    )


class UnifiedMrzResponseSchema(BaseModel):
    success: bool
    data: UnifiedMrzDataSchema | None
    error: ErrorSchema | None
