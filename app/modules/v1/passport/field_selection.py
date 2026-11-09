from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any

from pydantic import BeforeValidator, Field, TypeAdapter, ValidationError

from app.core.errors import AppError, ErrorCode


class PassportInternalField(StrEnum):
    ALL = "all"
    MRZ = "mrz"
    PASSPORT_TYPE = "passport_type"
    SURNAME = "surname"
    NAME = "name"
    PATRONYMIC = "patronymic"
    SEX = "sex"
    DATE_OF_BIRTH = "date_of_birth"
    BIRTH_PLACE_RU = "birth_place_ru"
    PASSPORT_SERIES = "passport_series"
    PASSPORT_NUMBER = "passport_number"
    ISSUE_DATE = "issue_date"
    DEPARTMENT_CODE = "department_code"
    ISSUE_ORGANIZATION_RU = "Issue_organization_ru"


def _split_form_fields(value: Any) -> Any:
    # Repeated multipart keys are canonical; Swagger also sends CSV arrays.
    if isinstance(value, list):
        return [
            part.strip() if isinstance(part, str) else part
            for item in value
            for part in (item.split(",") if isinstance(item, str) else [item])
        ]
    return value


PassportInternalFields = Annotated[
    list[PassportInternalField],
    BeforeValidator(_split_form_fields),
    Field(min_length=1),
]
_FIELDS_ADAPTER = TypeAdapter(PassportInternalFields)
INTERNAL_FIELDS = tuple(
    field.value for field in PassportInternalField if field != "all"
)
FIELDS_DESCRIPTION = (
    "Поля для распознавания и ответа. По умолчанию all — все поля. "
    "Для выбора передайте несколько fields, например surname и name. "
    "Допускается также список через запятую. all нельзя сочетать с другими "
    "полями. Незапрошенные поля не распознаются и отсутствуют в ответе; "
    "непрочитанные выбранные поля равны null. meta возвращается всегда. "
    "Серия и номер читаются из общей области."
)


def validate_internal_fields(fields: Any = None) -> list[str]:
    """Validate both HTTP input and persisted task options before inference."""
    try:
        selected = _FIELDS_ADAPTER.validate_python(
            ["all"] if fields is None else fields
        )
    except ValidationError as error:
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            "fields должен содержать непустой список допустимых полей.",
        ) from error
    if "all" in selected and len(selected) != 1:
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            "Значение all нельзя сочетать с другими полями.",
        )
    return list(dict.fromkeys(field.value for field in selected))


def resolve_internal_fields(fields: Any = None) -> tuple[str, ...]:
    selected = validate_internal_fields(fields)
    return INTERNAL_FIELDS if selected == ["all"] else tuple(selected)
