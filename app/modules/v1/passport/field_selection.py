from enum import StrEnum
from typing import Any

from app.core.field_selection import (
    VISUAL_FIELDS_DESCRIPTION,
    SelectedFields,
    resolve_fields,
    validate_fields,
)


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


PassportInternalFields = SelectedFields[PassportInternalField]
INTERNAL_FIELDS = tuple(
    field.value for field in PassportInternalField if field != "all"
)
FIELDS_DESCRIPTION = VISUAL_FIELDS_DESCRIPTION + (
    " Серия и номер читаются из общей области."
)


def validate_internal_fields(fields: Any = None) -> list[str]:
    return validate_fields(fields, PassportInternalField)


def resolve_internal_fields(fields: Any = None) -> tuple[str, ...]:
    return resolve_fields(fields, PassportInternalField)
