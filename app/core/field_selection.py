"""Shared validation for selectable document response fields."""

from enum import StrEnum
from functools import lru_cache
from typing import Annotated, Any, TypeVar

from pydantic import BeforeValidator, Field, TypeAdapter, ValidationError

from app.core.errors import AppError, ErrorCode

FieldName = TypeVar("FieldName", bound=StrEnum)


def split_form_fields(value: Any) -> Any:
    """Accept repeated multipart keys and Swagger's comma-separated arrays."""
    if isinstance(value, list):
        return [
            part.strip() if isinstance(part, str) else part
            for item in value
            for part in (item.split(",") if isinstance(item, str) else [item])
        ]
    return value


SelectedFields = Annotated[
    list[FieldName], BeforeValidator(split_form_fields), Field(min_length=1)
]
FIELDS_DESCRIPTION = (
    "Поля для ответа. По умолчанию all — все поля. Передайте несколько "
    "одноимённых fields или список через запятую. all используется отдельно. "
    "Незапрошенные поля отсутствуют в JSON; непрочитанные выбранные поля "
    "равны null (список категорий может быть пустым). "
    "meta возвращается всегда."
)
VISUAL_FIELDS_DESCRIPTION = FIELDS_DESCRIPTION + (
    " В OCR передаются только выбранные области и общие для них вырезки. "
    "Поиск документа и областей выполняется целиком."
)
MRZ_FIELDS_DESCRIPTION = FIELDS_DESCRIPTION + (
    " MRZ всегда читается целиком для определения формата и проверки "
    "контрольных цифр; fields ограничивает только JSON. Если normalization "
    "не выбран, журнал отсутствует; если выбран при "
    "normalization=false — null."
)


@lru_cache
def _adapter(field_type: type[StrEnum]) -> TypeAdapter:
    return TypeAdapter(SelectedFields[field_type])


def validate_fields(fields: Any, field_type: type[StrEnum]) -> list[str]:
    """Validate HTTP input and stored task options before model execution."""
    try:
        selected = _adapter(field_type).validate_python(
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


def resolve_fields(fields: Any, field_type: type[StrEnum]) -> tuple[str, ...]:
    selected = validate_fields(fields, field_type)
    if selected == ["all"]:
        return tuple(field.value for field in field_type if field != "all")
    return tuple(selected)
