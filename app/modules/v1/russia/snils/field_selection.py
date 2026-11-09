from enum import StrEnum

from app.core.field_selection import SelectedFields


class SnilsField(StrEnum):
    ALL = "all"
    SNILS_NUMBER = "snils_number"
    SURNAME = "surname"
    NAME = "name"
    PATRONYMIC = "patronymic"
    DATE_OF_BIRTH = "date_of_birth"
    BIRTH_PLACE = "birth_place"
    SEX = "sex"
    REGISTRATION_DATE = "registration_date"


SnilsFields = SelectedFields[SnilsField]
