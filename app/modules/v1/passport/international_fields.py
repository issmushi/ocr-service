from enum import StrEnum

from app.core.field_selection import SelectedFields


class PassportInternationalField(StrEnum):
    ALL = "all"
    MRZ = "mrz"
    PASSPORT_TYPE = "passport_type"
    DOCUMENT_CODE = "document_code"
    ISSUING_COUNTRY = "issuing_country"
    SURNAME = "surname"
    NAME = "name"
    PATRONYMIC = "patronymic"
    SURNAME_LATIN = "surname_latin"
    NAME_LATIN = "name_latin"
    PATRONYMIC_LATIN = "patronymic_latin"
    SEX = "sex"
    DATE_OF_BIRTH = "date_of_birth"
    BIRTH_PLACE_RU = "birth_place_ru"
    BIRTH_PLACE_EN = "birth_place_en"
    PASSPORT_NUMBER = "passport_number"
    NATIONALITY = "nationality"
    ISSUE_DATE = "issue_date"
    EXPIRY_DATE = "expiry_date"
    ISSUE_ORGANIZATION_RU = "Issue_organization_ru"
    ISSUE_ORGANIZATION_CODE = "issue_organization_code"


PassportInternationalFields = SelectedFields[PassportInternationalField]
