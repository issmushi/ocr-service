from enum import StrEnum

from app.core.field_selection import SelectedFields


class MrzField(StrEnum):
    ALL = "all"
    MRZ = "mrz"
    NORMALIZATION = "normalization"
    DOCUMENT_TYPE = "document_type"
    DOCUMENT_CODE = "document_code"
    ISSUING_COUNTRY = "issuing_country"
    SURNAME = "surname"
    NAME = "name"
    PATRONYMIC = "patronymic"
    SEX = "sex"
    DATE_OF_BIRTH = "date_of_birth"
    DOCUMENT_SERIES = "document_series"
    DOCUMENT_NUMBER = "document_number"
    NATIONALITY = "nationality"
    ISSUE_DATE = "issue_date"
    EXPIRY_DATE = "expiry_date"
    DEPARTMENT_CODE = "department_code"
    PERSONAL_NUMBER = "personal_number"
    MRZ_VALID = "mrz_valid"


MrzFields = SelectedFields[MrzField]
