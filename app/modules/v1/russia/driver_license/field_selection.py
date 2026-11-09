from enum import StrEnum

from app.core.field_selection import SelectedFields


class DriverLicenseField(StrEnum):
    ALL = "all"
    LICENSE_SERIES = "license_series"
    LICENSE_NUMBER = "license_number"
    SURNAME = "surname"
    NAME = "name"
    PATRONYMIC = "patronymic"
    SURNAME_LATIN = "surname_latin"
    NAME_LATIN = "name_latin"
    PATRONYMIC_LATIN = "patronymic_latin"
    DATE_OF_BIRTH = "date_of_birth"
    BIRTH_PLACE_RU = "birth_place_ru"
    BIRTH_PLACE_EN = "birth_place_en"
    ISSUE_DATE = "issue_date"
    EXPIRY_DATE = "expiry_date"
    ISSUE_ORGANIZATION_RU = "issue_organization_ru"
    ISSUE_ORGANIZATION_EN = "issue_organization_en"
    DEPARTMENT_CODE = "department_code"
    RESIDENCE_REGION_RU = "residence_region_ru"
    RESIDENCE_REGION_EN = "residence_region_en"
    CATEGORIES = "categories"


DriverLicenseFields = SelectedFields[DriverLicenseField]
