from enum import StrEnum

from app.core.field_selection import SelectedFields


class EgrnField(StrEnum):
    ALL = "all"
    APPLICATION_NUMBER = "application_number"
    OBJECT_TYPE = "object_type"
    CADASTRAL_NUMBER = "cadastral_number"
    CADASTRAL_QUARTER_NUMBER = "cadastral_quarter_number"
    CADASTRAL_NUMBER_ASSIGNMENT_DATE = "cadastral_number_assignment_date"
    PREVIOUS_REGISTRATION_NUMBER = "previous_registration_number"
    ADDRESS = "address"
    FLOOR = "floor"
    AREA = "area"
    PURPOSE = "purpose"
    OBJECT_NAME = "object_name"
    RESIDENTIAL_PREMISES_TYPE = "residential_premises_type"
    CADASTRAL_VALUE = "cadastral_value"
    PARENT_CADASTRAL_NUMBERS = "parent_cadastral_numbers"
    CADASTRAL_ENGINEER = "cadastral_engineer"
    CULTURAL_HERITAGE = "cultural_heritage"
    CLAIMS = "claims"
    SPECIALIZED_HOUSING = "specialized_housing"
    ACTUALITY_DATE = "actuality_date"


EgrnFields = SelectedFields[EgrnField]
