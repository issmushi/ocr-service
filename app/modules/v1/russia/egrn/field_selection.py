from enum import StrEnum

from app.core.field_selection import SelectedFields


class EgrnField(StrEnum):
    ALL = "all"
    APPLICATION_NUMBER = "application_number"
    CADASTRAL_NUMBER = "cadastral_number"
    ADDRESS = "address"
    OBJECT_NAME = "object_name"
    ACTUALITY_DATE = "actuality_date"


EgrnFields = SelectedFields[EgrnField]
