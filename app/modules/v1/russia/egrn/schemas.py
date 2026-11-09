from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from app.core.api_schemas import ErrorSchema


class EgrnModel(StrEnum):
    PADDLE = "paddle"
    CYRILLIC = "paddle-cyrillic"


class EgrnMeta(BaseModel):
    engine: Literal["paddleocr"] = "paddleocr"
    model: EgrnModel
    duration_seconds: float = Field(ge=0)


class EgrnData(BaseModel):
    application_number: str | None = Field(None, description="Номер заявления")
    object_type: str | None = Field(None, description="Вид объекта")
    cadastral_number: str | None = Field(None, description="Кадастровый номер")
    cadastral_quarter_number: str | None = Field(
        None, description="Номер кадастрового квартала"
    )
    cadastral_number_assignment_date: str | None = Field(
        None, description="Дата присвоения кадастрового номера"
    )
    previous_registration_number: str | None = Field(
        None, description="Ранее присвоенный учётный номер"
    )
    address: str | None = Field(None, description="Адрес (местоположение)")
    floor: str | None = Field(None, description="Номер, тип этажа")
    area: str | None = Field(None, description="Площадь, м²")
    purpose: str | None = Field(None, description="Назначение")
    object_name: str | None = Field(None, description="Наименование объекта")
    residential_premises_type: str | None = Field(
        None, description="Вид жилого помещения"
    )
    cadastral_value: str | None = Field(
        None, description="Кадастровая стоимость, руб."
    )
    parent_cadastral_numbers: str | None = Field(
        None, description="Кадастровые номера содержащих объектов"
    )
    cadastral_engineer: str | None = Field(
        None, description="Сведения о кадастровом инженере"
    )
    cultural_heritage: str | None = Field(
        None, description="Сведения о включении в реестр культурного наследия"
    )
    claims: str | None = Field(None, description="Сведения о правопритязаниях")
    specialized_housing: str | None = Field(
        None, description="Отнесение к специализированному жилищному фонду"
    )
    actuality_date: str | None = Field(None, description="Дата актуальности")
    meta: EgrnMeta


class EgrnResponse(BaseModel):
    success: bool
    data: EgrnData | None
    error: ErrorSchema | None
