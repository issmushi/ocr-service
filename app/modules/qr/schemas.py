from __future__ import annotations

from pydantic import BaseModel, Field


class QRGenerateRequest(BaseModel):
    surname: str = Field(
        ...,
        min_length=1,
        max_length=100,
        examples=["ТЕСТОВ"],
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        examples=["ТЕСТ"],
    )
    patronymic: str | None = Field(
        default=None,
        max_length=100,
        examples=["ТЕСТОВИЧ"],
    )
    sex: str = Field(
        ...,
        min_length=1,
        max_length=16,
        examples=["МУЖ."],
    )
    date_of_birth: str = Field(
        ...,
        min_length=1,
        max_length=32,
        examples=["15.01.1990"],
    )
    passport_series: str = Field(
        ...,
        min_length=1,
        max_length=32,
        examples=["45 15"],
    )
    passport_number: str = Field(
        ...,
        min_length=1,
        max_length=32,
        examples=["123456"],
    )
