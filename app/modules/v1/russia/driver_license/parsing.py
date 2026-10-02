"""Map RussianDocsOCR driving licence fields to the public response."""

from __future__ import annotations

import re
from typing import Any

from app.ocr_engine.russian.vendor.document_processing.pipeline.dates import (
    to_ddmmyyyy,
)

_NUMBER_TEXT = re.compile(r"[0-9\s-]+")
_DATE_TEXT = re.compile(r"\s*(\d{2})\s*[./-]\s*(\d{2})\s*[./-]\s*(\d{4})\s*")
_CATEGORY_TOKENS = (
    "C1E",
    "D1E",
    "A1",
    "B1",
    "C1",
    "D1",
    "BE",
    "CE",
    "DE",
    "A",
    "B",
    "C",
    "D",
    "M",
)


def _text(value: Any) -> str | None:
    text = value.strip() if isinstance(value, str) else ""
    return text or None


def parse_driver_license_date(value: Any) -> str | None:
    """Accept one complete printed date, never an adjacent pair of dates."""
    text = _text(value)
    match = _DATE_TEXT.fullmatch(text) if text else None
    if match is None:
        return None
    day, month, year = match.groups()
    return to_ddmmyyyy(f"{day}.{month}.{year}")


def reconcile_date_reading(raw: Any, reread: Any) -> str | None:
    """Fill unclear date characters while preserving Russian OCR's digits."""
    candidate = parse_driver_license_date(reread)
    if candidate is None:
        return None
    original = _text(raw)
    if original is None:
        return candidate
    compact = re.sub(r"[\s./-]", "", original.upper())
    if len(compact) != 8 or not compact.isalnum():
        return candidate
    unknown = [
        index for index, char in enumerate(compact) if not char.isdigit()
    ]
    if not 1 <= len(unknown) <= 2:
        return None
    candidate_digits = candidate.replace(".", "")
    # Keep a date only when the second model agrees on every legible digit.
    if any(
        compact[index].isdigit() and compact[index] != candidate_digits[index]
        for index in range(8)
    ):
        return None
    recovered = "".join(
        char if char.isdigit() else candidate_digits[index]
        for index, char in enumerate(compact)
    )
    return parse_driver_license_date(
        f"{recovered[:2]}.{recovered[2:4]}.{recovered[4:]}"
    )


def _license_number(value: Any) -> tuple[str | None, str | None]:
    """Accept only a complete ten-digit series and number."""
    text = _text(value)
    if not text or _NUMBER_TEXT.fullmatch(text) is None:
        return None, None
    digits = re.sub(r"\D", "", text)
    if len(digits) != 10:
        return None, None
    return f"{digits[:2]} {digits[2:4]}", digits[4:]


def _categories(value: Any) -> list[str]:
    """Split known categories without guessing from invalid OCR characters."""
    text = _text(value)
    if not text:
        return []
    compact = re.sub(r"[\s,;/]+", "", text.upper())
    categories: list[str] = []
    while compact:
        token = next(
            (item for item in _CATEGORY_TOKENS if compact.startswith(item)),
            None,
        )
        if token is None:
            return []
        if token not in categories:
            categories.append(token)
        compact = compact[len(token) :]
    return categories


def driver_license_fields(raw_ocr: dict[str, Any]) -> dict[str, Any]:
    series, number = _license_number(raw_ocr.get("Licence_number"))
    return {
        "license_series": series,
        "license_number": number,
        "surname": _text(raw_ocr.get("Last_name_ru")),
        "name": _text(raw_ocr.get("First_name_ru")),
        "patronymic": _text(raw_ocr.get("Middle_name_ru")),
        "surname_latin": _text(raw_ocr.get("Last_name_en")),
        "name_latin": _text(raw_ocr.get("First_name_en")),
        "patronymic_latin": _text(raw_ocr.get("Middle_name_en")),
        "date_of_birth": parse_driver_license_date(raw_ocr.get("Birth_date")),
        "birth_place_ru": _text(raw_ocr.get("Birth_place_ru")),
        "birth_place_en": _text(raw_ocr.get("Birth_place_en")),
        "issue_date": parse_driver_license_date(raw_ocr.get("Issue_date")),
        "expiry_date": parse_driver_license_date(
            raw_ocr.get("Expiration_date")
        ),
        "issue_organization_ru": _text(raw_ocr.get("Issue_organization_ru")),
        "issue_organization_en": _text(raw_ocr.get("Issue_organization_en")),
        "department_code": _text(raw_ocr.get("Issue_organisation_code")),
        "residence_region_ru": _text(raw_ocr.get("Living_region_ru")),
        "residence_region_en": _text(raw_ocr.get("Living_region_en")),
        "categories": _categories(raw_ocr.get("Driver_class")),
    }
