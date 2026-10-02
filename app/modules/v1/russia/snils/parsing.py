"""Public field formatting for a Russian SNILS card."""

from __future__ import annotations

import re
from typing import Any

from app.modules.v1.passport.fields import normalize_sex
from app.ocr_engine.russian.vendor.document_processing.pipeline.dates import (
    to_ddmmyyyy,
)


def _text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def format_snils_number(value: Any) -> str | None:
    """Format exactly eleven read digits, without a checksum assumption."""
    text = _text(value)
    if text is None:
        return None
    digits = re.sub(r"\D", "", text)
    if len(digits) != 11:
        return None
    return f"{digits[:3]}-{digits[3:6]}-{digits[6:9]} {digits[9:]}"


def snils_fields(
    raw_ocr: dict[str, Any],
    *,
    number_text: str | None,
    birth_date_text: str | None,
    issue_date_text: str | None,
) -> dict[str, Any]:
    """Use Paddle's field readings for dates and the complete card number."""
    sex = normalize_sex(raw_ocr.get("Sex_ru"))
    if sex not in {"МУЖ.", "ЖЕН."}:
        sex = None

    return {
        "snils_number": format_snils_number(number_text)
        or format_snils_number(raw_ocr.get("Licence_number")),
        "surname": _text(raw_ocr.get("Last_name_ru")),
        "name": _text(raw_ocr.get("First_name_ru")),
        "patronymic": _text(raw_ocr.get("Middle_name_ru")),
        "date_of_birth": to_ddmmyyyy(birth_date_text or ""),
        "birth_place": _text(raw_ocr.get("Birth_place_ru")),
        "sex": sex,
        "registration_date": to_ddmmyyyy(issue_date_text or ""),
    }
