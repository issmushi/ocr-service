from __future__ import annotations

import re
from typing import Any


def normalize_sex(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    if not text:
        return None
    if text.startswith(("М", "M")):
        return "МУЖ."
    if text.startswith(("Ж", "F")):
        return "ЖЕН."
    return text


def normalize_date(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None

    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
    if match:
        year, month, day = match.groups()
        return f"{day}.{month}.{year}"
    return text


def normalize_series(value: Any) -> str | None:
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value))
    if len(digits) == 4:
        return f"{digits[:2]} {digits[2:]}"
    return str(value).strip() or None


def normalize_number(value: Any) -> str | None:
    if value is None:
        return None
    normalized = re.sub(r"\s+", "", str(value)).upper()
    return normalized or None
