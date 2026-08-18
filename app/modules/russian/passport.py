from __future__ import annotations

import re
from datetime import date
from typing import Any

from app.ocr_engine.russian.recognizer import (
    decode_image_bytes,
    recognize_image,
)


def normalize_date(value: Any) -> str | None:
    """Normalize common RussianDocsOCR date outputs to ISO date format."""
    text = str(value or "").strip()
    if not text:
        return None

    match = re.search(r"(\d{2})[.\-/ ](\d{2})[.\-/ ](\d{4})", text)
    if match:
        day, month, year = map(int, match.groups())
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            return None

    match = re.search(r"(\d{4})[.\-/ ](\d{2})[.\-/ ](\d{2})", text)
    if match:
        year, month, day = map(int, match.groups())
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            return None

    digits = re.sub(r"\D", "", text)
    if len(digits) == 8:
        day, month, year = int(digits[:2]), int(digits[2:4]), int(digits[4:])
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            return None

    return text


def normalize_sex(value: Any) -> str | None:
    """Normalize RussianDocsOCR sex values to PaddleOCR's M/F convention."""
    text = str(value or "").strip().upper()
    if not text:
        return None

    if text.startswith(("М", "M")):
        return "M"
    if text.startswith(("Ж", "F")):
        return "F"

    return text


def normalize_text(value: Any) -> str | None:
    """Return one non-empty OCR field without changing its characters."""
    text = str(value).strip() if value is not None else ""
    return text or None


def first_text_field(ocr_result: dict[str, Any], *names: str) -> str | None:
    for name in names:
        value = normalize_text(ocr_result.get(name))
        if value is not None:
            return value
    return None


def parse_licence_number(value: Any) -> tuple[str | None, str | None]:
    """Parse passport series and number from a recognized licence field."""
    digits = re.sub(r"\D", "", str(value or ""))
    if len(digits) >= 10:
        return digits[:4], digits[4:10]
    if len(digits) == 9:
        return digits[:3], digits[3:9]

    return None, digits or None


def parse_internal_passport_number(
    value: Any,
) -> tuple[str | None, str | None]:
    """Split the exact ten digits printed in a Russian internal passport."""
    digits = re.sub(r"\D", "", str(value or ""))
    if len(digits) != 10:
        return None, None
    return digits[:4], digits[4:]


def normalize_passport_text_result(
    ocr_result: dict[str, Any],
    *,
    doctype: Any = None,
) -> dict[str, Any]:
    """Map reliable RussianDocsOCR fields to common passport field names."""
    normalized_doctype = str(doctype or "").lower()
    if "intpassport" in normalized_doctype:
        passport_type = "internal"
        document_code = "PN"
    elif "extpassport" in normalized_doctype:
        passport_type = "international"
        document_code = "P"
    else:
        passport_type = None
        document_code = None

    issuing_country = "RUS" if passport_type is not None else None
    nationality = issuing_country
    passport_series, passport_number = parse_licence_number(
        ocr_result.get("Licence_number")
    )
    if passport_type == "international":
        passport_series = None
        passport_number = (
            re.sub(
                r"\s+",
                "",
                normalize_text(ocr_result.get("Licence_number")) or "",
            ).upper()
            or None
        )

    return {
        "passport_type": passport_type,
        "document_code": document_code,
        "issuing_country": issuing_country,
        "surname": first_text_field(
            ocr_result, "Last_name_ru", "Last_name_en"
        ),
        "name": first_text_field(ocr_result, "First_name_ru", "First_name_en"),
        "patronymic": first_text_field(
            ocr_result,
            "Middle_name_ru",
            "Middle_name_en",
        ),
        "sex": normalize_sex(
            ocr_result.get("Sex_ru") or ocr_result.get("Sex_en")
        ),
        "date_of_birth": normalize_date(ocr_result.get("Birth_date")),
        "passport_series": passport_series,
        "passport_number": passport_number,
        "nationality": nationality,
        "issue_date": normalize_date(ocr_result.get("Issue_date")),
        "expiry_date": normalize_date(ocr_result.get("Expiration_date")),
        "department_code": normalize_text(
            ocr_result.get("Issue_organisation_code")
        ),
        "personal_number": None,
        "mrz_valid": None,
    }


def recognize_passport_text_bytes(
    content: bytes,
    pipeline: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Return RussianDocsOCR fields through the common text contract."""
    image = decode_image_bytes(content)
    height, width = image.shape[:2]
    result = recognize_image(image, pipeline=pipeline, **kwargs)
    raw_ocr = getattr(result, "ocr", None) or {}
    if not isinstance(raw_ocr, dict):
        raise ValueError("RussianDocsOCR returned invalid passport text")

    items = []
    for value in raw_ocr.values():
        text = str(value).strip() if value is not None else ""
        if text:
            items.append(
                {
                    "text": text,
                    "confidence": None,
                    "box": None,
                    "detection_confidence": None,
                }
            )
    passport = normalize_passport_text_result(
        raw_ocr,
        doctype=getattr(result, "doctype", None),
    )
    passport_series, passport_number = parse_internal_passport_number(
        raw_ocr.get("Licence_number")
    )
    passport.update(
        {
            "passport_type": "internal",
            "document_code": "PN",
            "issuing_country": "RUS",
            "passport_series": passport_series,
            "passport_number": passport_number,
            "nationality": "RUS",
            "expiry_date": None,
            "personal_number": None,
            "mrz_valid": None,
        }
    )
    return {
        "items": items,
        "image_width": width,
        "image_height": height,
        "passport": passport,
    }
