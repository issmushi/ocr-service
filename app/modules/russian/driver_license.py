"""RussianDocsOCR adapter for the front of a Russian driving licence."""

from __future__ import annotations

from typing import Any

from app.modules.russian.crops import extract_field_crops
from app.modules.russian.field_selection import ocr_fields_for
from app.ocr_engine.russian.recognizer import (
    decode_image_bytes,
    recognize_image,
)

DATE_FIELDS = frozenset({"Birth_date", "Issue_date", "Expiration_date"})
DRIVER_LICENSE_OCR_FIELDS = {
    "license_series": "Licence_number",
    "license_number": "Licence_number",
    "surname": "Last_name_ru",
    "name": "First_name_ru",
    "patronymic": "Middle_name_ru",
    "surname_latin": "Last_name_en",
    "name_latin": "First_name_en",
    "patronymic_latin": "Middle_name_en",
    "date_of_birth": "Birth_date",
    "birth_place_ru": "Birth_place_ru",
    "birth_place_en": "Birth_place_en",
    "issue_date": "Issue_date",
    "expiry_date": "Expiration_date",
    "issue_organization_ru": "Issue_organization_ru",
    "issue_organization_en": "Issue_organization_en",
    "department_code": "Issue_organisation_code",
    "residence_region_ru": "Living_region_ru",
    "residence_region_en": "Living_region_en",
    "categories": "Driver_class",
}


def recognize_driver_license_bytes(
    content: bytes,
    pipeline: Any | None = None,
    *,
    fields: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Return the detected document type and the engine's field readings."""
    image = decode_image_bytes(content)
    selected = ocr_fields_for(DRIVER_LICENSE_OCR_FIELDS, fields)
    result = recognize_image(image, pipeline=pipeline, ocr_fields=selected)
    ocr = getattr(result, "ocr", None) or {}
    if not isinstance(ocr, dict):
        raise ValueError(
            "RussianDocsOCR returned invalid driving licence fields"
        )
    return {
        "doctype": getattr(result, "doctype", None),
        "ocr": ocr,
        "crops": extract_field_crops(
            result, DATE_FIELDS.intersection(selected)
        ),
    }
