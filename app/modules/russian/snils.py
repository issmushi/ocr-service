"""Read a SNILS card and retain its exact field crops for targeted OCR."""

from __future__ import annotations

from typing import Any

from app.modules.russian.crops import extract_field_crops
from app.modules.russian.field_selection import ocr_fields_for
from app.ocr_engine.russian.recognizer import (
    decode_image_bytes,
    recognize_image,
)

TARGET_FIELDS = frozenset({"Birth_date", "Issue_date", "Licence_number"})
SNILS_OCR_FIELDS = {
    "snils_number": "Licence_number",
    "surname": "Last_name_ru",
    "name": "First_name_ru",
    "patronymic": "Middle_name_ru",
    "date_of_birth": "Birth_date",
    "birth_place": "Birth_place_ru",
    "sex": "Sex_ru",
    "registration_date": "Issue_date",
}
SNILS_PADDLE_FIELDS = frozenset(
    {"snils_number", "date_of_birth", "registration_date"}
)


def recognize_snils_bytes(
    content: bytes,
    pipeline: Any | None = None,
    *,
    fields: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Return Russian OCR fields and selected lossless SNILS field crops."""
    image = decode_image_bytes(content)
    selected = ocr_fields_for(SNILS_OCR_FIELDS, fields)
    # Dates are read only by Paddle; retain Russian OCR's number fallback.
    result = recognize_image(
        image,
        pipeline=pipeline,
        ocr_fields=tuple(
            name
            for name in selected
            if name not in {"Birth_date", "Issue_date"}
        ),
    )
    raw_ocr = getattr(result, "ocr", None) or {}
    if not isinstance(raw_ocr, dict):
        raise ValueError("RussianDocsOCR returned invalid SNILS fields")
    return {
        "doctype": getattr(result, "doctype", None),
        "ocr": raw_ocr,
        "crops": extract_field_crops(
            result, TARGET_FIELDS.intersection(selected)
        ),
    }
