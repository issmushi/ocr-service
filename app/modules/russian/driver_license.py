"""RussianDocsOCR adapter for the front of a Russian driving licence."""

from __future__ import annotations

from typing import Any

from app.modules.russian.crops import extract_field_crops
from app.ocr_engine.russian.recognizer import (
    decode_image_bytes,
    recognize_image,
)

DATE_FIELDS = frozenset({"Birth_date", "Issue_date", "Expiration_date"})


def recognize_driver_license_bytes(
    content: bytes,
    pipeline: Any | None = None,
) -> dict[str, Any]:
    """Return the detected document type and the engine's field readings."""
    image = decode_image_bytes(content)
    result = recognize_image(image, pipeline=pipeline)
    ocr = getattr(result, "ocr", None) or {}
    if not isinstance(ocr, dict):
        raise ValueError(
            "RussianDocsOCR returned invalid driving licence fields"
        )
    return {
        "doctype": getattr(result, "doctype", None),
        "ocr": ocr,
        "crops": extract_field_crops(result, DATE_FIELDS),
    }
