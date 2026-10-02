"""Read a SNILS card and retain its exact field crops for targeted OCR."""

from __future__ import annotations

from typing import Any

from app.modules.russian.crops import extract_field_crops
from app.ocr_engine.russian.recognizer import (
    decode_image_bytes,
    recognize_image,
)

TARGET_FIELDS = frozenset({"Birth_date", "Issue_date", "Licence_number"})


def recognize_snils_bytes(
    content: bytes,
    pipeline: Any | None = None,
) -> dict[str, Any]:
    """Return Russian OCR fields and selected lossless SNILS field crops."""
    image = decode_image_bytes(content)
    result = recognize_image(image, pipeline=pipeline)
    raw_ocr = getattr(result, "ocr", None) or {}
    if not isinstance(raw_ocr, dict):
        raise ValueError("RussianDocsOCR returned invalid SNILS fields")
    return {
        "doctype": getattr(result, "doctype", None),
        "ocr": raw_ocr,
        "crops": extract_field_crops(result, TARGET_FIELDS),
    }
