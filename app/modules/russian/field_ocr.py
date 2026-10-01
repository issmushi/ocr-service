"""Decode field PNGs inside the Russian OCR inference worker."""

from __future__ import annotations

from typing import Any

from app.ocr_engine.russian.field_recognizer import recognize_field_text
from app.ocr_engine.russian.recognizer import decode_image_bytes


def recognize_field_text_bytes(
    content: bytes, *, reader: Any, field_name: str
) -> dict[str, Any]:
    return recognize_field_text(
        decode_image_bytes(content), reader, field_name=field_name
    )
