"""Decode a complete image for Russian OCR without passport field detection."""

from __future__ import annotations

from typing import Any

from app.ocr_engine.russian.full_recognizer import recognize_full_text
from app.ocr_engine.russian.recognizer import decode_image_bytes


def recognize_full_text_bytes(
    content: bytes, *, reader: Any
) -> dict[str, Any]:
    return recognize_full_text(decode_image_bytes(content), reader)
