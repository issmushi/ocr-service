from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.core.images.preprocessing import decode_image_bytes
from app.ocr_engine.paddle.recognizer import recognize_text_crops

if TYPE_CHECKING:
    from paddleocr import PaddleOCR


def recognize_field_text_bytes(
    content: bytes, *, ocr: PaddleOCR
) -> dict[str, Any]:
    """Read an already cropped and oriented field with a CPU recognizer."""
    image = decode_image_bytes(content)
    result = recognize_text_crops(ocr, [image])[0]
    return {
        "text": result.text,
        "confidence": result.confidence,
        "device": "cpu",
    }
