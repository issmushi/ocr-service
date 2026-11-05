from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.core.images.preprocessing import decode_image_bytes
from app.ocr_engine.contracts import RecognizedText
from app.ocr_engine.paddle.lines import split_field_lines
from app.ocr_engine.paddle.recognizer import (
    recognize_text_crops,
    recognize_text_model_crops,
)

if TYPE_CHECKING:
    from paddleocr import PaddleOCR, TextRecognition


def recognize_field_text_bytes(
    content: bytes, *, ocr: PaddleOCR, split_lines: bool = False
) -> dict[str, Any]:
    """Read an already cropped and oriented field with a CPU recognizer."""
    image = decode_image_bytes(content)
    crops = split_field_lines(image) if split_lines else [image]
    return _join_lines(recognize_text_crops(ocr, crops))


def recognize_cyrillic_field_text_bytes(
    content: bytes, *, recognizer: TextRecognition, split_lines: bool = False
) -> dict[str, Any]:
    """Read an oriented crop with the standalone Cyrillic CPU model."""
    image = decode_image_bytes(content)
    crops = split_field_lines(image) if split_lines else [image]
    return _join_lines(recognize_text_model_crops(recognizer, crops))


def _join_lines(lines: list[RecognizedText]) -> dict[str, Any]:
    return {
        "text": "\n".join(line.text for line in lines),
        # Conservative field score: a weak line must remain visible.
        "confidence": min(line.confidence for line in lines),
        "device": "cpu",
    }
