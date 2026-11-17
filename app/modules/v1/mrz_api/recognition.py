from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.core.images.preprocessing import decode_image_bytes, image_dimensions
from app.ocr_engine.mrz.recognizer import recognize_mrz

if TYPE_CHECKING:
    from mrzscanner import MRZScanner


def recognize_mrz_text_bytes(
    content: bytes,
    *,
    scanner: MRZScanner,
) -> dict[str, Any]:
    """Decode the original image and pass it directly to MRZ Scanner."""
    image = decode_image_bytes(content)
    width, height = image_dimensions(image)
    recognition = recognize_mrz(image, scanner)
    return {
        "items": [
            {
                "text": text,
                "confidence": None,
                "box": None,
                "detection_confidence": None,
            }
            for text in recognition.lines
        ],
        "image_width": width,
        "image_height": height,
    }
