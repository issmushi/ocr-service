from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.core.images.preprocessing import decode_image_bytes, image_dimensions
from app.ocr_engine.mrz.recognizer import recognize_mrz

if TYPE_CHECKING:
    from mrzscanner import MRZScanner


def process_mrz_crop_text_bytes_with_mrz_scanner(
    content: bytes,
    *,
    scanner: MRZScanner,
) -> dict[str, Any]:
    """Read a prepared crop with scanner postprocessing explicitly disabled."""
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
