from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.core.images.preprocessing import (
    decode_image_bytes,
    prepare_document_image,
)
from app.modules.v1.passport.recognition import recognize_raw_passport_mrz

if TYPE_CHECKING:
    from paddleocr import PaddleOCR


def process_idcard_text_bytes(
    content: bytes, *, ocr: PaddleOCR
) -> dict[str, Any]:
    """Use the passport image preparation while retaining every OCR row."""
    prepared = prepare_document_image(decode_image_bytes(content))
    recognition = recognize_raw_passport_mrz(prepared.image, ocr)
    return {"items": [{"text": text} for text in recognition.lines]}
