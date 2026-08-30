from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from app.core.images.preprocessing import (
    decode_image_bytes,
    resize_image_max_side,
)
from app.modules.paddle.text import recognize_all_text_regions
from app.ocr_engine.paddle.factory import create_ocr

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from paddleocr import PaddleOCR


def line_text(line: dict[str, Any]) -> str:
    return str(line.get("text", "")).strip()


def lines_to_text(lines: list[dict[str, Any]]) -> str:
    values = [line_text(line) for line in lines]
    return "\n".join(value for value in values if value)


def recognize_full_image(
    image: np.ndarray,
    suffix: str = ".jpg",
    ocr: PaddleOCR | None = None,
    max_side: int | None = 1600,
) -> dict[str, Any]:
    ocr = create_ocr() if ocr is None else ocr
    image = resize_image_max_side(image, max_side=max_side)
    lines = recognize_all_text_regions(ocr, image)
    text = lines_to_text(lines)
    logger.info(
        "PaddleOCR full text extracted raw_text_count=%s text_length=%s",
        len(lines),
        len(text),
    )
    return {
        "text": text,
        "raw_text_count": len([line for line in lines if line_text(line)]),
    }


def recognize_full_bytes(
    content: bytes,
    suffix: str = ".jpg",
    ocr: PaddleOCR | None = None,
    max_side: int | None = 1600,
) -> dict[str, Any]:
    image = decode_image_bytes(content)
    return recognize_full_image(
        image,
        suffix=suffix,
        ocr=ocr,
        max_side=max_side,
    )
