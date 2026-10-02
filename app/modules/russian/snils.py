"""Read a SNILS card and retain its exact field crops for targeted OCR."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from app.ocr_engine.russian.recognizer import (
    decode_image_bytes,
    recognize_image,
)

TARGET_FIELDS = frozenset({"Birth_date", "Issue_date", "Licence_number"})


def _png_bytes(patch: Any) -> bytes | None:
    if (
        not isinstance(patch, np.ndarray)
        or patch.dtype != np.uint8
        or patch.ndim not in (2, 3)
        or not patch.size
    ):
        return None
    if patch.ndim == 3 and patch.shape[2] not in (3, 4):
        return None
    output = BytesIO()
    Image.fromarray(patch).save(output, format="PNG")
    return output.getvalue()


def _target_crops(result: Any) -> dict[str, bytes]:
    """Reuse aligned field patches without running detection again."""
    text_fields = getattr(result, "text_fields", None)
    if not isinstance(text_fields, (list, tuple)) or len(text_fields) != 2:
        return {}
    boxes, patches = text_fields
    if not isinstance(boxes, (list, tuple)) or not isinstance(
        patches, (list, tuple)
    ):
        return {}

    selected: dict[str, tuple[float, bytes]] = {}
    for box, patch in zip(boxes, patches):
        if not isinstance(box, (list, tuple)) or len(box) < 7:
            continue
        name = str(box[-1])
        if name not in TARGET_FIELDS:
            continue
        try:
            confidence = float(box[4])
        except (TypeError, ValueError):
            continue
        if not 0 <= confidence <= 1:
            continue
        encoded = _png_bytes(patch)
        if encoded is None:
            continue
        if name not in selected or confidence > selected[name][0]:
            selected[name] = (confidence, encoded)
    return {name: content for name, (_, content) in selected.items()}


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
        "crops": _target_crops(result),
    }
