"""Encode registration crop previews inside the resident OCR worker."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from app.core.images.decoding import decode_image
from app.modules.russian.fields import _preview
from app.ocr_engine.russian.registration_fields import (
    detect_registration_fields,
)


def detect_registration_fields_bytes(
    content: bytes, pipeline: Any, *, padding: int = 4
) -> dict[str, Any]:
    with decode_image(content) as image:
        result = detect_registration_fields(
            np.asarray(image), pipeline=pipeline, padding=padding
        )
    started = perf_counter()
    fields = [
        {
            **{key: value for key, value in field.items() if key != "image"},
            **_preview(field["image"], image_format="PNG"),
        }
        for field in result["fields"]
    ]
    return {
        "document_type": result["document_type"],
        "document": {
            **_preview(result["document_image"]),
            "detected": result["document_detected"],
            "rotation_degrees": result["rotation_degrees"],
            "orientation_confidence": result["orientation_confidence"],
        },
        "fields": fields,
        "timings": {
            **result["timings"],
            "encoding": perf_counter() - started,
        },
    }
