"""Read registration pages with the resident RussianDocsOCR pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.ocr_engine.russian.recognizer import (
    decode_image_bytes,
    recognize_image,
)


def _probability(value: Any) -> float:
    probability = float(value)
    # Float32 model outputs may drift a few ulps beyond the probability range.
    # Keep larger errors intact so the response schema rejects invalid output.
    if -1e-6 <= probability <= 1 + 1e-6:
        return min(1.0, max(0.0, probability))
    return probability


def recognize_registration_bytes(
    content: bytes,
    pipeline: Any | None = None,
) -> dict[str, Any]:
    """Return text and geometry without image arrays in the IPC result."""
    result = recognize_image(decode_image_bytes(content), pipeline=pipeline)
    ocr = result.ocr or {}
    metadata = result.meta_results
    if not isinstance(ocr, dict) or not isinstance(metadata, dict):
        raise ValueError("RussianDocsOCR returned invalid registration data")
    quads = result.address_line_quads
    lines = []
    for index, line in enumerate(metadata.get("Address_lines") or []):
        quad = (
            quads[index] if quads is not None and index < len(quads) else None
        )
        lines.append(
            {
                "kind": line["kind"],
                "text": line["text"],
                "p_handwritten": _probability(line["p_handwritten"]),
                "stamp": line.get("stamp"),
                "quad": (
                    np.asarray(quad, dtype=float).tolist()
                    if quad is not None
                    else None
                ),
            }
        )
    return {
        "doctype": result.doctype,
        "address": ocr.get("Address") or None,
        "has_handwritten": ocr.get("Address_has_handwritten", False),
        "address_lines": lines,
        "registration_stamps": result.registration_stamps,
    }
