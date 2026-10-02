"""Lossless field crops from a RussianDocsOCR pipeline result."""

from __future__ import annotations

from collections.abc import Collection
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image


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


def extract_field_crops(
    result: Any, target_fields: Collection[str]
) -> dict[str, bytes]:
    """Select the highest-confidence patch for each requested field."""
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
        if name not in target_fields:
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
