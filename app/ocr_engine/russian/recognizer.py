from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.ocr_engine.russian.factory import create_pipeline


def decode_image_bytes(content: bytes) -> np.ndarray:
    """Decode bytes to RGB for the embedded RussianDocsOCR engine."""
    if not content:
        raise ValueError("Cannot read image: empty file content")
    image = cv2.imdecode(
        np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if image is None:
        raise ValueError("Cannot read image from uploaded file content")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def recognize_image(
    image: np.ndarray | str | Path,
    pipeline: Any | None = None,
    *,
    device: str = "cpu",
    img_size: int = 1500,
) -> Any:
    """Return raw output; document normalization belongs to modules."""
    if pipeline is None:
        pipeline = create_pipeline(device=device)
    return pipeline(image, img_size=img_size)
