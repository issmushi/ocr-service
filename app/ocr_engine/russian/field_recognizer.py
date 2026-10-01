"""Read a displayed passport crop with the current accurate OCR weights."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np


def create_cpu_field_reader(pipeline: Any, *, device: str) -> Any:
    """Reuse CPU models; a GPU worker gets only the three needed CPU models."""
    if device == "cpu":
        return pipeline

    from app.ocr_engine.russian.vendor.document_processing.pipeline_modules import (  # noqa: E501
        OCRCyrillic,
        OCRLatin,
        WordsDetector,
    )

    return SimpleNamespace(
        words_detector=WordsDetector(model_format="ONNX", device="cpu"),
        ocr_cyr=OCRCyrillic(tier="accurate", device="cpu"),
        ocr_lat=OCRLatin(tier="accurate", device="cpu"),
    )


def recognize_field_text(
    image: np.ndarray, reader: Any, *, field_name: str
) -> dict[str, Any]:
    """Return raw OCR, without upstream field corrections or normalization.

    Accurate models recognize words without a space character. Use their
    passport word-splitting rules for multiword fields, retaining reading
    order. The input is the exact displayed RGB crop, without recompression.
    """
    from app.ocr_engine.russian.vendor.document_processing.pipeline.pipeline import (  # noqa: E501
        OCROptionsINTPassport,
    )

    options = OCROptionsINTPassport
    patches = [image]
    if field_name in options.needed_split:
        patches = reader.words_detector.predict_transform(image)[
            "WordsDetector"
        ]["warped_img"]

    latin = field_name in options.en_fields or field_name.endswith("_en")
    engine = reader.ocr_lat if latin else reader.ocr_cyr
    key = "OCRLatin" if latin else "OCRCyrillic"
    words = [
        engine.predict(patch)[key]["ocr_output"]
        for patch in patches
        if patch.size
    ]
    return {"text": " ".join(words), "confidence": None, "device": "cpu"}
