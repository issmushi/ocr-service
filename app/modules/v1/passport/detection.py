from __future__ import annotations

from app.core.images.regions import vertical_position
from app.ocr_engine.contracts import DetectedTextRegion, TextDetectionResult


def select_passport_mrz_regions(
    detection: TextDetectionResult,
) -> list[DetectedTextRegion]:
    """Return every detected row in visual order for post-OCR MRZ filtering."""
    return sorted(
        detection.regions,
        key=lambda region: vertical_position(region.box),
    )
