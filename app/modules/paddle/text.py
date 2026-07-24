from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from app.core.images.regions import crop_regions, vertical_position
from app.ocr_engine.contracts import DetectedTextRegion, TextDetectionResult
from app.ocr_engine.paddle.detector import detect_text_regions
from app.ocr_engine.paddle.recognizer import recognize_text_crops

if TYPE_CHECKING:
    from paddleocr import PaddleOCR


def order_regions(
    regions: Sequence[DetectedTextRegion],
) -> list[DetectedTextRegion]:
    return sorted(regions, key=lambda region: vertical_position(region.box))


def select_all_regions(
    detection: TextDetectionResult,
) -> list[DetectedTextRegion]:
    return order_regions(detection.regions)


def recognize_selected_regions(
    ocr: PaddleOCR,
    detection: TextDetectionResult,
    selected_regions: Sequence[DetectedTextRegion],
    *,
    use_textline_orientation: bool = False,
) -> list[dict[str, Any]]:
    crops = crop_regions(
        detection.image,
        [region.box for region in selected_regions],
    )
    recognized = recognize_text_crops(
        ocr,
        crops,
        use_textline_orientation=use_textline_orientation,
    )
    if len(recognized) != len(selected_regions):
        raise RuntimeError(
            "PaddleOCR recognition result count "
            "does not match selected regions"
        )
    return [
        {
            "text": text.text,
            "confidence": text.confidence,
            "box": region.box,
            "detection_confidence": region.confidence,
        }
        for region, text in zip(selected_regions, recognized)
    ]


def recognize_all_text_regions(
    ocr: PaddleOCR,
    image: np.ndarray,
    *,
    use_textline_orientation: bool = False,
) -> list[dict[str, Any]]:
    detection = detect_text_regions(ocr, image)
    selected = select_all_regions(detection)
    return recognize_selected_regions(
        ocr,
        detection,
        selected,
        use_textline_orientation=use_textline_orientation,
    )
