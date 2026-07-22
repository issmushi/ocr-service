from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from app.ocr_engine.contracts import DetectedTextRegion, TextDetectionResult
from app.ocr_engine.paddle.factory import direct_ocr_pipeline

if TYPE_CHECKING:
    from paddleocr import PaddleOCR


def normalize_box(box: Any) -> list[list[float]]:
    """Convert a Paddle box into a four-point polygon where possible."""
    points = np.asarray(box, dtype=float)
    if points.ndim == 1 and len(points) == 4:
        x1, y1, x2, y2 = points.tolist()
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return points.reshape(-1, 2).tolist()


def detection_confidence(scores: Any, index: int) -> float | None:
    if scores is None:
        return None
    try:
        return float(scores[index])
    except (IndexError, TypeError, ValueError):
        return None


def detect_text_regions(
    ocr: PaddleOCR,
    image: np.ndarray,
) -> TextDetectionResult:
    """Detect every text region without running text recognition."""
    pipeline = direct_ocr_pipeline(ocr)
    raw_result = next(pipeline.text_det_model.predict(image), None)
    if raw_result is None:
        return TextDetectionResult(image=image, regions=())

    detection_image = raw_result.get("input_img")
    if not isinstance(detection_image, np.ndarray):
        detection_image = image

    polygons = raw_result.get("dt_polys")
    if polygons is None or len(polygons) == 0:
        return TextDetectionResult(image=detection_image, regions=())

    scores = raw_result.get("dt_scores")
    regions = [
        DetectedTextRegion(
            box=normalize_box(polygon),
            confidence=detection_confidence(scores, index),
        )
        for index, polygon in enumerate(polygons)
    ]
    return TextDetectionResult(
        image=detection_image,
        regions=tuple(regions),
    )
