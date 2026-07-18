from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Polygon = list[list[float]]


@dataclass(frozen=True, slots=True)
class DocumentOrientation:
    """Predicted clockwise correction and the classifier's confidence."""

    rotation_degrees: int
    confidence: float


@dataclass(frozen=True, slots=True)
class DetectedTextRegion:
    box: Polygon
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class TextDetectionResult:
    image: np.ndarray
    regions: tuple[DetectedTextRegion, ...]


@dataclass(frozen=True, slots=True)
class RecognizedText:
    text: str
    confidence: float
