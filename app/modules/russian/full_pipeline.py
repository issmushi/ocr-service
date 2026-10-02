"""Detect text lines with Paddle and read their words with Russian OCR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from app.core.images.regions import crop_quadrilateral
from app.ocr_engine.contracts import DetectedTextRegion, Polygon
from app.ocr_engine.paddle.detector import detect_text_model_regions


@dataclass
class _LineRegion:
    box: Polygon
    left: float
    center_y: float
    height: float


@dataclass
class _TextRow:
    center_y: float
    height: float
    regions: list[_LineRegion] = field(default_factory=list)

    def add(self, region: _LineRegion) -> None:
        count = len(self.regions)
        self.center_y = (self.center_y * count + region.center_y) / (count + 1)
        self.height = (self.height * count + region.height) / (count + 1)
        self.regions.append(region)


def _prepare_region(
    region: DetectedTextRegion, *, width: int, height: int
) -> _LineRegion | None:
    """Clip usable polygons to original pixels before perspective cropping."""
    try:
        points = np.asarray(region.box, dtype=np.float32).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    if len(points) < 4 or not np.isfinite(points).all():
        return None
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    if cv2.contourArea(cv2.convexHull(points)) < 1:
        return None
    left, top = points.min(axis=0)
    right, bottom = points.max(axis=0)
    if right - left < 2 or bottom - top < 2:
        return None
    return _LineRegion(
        box=points.tolist(),
        left=float(left),
        center_y=float((top + bottom) / 2),
        height=float(bottom - top),
    )


def _reading_rows(regions: list[_LineRegion]) -> list[list[_LineRegion]]:
    """Order horizontal rows top-to-bottom and their regions left-to-right."""
    rows: list[_TextRow] = []
    for region in sorted(regions, key=lambda item: (item.center_y, item.left)):
        candidates = [
            row
            for row in rows
            if abs(region.center_y - row.center_y)
            < 0.5 * max(region.height, row.height)
        ]
        if candidates:
            row = min(
                candidates,
                key=lambda item: abs(region.center_y - item.center_y),
            )
        else:
            row = _TextRow(center_y=region.center_y, height=region.height)
            rows.append(row)
        row.add(region)
    return [
        sorted(row.regions, key=lambda region: region.left)
        for row in sorted(rows, key=lambda row: row.center_y)
    ]


def _recognize_line(image: np.ndarray, reader: Any) -> str:
    patches = reader.words_detector.predict_transform(image)["WordsDetector"][
        "warped_img"
    ]
    valid_patches = [
        patch
        for patch in patches
        if isinstance(patch, np.ndarray)
        and patch.ndim == 3
        and patch.shape[2] == 3
        and patch.size
    ]
    # Keep detected lines even when their individual words were not located.
    if not valid_patches:
        valid_patches = [image]
    words = [
        reader.ocr_cyr.predict(patch)["OCRCyrillic"]["ocr_output"]
        for patch in valid_patches
    ]
    return " ".join(word for word in words if word)


def recognize_full_text(
    image: np.ndarray, reader: Any, detector: Any
) -> dict[str, str]:
    """Read all detected regions without field filtering or text corrections.

    Paddle only finds lines and expects BGR. Both perspective crops and Russian
    OCR use original RGB pixels without document cropping or orientation.
    Ordering assumes horizontal rows, not arbitrary multi-column layouts.
    """
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or not image.size
    ):
        raise ValueError("Expected a nonempty RGB uint8 image")

    detection = detect_text_model_regions(
        detector, cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    )
    height, width = image.shape[:2]
    regions = [
        prepared
        for region in detection.regions
        if (prepared := _prepare_region(region, width=width, height=height))
        is not None
    ]
    lines: list[str] = []
    for row in _reading_rows(regions):
        texts: list[str] = []
        for region in row:
            try:
                line_image = crop_quadrilateral(image, region.box)
            except (ValueError, cv2.error):
                # A degenerate region is not an inference error; skip it only.
                continue
            text = _recognize_line(line_image, reader)
            if text:
                texts.append(text)
        if texts:
            lines.append(" ".join(texts))
    return {"text": "\n".join(lines)}
