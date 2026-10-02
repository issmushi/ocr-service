"""Read horizontal Russian text directly from a whole RGB image."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

WordBox = tuple[int, int, int, int]


@dataclass
class _TextLine:
    center_y: float
    height: float
    boxes: list[WordBox] = field(default_factory=list)

    def add(self, box: WordBox) -> None:
        count = len(self.boxes)
        self.center_y = (self.center_y * count + (box[1] + box[3]) / 2) / (
            count + 1
        )
        self.height = (self.height * count + box[3] - box[1]) / (count + 1)
        self.boxes.append(box)


def _reading_lines(boxes: list[WordBox]) -> list[list[WordBox]]:
    """Group by vertical centers, as the vendor does, retaining line breaks."""
    lines: list[_TextLine] = []
    for box in sorted(boxes, key=lambda box: ((box[1] + box[3]) / 2, box[0])):
        center_y = (box[1] + box[3]) / 2
        height = box[3] - box[1]
        candidates = [
            line
            for line in lines
            if abs(center_y - line.center_y) < 0.5 * max(height, line.height)
        ]
        if candidates:
            line = min(
                candidates, key=lambda line: abs(center_y - line.center_y)
            )
        else:
            line = _TextLine(center_y=center_y, height=height)
            lines.append(line)
        line.add(box)
    return [
        sorted(line.boxes, key=lambda box: box[0])
        for line in sorted(lines, key=lambda line: line.center_y)
    ]


def recognize_full_text(image: np.ndarray, reader: Any) -> dict[str, str]:
    """Detect words once and return their raw Cyrillic OCR in reading order.

    The loaded reader supplies ``words_detector`` and ``ocr_cyr``. The vendor
    detector maps boxes back to the original image, so OCR receives original
    RGB pixels. No document crop, orientation, field routing, or corrections
    are applied. This simple order is for horizontal rows, not column layouts.
    """
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or not image.size
    ):
        raise ValueError("Expected a nonempty RGB uint8 image")

    detections = reader.words_detector.predict(image)["WordsDetector"]["bbox"]
    height, width = image.shape[:2]
    boxes: list[WordBox] = []
    for detection in detections:
        # Vendor boxes: x1, y1, x2, y2, confidence, class_id, class_name.
        # Match its exclusive slice edges, clipping both ends before slicing.
        x1, y1, x2, y2 = map(int, detection[:4])
        x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
        y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))

    lines: list[str] = []
    for line in _reading_lines(boxes):
        words: list[str] = []
        for x1, y1, x2, y2 in line:
            patch = image[y1:y2, x1:x2]
            text = reader.ocr_cyr.predict(patch)["OCRCyrillic"]["ocr_output"]
            if text:
                words.append(text)
        if words:
            lines.append(" ".join(words))
    return {"text": "\n".join(lines)}
