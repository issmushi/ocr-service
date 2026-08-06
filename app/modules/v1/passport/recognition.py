from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from app.core.images.regions import crop_regions
from app.modules.v1.passport.detection import select_passport_mrz_regions
from app.ocr_engine.paddle.detector import detect_text_regions
from app.ocr_engine.paddle.recognizer import recognize_text_crops

if TYPE_CHECKING:
    from paddleocr import PaddleOCR


PASSPORT_MRZ_LINE_COUNT = 2


@dataclass(frozen=True, slots=True)
class RawPassportMrzRecognition:
    lines: tuple[str, ...]
    detected_regions_count: int
    selected_regions_count: int


def select_longest_passport_mrz_lines(lines: Sequence[str]) -> tuple[str, ...]:
    """Compact OCR rows and select the two longest in their source order."""
    compact_lines: list[tuple[int, str]] = []
    source_index = 0
    for text in lines:
        for line in text.splitlines():
            compact = "".join(line.split()).upper()
            if compact:
                compact_lines.append((source_index, compact))
            source_index += 1

    selected = sorted(
        compact_lines,
        key=lambda item: (-len(item[1]), item[0]),
    )[:PASSPORT_MRZ_LINE_COUNT]
    selected.sort(key=lambda item: item[0])
    return tuple(line for _, line in selected)


def recognize_raw_passport_mrz(
    image: np.ndarray,
    ocr: PaddleOCR,
) -> RawPassportMrzRecognition:
    """Recognize every detected row before the shared longest-
    line selection.
    """
    detection = detect_text_regions(ocr, image)
    selected_regions = select_passport_mrz_regions(detection)
    crops = crop_regions(
        detection.image,
        [region.box for region in selected_regions],
    )
    recognized = recognize_text_crops(ocr, crops)
    return RawPassportMrzRecognition(
        lines=tuple(item.text for item in recognized if item.text),
        detected_regions_count=len(detection.regions),
        selected_regions_count=len(selected_regions),
    )
