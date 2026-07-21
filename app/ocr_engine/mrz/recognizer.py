from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mrzscanner import MRZScanner


@dataclass(frozen=True, slots=True)
class MrzScanResult:
    lines: tuple[str, ...]
    detected_regions_count: int


def _detected_regions_count(polygon: Any) -> int:
    if polygon is None:
        return 0

    try:
        points = np.asarray(polygon, dtype=float)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "MRZScanner returned an invalid MRZ polygon"
        ) from error

    if points.size == 0:
        return 0
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise RuntimeError("MRZScanner returned an invalid MRZ polygon")
    return 1


def recognize_mrz(image: np.ndarray, scanner: MRZScanner) -> MrzScanResult:
    """Scan a BGR image without changing the recognized MRZ characters."""
    try:
        result = scanner(image, do_center_crop=False, do_postprocess=False)
    except Exception as error:
        raise RuntimeError("MRZScanner recognition failed") from error

    if result is None:
        return MrzScanResult(lines=(), detected_regions_count=0)
    if (
        not isinstance(result, dict)
        or not {"mrz_polygon", "mrz_texts"} <= result.keys()
    ):
        raise RuntimeError("MRZScanner returned an invalid recognition result")

    detected_regions_count = _detected_regions_count(result["mrz_polygon"])
    texts = result["mrz_texts"]
    if texts is None:
        texts = []
    if not isinstance(texts, (list, tuple)) or any(
        not isinstance(text, str) for text in texts
    ):
        raise RuntimeError("MRZScanner returned invalid MRZ texts")

    return MrzScanResult(
        lines=tuple(texts) if detected_regions_count else (),
        detected_regions_count=detected_regions_count,
    )
