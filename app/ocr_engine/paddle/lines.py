"""Split deskewed field images into text lines without another model."""

from __future__ import annotations

import cv2
import numpy as np


def split_field_lines(image: np.ndarray) -> list[np.ndarray]:
    """Keep original pixels and reading order; leave single lines untouched."""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10,
    )
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    glyphs = [
        (int(y), int(h))
        for _, y, w, h, area in stats[1:]
        if area >= 5 and h >= 3 and w < width * 0.9 and area >= w * h * 0.12
    ]
    if not glyphs:
        return [image]
    typical_height = float(np.median([h for _, h in glyphs]))
    bands: list[list[int]] = []
    gap = max(1, round(typical_height * 0.15))
    for y, h in sorted(glyphs):
        if h < typical_height * 0.4:
            continue
        if bands and y <= bands[-1][1] + gap:
            bands[-1][1] = max(bands[-1][1], y + h)
        else:
            bands.append([y, y + h])
    if len(bands) < 2:
        return [image]
    # Cut in the whitespace between lines, preserving accents and punctuation.
    boundaries = (
        [0]
        + [
            (first[1] + second[0]) // 2
            for first, second in zip(bands, bands[1:])
        ]
        + [height]
    )
    return [
        image[top:bottom] for top, bottom in zip(boundaries, boundaries[1:])
    ]
