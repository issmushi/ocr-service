from __future__ import annotations

import cv2
import numpy as np


def order_quad_points(points: np.ndarray) -> np.ndarray:
    """Return four distinct convex corners clockwise, starting at top-left."""
    polygon = np.asarray(points, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(polygon).all() or len(np.unique(polygon, axis=0)) != 4:
        raise ValueError(
            "A quadrilateral requires four distinct finite corners"
        )
    center = polygon.mean(axis=0)
    angles = np.arctan2(polygon[:, 1] - center[1], polygon[:, 0] - center[0])
    polygon = polygon[np.argsort(angles)]
    if not cv2.isContourConvex(polygon) or cv2.contourArea(polygon) < 1e-4:
        raise ValueError("A quadrilateral must be convex and non-degenerate")
    # Selecting the start of a cyclic order cannot reuse or lose a corner.
    start = np.lexsort((polygon[:, 0], polygon[:, 1], polygon.sum(axis=1)))[0]
    return np.roll(polygon, -int(start), axis=0).copy()
