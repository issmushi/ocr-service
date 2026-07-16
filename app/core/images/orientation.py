from __future__ import annotations

from numbers import Integral

import cv2
import numpy as np


def rotate_document(image: np.ndarray, rotation_degrees: int) -> np.ndarray:
    """Apply a clockwise quarter turn without interpolation or resizing."""
    if (
        isinstance(rotation_degrees, bool)
        or not isinstance(rotation_degrees, Integral)
        or rotation_degrees not in (0, 90, 180, 270)
    ):
        raise ValueError(
            "Document rotation must be 0, 90, 180 or 270 clockwise degrees"
        )
    if (
        not isinstance(image, np.ndarray)
        or image.ndim not in (2, 3)
        or 0 in image.shape
    ):
        raise ValueError("Document image must be a nonempty image array")
    if rotation_degrees == 0:
        return image
    code = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }[rotation_degrees]
    return cv2.rotate(image, code)
