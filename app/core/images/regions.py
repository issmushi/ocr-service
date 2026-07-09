from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from app.core.images.preprocessing import order_quad_points

Polygon = list[list[float]]


def crop_quadrilateral(image: np.ndarray, box: Polygon) -> np.ndarray:
    points = np.asarray(box, dtype=np.float32).reshape(-1, 2)
    if len(points) < 4:
        raise ValueError("Image region has fewer than four points")

    if points.shape != (4, 2):
        rectangle = cv2.minAreaRect(points)
        points = cv2.boxPoints(rectangle)

    top_left, top_right, bottom_right, bottom_left = order_quad_points(points)
    width = max(
        int(round(np.linalg.norm(bottom_right - bottom_left))),
        int(round(np.linalg.norm(top_right - top_left))),
    )
    height = max(
        int(round(np.linalg.norm(top_right - bottom_right))),
        int(round(np.linalg.norm(top_left - bottom_left))),
    )
    if width < 2 or height < 2:
        raise ValueError("Image region is too small to crop")

    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(
        np.array([top_left, top_right, bottom_right, bottom_left]),
        destination,
    )
    crop = cv2.warpPerspective(
        image,
        transform,
        (width, height),
        borderMode=cv2.BORDER_REPLICATE,
    )
    if crop.shape[0] / max(1, crop.shape[1]) >= 1.5:
        crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return crop


def crop_regions(
    image: np.ndarray,
    boxes: Sequence[Polygon],
) -> list[np.ndarray]:
    return [crop_quadrilateral(image, box) for box in boxes]


def box_center(box: Polygon) -> tuple[float, float]:
    points = np.asarray(box, dtype=np.float32).reshape(-1, 2)
    return float(points[:, 0].mean()), float(points[:, 1].mean())


def vertical_position(box: Polygon) -> tuple[float, float]:
    x, y = box_center(box)
    return y, x
