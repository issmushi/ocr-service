from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.images.geometry import order_quad_points

ANALYSIS_MAX_SIDE = 1200
MIN_DOCUMENT_AREA_RATIO = 0.20
MIN_DOCUMENT_SIDE = 96
MAX_DOCUMENT_ASPECT_RATIO = 3.0
MAX_DESKEW_DEGREES = 12.0
MIN_DESKEW_DEGREES = 0.35
TARGET_LONG_SIDE = 1800
MAX_LONG_SIDE = 2400
MAX_UPSCALE = 3.0


@dataclass(frozen=True)
class PreparedImage:
    image: np.ndarray
    metadata: dict[str, Any]


def read_image(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return image


def decode_image_bytes(content: bytes) -> np.ndarray:
    if not content:
        raise ValueError("Cannot read image: empty file content")

    image = cv2.imdecode(
        np.frombuffer(content, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise ValueError("Cannot read image from uploaded file content")
    return image


def resize_image_max_side(
    image: np.ndarray,
    max_side: int | None = None,
) -> np.ndarray:
    if not max_side or max_side <= 0:
        return image

    width, height = image_dimensions(image)
    largest_side = max(width, height)
    if largest_side <= max_side:
        return image

    scale = max_side / largest_side
    return cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def build_image_variants(image: np.ndarray) -> list[np.ndarray]:
    """Build original, contrast-enhanced, and binary image variants."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    denoised = cv2.fastNlMeansDenoising(clahe, h=8)
    adaptive = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )
    return [
        image,
        cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR),
    ]


def image_dimensions(image: np.ndarray) -> tuple[int, int]:
    if image.size == 0 or image.ndim not in {2, 3}:
        raise ValueError("Cannot preprocess an empty or malformed image")

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Cannot preprocess an image with invalid dimensions")
    return width, height


def resize_for_analysis(
    image: np.ndarray,
    max_side: int = ANALYSIS_MAX_SIDE,
) -> tuple[np.ndarray, float, float]:
    width, height = image_dimensions(image)
    largest_side = max(width, height)
    if largest_side <= max_side:
        return image, 1.0, 1.0

    scale = max_side / largest_side
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    resized_width, resized_height = image_dimensions(resized)
    return resized, width / resized_width, height / resized_height


def document_edge_map(image: np.ndarray) -> np.ndarray:
    gray = (
        image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    )
    normalized = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    ).apply(gray)
    blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
    median = float(np.median(blurred))
    lower = int(max(20, 0.66 * median))
    upper = int(min(240, max(lower + 30, 1.33 * median)))
    edges = cv2.Canny(blurred, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)


def valid_document_quad(
    points: np.ndarray,
    image_width: int,
    image_height: int,
) -> bool:
    try:
        polygon = order_quad_points(points)
    except ValueError:
        return False
    area = abs(float(cv2.contourArea(polygon)))
    area_ratio = area / float(image_width * image_height)
    if area_ratio < MIN_DOCUMENT_AREA_RATIO:
        return False

    if not cv2.isContourConvex(polygon.astype(np.int32)):
        return False

    ordered = order_quad_points(polygon)
    top_width = np.linalg.norm(ordered[1] - ordered[0])
    bottom_width = np.linalg.norm(ordered[2] - ordered[3])
    left_height = np.linalg.norm(ordered[3] - ordered[0])
    right_height = np.linalg.norm(ordered[2] - ordered[1])
    width = max(top_width, bottom_width)
    height = max(left_height, right_height)
    if width < MIN_DOCUMENT_SIDE or height < MIN_DOCUMENT_SIDE:
        return False

    aspect_ratio = width / height
    return (
        1 / MAX_DOCUMENT_ASPECT_RATIO
        <= aspect_ratio
        <= MAX_DOCUMENT_ASPECT_RATIO
    )


def find_document_quad(
    image: np.ndarray,
    *,
    required_region: np.ndarray | None = None,
    prefer_landscape: bool = False,
) -> np.ndarray | None:
    from app.core.images.document_detection import detect_document_quad

    analysis_image, scale_x, scale_y = resize_for_analysis(image)
    if required_region is None and not prefer_landscape:
        points = detect_document_quad(analysis_image)
    else:
        analysis_region = (
            np.asarray(required_region, dtype=np.float32)
            / np.array([scale_x, scale_y], np.float32)
            if required_region is not None
            else None
        )
        points = detect_document_quad(
            analysis_image,
            required_region=analysis_region,
            prefer_landscape=prefer_landscape,
        )
    if points is None:
        return None
    points[:, 0] *= scale_x
    points[:, 1] *= scale_y
    return points


def warp_document(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    top_left, top_right, bottom_right, bottom_left = order_quad_points(points)
    width = max(
        np.linalg.norm(top_right - top_left),
        np.linalg.norm(bottom_right - bottom_left),
    )
    height = max(
        np.linalg.norm(bottom_left - top_left),
        np.linalg.norm(bottom_right - top_right),
    )
    target_width = max(MIN_DOCUMENT_SIDE, round(float(width)))
    target_height = max(MIN_DOCUMENT_SIDE, round(float(height)))
    destination = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(
        order_quad_points(points),
        destination,
    )
    return cv2.warpPerspective(
        image,
        matrix,
        (target_width, target_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def estimate_skew_angle(
    image: np.ndarray,
    max_degrees: float = MAX_DESKEW_DEGREES,
) -> float:
    gray = (
        image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    )
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    height, width = gray.shape[:2]
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(30, min(width, height) // 12),
        minLineLength=max(30, width // 5),
        maxLineGap=max(10, width // 50),
    )
    if lines is None:
        return 0.0

    angles: list[float] = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        angle = degrees(atan2(float(y2 - y1), float(x2 - x1)))
        if -max_degrees <= angle <= max_degrees:
            angles.append(angle)

    if not angles:
        return 0.0
    return float(np.median(np.asarray(angles, dtype=np.float32)))


def rotate_image_bound(image: np.ndarray, angle: float) -> np.ndarray:
    width, height = image_dimensions(image)
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    target_width = max(1, round((height * sine) + (width * cosine)))
    target_height = max(1, round((height * cosine) + (width * sine)))
    matrix[0, 2] += (target_width / 2) - center[0]
    matrix[1, 2] += (target_height / 2) - center[1]
    border_value: int | tuple[int, int, int]
    border_value = 255 if image.ndim == 2 else (255, 255, 255)
    return cv2.warpAffine(
        image,
        matrix,
        (target_width, target_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def adaptive_resize(
    image: np.ndarray,
    target_long_side: int = TARGET_LONG_SIDE,
    max_long_side: int = MAX_LONG_SIDE,
    max_upscale: float = MAX_UPSCALE,
) -> tuple[np.ndarray, float]:
    width, height = image_dimensions(image)
    largest_side = max(width, height)
    scale = 1.0
    if largest_side < target_long_side:
        scale = min(target_long_side / largest_side, max_upscale)
    elif largest_side > max_long_side:
        scale = max_long_side / largest_side

    if abs(scale - 1.0) < 0.01:
        return image, 1.0

    interpolation = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=interpolation,
    )
    return resized, float(scale)


def prepare_document_image(image: np.ndarray) -> PreparedImage:
    original_width, original_height = image_dimensions(image)
    document_quad = find_document_quad(image)
    perspective_corrected = document_quad is not None
    prepared = (
        warp_document(image, document_quad)
        if document_quad is not None
        else image.copy()
    )

    skew_angle = estimate_skew_angle(prepared)
    deskew_applied = abs(skew_angle) >= MIN_DESKEW_DEGREES
    if deskew_applied:
        prepared = rotate_image_bound(prepared, skew_angle)

    prepared, scale = adaptive_resize(prepared)
    output_width, output_height = image_dimensions(prepared)
    return PreparedImage(
        image=prepared,
        metadata={
            "original_width": original_width,
            "original_height": original_height,
            "output_width": output_width,
            "output_height": output_height,
            "document_detected": document_quad is not None,
            "perspective_corrected": perspective_corrected,
            "deskew_applied": deskew_applied,
            "deskew_angle_degrees": round(skew_angle, 2),
            "scale_factor": round(scale, 3),
        },
    )
