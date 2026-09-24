from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.core.config import settings
from app.core.images.validation import validate_image_dimensions

QUALITY_WEIGHTS = {
    "document_completeness": 0.25,
    "sharpness": 0.20,
    "glare": 0.15,
    "exposure": 0.10,
    "perspective": 0.10,
    "local_contrast": 0.10,
    "resolution": 0.05,
    "noise_and_blur": 0.05,
}

QR_QUALITY_WEIGHTS = {
    "sharpness": 0.30,
    "glare": 0.20,
    "exposure": 0.15,
    "local_contrast": 0.15,
    "resolution": 0.10,
    "noise_and_blur": 0.10,
}


@dataclass(frozen=True)
class ImageQualityAssessment:
    score: float
    grade: str
    width: int
    height: int
    document_detected: bool
    touches_frame: bool
    document_area_ratio: float
    border_margin_ratio: float
    sharpness_variance: float
    glare_ratio: float
    brightness: float
    underexposure_ratio: float
    overexposure_ratio: float
    illumination_spread: float
    local_contrast: float
    noise_level: float
    perspective_distortion: float
    component_scores: dict[str, float]


@dataclass(frozen=True)
class _DocumentGeometry:
    points: np.ndarray | None
    detected: bool
    touches_frame: bool
    area_ratio: float
    border_margin_ratio: float
    completeness_score: float
    perspective_score: float
    perspective_distortion: float


@dataclass(frozen=True)
class _ReadabilityMetrics:
    sharpness_variance: float
    glare_ratio: float
    brightness: float
    underexposure_ratio: float
    overexposure_ratio: float
    illumination_spread: float
    local_contrast: float
    noise_level: float
    component_scores: dict[str, float]


def _clip(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _scaled_score(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        return 0.0
    return _clip((value - minimum) / (maximum - minimum))


def _load_analysis_image(
    image_source: str | Path | BinaryIO,
) -> tuple[np.ndarray, int, int]:
    stream = image_source if hasattr(image_source, "read") else None
    source_value = stream if stream is not None else Path(image_source)
    if stream is not None:
        stream.seek(0)

    try:
        with Image.open(source_value) as source:
            validate_image_dimensions(*source.size)
            width, height = source.size
            orientation = int(source.getexif().get(274, 1))
            if orientation in {5, 6, 7, 8}:
                width, height = height, width

            max_side = max(1, settings.ocr_quality_max_side)
            source.draft("RGB", (max_side, max_side))
            normalized = ImageOps.exif_transpose(source).convert("RGB")
            normalized.thumbnail(
                (max_side, max_side),
                Image.Resampling.LANCZOS,
            )
            rgb = np.array(normalized, dtype=np.uint8, copy=True)
    finally:
        if stream is not None:
            stream.seek(0)

    if rgb.size == 0:
        raise ValueError("Cannot analyze an empty image")
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), width, height


def _order_quad_points(points: np.ndarray) -> np.ndarray:
    polygon = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = polygon.sum(axis=1)
    differences = np.diff(polygon, axis=1).reshape(-1)
    ordered[0] = polygon[np.argmin(sums)]
    ordered[1] = polygon[np.argmin(differences)]
    ordered[2] = polygon[np.argmax(sums)]
    ordered[3] = polygon[np.argmax(differences)]
    return ordered


def _find_document_quad(image: np.ndarray) -> np.ndarray | None:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    best_quad: np.ndarray | None = None
    best_area = 0.0
    image_area = float(width * height)
    minimum_side = max(40.0, min(width, height) * 0.15)

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        for epsilon_ratio in (0.015, 0.025, 0.04):
            approximation = cv2.approxPolyDP(
                contour,
                epsilon_ratio * perimeter,
                True,
            )
            if len(approximation) != 4:
                continue

            points = approximation.reshape(4, 2).astype(np.float32)
            if not cv2.isContourConvex(points.astype(np.int32)):
                continue

            area = abs(float(cv2.contourArea(points)))
            if area / image_area < 0.12 or area <= best_area:
                continue

            ordered = _order_quad_points(points)
            sides = [
                float(
                    np.linalg.norm(ordered[(index + 1) % 4] - ordered[index])
                )
                for index in range(4)
            ]
            if min(sides) < minimum_side:
                continue

            long_side = max(sides)
            short_side = max(1.0, min(sides))
            if long_side / short_side > 3.2:
                continue

            best_quad = ordered
            best_area = area
            break

    return best_quad


def _perspective_distortion(points: np.ndarray) -> float:
    polygon = _order_quad_points(points)
    sides = [
        float(np.linalg.norm(polygon[(index + 1) % 4] - polygon[index]))
        for index in range(4)
    ]
    opposite_imbalance = (
        abs(sides[0] - sides[2]) / max(sides[0], sides[2], 1.0)
        + abs(sides[1] - sides[3]) / max(sides[1], sides[3], 1.0)
    ) / 2

    corner_cosines: list[float] = []
    for index in range(4):
        previous = polygon[(index - 1) % 4] - polygon[index]
        following = polygon[(index + 1) % 4] - polygon[index]
        denominator = float(
            np.linalg.norm(previous) * np.linalg.norm(following)
        )
        if denominator > 0:
            corner_cosines.append(
                abs(float(np.dot(previous, following)) / denominator)
            )

    angle_distortion = (
        float(np.mean(corner_cosines)) if corner_cosines else 1.0
    )
    return _clip((opposite_imbalance * 0.55) + (angle_distortion * 0.45))


def _document_geometry(image: np.ndarray) -> _DocumentGeometry:
    height, width = image.shape[:2]
    points = _find_document_quad(image)
    if points is None:
        return _DocumentGeometry(
            points=None,
            detected=False,
            touches_frame=False,
            area_ratio=0.0,
            border_margin_ratio=0.0,
            completeness_score=0.10,
            perspective_score=0.40,
            perspective_distortion=0.60,
        )

    area_ratio = abs(float(cv2.contourArea(points))) / float(width * height)
    margins = np.concatenate(
        (
            points[:, 0] / max(1, width),
            (width - 1 - points[:, 0]) / max(1, width),
            points[:, 1] / max(1, height),
            (height - 1 - points[:, 1]) / max(1, height),
        )
    )
    border_margin_ratio = max(0.0, float(np.min(margins)))
    touches_frame = border_margin_ratio <= 0.012
    area_score = _scaled_score(area_ratio, 0.12, 0.40)
    margin_score = _scaled_score(border_margin_ratio, 0.005, 0.03)
    completeness_score = 0.65 + (area_score * 0.20) + (margin_score * 0.15)
    if touches_frame:
        completeness_score *= 0.55

    distortion = _perspective_distortion(points)
    return _DocumentGeometry(
        points=points,
        detected=True,
        touches_frame=touches_frame,
        area_ratio=area_ratio,
        border_margin_ratio=border_margin_ratio,
        completeness_score=_clip(completeness_score),
        perspective_score=_clip(1.0 - (distortion / 0.55)),
        perspective_distortion=distortion,
    )


def _analysis_region(
    gray: np.ndarray, geometry: _DocumentGeometry
) -> np.ndarray:
    if geometry.points is None:
        return gray

    x, y, width, height = cv2.boundingRect(geometry.points.astype(np.int32))
    x = max(0, x)
    y = max(0, y)
    return gray[y : y + max(1, height), x : x + max(1, width)]


def _glare_ratio(image: np.ndarray) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    kernel = np.ones((11, 11), dtype=np.uint8)
    local_range = cv2.dilate(value, kernel) - cv2.erode(value, kernel)
    glare = (value >= 248) & (saturation <= 35) & (local_range >= 12)
    return float(np.mean(glare))


def _qr_readability(
    image: np.ndarray,
    original_width: int,
    original_height: int,
    geometry: _DocumentGeometry,
) -> _ReadabilityMetrics:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    region = _analysis_region(gray, geometry)

    sharpness_variance = float(cv2.Laplacian(region, cv2.CV_64F).var())
    sharpness_score = _scaled_score(sharpness_variance, 45.0, 450.0)

    brightness = float(np.mean(region))
    underexposure_ratio = float(np.mean(region <= 20))
    overexposure_ratio = float(np.mean(region >= 245))
    brightness_score = 1.0 - _clip(abs(brightness - 155.0) / 125.0)
    clipping_score = 1.0 - _clip(
        (underexposure_ratio + overexposure_ratio) / 0.35
    )

    sigma = max(5.0, min(region.shape[:2]) / 30.0)
    illumination = cv2.GaussianBlur(region, (0, 0), sigmaX=sigma)
    illumination_spread = float(
        np.percentile(illumination, 90) - np.percentile(illumination, 10)
    )
    illumination_score = 1.0 - _scaled_score(
        illumination_spread,
        35.0,
        150.0,
    )
    exposure_score = _clip(
        (brightness_score * 0.45)
        + (clipping_score * 0.35)
        + (illumination_score * 0.20)
    )

    region_float = region.astype(np.float32)
    local_mean = cv2.GaussianBlur(region_float, (0, 0), sigmaX=3.0)
    local_square_mean = cv2.GaussianBlur(
        region_float * region_float,
        (0, 0),
        sigmaX=3.0,
    )
    local_variance = np.maximum(
        local_square_mean - (local_mean * local_mean),
        0.0,
    )
    local_contrast = float(np.mean(np.sqrt(local_variance)))
    contrast_score = _scaled_score(local_contrast, 5.0, 32.0)

    median = cv2.medianBlur(region, 3)
    noise_level = float(
        np.mean(np.abs(region.astype(np.float32) - median.astype(np.float32)))
    )
    noise_score = 1.0 - _scaled_score(noise_level, 2.0, 16.0)
    noise_and_blur_score = _clip(
        (noise_score * 0.60) + (sharpness_score * 0.40)
    )

    glare_ratio = _glare_ratio(image)
    glare_score = 1.0 - _scaled_score(glare_ratio, 0.005, 0.08)
    resolution_score = _scaled_score(
        float(min(original_width, original_height)),
        400.0,
        1_200.0,
    )

    component_scores = {
        "document_completeness": geometry.completeness_score,
        "sharpness": sharpness_score,
        "glare": glare_score,
        "exposure": exposure_score,
        "perspective": geometry.perspective_score,
        "local_contrast": contrast_score,
        "resolution": resolution_score,
        "noise_and_blur": noise_and_blur_score,
    }
    return _ReadabilityMetrics(
        sharpness_variance=sharpness_variance,
        glare_ratio=glare_ratio,
        brightness=brightness,
        underexposure_ratio=underexposure_ratio,
        overexposure_ratio=overexposure_ratio,
        illumination_spread=illumination_spread,
        local_contrast=local_contrast,
        noise_level=noise_level,
        component_scores=component_scores,
    )


def _document_mask(
    gray: np.ndarray, geometry: _DocumentGeometry
) -> np.ndarray:
    """Exclude the desk and document outline from readability measurements."""
    mask = np.zeros(gray.shape, dtype=np.uint8)
    if geometry.points is None:
        mask[:] = 255
    else:
        cv2.fillConvexPoly(mask, geometry.points.astype(np.int32), 255)
    inset = max(2, round(min(gray.shape) * 0.015))
    eroded = cv2.erode(
        mask,
        np.ones((inset * 2 + 1, inset * 2 + 1), dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return eroded > 0 if np.any(eroded) else mask > 0


def _document_detail(
    gray: np.ndarray, mask: np.ndarray
) -> tuple[float, float, float]:
    """Measure text detail after removing the estimated sensor-noise energy."""
    gray_float = gray.astype(np.float32)
    smooth = cv2.GaussianBlur(gray_float, (5, 5), 0.8)
    gradient_x = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3) / 8
    gradient_y = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3) / 8
    gradient = cv2.magnitude(gradient_x, gradient_y)
    flat = mask & (gradient <= np.percentile(gradient[mask], 50))
    # MAD on low-gradient pixels avoids counting printed strokes as noise.
    high_pass = cv2.filter2D(
        gray_float,
        -1,
        np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32),
    )
    noise = float(np.median(np.abs(high_pass[flat]))) / (6 * 0.6745)

    laplacian = cv2.Laplacian(smooth, cv2.CV_32F)
    # The squared filter norm is the response variance to unit white noise.
    impulse = np.zeros((9, 9), dtype=np.float32)
    impulse[4, 4] = 1
    response = cv2.Laplacian(
        cv2.GaussianBlur(impulse, (5, 5), 0.8), cv2.CV_32F
    )
    noise_gain = float(np.sum(response * response))
    sharpness = max(
        0.0, float(np.var(laplacian[mask])) - noise * noise * noise_gain
    )

    local_mean = cv2.GaussianBlur(gray_float, (0, 0), 3.0)
    local_variance = cv2.GaussianBlur(gray_float**2, (0, 0), 3.0)
    local_variance -= local_mean**2
    local_variance -= noise * noise
    contrast = float(np.mean(np.sqrt(np.maximum(local_variance[mask], 0))))
    return sharpness, noise, contrast


def _document_glare(image: np.ndarray, mask: np.ndarray) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    # A uniformly white sheet is expected. Glare is a localized clipped patch
    # brighter than the surrounding paper, including its textureless center.
    paper_brightness = float(np.percentile(value[mask], 60))
    if paper_brightness >= 245:
        return 0.0
    clipped = (mask & (value >= 250) & (hsv[:, :, 1] <= 35)).astype(np.uint8)
    clipped = cv2.morphologyEx(
        clipped, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8)
    )
    return float(np.count_nonzero(clipped) / np.count_nonzero(mask))


def _document_resolution(
    image: np.ndarray,
    geometry: _DocumentGeometry,
    original_width: int,
    original_height: int,
) -> float:
    if geometry.points is None:
        return float(min(original_width, original_height))
    points = geometry.points * np.array(
        [original_width / image.shape[1], original_height / image.shape[0]]
    )
    sides = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    return float(min((sides[0] + sides[2]) / 2, (sides[1] + sides[3]) / 2))


def _document_readability(
    image: np.ndarray,
    original_width: int,
    original_height: int,
    geometry: _DocumentGeometry,
) -> _ReadabilityMetrics:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = _document_mask(gray, geometry)
    pixels = gray[mask]
    sharpness, noise, contrast = _document_detail(gray, mask)
    resolution_score = _scaled_score(
        _document_resolution(image, geometry, original_width, original_height),
        200.0,
        1_200.0,
    )
    # Smooth response curves still distinguish already-sharp burst frames.
    noise_score = 1.0 / (1.0 + (noise / 10.0) ** 2)
    sharpness_score = sharpness / (sharpness + 35.0) * noise_score
    # Downsampling narrows strokes and can increase pixel-level sharpness.
    # A sharp thumbnail still contains fewer readable document samples.
    sharpness_score *= 0.5 + 0.5 * resolution_score
    contrast_score = contrast / (contrast + 8.0) * noise_score**0.5

    brightness = float(np.mean(pixels))
    underexposure_ratio = float(np.mean(pixels <= 20))
    overexposure_ratio = float(np.mean(pixels >= 245))
    # Paper can legitimately be white and text black; neither is itself proof
    # of clipping damage. Lost text is measured by focus and local contrast.
    paper_brightness = float(np.percentile(pixels, 90))
    brightness_score = _scaled_score(paper_brightness, 45.0, 180.0)
    illumination = cv2.GaussianBlur(
        gray, (0, 0), sigmaX=max(5.0, min(gray.shape) / 30.0)
    )
    illumination_spread = float(
        np.percentile(illumination[mask], 90)
        - np.percentile(illumination[mask], 10)
    )
    illumination_score = 1.0 - _scaled_score(illumination_spread, 35.0, 150.0)
    glare = _document_glare(image, mask)
    component_scores = {
        "document_completeness": geometry.completeness_score,
        "sharpness": sharpness_score,
        "glare": 1.0 - _scaled_score(glare, 0.005, 0.20),
        "exposure": brightness_score * 0.75 + illumination_score * 0.25,
        "perspective": geometry.perspective_score,
        "local_contrast": contrast_score,
        "resolution": resolution_score,
        "noise_and_blur": noise_score * 0.6 + sharpness_score * 0.4,
    }
    return _ReadabilityMetrics(
        sharpness_variance=sharpness,
        glare_ratio=glare,
        brightness=brightness,
        underexposure_ratio=underexposure_ratio,
        overexposure_ratio=overexposure_ratio,
        illumination_spread=illumination_spread,
        local_contrast=contrast,
        noise_level=noise,
        component_scores=component_scores,
    )


def _assess_image_quality(
    image_source: str | Path | BinaryIO, *, qr: bool
) -> ImageQualityAssessment:
    image, original_width, original_height = _load_analysis_image(image_source)
    geometry = _document_geometry(image)
    measure = _qr_readability if qr else _document_readability
    metrics = measure(image, original_width, original_height, geometry)
    score = sum(
        metrics.component_scores[name] * weight
        for name, weight in QUALITY_WEIGHTS.items()
    )
    grade = "good" if score >= 0.75 else "medium" if score >= 0.50 else "poor"
    return ImageQualityAssessment(
        score=round(score, 4),
        grade=grade,
        width=original_width,
        height=original_height,
        document_detected=geometry.detected,
        touches_frame=geometry.touches_frame,
        document_area_ratio=round(geometry.area_ratio, 4),
        border_margin_ratio=round(geometry.border_margin_ratio, 4),
        sharpness_variance=round(metrics.sharpness_variance, 2),
        glare_ratio=round(metrics.glare_ratio, 4),
        brightness=round(metrics.brightness, 2),
        underexposure_ratio=round(metrics.underexposure_ratio, 4),
        overexposure_ratio=round(metrics.overexposure_ratio, 4),
        illumination_spread=round(metrics.illumination_spread, 2),
        local_contrast=round(metrics.local_contrast, 2),
        noise_level=round(metrics.noise_level, 2),
        perspective_distortion=round(geometry.perspective_distortion, 4),
        component_scores={
            name: round(value, 4)
            for name, value in metrics.component_scores.items()
        },
    )


def assess_image_quality(
    image_source: str | Path | BinaryIO,
) -> ImageQualityAssessment:
    """Rank document frames using bounded, CPU-only readability heuristics."""
    return _assess_image_quality(image_source, qr=False)


def assess_qr_image_quality(
    image_source: str | Path | BinaryIO,
) -> ImageQualityAssessment:
    """Preserve QR measurements independently of document-specific scoring."""
    return _assess_image_quality(image_source, qr=True)


def qr_quality_score(assessment: ImageQualityAssessment) -> float:
    score = sum(
        assessment.component_scores[name] * weight
        for name, weight in QR_QUALITY_WEIGHTS.items()
    )
    return round(score, 4)
