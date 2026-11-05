"""Crop the printed area of a light paper scan without running OCR."""

from __future__ import annotations

import math

import cv2
import numpy as np

_ANALYSIS_SIZE = 1600


def _paper_mask(gray: np.ndarray) -> np.ndarray:
    # Fill holes so text and table cells remain inside the paper mask.
    threshold = max(100.0, float(np.percentile(gray, 90)) * 0.75)
    _, paper = cv2.threshold(
        cv2.GaussianBlur(gray, (9, 9), 0),
        threshold,
        255,
        cv2.THRESH_BINARY,
    )
    contours, _ = cv2.findContours(
        paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    paper[:] = 0
    if contours:
        cv2.drawContours(
            paper,
            [max(contours, key=cv2.contourArea)],
            -1,
            255,
            cv2.FILLED,
        )
    # Exclude scanner edges and their shadows from content detection.
    return cv2.erode(
        paper,
        np.ones((11, 11), np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _content_mask(gray: np.ndarray) -> np.ndarray:
    # Local contrast also retains faint text on an unevenly lit scan.
    mask = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (3, 3), 0),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        12,
    )
    mask = cv2.bitwise_and(mask, _paper_mask(gray))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    height, width = gray.shape
    keep = np.zeros(count, np.uint8)
    for index, (x, y, w, h, area) in enumerate(stats[1:], 1):
        if x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2:
            continue
        if area >= 8 and ((h >= 4 and w >= 2) or w > width * 0.15):
            keep[index] = 255

    # Join adjacent characters into words; isolated dust should not expand
    # the crop. Keep every surviving block, including detached footers.
    grouped = cv2.morphologyEx(
        keep[labels], cv2.MORPH_CLOSE, np.ones((3, 9), np.uint8)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, 8)
    keep = np.zeros(count, np.uint8)
    for index, (_, _, w, h, area) in enumerate(stats[1:], 1):
        if area >= 45 and w >= 12 and h >= 5:
            keep[index] = 255
    return keep[labels]


def _skew_angle(mask: np.ndarray) -> float:
    lines = cv2.HoughLinesP(
        mask,
        1,
        np.pi / 1800,
        threshold=100,
        minLineLength=mask.shape[1] * 0.3,
        maxLineGap=20,
    )
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(angle) <= 10:
            angles.append(angle)
    # Several long table rules must agree before rotating the scan.
    if len(angles) < 3:
        return 0.0
    median = float(np.median(angles))
    agreeing = [angle for angle in angles if abs(angle - median) <= 0.5]
    if len(agreeing) < max(3, len(angles) * 0.6):
        return 0.0
    angle = float(np.median(agreeing))
    return angle if abs(angle) >= 0.05 else 0.0


def crop_printed_content(image: np.ndarray) -> np.ndarray:
    """Trim margins and deskew a scan; preserve original pixel resolution.

    Intended for light paper scans with dark text and tables. This finds
    printed regions, not the document type or individual semantic fields.
    Raises ValueError if no usable printed region is found.
    """
    height, width = image.shape[:2]
    scale = min(1.0, _ANALYSIS_SIZE / max(height, width))
    small = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    mask = _content_mask(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    points = cv2.findNonZero(mask)
    if points is None or len(points) < 100:
        raise ValueError("No printed content found")

    angle = _skew_angle(mask)
    rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    # Transform detected content, not a new mask of the rotated image:
    # otherwise the added white corners can look like new paper edges.
    coordinates = points.astype(np.float32)
    coordinates[:, :, 0] *= width / small.shape[1]
    coordinates[:, :, 1] *= height / small.shape[0]
    aligned = cv2.transform(coordinates, rotation).reshape(-1, 2)
    padding = max(6, round(min(width, height) * 0.01))
    left, top = np.floor(aligned.min(axis=0)).astype(int) - padding
    right, bottom = np.ceil(aligned.max(axis=0)).astype(int) + padding + 1
    # Render the crop directly, avoiding a second full-resolution canvas.
    rotation[:, 2] -= (left, top)
    return cv2.warpAffine(
        image,
        rotation,
        (int(right - left), int(bottom - top)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
