"""Locate a plausible MRZ from text geometry, without OCR or a fallback band.

The input is an upright, rectified document in BGR or grayscale. Geometry
cannot prove that text is an MRZ: OCR and check digits remain a separate step.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import ceil, floor

import cv2
import numpy as np


@dataclass(frozen=True)
class _Line:
    left: int
    top: int
    right: int
    bottom: int
    character_height: float
    pitch: float
    slope: float
    center_y: float
    characters: int

    @property
    def width(self) -> int:
        return self.right - self.left


def _character_line(
    binary: np.ndarray, bbox: tuple[int, int, int, int]
) -> _Line | None:
    x, y, width, height = bbox
    patch = binary[y : y + height, x : x + width]
    _, _, stats, centroids = cv2.connectedComponentsWithStats(
        patch, connectivity=8
    )
    stats, centroids = stats[1:], centroids[1:]
    if not len(stats):
        return None
    all_stats, all_centroids = stats, centroids
    # Long rules/borders and small texture speckles are not characters.
    valid = (
        (stats[:, cv2.CC_STAT_HEIGHT] >= max(4, height * 0.25))
        & (stats[:, cv2.CC_STAT_WIDTH] >= 2)
        & (stats[:, cv2.CC_STAT_WIDTH] <= height * 1.25)
        & (stats[:, cv2.CC_STAT_AREA] >= 6)
    )
    stats, centroids = stats[valid], centroids[valid]
    if len(stats) < 20:
        return None
    median_height = float(np.median(stats[:, cv2.CC_STAT_HEIGHT]))
    valid = (stats[:, cv2.CC_STAT_HEIGHT] >= median_height * 0.55) & (
        stats[:, cv2.CC_STAT_HEIGHT] <= median_height * 1.7
    )
    stats, centroids = stats[valid], centroids[valid]
    if not 20 <= len(stats) <= 65:
        return None
    order = np.argsort(centroids[:, 0])
    stats, centroids = stats[order], centroids[order]
    spacing = np.diff(centroids[:, 0])
    pitch = float(np.median(spacing))
    if pitch <= 2 or np.percentile(spacing, 90) > pitch * 1.8:
        return None
    if np.median(np.abs(spacing - pitch)) > pitch * 0.28:
        return None
    if spacing.max() > pitch * 3.2:
        return None
    slope, intercept = np.polyfit(centroids[:, 0], centroids[:, 1], 1)
    residual = np.abs(centroids[:, 1] - (slope * centroids[:, 0] + intercept))
    if abs(slope) > 0.14 or np.percentile(residual, 80) > median_height * 0.45:
        return None
    # Keep touching edge glyphs after the independent characters have proved
    # the row. Blur can join two letters into a component too wide to count as
    # one character; discarding that component from the final envelope clips
    # otherwise valid MRZ text. Only recover short adjacent components on the
    # fitted baseline, not rules or unrelated texture connected to the row.
    row_left = int(stats[:, cv2.CC_STAT_LEFT].min())
    row_right = int(
        (stats[:, cv2.CC_STAT_LEFT] + stats[:, cv2.CC_STAT_WIDTH]).max()
    )
    if row_right - row_left < width * 0.82:
        return None
    merged = (
        (all_stats[:, cv2.CC_STAT_WIDTH] > height * 1.25)
        & (all_stats[:, cv2.CC_STAT_WIDTH] <= pitch * 3)
        & (all_stats[:, cv2.CC_STAT_HEIGHT] >= median_height * 0.55)
        & (all_stats[:, cv2.CC_STAT_HEIGHT] <= median_height * 1.7)
        & (all_stats[:, cv2.CC_STAT_AREA] >= 6)
        & (all_stats[:, cv2.CC_STAT_LEFT] <= row_right + pitch * 1.5)
        & (
            all_stats[:, cv2.CC_STAT_LEFT] + all_stats[:, cv2.CC_STAT_WIDTH]
            >= row_left - pitch * 1.5
        )
        & (
            np.abs(
                all_centroids[:, 1] - (slope * all_centroids[:, 0] + intercept)
            )
            <= median_height * 0.45
        )
    )
    bounds = np.concatenate((stats, all_stats[merged]))
    left = x + int(bounds[:, cv2.CC_STAT_LEFT].min())
    right = x + int(
        (bounds[:, cv2.CC_STAT_LEFT] + bounds[:, cv2.CC_STAT_WIDTH]).max()
    )
    top = y + int(bounds[:, cv2.CC_STAT_TOP].min())
    bottom = y + int(
        (bounds[:, cv2.CC_STAT_TOP] + bounds[:, cv2.CC_STAT_HEIGHT]).max()
    )
    return _Line(
        left,
        top,
        right,
        bottom,
        median_height,
        pitch,
        float(slope),
        float(y + slope * (width / 2) + intercept),
        len(stats),
    )


def _line_candidates(
    gray: np.ndarray, threshold_scale: float = 1.0
) -> list[_Line]:
    height, width = gray.shape
    roi_top = int(height * 0.45)
    roi = gray[roi_top:]
    blackhat = cv2.morphologyEx(
        roi,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(15, round(width * 0.025)), max(7, round(height * 0.025))),
        ),
    )
    if int(blackhat.max()) < 12:
        return []
    threshold, binary = cv2.threshold(
        blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    if threshold_scale != 1.0:
        _, binary = cv2.threshold(
            blackhat,
            threshold * threshold_scale,
            255,
            cv2.THRESH_BINARY,
        )
    lines: list[_Line] = []
    # Short vertical kernels connect glyphs along a row, not adjacent MRZ rows.
    for relative_width in (0.014, 0.022):
        connected = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (max(9, round(width * relative_width)), 3),
            ),
        )
        contours, _ = cv2.findContours(
            connected,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if (
                box_width < width * 0.55
                or box_height < 5
                or box_height > height * 0.13
                or box_width / box_height < 8
            ):
                continue
            line = _character_line(binary, (x, y, box_width, box_height))
            if line is None:
                continue
            line = _Line(
                line.left,
                line.top + roi_top,
                line.right,
                line.bottom + roi_top,
                line.character_height,
                line.pitch,
                line.slope,
                line.center_y + roi_top,
                line.characters,
            )
            if not any(
                abs(line.center_y - old.center_y) < line.character_height * 0.5
                and abs(line.left - old.left) < width * 0.05
                for old in lines
            ):
                lines.append(line)
    return sorted(lines, key=lambda line: line.center_y)


def _compatible(first: _Line, second: _Line) -> bool:
    height = (first.character_height + second.character_height) / 2
    gap = second.center_y - first.center_y
    overlap = max(
        0, min(first.right, second.right) - max(first.left, second.left)
    )
    return (
        height * 1.25 <= gap <= height * 3.5
        and overlap / max(first.width, second.width) >= 0.78
        and (
            min(first.character_height, second.character_height)
            / max(first.character_height, second.character_height)
            >= 0.65
        )
        and min(first.pitch, second.pitch) / max(first.pitch, second.pitch)
        >= 0.72
        and abs(first.slope - second.slope) <= 0.045
        and min(first.characters, second.characters)
        / max(first.characters, second.characters)
        >= 0.70
    )


def find_mrz_bbox(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return a padded box for 2–3 MRZ-like rows; x2/y2 are exclusive.

    No suitable rows means None, including blank images and a lone text row.
    No OCR models are loaded and the input array is never modified.
    """
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim not in (2, 3)
        or image.size == 0
        or (image.ndim == 3 and image.shape[2] != 3)
    ):
        return None
    original_height, original_width = image.shape[:2]
    if original_height < 80 or original_width < 160:
        return None
    scale = min(1.0, 1200 / max(original_height, original_width))
    analysis = (
        image
        if scale == 1
        else cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    )
    gray = (
        analysis
        if analysis.ndim == 2
        else cv2.cvtColor(analysis, cv2.COLOR_BGR2GRAY)
    )
    height, width = gray.shape
    groups = []
    # A stronger threshold can separate touching glyphs in small/blurred scans.
    # It must still produce the same independent character and row evidence.
    for threshold_scale in (1.0, 1.8):
        lines = _line_candidates(gray, threshold_scale)
        groups = [
            group
            for count in (2, 3)
            for group in combinations(lines, count)
            if all(
                _compatible(first, second)
                for first, second in zip(group, group[1:])
            )
        ]
        if groups:
            break
    if not groups:
        return None
    group = max(
        groups,
        key=lambda rows: (
            sum(line.width for line in rows) / (len(rows) * width)
            + 0.25 * rows[-1].center_y / height
            + 0.12 * (len(rows) - 2)
        ),
    )
    margin_x = max(5, round(width * 0.015))
    margin_y = max(
        4,
        round(np.median([line.character_height for line in group]) * 0.45),
    )
    return (
        max(0, floor((min(line.left for line in group) - margin_x) / scale)),
        max(0, floor((min(line.top for line in group) - margin_y) / scale)),
        min(
            original_width,
            ceil((max(line.right for line in group) + margin_x) / scale),
        ),
        min(
            original_height,
            ceil((max(line.bottom for line in group) + margin_y) / scale),
        ),
    )
