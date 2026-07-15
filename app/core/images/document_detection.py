from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import chain, combinations

import cv2
import numpy as np

from app.core.images.geometry import order_quad_points


@dataclass(frozen=True)
class DocumentCandidate:
    quad: np.ndarray
    score: float
    method: str
    side_support: tuple[float, ...]
    interior_contrast: float


@dataclass
class _Line:
    coefficients: np.ndarray
    length: float
    position: float
    endpoints: np.ndarray


def _cross2(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def _geometry(
    quad: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, float, float] | None:
    try:
        quad = order_quad_points(quad)
    except ValueError:
        return None
    if (
        (quad < 0).any()
        or (quad[:, 0] > width - 1).any()
        or (quad[:, 1] > height - 1).any()
    ):
        return None
    area = float(cv2.contourArea(quad)) / (width * height)
    if area < 0.20:
        return None
    vectors = np.roll(quad, -1, axis=0) - quad
    lengths = np.linalg.norm(vectors, axis=1)
    if min(lengths) < 96:
        return None
    aspect = max(lengths[0], lengths[2]) / max(lengths[1], lengths[3])
    if not 1 / 3 <= aspect <= 3:
        return None
    unit = vectors / lengths[:, None]
    cosines = np.sum(unit * np.roll(unit, -1, axis=0), axis=1)
    if np.max(np.abs(cosines)) > 0.75:
        return None
    touches = sum(
        (
            bool((quad[:, 0] <= 3).any()),
            bool((quad[:, 0] >= width - 4).any()),
            bool((quad[:, 1] <= 3).any()),
            bool((quad[:, 1] >= height - 4).any()),
        )
    )
    # Dense background texture can produce a contour around the entire image.
    if area > 0.90 and touches >= 3:
        return None
    rectangularity = float(1 - np.mean(np.abs(cosines)))
    return quad, area, rectangularity


def _masks(image: np.ndarray) -> Iterator[tuple[np.ndarray, str]]:
    gray = (
        image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    )
    small_kernel = np.ones((3, 3), np.uint8)
    for blur_size in (5, 11):
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        for lower, upper in ((30, 90), (60, 160), (100, 240)):
            edges = cv2.Canny(blurred, lower, upper)
            yield edges, "contour"
            yield (
                cv2.morphologyEx(edges, cv2.MORPH_CLOSE, small_kernel),
                "contour",
            )
    planes = [gray]
    if image.ndim == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        planes.extend((lab[:, :, 1], lab[:, :, 2]))
    for plane in planes:
        blurred = cv2.GaussianBlur(plane, (11, 11), 0)
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        for mask in (binary, 255 - binary):
            closed = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
            )
            yield (
                cv2.morphologyEx(closed, cv2.MORPH_OPEN, small_kernel),
                "region",
            )


def _contour_quads(image: np.ndarray) -> Iterator[tuple[np.ndarray, str]]:
    height, width = image.shape[:2]
    for max_side in (600, 1000):
        scale = min(1.0, max_side / max(width, height))
        work = cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
        work_height, work_width = work.shape[:2]
        factors = np.array(
            [width / work_width, height / work_height], np.float32
        )
        for mask, method in _masks(work):
            contours, _ = cv2.findContours(
                mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in sorted(contours, key=cv2.contourArea, reverse=True)[
                :35
            ]:
                area = cv2.contourArea(contour)
                if area < 0.15 * work_width * work_height:
                    continue
                hull = cv2.convexHull(contour)
                shapes = [contour]
                if area / max(cv2.contourArea(hull), 1) > 0.80:
                    shapes.append(hull)
                for shape in shapes:
                    perimeter = cv2.arcLength(shape, True)
                    for epsilon in (0.012, 0.022, 0.035):
                        approximation = cv2.approxPolyDP(
                            shape, epsilon * perimeter, True
                        )
                        if len(approximation) == 4:
                            yield (
                                approximation.reshape(4, 2).astype(np.float32)
                                * factors,
                                method,
                            )


def _long_lines(gray: np.ndarray) -> tuple[list[_Line], list[_Line]]:
    height, width = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    detected = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE).detect(
        blurred
    )[0]
    if detected is None:
        return [], []
    horizontal: list[_Line] = []
    vertical: list[_Line] = []
    for segment in sorted(
        detected.reshape(-1, 4),
        key=lambda s: -np.hypot(s[2] - s[0], s[3] - s[1]),
    ):
        p, q = segment[:2], segment[2:]
        vector = q - p
        length = float(np.linalg.norm(vector))
        if length < max(55, 0.10 * min(width, height)):
            continue
        normal = np.array([-vector[1], vector[0]], dtype=np.float32) / length
        is_horizontal = abs(vector[0]) >= abs(vector[1])
        index = 1 if is_horizontal else 0
        if normal[index] < 0:
            normal = -normal
        coefficients = np.append(normal, -np.dot(normal, p))
        extent = width if is_horizontal else height
        position = float(
            -(coefficients[2] + coefficients[1 - index] * extent / 2)
            / coefficients[index]
        )
        group = horizontal if is_horizontal else vertical
        duplicate = False
        for existing in group:
            if (
                abs(
                    float(_cross2(coefficients[:2], existing.coefficients[:2]))
                )
                < 0.025
                and abs(position - existing.position) < 6
            ):
                # Collinear fragments can be separated by a finger or a glare.
                endpoints = np.vstack((existing.endpoints, p, q))
                tangent = np.array(
                    [existing.coefficients[1], -existing.coefficients[0]]
                )
                projected = endpoints @ tangent
                existing.endpoints = endpoints[
                    [np.argmin(projected), np.argmax(projected)]
                ]
                existing.length = float(np.ptp(projected))
                duplicate = True
                break
        if not duplicate:
            group.append(
                _Line(coefficients, length, position, np.array([p, q]))
            )
    return (
        sorted(
            sorted(horizontal, key=lambda line: -line.length)[:18],
            key=lambda line: line.position,
        ),
        sorted(
            sorted(vertical, key=lambda line: -line.length)[:18],
            key=lambda line: line.position,
        ),
    )


def _line_quads(gray: np.ndarray) -> Iterator[tuple[np.ndarray, str]]:
    height, width = gray.shape
    horizontal, vertical = _long_lines(gray)
    if len(horizontal) < 2 or len(vertical) < 2:
        return

    def pairs(lines: list[_Line], extent: int) -> list[tuple[int, int]]:
        return [
            (i, j)
            for i, j in combinations(range(len(lines)), 2)
            if lines[j].position - lines[i].position > 0.18 * extent
            and abs(
                float(
                    _cross2(
                        lines[i].coefficients[:2], lines[j].coefficients[:2]
                    )
                )
            )
            < 0.32
        ]

    hp, vp = pairs(horizontal, height), pairs(vertical, width)
    if not hp or not vp:
        return
    h = np.array([line.coefficients for line in horizontal])[:, None, :]
    v = np.array([line.coefficients for line in vertical])[None, :, :]
    cross = np.cross(h, v)
    intersections = cross[:, :, :2] / np.where(
        abs(cross[:, :, 2:]) > 1e-6,
        cross[:, :, 2:],
        np.nan,
    )
    hi, hj = np.array(hp).T
    vi, vj = np.array(vp).T
    quads = np.stack(
        (
            intersections[hi[:, None], vi],
            intersections[hi[:, None], vj],
            intersections[hj[:, None], vj],
            intersections[hj[:, None], vi],
        ),
        axis=2,
    ).reshape(-1, 4, 2)
    # A visible edge may extrapolate a few pixels beyond the camera frame.
    # Limit that tolerance, then keep the output inside the available pixels.
    margin = 0.02 * min(width, height)
    valid = np.isfinite(quads).all(axis=(1, 2)) & (quads >= -margin).all(
        axis=(1, 2)
    )
    valid &= (quads[:, :, 0] <= width - 1 + margin).all(axis=1) & (
        quads[:, :, 1] <= height - 1 + margin
    ).all(axis=1)
    quads = quads[valid]
    if not len(quads):
        return
    quads[:, :, 0] = np.clip(quads[:, :, 0], 0, width - 1)
    quads[:, :, 1] = np.clip(quads[:, :, 1], 0, height - 1)
    edges = np.roll(quads, -1, axis=1) - quads
    lengths = np.linalg.norm(edges, axis=2)
    unit = edges / np.maximum(lengths[:, :, None], 1e-6)
    turns = _cross2(unit, np.roll(unit, -1, axis=1))
    cosines = np.sum(unit * np.roll(unit, -1, axis=1), axis=2)
    area = np.abs(_cross2(quads, np.roll(quads, -1, axis=1)).sum(axis=1)) / (
        2 * width * height
    )
    valid = (area >= 0.20) & (area <= 0.95) & (lengths.min(axis=1) >= 96)
    valid &= (turns > 0.65).all(axis=1)
    indices = np.flatnonzero(valid)
    if not len(indices):
        return
    # Geometry alone favors very rectangular combinations of unrelated lines.
    # Check actual image edges before reducing the candidate set.
    selected = quads[indices]
    selected_edges = edges[indices]
    selected_lengths = lengths[indices]
    normals = (
        np.stack((-selected_edges[:, :, 1], selected_edges[:, :, 0]), axis=2)
        / selected_lengths[:, :, None]
    )
    fractions = np.linspace(0.08, 0.92, 10, dtype=np.float32)
    offsets = np.array([-3, 0, 3], np.float32)
    samples = (
        selected[:, :, None, None, :]
        + selected_edges[:, :, None, None, :]
        * fractions[None, None, :, None, None]
    )
    samples = (
        samples
        + normals[:, :, None, None, :] * offsets[None, None, None, :, None]
    )
    smoothed = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3, scale=0.25)
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3, scale=0.25)
    maps = [
        samples[..., axis].reshape(len(selected), -1).astype(np.float32)
        for axis in (0, 1)
    ]
    dx = cv2.remap(gx, *maps, cv2.INTER_LINEAR).reshape(
        len(selected), 4, 10, 3
    )
    dy = cv2.remap(gy, *maps, cv2.INTER_LINEAR).reshape(
        len(selected), 4, 10, 3
    )
    strength = np.abs(
        dx * normals[:, :, 0, None, None] + dy * normals[:, :, 1, None, None]
    )
    aligned = strength >= 0.8 * np.hypot(dx, dy)
    supported = ((strength >= 9) & aligned).any(axis=3).mean(axis=2)
    proxy = 0.5 * supported.mean(axis=1) + 0.3 * supported.min(axis=1)
    proxy += (
        0.15 * (1 - np.abs(cosines[indices]).mean(axis=1))
        + 0.05 * area[indices]
    )
    for index in indices[np.argsort(proxy)[-180:]]:
        yield quads[index].astype(np.float32), "lines"


def rank_document_candidates(image: np.ndarray) -> list[DocumentCandidate]:
    """Return candidates in preference order.

    Scores are heuristic rather than confidence values.
    """
    gray = (
        image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    )
    height, width = gray.shape
    if min(height, width) < 96 or float(gray.std()) < 2:
        return []
    candidates = []
    seen = set()
    for quad, method in chain(_contour_quads(image), _line_quads(gray)):
        geometry = _geometry(quad, width, height)
        if geometry is None:
            continue
        quad, area, rectangularity = geometry
        key = tuple(np.round(quad.reshape(-1) / 5).astype(int))
        if key in seen:
            continue
        seen.add(key)
        candidates.append((quad, method, area, rectangularity))
    if not candidates:
        return []

    smoothed = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3, scale=0.25)
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3, scale=0.25)
    fractions = np.linspace(0.05, 0.95, 80, dtype=np.float32)
    offsets = np.arange(-4, 5, dtype=np.float32)
    ranked = []
    for quad, method, area, rectangularity in candidates:
        sides = np.roll(quad, -1, axis=0) - quad
        lengths = np.linalg.norm(sides, axis=1)
        normals = (
            np.stack((-sides[:, 1], sides[:, 0]), axis=1) / lengths[:, None]
        )
        samples = (
            quad[:, None, None, :]
            + sides[:, None, None, :] * fractions[None, :, None, None]
        )
        samples = (
            samples + normals[:, None, None, :] * offsets[None, None, :, None]
        )
        map_x = samples[:, :, :, 0].reshape(4, -1).astype(np.float32)
        map_y = samples[:, :, :, 1].reshape(4, -1).astype(np.float32)
        dx = cv2.remap(
            gx,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        ).reshape(4, 80, -1)
        dy = cv2.remap(
            gy,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        ).reshape(4, 80, -1)
        strength = np.abs(
            dx * normals[:, 0, None, None] + dy * normals[:, 1, None, None]
        )
        alignment = strength / np.maximum(np.hypot(dx, dy), 1e-6)
        strength = np.where(alignment >= 0.80, strength, 0).max(axis=2)
        support = (strength >= 9).mean(axis=1)
        if np.min(support) < 0.40 or np.mean(support) < 0.68:
            continue
        edge_strength = float(np.minimum(strength / 35, 1).mean())
        # A printed rule has an edge but usually no sustained color transition
        # across it. Sample beyond thin ink/borders, on both sides of
        # each edge.
        centers = (
            quad[:, None, :] + sides[:, None, :] * fractions[None, :, None]
        )
        inner = centers + normals[:, None, :] * 10
        outer = centers - normals[:, None, :] * 10
        inside = cv2.remap(
            smoothed,
            inner[:, :, 0],
            inner[:, :, 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        outside = cv2.remap(
            smoothed,
            outer[:, :, 0],
            outer[:, :, 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        contrast = float(np.minimum(np.abs(inside - outside) / 40, 1).mean())
        interior_contrast = float(
            np.clip((inside - outside) / 40, 0, 1).mean()
        )
        score = (
            0.30 * float(support.mean())
            + 0.12 * float(support.min())
            + 0.14 * edge_strength
            + 0.14 * rectangularity
            + 0.12 * contrast
            + 0.10 * area
            + 0.08 * interior_contrast
        )
        if score >= 0.66:
            ranked.append(
                DocumentCandidate(
                    quad,
                    score,
                    method,
                    tuple(float(x) for x in support),
                    interior_contrast,
                )
            )
    ranked.sort(key=lambda candidate: candidate.score, reverse=True)
    if ranked:
        best = ranked[0]
        best_area = cv2.contourArea(best.quad)
        containers = []
        for candidate in ranked:
            candidate_area = cv2.contourArea(candidate.quad)
            if not 1.02 * best_area <= candidate_area <= 1.25 * best_area:
                continue
            if (
                candidate.score < best.score - 0.05
                or min(candidate.side_support) < 0.65
            ):
                continue
            if candidate.interior_contrast < best.interior_contrast - 0.20:
                continue
            intersection, _ = cv2.intersectConvexConvex(
                best.quad, candidate.quad
            )
            if intersection / best_area >= 0.985:
                containers.append(candidate)
        if containers:
            # A well-supported enclosing border wins over a printed header
            # separator or an inner frame, preserving the complete document.
            enclosing = max(
                containers,
                key=lambda candidate: cv2.contourArea(candidate.quad),
            )
            ranked = [enclosing] + [
                candidate for candidate in ranked if candidate is not enclosing
            ]
    return ranked


def _region_polygon(region: np.ndarray) -> np.ndarray:
    points = np.asarray(region, dtype=np.float32)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) < 3
        or not np.isfinite(points).all()
    ):
        raise ValueError(
            "The required document region must be a finite polygon"
        )
    polygon = cv2.convexHull(points).reshape(-1, 2)
    if cv2.contourArea(polygon) <= 0:
        raise ValueError(
            "The required document region must have positive area"
        )
    return polygon


def _covers_region(quad: np.ndarray, region: np.ndarray) -> bool:
    intersection, _ = cv2.intersectConvexConvex(quad, region)
    return intersection / cv2.contourArea(region) >= 0.985


def _quad_aspect(quad: np.ndarray) -> float:
    lengths = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    return float(max(lengths[0], lengths[2]) / max(lengths[1], lengths[3]))


def _frame_assisted_page(
    image: np.ndarray, region: np.ndarray
) -> np.ndarray | None:
    """Use a visible fold when the other page boundaries lie outside the frame.

    This is restricted to upright portrait spreads with a wide MRZ at the
    bottom. Frame limits alone are never sufficient evidence for a crop.
    """
    gray = (
        image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    )
    height, width = gray.shape
    if (
        height <= width
        or np.ptp(region[:, 0]) < 0.80 * width
        or region[:, 1].max() < 0.85 * height
    ):
        return None
    horizontal, _ = _long_lines(gray)
    smoothed = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3, scale=0.25)
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3, scale=0.25)
    xs = np.linspace(0.03 * width, 0.97 * width, 160, dtype=np.float32)
    offsets = np.arange(-6, 7, dtype=np.float32)
    # Starting with the uppermost supported fold preserves the whole bio page
    # when additional printed rules below it have similar edge strength.
    for line in sorted(horizontal, key=lambda item: item.position):
        if line.length < 0.40 * width or abs(
            line.coefficients[0]
        ) > 0.06 * abs(line.coefficients[1]):
            continue
        if (
            line.endpoints[:, 0].min() > 0.60 * width
            or line.endpoints[:, 0].max() < 0.40 * width
        ):
            continue
        left_y, right_y = (
            -(
                line.coefficients[0] * np.array([0, width - 1])
                + line.coefficients[2]
            )
            / line.coefficients[1]
        )
        if (
            min(left_y, right_y) < 0
            or max(left_y, right_y) >= region[:, 1].min()
        ):
            continue
        quad = np.array(
            [
                [0, left_y],
                [width - 1, right_y],
                [width - 1, height - 1],
                [0, height - 1],
            ],
            dtype=np.float32,
        )
        if not 1.20 <= _quad_aspect(quad) <= 1.85 or not _covers_region(
            quad, region
        ):
            continue
        ys = (
            -(line.coefficients[0] * xs + line.coefficients[2])
            / line.coefficients[1]
        )
        map_x = np.repeat(xs[:, None], len(offsets), axis=1)
        map_y = ys[:, None] + offsets
        dx = cv2.remap(gx, map_x, map_y, cv2.INTER_LINEAR)
        dy = cv2.remap(gy, map_x, map_y, cv2.INTER_LINEAR)
        supported = (
            (np.abs(dy) >= 9) & (np.abs(dy) >= 0.8 * np.hypot(dx, dy))
        ).any(axis=1)
        if (
            supported.mean() < 0.80
            or min(part.mean() for part in np.array_split(supported, 4)) < 0.50
        ):
            continue
        # A thin printed rule has similar paper colors farther from its ink.
        # Require a sustained transition across the proposed binding/fold.
        before = cv2.remap(
            smoothed, xs[None, :], (ys - 20)[None, :], cv2.INTER_LINEAR
        )
        after = cv2.remap(
            smoothed, xs[None, :], (ys + 20)[None, :], cv2.INTER_LINEAR
        )
        if (np.abs(before - after) >= 8).mean() < 0.50:
            continue
        return quad
    return None


def detect_document_quad(
    image: np.ndarray,
    *,
    required_region: np.ndarray | None = None,
    prefer_landscape: bool = False,
) -> np.ndarray | None:
    """Select a page while preserving a known content region if needed."""
    candidates = rank_document_candidates(image)
    if required_region is None and not prefer_landscape:
        return candidates[0].quad.copy() if candidates else None
    region = (
        _region_polygon(required_region)
        if required_region is not None
        else None
    )
    if region is not None:
        candidates = [
            candidate
            for candidate in candidates
            if _covers_region(candidate.quad, region)
        ]
    if prefer_landscape:
        landscape = [
            candidate
            for candidate in candidates
            if 1.15 <= _quad_aspect(candidate.quad) <= 2.15
        ]
        if landscape:
            return landscape[0].quad.copy()
        if region is not None:
            page = _frame_assisted_page(image, region)
            if page is not None:
                return page
    return candidates[0].quad.copy() if candidates else None
