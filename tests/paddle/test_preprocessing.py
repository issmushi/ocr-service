from __future__ import annotations

import unittest
from itertools import permutations

import cv2
import numpy as np

from app.core.images.preprocessing import (
    adaptive_resize,
    estimate_skew_angle,
    find_document_quad,
    order_quad_points,
    prepare_document_image,
    rotate_image_bound,
    warp_document,
)


def _textured_background() -> np.ndarray:
    """Make deterministic fabric-like clutter without a document edge."""
    height, width = 720, 960
    yy, xx = np.mgrid[:height, :width]
    noise = np.random.default_rng(20260915).normal(0, 2.0, (height, width))
    gray = 68 + 13 * xx / width + 9 * yy / height
    gray += 5 * np.sin((xx + 0.7 * yy) / 13) + noise
    image = np.repeat(
        np.clip(gray, 0, 255).astype(np.uint8)[..., None], 3, axis=2
    )
    for x in range(-height, width, 42):
        cv2.line(image, (x, 0), (x + height, height - 1), (91, 88, 84), 1)
    for y in range(22, height, 57):
        cv2.line(image, (0, y), (width - 1, y + 95), (52, 55, 59), 2)
    return image


def _synthetic_document_scene(boundary: str) -> tuple[np.ndarray, np.ndarray]:
    """Make a shaded page on fabric with all four corners visible."""
    image = _textured_background()
    document = np.array(
        [[150, 90], [808, 136], [746, 635], [112, 570]],
        dtype=np.float32,
    )
    shadow = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(shadow, (document + [11, 14]).astype(np.int32), 255)
    shadow = cv2.GaussianBlur(shadow, (31, 31), 0).astype(np.float32) / 255
    image = (image.astype(np.float32) * (1 - 0.35 * shadow[..., None])).astype(
        np.uint8
    )

    page_height, page_width = 500, 650
    yy, xx = np.mgrid[:page_height, :page_width]
    shade = (228 - 15 * xx / page_width + 6 * np.sin(yy / 70)).astype(np.uint8)
    page = np.repeat(shade[..., None], 3, axis=2)
    # Printed bars and a diagram add internal edges without personal data.
    cv2.rectangle(page, (35, 40), (205, 175), (166, 163, 157), -1)
    cv2.circle(page, (120, 105), 37, (112, 116, 120), 3)
    for row, y in enumerate(range(45, 450, 27)):
        left = 240 if row < 6 else 40
        right = page_width - 38 - (row % 4) * 31
        cv2.line(page, (left, y), (right, y), (113, 117, 122), 3)
    source = np.array(
        [
            [0, 0],
            [page_width - 1, 0],
            [page_width - 1, page_height - 1],
            [0, page_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, document)
    page_on_background = cv2.warpPerspective(
        page, matrix, (image.shape[1], image.shape[0])
    )
    mask = cv2.warpPerspective(
        np.full((page_height, page_width), 255, dtype=np.uint8),
        matrix,
        (image.shape[1], image.shape[0]),
    )
    image[mask > 127] = page_on_background[mask > 127]

    if boundary == "occluded":
        # An opaque object covers a short part of the right edge, not a corner.
        cv2.ellipse(image, (777, 385), (28, 52), 7, 0, 360, (73, 107, 141), -1)
    elif boundary == "broken":
        # A local glare spot washes out the middle of the top boundary.
        yy, xx = np.mgrid[: image.shape[0], : image.shape[1]]
        glare = np.exp(-(((xx - 480) / 85) ** 4) - ((yy - 113) / 32) ** 4)
        image = np.clip(
            image.astype(np.float32) * (1 - glare[..., None])
            + 246 * glare[..., None],
            0,
            255,
        ).astype(np.uint8)
    else:
        raise ValueError(f"Unknown boundary case: {boundary}")

    return image, document


class PreprocessingTests(unittest.TestCase):
    def test_order_quad_points_returns_clockwise_corners(self) -> None:
        points = np.array(
            [
                [90, 90],
                [10, 10],
                [10, 90],
                [90, 10],
            ],
            dtype=np.float32,
        )

        ordered = order_quad_points(points)

        np.testing.assert_array_equal(
            ordered,
            np.array(
                [
                    [10, 10],
                    [90, 10],
                    [90, 90],
                    [10, 90],
                ],
                dtype=np.float32,
            ),
        )

    def test_order_quad_points_preserves_distinct_corners_for_all_permutations(
        self,
    ) -> None:
        cases = {
            "perspective_quad": [
                [39, 162],
                [582, 674],
                [1143, 528],
                [1045, 14],
            ],
            "diamond_with_equal_coordinate_sums": [
                [20, 100],
                [100, 20],
                [180, 100],
                [100, 180],
            ],
        }
        for name, coordinates in cases.items():
            points = np.asarray(coordinates, dtype=np.float32)
            canonical = order_quad_points(points)
            for indices in permutations(range(4)):
                with self.subTest(case=name, permutation=indices):
                    ordered = order_quad_points(points[list(indices)])

                    self.assertEqual(ordered.shape, (4, 2))
                    self.assertEqual(len(np.unique(ordered, axis=0)), 4)
                    self.assertEqual(
                        {tuple(point) for point in ordered},
                        {tuple(point) for point in points},
                    )
                    np.testing.assert_array_equal(ordered, canonical)
                    edges = np.roll(ordered, -1, axis=0) - ordered
                    following = np.roll(edges, -1, axis=0)
                    turns = (
                        edges[:, 0] * following[:, 1]
                        - edges[:, 1] * following[:, 0]
                    )
                    self.assertTrue(
                        np.all(turns > 0),
                        "Corners must form a convex clockwise cycle",
                    )

    def test_order_and_warp_reject_invalid_quadrilaterals(self) -> None:
        image = np.full((240, 320, 3), 200, dtype=np.uint8)
        cases = {
            "duplicate_corner": [[20, 20], [280, 20], [280, 200], [20, 20]],
            "collinear": [[20, 20], [80, 80], [140, 140], [200, 200]],
            "three_collinear_corners": [
                [20, 20],
                [140, 20],
                [280, 20],
                [20, 200],
            ],
            "nan": [[20, 20], [280, 20], [280, np.nan], [20, 200]],
            "positive_infinity": [
                [20, 20],
                [np.inf, 20],
                [280, 200],
                [20, 200],
            ],
            "negative_infinity": [
                [20, -np.inf],
                [280, 20],
                [280, 200],
                [20, 200],
            ],
        }
        for name, coordinates in cases.items():
            points = np.asarray(coordinates, dtype=np.float32)
            for operation in (
                order_quad_points,
                lambda quad: warp_document(image, quad),
            ):
                with self.subTest(case=name, operation=operation.__name__):
                    with self.assertRaises(ValueError):
                        operation(points)

    def test_find_document_quad_rejects_blank_and_texture_only_images(
        self,
    ) -> None:
        cases = {
            "blank_dark": np.zeros((720, 960, 3), dtype=np.uint8),
            "blank_light": np.full((720, 960, 3), 235, dtype=np.uint8),
            "texture_only": _textured_background(),
        }
        for name, image in cases.items():
            with self.subTest(case=name):
                self.assertIsNone(find_document_quad(image))

    def test_finds_document_with_broken_boundary_on_texture(
        self,
    ) -> None:
        for boundary in ("occluded", "broken"):
            with self.subTest(boundary=boundary):
                image, expected = _synthetic_document_scene(boundary)

                quad = find_document_quad(image)

                self.assertIsNotNone(
                    quad,
                    "A local boundary interruption must not hide the document",
                )
                self.assertEqual(quad.shape, (4, 2))
                errors = np.linalg.norm(quad - expected, axis=1)
                self.assertLessEqual(
                    float(errors.max()),
                    30.0,
                    (
                        "Detected corners must follow the page, not the "
                        f"background or printed bars: {quad}"
                    ),
                )

    def test_adaptive_resize_upscales_small_image_with_cap(self) -> None:
        image = np.zeros((300, 500, 3), dtype=np.uint8)

        resized, scale = adaptive_resize(image)

        self.assertEqual(resized.shape[:2], (900, 1500))
        self.assertEqual(scale, 3.0)

    def test_adaptive_resize_downscales_oversized_image(self) -> None:
        image = np.zeros((3000, 4000, 3), dtype=np.uint8)

        resized, scale = adaptive_resize(image)

        self.assertEqual(resized.shape[:2], (1800, 2400))
        self.assertAlmostEqual(scale, 0.6)

    def test_finds_and_rectifies_document_quadrilateral(self) -> None:
        image = np.full((800, 1000, 3), 25, dtype=np.uint8)
        document = np.array(
            [
                [120, 100],
                [900, 130],
                [850, 700],
                [100, 670],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(image, document, (235, 235, 235))
        cv2.polylines(image, [document], True, (255, 255, 255), 8)

        quad = find_document_quad(image)
        prepared = prepare_document_image(image)

        self.assertIsNotNone(quad)
        self.assertTrue(prepared.metadata["document_detected"])
        self.assertTrue(prepared.metadata["perspective_corrected"])
        self.assertGreater(prepared.metadata["output_width"], 0)
        self.assertGreater(prepared.metadata["output_height"], 0)

    def test_estimates_and_reduces_small_skew(self) -> None:
        image = np.full((400, 800, 3), 255, dtype=np.uint8)
        for y in range(80, 340, 50):
            cv2.line(image, (80, y), (720, y), (0, 0, 0), 3)
        rotated = rotate_image_bound(image, 6.0)

        angle = estimate_skew_angle(rotated)
        corrected = rotate_image_bound(rotated, angle)
        corrected_angle = estimate_skew_angle(corrected)

        self.assertGreater(abs(angle), 3.0)
        self.assertLessEqual(abs(angle), 8.0)
        self.assertLess(abs(corrected_angle), 1.5)


if __name__ == "__main__":
    unittest.main()
