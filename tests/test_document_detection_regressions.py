from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from app.core.images.document_detection import (
    DocumentCandidate,
    detect_document_quad,
)
from app.core.images.preprocessing import find_document_quad


def _rectangle(
    left: float, top: float, right: float, bottom: float
) -> np.ndarray:
    return np.array(
        [[left, top], [right, top], [right, bottom], [left, bottom]],
        np.float32,
    )


def _candidate(quad: np.ndarray, score: float) -> DocumentCandidate:
    return DocumentCandidate(quad, score, "lines", (1.0, 1.0, 1.0, 1.0), 0.7)


def _spread(fold: int = 616) -> np.ndarray:
    image = np.full((1280, 960, 3), 220, np.uint8)
    image[:fold] = 180
    cv2.line(image, (0, fold), (959, fold), (90, 90, 90), 3)
    return image


class DocumentDetectionRegressions(unittest.TestCase):
    def test_default_selection_preserves_original_preference_order(
        self,
    ) -> None:
        spread = _candidate(_rectangle(20, 20, 580, 800), 0.91)
        page = _candidate(_rectangle(20, 400, 580, 800), 0.78)
        with patch(
            "app.core.images.document_detection.rank_document_candidates",
            return_value=[spread, page],
        ):
            result = detect_document_quad(np.zeros((820, 600, 3), np.uint8))
        np.testing.assert_array_equal(result, spread.quad)
        self.assertIsNot(result, spread.quad)

    def test_required_mrz_rejects_an_internal_separator_crop(self) -> None:
        upper = _candidate(_rectangle(0, 0, 799, 280), 0.91)
        card = _candidate(_rectangle(0, 0, 799, 499), 0.72)
        with patch(
            "app.core.images.document_detection.rank_document_candidates",
            return_value=[upper, card],
        ):
            result = detect_document_quad(
                np.zeros((500, 800, 3), np.uint8),
                required_region=_rectangle(40, 330, 760, 465),
            )
        np.testing.assert_array_equal(result, card.quad)

    def test_losing_five_percent_of_mrz_rejects_the_entire_crop(self) -> None:
        partial = _candidate(_rectangle(86, 0, 799, 499), 0.90)
        # 36 of the MRZ's 720 pixels would be lost at its left edge.
        with patch(
            "app.core.images.document_detection.rank_document_candidates",
            return_value=[partial],
        ):
            result = detect_document_quad(
                np.zeros((500, 800, 3), np.uint8),
                required_region=_rectangle(50, 330, 770, 465),
                prefer_landscape=True,
            )
        self.assertIsNone(result)

    def test_landscape_page_can_win_despite_a_large_score_gap(self) -> None:
        spread = _candidate(_rectangle(20, 20, 580, 800), 0.92)
        page = _candidate(_rectangle(20, 400, 580, 800), 0.75)
        with patch(
            "app.core.images.document_detection.rank_document_candidates",
            return_value=[spread, page],
        ):
            result = detect_document_quad(
                np.zeros((820, 600, 3), np.uint8),
                required_region=_rectangle(50, 720, 550, 775),
                prefer_landscape=True,
            )
        np.testing.assert_array_equal(result, page.quad)

    def test_keeps_covering_outer_candidate_when_no_landscape_page_exists(
        self,
    ) -> None:
        outer = _candidate(_rectangle(20, 20, 580, 800), 0.92)
        with patch(
            "app.core.images.document_detection.rank_document_candidates",
            return_value=[outer],
        ):
            result = detect_document_quad(
                np.zeros((820, 600, 3), np.uint8),
                required_region=_rectangle(50, 720, 550, 775),
                prefer_landscape=True,
            )
        np.testing.assert_array_equal(result, outer.quad)

    def test_required_region_accepts_a_convex_polygon_not_only_a_rectangle(
        self,
    ) -> None:
        page = _candidate(_rectangle(0, 0, 799, 499), 0.8)
        region = np.array(
            [[30, 350], [300, 335], [750, 350], [760, 450], [45, 465]],
            np.float32,
        )
        with patch(
            "app.core.images.document_detection.rank_document_candidates",
            return_value=[page],
        ):
            result = detect_document_quad(
                np.zeros((500, 800, 3), np.uint8), required_region=region
            )
        np.testing.assert_array_equal(result, page.quad)

    def test_real_fold_with_frame_clipped_sides_preserves_full_bio_page(
        self,
    ) -> None:
        image = _spread()
        region = _rectangle(22, 1170, 943, 1265)
        original_region = region.copy()
        result = find_document_quad(
            image, required_region=region, prefer_landscape=True
        )
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result[:2, 1], [616, 616], atol=7)
        np.testing.assert_allclose(
            result[2:], [[959, 1279], [0, 1279]], atol=2
        )
        self.assertGreaterEqual(
            cv2.intersectConvexConvex(result, region)[0]
            / cv2.contourArea(region),
            0.985,
        )
        np.testing.assert_array_equal(region, original_region)

    def test_required_region_and_result_scale_with_source_resolution(
        self,
    ) -> None:
        image = _spread()
        region = _rectangle(22, 1170, 943, 1265)
        regular = find_document_quad(
            image, required_region=region, prefer_landscape=True
        )
        enlarged = find_document_quad(
            cv2.resize(
                image, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST
            ),
            required_region=region * 2,
            prefer_landscape=True,
        )
        self.assertIsNotNone(regular)
        self.assertIsNotNone(enlarged)
        np.testing.assert_allclose(enlarged / 2, regular, atol=3)

    def test_frame_edges_require_both_mrz_and_landscape_preference(
        self,
    ) -> None:
        image = _spread()
        region = _rectangle(22, 1170, 943, 1265)
        with patch(
            "app.core.images.document_detection.rank_document_candidates",
            return_value=[],
        ):
            self.assertIsNone(
                detect_document_quad(image, prefer_landscape=True)
            )
            self.assertIsNone(
                detect_document_quad(image, required_region=region)
            )

    def test_frame_fallback_requires_visible_fold_and_plausible_page(
        self,
    ) -> None:
        blank = np.full((1280, 960, 3), 220, np.uint8)
        thin_rule = blank.copy()
        cv2.line(thin_rule, (0, 616), (959, 616), (90, 90, 90), 3)
        short_step = blank.copy()
        short_step[:616, 350:610] = 180
        wrong_aspect = _spread(fold=1050)
        normal_region = _rectangle(22, 1170, 943, 1265)
        cases = [
            ("blank", blank, normal_region),
            ("thin_printed_rule", thin_rule, normal_region),
            ("short_fragment", short_step, normal_region),
            ("wrong_page_aspect", wrong_aspect, normal_region),
            ("narrow_mrz", _spread(), _rectangle(220, 1170, 740, 1265)),
            ("mrz_above_bottom", _spread(), _rectangle(22, 830, 943, 930)),
            ("landscape_source", blank[:700], _rectangle(22, 600, 943, 680)),
        ]
        for name, image, region in cases:
            with (
                self.subTest(name=name),
                patch(
                    "app.core.images.document_detection.rank_document_candidates",
                    return_value=[],
                ),
            ):
                self.assertIsNone(
                    detect_document_quad(
                        image, required_region=region, prefer_landscape=True
                    )
                )


if __name__ == "__main__":
    unittest.main()
