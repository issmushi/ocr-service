from __future__ import annotations

import unittest
from math import ceil, floor

import cv2
import numpy as np

from app.core.images.mrz_detection import find_mrz_bbox


def document_with_rows(
    count: int = 2,
    *,
    near_bottom: bool = False,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Invented monospaced glyphs, with no document or personal data."""
    image = np.full((640, 1000, 3), (224, 239, 245), dtype=np.uint8)
    characters = 44 if count == 2 else 30
    pitch = 20 if count == 2 else 28
    left = 55
    baseline = (
        (630 - (count - 1) * 34) if near_bottom else (510 - (count - 2) * 34)
    )
    ink = np.zeros(image.shape[:2], dtype=np.uint8)
    for row in range(count):
        text = ("P<UTOEXAMPLE<<TEST1234567890<" * 3)[:characters]
        for column, character in enumerate(text):
            cv2.putText(
                ink,
                character,
                (left + column * pitch, baseline + row * 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                255,
                2,
                cv2.LINE_AA,
            )
    image[ink > 0] = (25, 25, 25)
    y, x = np.where(ink > 0)
    return image, (
        int(x.min()),
        int(y.min()),
        int(x.max()) + 1,
        int(y.max()) + 1,
    )


class MrzDetectionTests(unittest.TestCase):
    def assert_contains_rows(
        self,
        image: np.ndarray,
        ink_bbox: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        bbox = find_mrz_bbox(image)
        self.assertIsNotNone(bbox)
        assert bbox is not None
        x1, y1, x2, y2 = bbox
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, image.shape[1])
        self.assertLessEqual(y2, image.shape[0])
        self.assertLessEqual(x1, ink_bbox[0])
        self.assertLessEqual(y1, ink_bbox[1])
        self.assertGreaterEqual(x2, ink_bbox[2])
        self.assertGreaterEqual(y2, ink_bbox[3])
        self.assertLess(y2 - y1, image.shape[0] * 0.30)
        return bbox

    def test_two_dense_rows_are_found_with_padding(self) -> None:
        image, ink = document_with_rows()
        before = image.copy()
        self.assert_contains_rows(image, ink)
        np.testing.assert_array_equal(image, before)

    def test_three_rows_are_not_truncated_to_two(self) -> None:
        image, ink = document_with_rows(3)
        self.assert_contains_rows(image, ink)

    def test_scaled_image_returns_original_pixel_coordinates(self) -> None:
        image, ink = document_with_rows()
        enlarged = cv2.resize(
            image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
        )
        self.assert_contains_rows(enlarged, tuple(value * 2 for value in ink))

    def test_grayscale_and_blur(self) -> None:
        image, ink = document_with_rows()
        gray = cv2.cvtColor(
            cv2.GaussianBlur(image, (3, 3), 0), cv2.COLOR_BGR2GRAY
        )
        self.assert_contains_rows(gray, ink)

    def test_touching_characters_at_either_end_are_not_clipped(self) -> None:
        image, ink = document_with_rows()
        # This blur joins the final two glyphs. They still belong in the crop
        # even though their connected component cannot count as one character.
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        self.assert_contains_rows(blurred, ink)
        mirrored_ink = (
            image.shape[1] - ink[2],
            ink[1],
            image.shape[1] - ink[0],
            ink[3],
        )
        self.assert_contains_rows(cv2.flip(blurred, 1), mirrored_ink)

    def test_small_blurred_td3_keeps_both_rows_and_their_ends(self) -> None:
        image, ink = document_with_rows()
        scale = 0.4
        small = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        small = cv2.GaussianBlur(small, (5, 5), 0)
        self.assert_contains_rows(
            small,
            (
                floor(ink[0] * scale),
                floor(ink[1] * scale),
                ceil(ink[2] * scale),
                ceil(ink[3] * scale),
            ),
        )

    def test_small_id_retains_all_three_rows_including_near_edge(self) -> None:
        for near_bottom in (False, True):
            with self.subTest(near_bottom=near_bottom):
                image, ink = document_with_rows(3, near_bottom=near_bottom)
                scale = 0.32
                small = cv2.resize(
                    image,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )
                small = cv2.GaussianBlur(small, (3, 3), 0)
                self.assert_contains_rows(
                    small,
                    (
                        floor(ink[0] * scale),
                        floor(ink[1] * scale),
                        ceil(ink[2] * scale),
                        ceil(ink[3] * scale),
                    ),
                )

    def test_rows_near_bottom_stay_inside_image(self) -> None:
        image, ink = document_with_rows(3, near_bottom=True)
        self.assert_contains_rows(image, ink)

    def test_blank_and_low_contrast_have_no_fallback(self) -> None:
        for value in (0, 127, 255):
            with self.subTest(value=value):
                self.assertIsNone(
                    find_mrz_bbox(np.full((640, 1000, 3), value, np.uint8))
                )
        image, _ = document_with_rows()
        image[:] = 240 + (image[:, :, :1] > 100).astype(np.uint8) * 3
        self.assertIsNone(find_mrz_bbox(image))

    def test_single_long_text_row_is_not_mrz(self) -> None:
        image, _ = document_with_rows(1)
        self.assertIsNone(find_mrz_bbox(image))

    def test_ordinary_paragraph_is_not_mrz(self) -> None:
        image = np.full((640, 1000, 3), 245, np.uint8)
        for index, line in enumerate(
            (
                "This document contains ordinary printed text.",
                "Please review the address and issue date below.",
                "Office opening hours are Monday to Friday.",
            )
        ):
            cv2.putText(
                image,
                line,
                (40, 470 + index * 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )
        self.assertIsNone(find_mrz_bbox(image))

    def test_rules_and_frame_are_not_text_rows(self) -> None:
        image = np.full((640, 1000, 3), 245, np.uint8)
        cv2.rectangle(image, (12, 12), (987, 627), (0, 0, 0), 4)
        for y in (465, 500, 535, 570):
            cv2.line(image, (30, y), (970, y), (0, 0, 0), 3)
        self.assertIsNone(find_mrz_bbox(image))

    def test_card_fields_and_barcode_have_no_mrz_fallback(self) -> None:
        image = np.full((400, 640, 3), 245, np.uint8)
        for index, line in enumerate(
            (
                "IDENTITY CARD",
                "EXAMPLE SURNAME",
                "TEST GIVEN NAME",
                "01.01.2000     01.01.2030",
                "ISSUING AUTHORITY",
            )
        ):
            cv2.putText(
                image,
                line,
                (190, 115 + index * 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
        for x in range(35, 605, 7):
            cv2.rectangle(image, (x, 320), (x + 2, 370), (20, 20, 20), -1)
        self.assertIsNone(find_mrz_bbox(image))

    def test_unrelated_rows_with_different_pitch_are_not_grouped(self) -> None:
        first, _ = document_with_rows(2)
        second, _ = document_with_rows(3)
        image = first.copy()
        image[525:] = (224, 239, 245)
        image[535:570] = second[480:515]
        self.assertIsNone(find_mrz_bbox(image))

    def test_malformed_images_return_none(self) -> None:
        for image in (
            np.zeros((0, 10), np.uint8),
            np.zeros((40, 40, 3), np.uint8),
            np.zeros((100, 200, 4), np.uint8),
            np.zeros((100, 200), np.float32),
        ):
            with self.subTest(shape=image.shape, dtype=image.dtype):
                self.assertIsNone(find_mrz_bbox(image))


if __name__ == "__main__":
    unittest.main()
