from __future__ import annotations

import unittest
from itertools import product

from app.modules.v1.passport.normalization.countries.uzbekistan import (
    has_valid_uzbek_mrz_lower_line_prefix,
    normalize_uzbek_mrz_lower_line,
)

# Independent checks: number 8, birth 8, expiry 1, optional 5, total 6.
LOWER_LINE = "AA12345678UZB9001158M35011511234567890123456"


class UzbekPassportLowerLineTests(unittest.TestCase):
    def test_normalizes_fields_without_changing_numeric_data(
        self,
    ) -> None:
        for (nationality, expected_nationality), (
            raw_prefix,
            expected_prefix,
            raw_sex,
            expected_sex,
        ) in product(
            (
                ("UZB", "UZB"),
                ("UZ8", "UZB"),
                ("U2B", "UZB"),
                ("U28", "UZB"),
                ("U08", "UZB"),
                ("UZX", "UZB"),
                ("ARM", "ARM"),
            ),
            (
                ("44", "AA", "H", "M"),
                ("01", "OI", "E", "F"),
                ("AA", "AA", "N", "M"),
            ),
        ):
            with self.subTest(
                nationality=nationality, prefix=raw_prefix, sex=raw_sex
            ):
                raw = (
                    raw_prefix
                    + LOWER_LINE[2:10]
                    + nationality
                    + LOWER_LINE[13:20]
                    + raw_sex
                    + LOWER_LINE[21:]
                )
                expected = (
                    expected_prefix
                    + LOWER_LINE[2:10]
                    + expected_nationality
                    + LOWER_LINE[13:20]
                    + expected_sex
                    + LOWER_LINE[21:]
                )
                self.assertTrue(has_valid_uzbek_mrz_lower_line_prefix(raw))
                self.assertEqual(normalize_uzbek_mrz_lower_line(raw), expected)

    def test_restores_only_the_missing_check_digits_after_field_corrections(
        self,
    ) -> None:
        expected = LOWER_LINE[:20] + "F" + LOWER_LINE[21:]
        for nationality, length in product(
            ("UZB", "UZ8", "U2B", "U28", "U08"),
            (42, 43),
        ):
            with self.subTest(nationality=nationality, length=length):
                raw = (
                    "44"
                    + LOWER_LINE[2:10]
                    + nationality
                    + LOWER_LINE[13:20]
                    + "E"
                    + LOWER_LINE[21:]
                )[:length]
                self.assertEqual(normalize_uzbek_mrz_lower_line(raw), expected)

    def test_invalid_prefix_preserves_entire_raw_line_before_any_corrections(
        self,
    ) -> None:
        raw = (
            "44"
            + LOWER_LINE[2:10]
            + "U28"
            + LOWER_LINE[13:20]
            + "H"
            + LOWER_LINE[21:]
        )
        for line in (
            "",
            raw[:9],
            raw[1:],
            "4" + raw,
            raw[:4] + " " + raw[4:],
            "?" + raw[1:],
            raw[:3] + "١" + raw[4:],
        ):
            with self.subTest(line=line):
                self.assertFalse(has_valid_uzbek_mrz_lower_line_prefix(line))
                self.assertEqual(normalize_uzbek_mrz_lower_line(line), line)

    def test_unrecognized_nationality_does_not_block_normalization(
        self,
    ) -> None:
        raw = (
            "44"
            + LOWER_LINE[2:10]
            + "UXX"
            + LOWER_LINE[13:20]
            + "H"
            + LOWER_LINE[21:]
        )[:42]
        expected = "AA" + LOWER_LINE[2:10] + "UXX" + LOWER_LINE[13:]

        self.assertTrue(has_valid_uzbek_mrz_lower_line_prefix(raw))
        self.assertEqual(normalize_uzbek_mrz_lower_line(raw), expected)

    def test_does_not_fill_missing_optional_data_or_invalid_required_fields(
        self,
    ) -> None:
        for length in (28, 35, 41):
            with self.subTest(length=length):
                line = LOWER_LINE[:length]
                self.assertEqual(normalize_uzbek_mrz_lower_line(line), line)
        for index, character in (
            (13, "X"),
            (20, "X"),
            (21, "<"),
            (30, "X"),
            (41, "<"),
        ):
            with self.subTest(position=index + 1, character=character):
                line = (
                    LOWER_LINE[:index] + character + LOWER_LINE[index + 1 : 42]
                )
                self.assertEqual(normalize_uzbek_mrz_lower_line(line), line)

    def test_wrong_observed_optional_check_digit_blocks_completion(
        self,
    ) -> None:
        for observed, expected in (("4", "4"), ("E", "3"), ("<", "<")):
            with self.subTest(observed=observed):
                line = LOWER_LINE[:42] + observed
                self.assertEqual(
                    normalize_uzbek_mrz_lower_line(line),
                    LOWER_LINE[:42] + expected,
                )

    def test_complete_and_overlong_lines_keep_existing_checks(self) -> None:
        for line in (LOWER_LINE, LOWER_LINE[:-2] + "99", LOWER_LINE + "<"):
            with self.subTest(line=line):
                self.assertEqual(normalize_uzbek_mrz_lower_line(line), line)


if __name__ == "__main__":
    unittest.main()
