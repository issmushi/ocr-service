from __future__ import annotations

import unittest
from itertools import product

from app.modules.v1.passport.normalization.countries.tajikistan import (
    has_valid_tajik_mrz_lower_line_prefix,
    normalize_tajik_mrz_lower_line,
    normalize_tajik_mrz_upper_line,
)

# Independent checks: number 6, birth 8, expiry 1, optional 0, total 0.
LOWER_LINE = "4012345676TJK9001158M35011513500012345678<00"


class TajikPassportNormalizerTests(unittest.TestCase):
    def test_normalizes_all_upper_digits_and_pads_without_truncating(
        self,
    ) -> None:
        name = "0123456789<<TEST"
        expected_name = "OIZEASGTBG<<TEST"
        for width in (len(name), 39, 42):
            with self.subTest(name_width=width):
                raw = "PPTJK" + name.ljust(width, "<")
                expected = ("PPTJK" + expected_name.ljust(width, "<")).ljust(
                    44, "<"
                )
                self.assertEqual(normalize_tajik_mrz_upper_line(raw), expected)

    def test_normalizes_fields_and_restores_complete_additional_data(
        self,
    ) -> None:
        for (country, expected_country), (
            raw_sex,
            expected_sex,
        ), missing_tail in product(
            (
                ("TJK", "TJK"),
                ("7JK", "TJK"),
                ("TJX", "TJK"),
                ("7J?", "TJK"),
                ("tjk", "TJK"),
                ("ARM", "ARM"),
            ),
            (
                ("N", "M"),
                ("H", "M"),
                ("E", "F"),
                ("M", "M"),
                ("F", "F"),
                ("<", "<"),
            ),
            range(4),
        ):
            with self.subTest(
                country=country, sex=raw_sex, missing_tail=missing_tail
            ):
                raw = (
                    LOWER_LINE[:10]
                    + country
                    + LOWER_LINE[13:20]
                    + raw_sex
                    + LOWER_LINE[21:]
                )[: 44 - missing_tail]
                expected = (
                    LOWER_LINE[:10]
                    + expected_country
                    + LOWER_LINE[13:20]
                    + expected_sex
                    + LOWER_LINE[21:]
                )
                self.assertTrue(has_valid_tajik_mrz_lower_line_prefix(raw))
                self.assertEqual(normalize_tajik_mrz_lower_line(raw), expected)

    def test_preserves_entire_lower_line_when_prefix_is_invalid(self) -> None:
        raw = (
            LOWER_LINE[:10]
            + "7JK"
            + LOWER_LINE[13:20]
            + "H"
            + LOWER_LINE[21:42]
        )
        cases = {
            "missing_first_digit": raw[1:],
            "missing_first_two_digits": raw[2:],
            "eleven_initial_digits": "0" + raw,
            "space_inserted_in_number": raw[:4] + " " + raw[4:],
            "space_in_number": raw[:4] + " " + raw[5:],
            "letter_in_number": "X" + raw[1:],
            "non_ascii_digit": "４" + raw[1:],
        }
        cases.update(
            {f"short_{length}": raw[:length] for length in (0, 9, 10, 12)}
        )
        for case, candidate in cases.items():
            with self.subTest(case=case):
                self.assertFalse(
                    has_valid_tajik_mrz_lower_line_prefix(candidate)
                )
                self.assertEqual(
                    normalize_tajik_mrz_lower_line(candidate), candidate
                )

    def test_unrecognized_nationality_does_not_block_sex_or_tail_normalization(
        self,
    ) -> None:
        raw = (
            LOWER_LINE[:10] + "RXX" + LOWER_LINE[13:20] + "H" + LOWER_LINE[21:]
        )[:42]
        expected = LOWER_LINE[:10] + "RXX" + LOWER_LINE[13:]

        self.assertTrue(has_valid_tajik_mrz_lower_line_prefix(raw))
        self.assertEqual(normalize_tajik_mrz_lower_line(raw), expected)

    def test_normalizes_without_padding_incomplete_required_data(
        self,
    ) -> None:
        for length in (13, 20, 21, 27, 28, 35, 40):
            with self.subTest(length=length):
                raw = (
                    LOWER_LINE[:10]
                    + "7JK"
                    + LOWER_LINE[13:20]
                    + "H"
                    + LOWER_LINE[21:]
                )[:length]
                self.assertTrue(has_valid_tajik_mrz_lower_line_prefix(raw))
                self.assertEqual(
                    normalize_tajik_mrz_lower_line(raw), LOWER_LINE[:length]
                )

    def test_does_not_change_damaged_numeric_fields_or_pad_invalid_data(
        self,
    ) -> None:
        for index, character in (
            (13, "X"),
            (19, "X"),
            (20, "X"),
            (21, "X"),
            (27, "X"),
            (28, "<"),
            (40, "X"),
            (41, "0"),
        ):
            with self.subTest(position=index + 1, character=character):
                raw = (
                    LOWER_LINE[:index] + character + LOWER_LINE[index + 1 : 42]
                )
                self.assertTrue(has_valid_tajik_mrz_lower_line_prefix(raw))
                self.assertEqual(normalize_tajik_mrz_lower_line(raw), raw)

    def test_bad_observed_optional_check_blocks_tail_but_not_field_corrections(
        self,
    ) -> None:
        for observed_check, expected_check in (
            ("9", "9"),
            ("E", "3"),
            ("<", "<"),
        ):
            with self.subTest(check=observed_check):
                raw = (
                    LOWER_LINE[:10]
                    + "7JK"
                    + LOWER_LINE[13:20]
                    + "H"
                    + LOWER_LINE[21:42]
                    + observed_check
                )
                expected = LOWER_LINE[:42] + expected_check
                self.assertEqual(normalize_tajik_mrz_lower_line(raw), expected)

    def test_preserves_complete_and_overlong_lines_including_existing_checks(
        self,
    ) -> None:
        for raw in (LOWER_LINE, LOWER_LINE[:42] + "99", LOWER_LINE + "<"):
            with self.subTest(raw=raw):
                self.assertEqual(normalize_tajik_mrz_lower_line(raw), raw)


if __name__ == "__main__":
    unittest.main()
