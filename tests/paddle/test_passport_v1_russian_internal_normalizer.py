from __future__ import annotations

import unittest
from itertools import product

from app.modules.v1.passport.normalization.countries.russian_internal import (
    has_valid_russian_internal_mrz_lower_line_prefix,
    normalize_russian_internal_mrz_lower_line,
    normalize_russian_internal_mrz_upper_line,
)

# Independent check digits: number 9, birth 8, optional 8, total 8.
LOWER_LINE = "4511234569RUS9001158M<<<<<<<5150620770001<88"


class RussianInternalPassportLowerLineTests(unittest.TestCase):
    def test_normalizes_only_ambiguous_digits_in_upper_name(self) -> None:
        raw = "PNRUS0152346789<<TEST"
        expected = "PNRUSOIS2346789<<TEST".ljust(44, "<")

        self.assertEqual(
            normalize_russian_internal_mrz_upper_line(raw), expected
        )

    def test_normalizes_citizenship_digits_without_changing_numeric_fields(
        self,
    ) -> None:
        for raw_country, expected_country in (
            ("RUS", "RUS"),
            ("RU5", "RUS"),
            ("RUB", "RUS"),
            ("RU?", "RUS"),
            ("rus", "RUS"),
            ("ARM", "ARM"),
        ):
            with self.subTest(citizenship=raw_country):
                raw = LOWER_LINE[:10] + raw_country + LOWER_LINE[13:]
                expected = LOWER_LINE[:10] + expected_country + LOWER_LINE[13:]
                self.assertTrue(
                    has_valid_russian_internal_mrz_lower_line_prefix(raw)
                )
                self.assertEqual(
                    normalize_russian_internal_mrz_lower_line(raw), expected
                )

    def test_preserves_entire_lower_line_when_prefix_is_invalid(self) -> None:
        raw = (
            LOWER_LINE[:10]
            + "RU5"
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
                    has_valid_russian_internal_mrz_lower_line_prefix(candidate)
                )
                self.assertEqual(
                    normalize_russian_internal_mrz_lower_line(candidate),
                    candidate,
                )

    def test_unrecognized_nationality_does_not_block_sex_or_tail_normalization(
        self,
    ) -> None:
        raw = (
            LOWER_LINE[:10] + "RXX" + LOWER_LINE[13:20] + "H" + LOWER_LINE[21:]
        )[:42]
        expected = LOWER_LINE[:10] + "RXX" + LOWER_LINE[13:]

        self.assertTrue(has_valid_russian_internal_mrz_lower_line_prefix(raw))
        self.assertEqual(
            normalize_russian_internal_mrz_lower_line(raw), expected
        )

    def test_accepts_correctable_citizenship_when_only_tail_is_missing(
        self,
    ) -> None:
        for length in (13, 41, 42, 43, 44):
            with self.subTest(length=length):
                raw = (
                    LOWER_LINE[:10]
                    + "RU5"
                    + LOWER_LINE[13:20]
                    + "H"
                    + LOWER_LINE[21:]
                )[:length]
                self.assertTrue(
                    has_valid_russian_internal_mrz_lower_line_prefix(raw)
                )
                expected = LOWER_LINE[:13] if length == 13 else LOWER_LINE
                self.assertEqual(
                    normalize_russian_internal_mrz_lower_line(raw), expected
                )

    def test_normalizes_sex_and_restores_each_missing_tail_symbol(
        self,
    ) -> None:
        for (raw_sex, expected_sex), missing_tail in product(
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
            with self.subTest(sex=raw_sex, missing_tail=missing_tail):
                raw = (LOWER_LINE[:20] + raw_sex + LOWER_LINE[21:])[
                    : 44 - missing_tail
                ]
                expected = LOWER_LINE[:20] + expected_sex + LOWER_LINE[21:]
                self.assertEqual(
                    normalize_russian_internal_mrz_lower_line(raw), expected
                )

    def test_does_not_pad_incomplete_additional_data_or_short_lines(
        self,
    ) -> None:
        for length in (0, 20, 28, 35, 40):
            with self.subTest(length=length):
                raw = LOWER_LINE[:length]
                self.assertEqual(
                    normalize_russian_internal_mrz_lower_line(raw), raw
                )

    def test_does_not_pad_invalid_required_fields_or_additional_data(
        self,
    ) -> None:
        for index, character in (
            (0, "X"),
            (19, "X"),
            (20, "X"),
            (21, "0"),
            (28, "<"),
            (40, "X"),
            (41, "0"),
        ):
            with self.subTest(position=index + 1, character=character):
                raw = (
                    LOWER_LINE[:index] + character + LOWER_LINE[index + 1 : 42]
                )
                self.assertEqual(
                    normalize_russian_internal_mrz_lower_line(raw), raw
                )

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
                    + "RU5"
                    + LOWER_LINE[13:20]
                    + "H"
                    + LOWER_LINE[21:42]
                    + observed_check
                )
                expected = LOWER_LINE[:42] + expected_check
                self.assertEqual(
                    normalize_russian_internal_mrz_lower_line(raw), expected
                )

    def test_preserves_complete_and_overlong_lines_including_existing_checks(
        self,
    ) -> None:
        for raw in (LOWER_LINE, LOWER_LINE[:-2] + "99", LOWER_LINE + "<"):
            with self.subTest(raw=raw):
                self.assertEqual(
                    normalize_russian_internal_mrz_lower_line(raw), raw
                )


if __name__ == "__main__":
    unittest.main()
