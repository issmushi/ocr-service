from __future__ import annotations

import unittest
from itertools import product

from app.modules.v1.passport.normalization.countries.russian_foreign import (
    has_valid_russian_foreign_mrz_lower_line_prefix,
    normalize_russian_foreign_mrz_lower_line,
)

# Independent checks: number 2, birth 8, expiry 1, optional 0, total 8.
LOWER_LINE = "7512345672RUS9001158M3501151<<<<<<<<<<<<<<08"


class RussianForeignPassportLowerLineTests(unittest.TestCase):
    def test_normalizes_citizenship_and_sex_and_restores_empty_tail(
        self,
    ) -> None:
        for (country, expected_country), (
            raw_sex,
            expected_sex,
        ), missing_tail in product(
            (
                ("RUS", "RUS"),
                ("RU5", "RUS"),
                ("RUB", "RUS"),
                ("RU?", "RUS"),
                ("rus", "RUS"),
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
            (0, 1, 2, 5, 16),
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
                self.assertTrue(
                    has_valid_russian_foreign_mrz_lower_line_prefix(raw)
                )
                self.assertEqual(
                    normalize_russian_foreign_mrz_lower_line(raw), expected
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
            "non_ascii_digit": "７" + raw[1:],
        }
        cases.update(
            {f"short_{length}": raw[:length] for length in (0, 9, 10, 12)}
        )
        for case, candidate in cases.items():
            with self.subTest(case=case):
                self.assertFalse(
                    has_valid_russian_foreign_mrz_lower_line_prefix(candidate)
                )
                self.assertEqual(
                    normalize_russian_foreign_mrz_lower_line(candidate),
                    candidate,
                )

    def test_unrecognized_nationality_does_not_block_sex_or_tail_normalization(
        self,
    ) -> None:
        raw = (
            LOWER_LINE[:10] + "RXX" + LOWER_LINE[13:20] + "H" + LOWER_LINE[21:]
        )[:42]
        expected = LOWER_LINE[:10] + "RXX" + LOWER_LINE[13:]

        self.assertTrue(has_valid_russian_foreign_mrz_lower_line_prefix(raw))
        self.assertEqual(
            normalize_russian_foreign_mrz_lower_line(raw), expected
        )

    def test_rejects_shifted_number_that_mimics_a_supported_nationality(
        self,
    ) -> None:
        for raw in (
            "01000000074RU",
            "01000000074RUS900101IM3501014<<<<<<<<<<<<<<0",
        ):
            with self.subTest(raw=raw):
                self.assertFalse(
                    has_valid_russian_foreign_mrz_lower_line_prefix(raw)
                )
                self.assertEqual(
                    normalize_russian_foreign_mrz_lower_line(raw), raw
                )

    def test_does_not_pad_before_required_fields_are_complete(self) -> None:
        for length in (13, 20, 21, 27):
            with self.subTest(length=length):
                raw = (
                    LOWER_LINE[:10]
                    + "RU5"
                    + LOWER_LINE[13:20]
                    + "H"
                    + LOWER_LINE[21:]
                )[:length]
                self.assertTrue(
                    has_valid_russian_foreign_mrz_lower_line_prefix(raw)
                )
                self.assertEqual(
                    normalize_russian_foreign_mrz_lower_line(raw),
                    LOWER_LINE[:length],
                )

    def test_keeps_damaged_numeric_fields_and_invalid_required_data(
        self,
    ) -> None:
        for index, character in (
            (13, "X"),
            (19, "X"),
            (20, "X"),
            (21, "X"),
            (27, "X"),
        ):
            with self.subTest(position=index + 1, character=character):
                raw = (
                    LOWER_LINE[:index] + character + LOWER_LINE[index + 1 : 42]
                )
                self.assertTrue(
                    has_valid_russian_foreign_mrz_lower_line_prefix(raw)
                )
                self.assertEqual(
                    normalize_russian_foreign_mrz_lower_line(raw), raw
                )

    def test_preserves_nonempty_optional_data_without_restoring_tail(
        self,
    ) -> None:
        for optional, length in product(
            ("1" + "<" * 13, "<" * 13 + "A", "12345678901234"), (42, 43, 44)
        ):
            with self.subTest(optional=optional, length=length):
                raw = (
                    LOWER_LINE[:10]
                    + "RU5"
                    + LOWER_LINE[13:20]
                    + "H"
                    + LOWER_LINE[21:28]
                    + optional
                    + "08"
                )[:length]
                normalized_optional = optional.replace("A", "4")
                expected = (LOWER_LINE[:28] + normalized_optional + "08")[
                    :length
                ]
                self.assertEqual(
                    normalize_russian_foreign_mrz_lower_line(raw), expected
                )
        for length in (29, 35, 41):
            with self.subTest(length=length):
                raw = LOWER_LINE[:28] + "1" + "<" * (length - 29)
                self.assertEqual(
                    normalize_russian_foreign_mrz_lower_line(raw), raw
                )

    def test_preserves_observed_valid_optional_check_and_adds_only_composite(
        self,
    ) -> None:
        for observed_check in ("0", "<"):
            with self.subTest(check=observed_check):
                raw = LOWER_LINE[:42] + observed_check
                self.assertEqual(
                    normalize_russian_foreign_mrz_lower_line(raw), raw + "8"
                )

    def test_bad_observed_optional_check_blocks_tail_but_not_field_corrections(
        self,
    ) -> None:
        for observed_check, expected_check in (
            ("1", "1"),
            ("8", "8"),
            ("E", "3"),
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
                    normalize_russian_foreign_mrz_lower_line(raw), expected
                )

    def test_preserves_complete_and_overlong_lines_including_existing_checks(
        self,
    ) -> None:
        for raw in (
            LOWER_LINE,
            LOWER_LINE[:42] + "99",
            LOWER_LINE[:42] + "<8",
            LOWER_LINE + "<",
        ):
            with self.subTest(raw=raw):
                self.assertEqual(
                    normalize_russian_foreign_mrz_lower_line(raw), raw
                )


if __name__ == "__main__":
    unittest.main()
