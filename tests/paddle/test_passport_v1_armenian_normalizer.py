from __future__ import annotations

import unittest

from app.modules.v1.passport.normalization.countries.armenia import (
    normalize_armenian_mrz_lower_line,
)

LOWER_LINE = "AA12345678ARM9001158M3001156<<<<<<<<<<<<<<06"


class ArmenianPassportLowerLineTests(unittest.TestCase):
    def test_restores_each_missing_suffix_of_an_empty_optional_field(
        self,
    ) -> None:
        for length in range(28, 44):
            with self.subTest(length=length):
                self.assertEqual(
                    normalize_armenian_mrz_lower_line(LOWER_LINE[:length]),
                    LOWER_LINE,
                )

    def test_restores_checks_for_complete_nonempty_optional_data(self) -> None:
        line = "AA12345678ARM9001158M30011561234567890<<<<70"
        for length in (42, 43):
            with self.subTest(length=length):
                self.assertEqual(
                    normalize_armenian_mrz_lower_line(line[:length]), line
                )

    def test_preserves_existing_filler_as_optional_check_digit(self) -> None:
        line = LOWER_LINE[:42] + "<"
        self.assertEqual(normalize_armenian_mrz_lower_line(line), line + "6")

    def test_incorrect_observed_check_digit_blocks_completion(self) -> None:
        for observed, expected in (("9", "9"), ("E", "3")):
            with self.subTest(observed=observed):
                line = LOWER_LINE[:42] + observed
                self.assertEqual(
                    normalize_armenian_mrz_lower_line(line),
                    LOWER_LINE[:42] + expected,
                )

    def test_does_not_pad_truncated_personal_data_or_invalid_characters(
        self,
    ) -> None:
        for optional_data in ("12345", "12345<", "#", "<a<"):
            with self.subTest(optional_data=optional_data):
                line = LOWER_LINE[:28] + optional_data
                self.assertEqual(normalize_armenian_mrz_lower_line(line), line)

    def test_does_not_repair_a_missing_start_or_incomplete_required_fields(
        self,
    ) -> None:
        for line in (LOWER_LINE[1:], LOWER_LINE[2:], LOWER_LINE[:27]):
            with self.subTest(line=line):
                self.assertEqual(
                    len(normalize_armenian_mrz_lower_line(line)), len(line)
                )
        invalid_birth_check = LOWER_LINE[:19] + "X" + LOWER_LINE[20:35]
        self.assertEqual(
            normalize_armenian_mrz_lower_line(invalid_birth_check),
            invalid_birth_check,
        )

    def test_preserves_complete_and_overlong_lines_including_their_checks(
        self,
    ) -> None:
        for line in (LOWER_LINE, LOWER_LINE[:-2] + "99", LOWER_LINE + "<"):
            with self.subTest(line=line):
                self.assertEqual(normalize_armenian_mrz_lower_line(line), line)


if __name__ == "__main__":
    unittest.main()
