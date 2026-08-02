from __future__ import annotations

import unittest

from app.modules.v1.passport.normalization.countries.kazakhstan import (
    is_kazakh_mrz_number_prefix_missing,
    normalize_kazakh_mrz_lower_line,
)

LOWER_LINE = "N123456785KAZ9001011M3501014900101300123<<82"


class KazakhPassportLowerLineTests(unittest.TestCase):
    def test_restores_missing_number_initial_before_other_fields_and_tail(
        self,
    ) -> None:
        for citizenship in ("KAZ", "K42", "KA2"):
            for missing_tail in range(5):
                with self.subTest(
                    citizenship=citizenship, missing_tail=missing_tail
                ):
                    line = LOWER_LINE[:10] + citizenship + LOWER_LINE[13:]
                    raw = line[1 : 44 - missing_tail]
                    self.assertTrue(is_kazakh_mrz_number_prefix_missing(raw))
                    self.assertEqual(
                        normalize_kazakh_mrz_lower_line(raw), LOWER_LINE
                    )

    def test_missing_number_initial_does_not_allow_padding_an_incomplete_iin(
        self,
    ) -> None:
        raw = LOWER_LINE[1:39]
        self.assertTrue(is_kazakh_mrz_number_prefix_missing(raw))
        self.assertEqual(normalize_kazakh_mrz_lower_line(raw), LOWER_LINE[:39])

    def test_distinguishes_missing_initial_from_other_damage(
        self,
    ) -> None:
        replaced_initial = "4" + LOWER_LINE[1:]
        self.assertFalse(is_kazakh_mrz_number_prefix_missing(replaced_initial))
        self.assertEqual(
            normalize_kazakh_mrz_lower_line(replaced_initial), LOWER_LINE
        )
        for raw in (LOWER_LINE[2:], LOWER_LINE[:40]):
            with self.subTest(raw=raw):
                self.assertFalse(is_kazakh_mrz_number_prefix_missing(raw))

    def test_restores_tail_only_after_all_twelve_iin_digits(self) -> None:
        for length in range(40, 44):
            with self.subTest(length=length):
                self.assertEqual(
                    normalize_kazakh_mrz_lower_line(LOWER_LINE[:length]),
                    LOWER_LINE,
                )

    def test_does_not_pad_incomplete_or_invalid_iin_and_fillers(self) -> None:
        for optional_data in (
            "90010130012",
            "90010",
            "",
            "9001X1300123",
            "90010<300123",
            "900101300123X",
            "9001013001230",
        ):
            with self.subTest(optional_data=optional_data):
                line = LOWER_LINE[:28] + optional_data
                self.assertEqual(normalize_kazakh_mrz_lower_line(line), line)

    def test_does_not_pad_two_missing_initials_or_invalid_required_fields(
        self,
    ) -> None:
        for line in (LOWER_LINE[2:], LOWER_LINE[:27]):
            with self.subTest(line=line):
                self.assertEqual(
                    len(normalize_kazakh_mrz_lower_line(line)), len(line)
                )
        for index in (8, 9, 19, 27):
            with self.subTest(invalid_position=index + 1):
                line = LOWER_LINE[:index] + "X" + LOWER_LINE[index + 1 : 40]
                self.assertEqual(normalize_kazakh_mrz_lower_line(line), line)

    def test_incorrect_observed_optional_check_digit_blocks_completion(
        self,
    ) -> None:
        for observed, expected in (("9", "9"), ("E", "3"), ("<", "<")):
            with self.subTest(observed=observed):
                line = LOWER_LINE[:42] + observed
                self.assertEqual(
                    normalize_kazakh_mrz_lower_line(line),
                    LOWER_LINE[:42] + expected,
                )

    def test_preserves_complete_and_overlong_lines_including_their_checks(
        self,
    ) -> None:
        for line in (LOWER_LINE, LOWER_LINE[:-2] + "99", LOWER_LINE + "<"):
            with self.subTest(line=line):
                self.assertEqual(normalize_kazakh_mrz_lower_line(line), line)

    def test_corrects_sex_even_when_tail_cannot_be_restored(self) -> None:
        for length, raw_sex, expected_sex in ((39, "H", "M"), (43, "E", "F")):
            with self.subTest(length=length, sex=raw_sex):
                line = LOWER_LINE[:length]
                if length == 43:
                    line = line[:42] + "9"
                raw = line[:20] + raw_sex + line[21:]
                expected = line[:20] + expected_sex + line[21:]
                self.assertEqual(
                    normalize_kazakh_mrz_lower_line(raw), expected
                )

    def test_preserves_unknown_sex_and_empty_line(self) -> None:
        self.assertEqual(normalize_kazakh_mrz_lower_line(""), "")
        for length in (40, 44):
            with self.subTest(length=length):
                line = LOWER_LINE[:20] + "X" + LOWER_LINE[21:length]
                self.assertEqual(normalize_kazakh_mrz_lower_line(line), line)


if __name__ == "__main__":
    unittest.main()
