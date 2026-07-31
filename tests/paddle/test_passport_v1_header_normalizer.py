from __future__ import annotations

import unittest

from app.modules.v1.passport.normalization import (
    AmbiguousPassportHeaderError,
    PassportCountryNotFoundError,
    PassportDocumentCodeError,
    normalize_passport_mrz_header,
)


class PassportV1HeaderNormalizerTests(unittest.TestCase):
    def test_normalizes_digit_lookalikes_only_inside_header(self) -> None:
        result = normalize_passport_mrz_header("p<u28ivan0v")

        self.assertEqual(result.normalized_mrz, "P<UZBivan0v")
        self.assertEqual(result.normalized_header, "P<UZB")
        self.assertEqual(result.document_code, "P")
        self.assertEqual(result.country, "UZB")

    def test_recovers_country_digits_before_complete_header_lookup(
        self,
    ) -> None:
        cases = (
            ("P<K4Z", "P<KAZ", [(3, "4", "A")]),
            ("P<K42", "P<KAZ", [(3, "4", "A"), (4, "2", "Z")]),
            ("P<4RM", "P<ARM", [(2, "4", "A")]),
            ("P<7JK", "P<TJK", [(2, "7", "T")]),
        )
        tail = "9IVAN0V"

        for raw_header, expected_header, expected_corrections in cases:
            with self.subTest(raw_header=raw_header):
                raw_mrz = raw_header + tail
                result = normalize_passport_mrz_header(raw_mrz)

                self.assertEqual(result.raw_mrz, raw_mrz)
                self.assertEqual(result.raw_window, raw_header)
                self.assertEqual(result.normalized_header, expected_header)
                self.assertEqual(result.country, expected_header[2:])
                self.assertEqual(result.normalized_mrz, expected_header + tail)
                self.assertEqual(result.consumed_characters, 5)
                self.assertFalse(result.restored_missing_character)
                self.assertEqual(
                    [
                        (
                            item.position,
                            item.source,
                            item.replacement,
                            item.reason,
                        )
                        for item in result.corrections
                    ],
                    [
                        (
                            position,
                            source,
                            replacement,
                            "normalize_header_digit_to_letter",
                        )
                        for (
                            position,
                            source,
                            replacement,
                        ) in expected_corrections
                    ],
                )

    def test_recovers_country_digits_and_a_missing_header_character(
        self,
    ) -> None:
        cases = (
            ("<K42", "P<KAZ", [(2, "4", "A"), (3, "2", "Z")], 0, "P"),
            ("PK42", "P<KAZ", [(2, "4", "A"), (3, "2", "Z")], 1, "<"),
            ("<4RM", "P<ARM", [(1, "4", "A")], 0, "P"),
            ("P7JK", "PPTJK", [(1, "7", "T")], 0, "P"),
        )
        tail = "4IVAN0V"

        for (
            raw_header,
            expected_header,
            digit_corrections,
            position,
            inserted,
        ) in cases:
            with self.subTest(raw_header=raw_header):
                raw_mrz = raw_header + tail
                result = normalize_passport_mrz_header(raw_mrz)

                self.assertEqual(result.raw_mrz, raw_mrz)
                self.assertEqual(result.raw_window, raw_mrz[:5])
                self.assertEqual(result.normalized_header, expected_header)
                self.assertEqual(result.normalized_mrz, expected_header + tail)
                self.assertEqual(result.consumed_characters, 4)
                self.assertTrue(result.restored_missing_character)
                self.assertEqual(
                    [
                        (
                            item.position,
                            item.source,
                            item.replacement,
                            item.reason,
                        )
                        for item in result.corrections
                    ],
                    [
                        (
                            digit_position,
                            source,
                            replacement,
                            "normalize_header_digit_to_letter",
                        )
                        for (
                            digit_position,
                            source,
                            replacement,
                        ) in digit_corrections
                    ]
                    + [
                        (
                            position,
                            "",
                            inserted,
                            "restore_missing_passport_header_character",
                        )
                    ],
                )

    def test_repairs_complete_passport_prefixes(self) -> None:
        cases = {
            "X<KAZIVANOV": "P<KAZIVANOV",
            "PCKAZIVANOV": "P<KAZIVANOV",
            "XNRUSIVANOV": "PNRUSIVANOV",
            "XPTJKIVANOV": "PPTJKIVANOV",
        }

        for raw_mrz, expected in cases.items():
            with self.subTest(raw_mrz=raw_mrz):
                result = normalize_passport_mrz_header(raw_mrz)

                self.assertEqual(result.normalized_mrz, expected)
                self.assertFalse(result.restored_missing_character)

    def test_restores_one_missing_passport_header_character(self) -> None:
        cases = {
            "<KAZ8IVANOV": "P<KAZ8IVANOV",
            "PKAZ8IVANOV": "P<KAZ8IVANOV",
            "PARMIVANOV": "P<ARMIVANOV",
            "PUZBIVANOV": "P<UZBIVANOV",
            "NRUSIVANOV": "PNRUSIVANOV",
            "PTJKIVANOV": "PPTJKIVANOV",
        }

        for raw_mrz, expected in cases.items():
            with self.subTest(raw_mrz=raw_mrz):
                result = normalize_passport_mrz_header(raw_mrz)

                self.assertEqual(result.normalized_mrz, expected)
                self.assertTrue(result.restored_missing_character)
                self.assertEqual(result.consumed_characters, 4)

    def test_does_not_normalize_first_name_character_after_country(
        self,
    ) -> None:
        result = normalize_passport_mrz_header("<KAZ8IVANOV")

        self.assertEqual(result.normalized_mrz, "P<KAZ8IVANOV")
        self.assertNotIn(
            4,
            {correction.position for correction in result.corrections},
        )

    def test_fuzzy_matches_each_supported_header_with_one_difference(
        self,
    ) -> None:
        cases = (
            ("P<ARX", "P<ARM"),
            ("P<KAX", "P<KAZ"),
            ("P<RUX", "P<RUS"),
            ("PNRUX", "PNRUS"),
            ("P<TJX", "P<TJK"),
            ("PPTJX", "PPTJK"),
            ("P<UZX", "P<UZB"),
        )
        tail = "IVANOV"

        for raw_header, expected_header in cases:
            with self.subTest(raw_header=raw_header):
                result = normalize_passport_mrz_header(raw_header + tail)

                self.assertEqual(result.normalized_header, expected_header)
                self.assertEqual(result.normalized_mrz, expected_header + tail)
                self.assertEqual(result.country, expected_header[2:])
                self.assertEqual(result.consumed_characters, 5)
                self.assertFalse(result.restored_missing_character)
                self.assertEqual(
                    [
                        (
                            item.position,
                            item.source,
                            item.replacement,
                            item.reason,
                        )
                        for item in result.corrections
                    ],
                    [
                        (
                            4,
                            "X",
                            expected_header[4],
                            "repair_passport_header_by_nearest_match",
                        )
                    ],
                )

    def test_fuzzy_budget_is_applied_after_digit_normalization(self) -> None:
        result = normalize_passport_mrz_header("P<K4XIVANOV")

        self.assertEqual(result.normalized_mrz, "P<KAZIVANOV")
        self.assertEqual(
            [
                (item.position, item.source, item.replacement, item.reason)
                for item in result.corrections
            ],
            [
                (3, "4", "A", "normalize_header_digit_to_letter"),
                (4, "X", "Z", "repair_passport_header_by_nearest_match"),
            ],
        )

    def test_rejects_ambiguous_nearest_supported_header(self) -> None:
        for raw_header in ("PXRUS", "PXTJK"):
            with self.subTest(raw_header=raw_header):
                with self.assertRaises(AmbiguousPassportHeaderError):
                    normalize_passport_mrz_header(raw_header + "IVANOV")

    def test_weight_prefers_known_filler_confusion(self) -> None:
        cases = {
            "PCRUSIVANOV": "P<RUSIVANOV",
            "PCTJKIVANOV": "P<TJKIVANOV",
        }

        for raw_mrz, expected in cases.items():
            with self.subTest(raw_mrz=raw_mrz):
                self.assertEqual(
                    normalize_passport_mrz_header(raw_mrz).normalized_mrz,
                    expected,
                )

    def test_rejects_unique_header_candidate_beyond_one_difference(
        self,
    ) -> None:
        for raw_header in ("P<KXX", "QNRUX"):
            with self.subTest(raw_header=raw_header):
                with self.assertRaises(PassportCountryNotFoundError):
                    normalize_passport_mrz_header(raw_header + "IVANOV")

    def test_does_not_combine_missing_character_and_fuzzy_repair(self) -> None:
        for raw_header in ("<ARX", "PARX"):
            with self.subTest(raw_header=raw_header):
                with self.assertRaises(PassportCountryNotFoundError):
                    normalize_passport_mrz_header(raw_header + "IVANOV")

    def test_rejects_ambiguous_russian_prefix(self) -> None:
        for raw_mrz in ("PRUSIVANOV", "PRU5IVANOV"):
            with self.subTest(raw_mrz=raw_mrz):
                with self.assertRaises(AmbiguousPassportHeaderError):
                    normalize_passport_mrz_header(raw_mrz)

    def test_rejects_identity_card_prefix(self) -> None:
        with self.assertRaises(PassportDocumentCodeError):
            normalize_passport_mrz_header("DKAZIVANOV")

    def test_rejects_unknown_country(self) -> None:
        for raw_header in ("P<XXX", "P<4B3", "P<9R7", "<4B3"):
            with self.subTest(raw_header=raw_header):
                with self.assertRaises(PassportCountryNotFoundError):
                    normalize_passport_mrz_header(raw_header + "IVANOV")


if __name__ == "__main__":
    unittest.main()
