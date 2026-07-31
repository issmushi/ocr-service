from __future__ import annotations

import unittest

from app.modules.v1.passport.normalization import normalize_passport_mrz_header
from app.modules.v1.passport.normalization.countries.armenia import (
    normalize_armenian_mrz_lower_line,
    normalize_armenian_mrz_upper_line,
)
from app.modules.v1.passport.normalization.diagnostics import (
    build_mrz_normalization_report,
)

LOWER_LINE = "AA12345678ARM9001158M3001156<<<<<<<<<<<<<<06"
UPPER_LINE = "P<ARMPETROSYAN<<ARAM".ljust(44, "<")


class PassportNormalizationReportTests(unittest.TestCase):
    def report(self, upper: str, lower: str = LOWER_LINE):
        raw = [upper, lower]
        header = normalize_passport_mrz_header(upper)
        normalized = [header.normalized_mrz, lower]
        if header.country == "ARM":
            normalized = [
                normalize_armenian_mrz_upper_line(normalized[0]),
                normalize_armenian_mrz_lower_line(lower),
            ]
        report = build_mrz_normalization_report(raw, normalized, header=header)

        # Every report must reproduce the MRZ, including shifted names.
        replay = [list(line) for line in raw]
        for correction in report["corrections"]:
            line = correction["line"] - 1
            index = correction["position"] - 1
            self.assertEqual(
                normalized[line][index], correction["replacement"]
            )
            if correction["operation"] == "вставка":
                self.assertIsNone(correction["source_position"])
                self.assertEqual(correction["source"], "")
                replay[line].insert(index, correction["replacement"])
            else:
                self.assertEqual(
                    raw[line][correction["source_position"] - 1],
                    correction["source"],
                )
                self.assertEqual(replay[line][index], correction["source"])
                replay[line][index] = correction["replacement"]
        self.assertEqual(["".join(line) for line in replay], normalized)
        self.assertEqual(
            report["corrections_count"], len(report["corrections"])
        )
        self.assertEqual(report["issues"], [])
        return report

    def test_preserves_source_positions_after_either_document_code_insertion(
        self,
    ):
        for prefix, inserted_position, inserted_character in (
            ("<ARM", 1, "P"),
            ("PARM", 2, "<"),
        ):
            with self.subTest(prefix=prefix):
                report = self.report(prefix + "8" + "<" * 38)
                inserted, name = report["corrections"]
                self.assertEqual(report["insertions_count"], 1)
                self.assertEqual(report["replacements_count"], 1)
                self.assertEqual(inserted["position"], inserted_position)
                self.assertEqual(inserted["replacement"], inserted_character)
                self.assertEqual(inserted["field"], "Тип документа")
                self.assertEqual(name["position"], 6)
                self.assertEqual(name["source_position"], 5)
                self.assertEqual(name["field"], "Фамилия и имена")
                self.assertEqual(
                    (name["source"], name["replacement"]), ("8", "B")
                )

    def test_assigns_issuing_country_by_final_position_after_insertion(self):
        report = self.report("<aRM8" + "<" * 38)
        country = report["corrections"][1]
        self.assertEqual(country["field"], "Страна выдачи")
        self.assertEqual(country["position"], 3)
        self.assertEqual(country["source_position"], 2)

    def test_collapses_intermediate_document_code_substitutions(self):
        for raw_code, source, replacement, position in (
            ("0<ARM", "0", "P", 1),
            ("PcARM", "c", "<", 2),
        ):
            with self.subTest(raw_code=raw_code):
                report = self.report(raw_code + "<" * 39)
                self.assertEqual(report["corrections_count"], 1)
                correction = report["corrections"][0]
                self.assertEqual(correction["source"], source)
                self.assertEqual(correction["replacement"], replacement)
                self.assertEqual(correction["position"], position)

    def test_uses_known_insertion_for_repeated_tajik_document_code_letter(
        self,
    ):
        report = self.report("PTJK" + "<" * 39)
        self.assertEqual(report["corrections_count"], 1)
        self.assertEqual(report["corrections"][0]["position"], 1)
        self.assertEqual(report["corrections"][0]["replacement"], "P")

    def test_records_each_appended_filler_and_generated_check_digit(self):
        report = self.report(UPPER_LINE, LOWER_LINE[:28])
        self.assertEqual(report["insertions_count"], 16)
        self.assertEqual(report["replacements_count"], 0)
        self.assertEqual(
            [item["position"] for item in report["corrections"]],
            list(range(29, 45)),
        )
        self.assertEqual(
            [item["field"] for item in report["corrections"]],
            ["Дополнительные данные"] * 14
            + [
                "Контрольная цифра дополнительных данных",
                "Общая контрольная цифра",
            ],
        )

    def test_records_upper_padding_in_the_name_field(self):
        upper = "P<ARMPETROSYAN<<ARAM"
        report = self.report(upper)
        self.assertEqual(report["insertions_count"], 44 - len(upper))
        self.assertTrue(
            all(
                item["field"] == "Фамилия и имена"
                and item["replacement"] == "<"
                for item in report["corrections"]
            )
        )

    def test_distinguishes_passport_number_nationality_and_sex(self):
        lower = "44123456784RM9001158E3001156<<<<<<<<<<<<<<06"
        report = self.report(UPPER_LINE, lower)
        self.assertEqual(report["replacements_count"], 4)
        self.assertEqual(
            [
                (item["position"], item["field"])
                for item in report["corrections"]
            ],
            [
                (1, "Номер паспорта"),
                (2, "Номер паспорта"),
                (11, "Гражданство"),
                (21, "Пол"),
            ],
        )

    def test_reports_only_missing_composite_digit_with_optional_filler(
        self,
    ):
        report = self.report(UPPER_LINE, LOWER_LINE[:42] + "<")
        self.assertEqual(report["corrections_count"], 1)
        self.assertEqual(report["corrections"][0]["position"], 44)

    def test_does_not_report_an_unsuccessful_tail_repair(self):
        report = self.report(UPPER_LINE, LOWER_LINE[:42] + "9")
        self.assertEqual(report["corrections"], [])

    def test_unchanged_mrz_has_zero_counts(self):
        self.assertEqual(
            self.report(UPPER_LINE),
            {
                "corrections_count": 0,
                "replacements_count": 0,
                "insertions_count": 0,
                "corrections": [],
                "issues": [],
            },
        )


if __name__ == "__main__":
    unittest.main()
