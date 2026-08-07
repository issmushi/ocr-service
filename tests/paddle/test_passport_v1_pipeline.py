from __future__ import annotations

import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.core.images.preprocessing import PreparedImage
from app.modules.v1.passport.pipeline import process_passport_image


class FakeModel:
    def __init__(self, results):
        self.results = results
        self.inputs = None

    def predict(self, inputs):
        self.inputs = inputs
        yield from self.results


class PassportV1PipelineTests(unittest.TestCase):
    def test_normalizes_armenian_lines_after_header_recovery(self) -> None:
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        full_raw_line_2 = "44123456784RM9001158N3001156<<<<<<<<<<<<<<06"
        expected_line_2 = "AA12345678ARM9001158M3001156<<<<<<<<<<<<<<06"
        name = "9R190RYAN<<AR4M"
        expected_line_1 = "P<ARMGRIGORYAN<<ARAM".ljust(44, "<")

        for header, missing_tail in product(
            ("P<ARM", "<ARM", "PARM"), (0, 1, 2, 5, 16)
        ):
            with self.subTest(header=header, missing_tail=missing_tail):
                raw_line_1 = header + name.ljust(39, "<")
                raw_line_2 = full_raw_line_2[: 44 - missing_tail]
                recognition = SimpleNamespace(
                    lines=(raw_line_1, raw_line_2),
                    detected_regions_count=2,
                    selected_regions_count=2,
                )
                with (
                    patch(
                        "app.modules.v1.passport.pipeline.prepare_document_image",
                        return_value=PreparedImage(image=image, metadata={}),
                    ),
                    patch(
                        "app.modules.v1.passport.pipeline.recognize_raw_passport_mrz",
                        return_value=recognition,
                    ),
                ):
                    result = process_passport_image(image, ocr=object())

                self.assertEqual(
                    result["mrz"], [expected_line_1, expected_line_2]
                )
                self.assertEqual(result["raw_mrz"], [raw_line_1, raw_line_2])
                header_diagnostics = result["diagnostics"][
                    "header_normalization"
                ]
                self.assertEqual(header_diagnostics["country"], "ARM")
                self.assertEqual(
                    header_diagnostics["restored_missing_character"],
                    len(header) == 4,
                )
                tail_diagnostics = result["diagnostics"][
                    "lower_line_tail_restoration"
                ]
                if missing_tail:
                    self.assertEqual(
                        tail_diagnostics["generated_check_digit_positions"],
                        [44] if missing_tail == 1 else [43, 44],
                    )
                    self.assertEqual(
                        tail_diagnostics["added_fillers"],
                        max(missing_tail - 2, 0),
                    )
                else:
                    self.assertIsNone(tail_diagnostics)
                report = result["diagnostics"]["normalization"]
                self.assertEqual(report["replacements_count"], 9)
                self.assertEqual(
                    report["insertions_count"],
                    missing_tail + (len(header) == 4),
                )
                self.assertEqual(
                    report["corrections_count"],
                    report["replacements_count"] + report["insertions_count"],
                )

    def test_normalizes_kazakh_upper_line_citizenship_and_iin(self) -> None:
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        full_raw_line_2 = "N123456785KA29001011M35010149001O1300123<<82"
        full_expected_line_2 = "N123456785KAZ9001011M3501014900101300123<<82"
        expected_line_1 = "P<KAZNURLANOV<<AZAMAT".ljust(44, "<")

        for (
            header,
            inserted_position,
            inserted_symbol,
        ), missing_tail in product(
            (("<KAZ", 1, "P"), ("PKAZ", 2, "<")), (0, 5)
        ):
            with self.subTest(header=header, missing_tail=missing_tail):
                raw_line_1 = header + "NURLAN0V<<A24MAT<<"
                raw_line_2 = full_raw_line_2[: 44 - missing_tail]
                expected_line_2 = full_expected_line_2[: 44 - missing_tail]
                recognition = SimpleNamespace(
                    lines=(raw_line_1, raw_line_2),
                    detected_regions_count=2,
                    selected_regions_count=2,
                )
                with (
                    patch(
                        "app.modules.v1.passport.pipeline.prepare_document_image",
                        return_value=PreparedImage(image=image, metadata={}),
                    ),
                    patch(
                        "app.modules.v1.passport.pipeline.recognize_raw_passport_mrz",
                        return_value=recognition,
                    ),
                ):
                    result = process_passport_image(image, ocr=object())

                self.assertEqual(
                    result["mrz"], [expected_line_1, expected_line_2]
                )
                self.assertEqual(result["raw_mrz"], [raw_line_1, raw_line_2])
                self.assertIsNone(
                    result["diagnostics"]["lower_line_tail_restoration"]
                )
                report = result["diagnostics"]["normalization"]
                self.assertEqual(report["replacements_count"], 5)
                self.assertEqual(report["insertions_count"], 22)
                self.assertEqual(report["corrections_count"], 27)
                expected_corrections = [
                    (
                        1,
                        inserted_position,
                        None,
                        "Тип документа",
                        "вставка",
                        "",
                        inserted_symbol,
                    ),
                    (1, 12, 11, "Фамилия и имена", "замена", "0", "O"),
                    (1, 17, 16, "Фамилия и имена", "замена", "2", "Z"),
                    (1, 18, 17, "Фамилия и имена", "замена", "4", "A"),
                ]
                expected_corrections.extend(
                    (1, position, None, "Фамилия и имена", "вставка", "", "<")
                    for position in range(24, 45)
                )
                expected_corrections.extend(
                    [
                        (2, 13, 13, "Гражданство", "замена", "2", "Z"),
                        (
                            2,
                            33,
                            33,
                            "Дополнительные данные",
                            "замена",
                            "O",
                            "0",
                        ),
                    ]
                )
                self.assertEqual(
                    [
                        (
                            item["line"],
                            item["position"],
                            item["source_position"],
                            item["field"],
                            item["operation"],
                            item["source"],
                            item["replacement"],
                        )
                        for item in report["corrections"]
                    ],
                    expected_corrections,
                )
                self.assertEqual(
                    report["corrections"][-1]["reason"],
                    "Замена буквы на похожую цифру",
                )

    def test_restores_kazakh_passport_number_initial_after_header_recovery(
        self,
    ) -> None:
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        name = "NURLANOV<<AZAMAT".ljust(39, "<")
        lower_tail = "123456785KAZ9001011M3501014900101300123<<82"

        for header, first in product(
            ("P<KAZ", "<KAZ", "PKAZ"), ("N", "H", "D", "4")
        ):
            with self.subTest(header=header, first=first):
                raw_line_1 = header + name
                raw_line_2 = first + lower_tail if first else ""
                recognition = SimpleNamespace(
                    lines=(raw_line_1, raw_line_2),
                    detected_regions_count=2,
                    selected_regions_count=2,
                )
                with (
                    patch(
                        "app.modules.v1.passport.pipeline.prepare_document_image",
                        return_value=PreparedImage(image=image, metadata={}),
                    ),
                    patch(
                        "app.modules.v1.passport.pipeline.recognize_raw_passport_mrz",
                        return_value=recognition,
                    ),
                ):
                    result = process_passport_image(image, ocr=object())

                expected_line_2 = "N" + lower_tail if first else ""
                self.assertEqual(
                    result["mrz"], ["P<KAZ" + name, expected_line_2]
                )
                self.assertEqual(result["raw_mrz"], [raw_line_1, raw_line_2])
                self.assertEqual(result["mrz"][1][1:], raw_line_2[1:])
                self.assertEqual(len(result["mrz"][1]), len(raw_line_2))
                report = result["diagnostics"]["normalization"]
                expected_replacements = int(bool(first) and first != "N")
                expected_insertions = int(len(header) == 4)
                self.assertEqual(
                    report["replacements_count"], expected_replacements
                )
                self.assertEqual(
                    report["insertions_count"], expected_insertions
                )
                self.assertEqual(
                    report["corrections_count"],
                    expected_replacements + expected_insertions,
                )
                lower_corrections = [
                    item for item in report["corrections"] if item["line"] == 2
                ]
                expected_lower_corrections = (
                    [
                        {
                            "line": 2,
                            "position": 1,
                            "source_position": 1,
                            "field": "Номер паспорта",
                            "operation": "замена",
                            "source": first,
                            "replacement": "N",
                            "reason": (
                                "Восстановление начальной буквы номера "
                                "паспорта Казахстана"
                            ),
                        }
                    ]
                    if expected_replacements
                    else []
                )
                self.assertEqual(lower_corrections, expected_lower_corrections)

    def test_normalizes_kazakh_citizenship_without_changing_other_fields(
        self,
    ) -> None:
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        raw_line_1 = "P<KAZNURLANOV<<AZAMAT".ljust(44, "<")
        number = "N123456785"
        lower_tail = "9001011M3501014900101300123<<82"

        for raw_country, expected_country, changes in (
            ("K42", "KAZ", ((12, "4", "A"), (13, "2", "Z"))),
            ("KAZ", "KAZ", ()),
            ("ARM", "ARM", ()),
        ):
            with self.subTest(citizenship=raw_country):
                raw_line_2 = number + raw_country + lower_tail
                recognition = SimpleNamespace(
                    lines=(raw_line_1, raw_line_2),
                    detected_regions_count=2,
                    selected_regions_count=2,
                )
                with (
                    patch(
                        "app.modules.v1.passport.pipeline.prepare_document_image",
                        return_value=PreparedImage(image=image, metadata={}),
                    ),
                    patch(
                        "app.modules.v1.passport.pipeline.recognize_raw_passport_mrz",
                        return_value=recognition,
                    ),
                ):
                    result = process_passport_image(image, ocr=object())

                self.assertEqual(
                    result["mrz"],
                    [raw_line_1, number + expected_country + lower_tail],
                )
                self.assertEqual(result["raw_mrz"], [raw_line_1, raw_line_2])
                self.assertEqual(
                    result["diagnostics"]["normalization"],
                    {
                        "corrections_count": len(changes),
                        "replacements_count": len(changes),
                        "insertions_count": 0,
                        "issues": [],
                        "corrections": [
                            {
                                "line": 2,
                                "position": position,
                                "source_position": position,
                                "field": "Гражданство",
                                "operation": "замена",
                                "source": source,
                                "replacement": replacement,
                                "reason": (
                                    "Нормализация кода гражданства по списку "
                                    "поддерживаемых значений"
                                ),
                            }
                            for position, source, replacement in changes
                        ],
                    },
                )

    def test_normalizes_kazakh_sex_and_restores_tail_after_complete_iin(
        self,
    ) -> None:
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        raw_line_1 = "P<KAZNURLANOV<<AZAMAT".ljust(44, "<")
        full_line_2 = "N123456785KAZ9001011M3501014900101300123<<82"
        tail_fields = {
            41: "Дополнительные данные",
            42: "Дополнительные данные",
            43: "Контрольная цифра дополнительных данных",
            44: "Общая контрольная цифра",
        }

        for (raw_sex, expected_sex), missing_tail in product(
            (("N", "M"), ("H", "M"), ("E", "F")), range(5)
        ):
            with self.subTest(sex=raw_sex, missing_tail=missing_tail):
                expected_line_2 = (
                    full_line_2[:20] + expected_sex + full_line_2[21:]
                )
                raw_line_2 = (full_line_2[:20] + raw_sex + full_line_2[21:])[
                    : 44 - missing_tail
                ]
                recognition = SimpleNamespace(
                    lines=(raw_line_1, raw_line_2),
                    detected_regions_count=2,
                    selected_regions_count=2,
                )
                with (
                    patch(
                        "app.modules.v1.passport.pipeline.prepare_document_image",
                        return_value=PreparedImage(image=image, metadata={}),
                    ),
                    patch(
                        "app.modules.v1.passport.pipeline.recognize_raw_passport_mrz",
                        return_value=recognition,
                    ),
                ):
                    result = process_passport_image(image, ocr=object())

                self.assertEqual(result["mrz"], [raw_line_1, expected_line_2])
                self.assertEqual(result["raw_mrz"], [raw_line_1, raw_line_2])
                expected_tail_diagnostics = (
                    {
                        "original_length": 44 - missing_tail,
                        "added_fillers": max(missing_tail - 2, 0),
                        "generated_check_digit_positions": (
                            [44] if missing_tail == 1 else [43, 44]
                        ),
                    }
                    if missing_tail
                    else None
                )
                self.assertEqual(
                    result["diagnostics"]["lower_line_tail_restoration"],
                    expected_tail_diagnostics,
                )
                report = result["diagnostics"]["normalization"]
                self.assertEqual(report["replacements_count"], 1)
                self.assertEqual(report["insertions_count"], missing_tail)
                self.assertEqual(report["corrections_count"], missing_tail + 1)
                expected_corrections = [
                    (2, 21, 21, "Пол", "замена", raw_sex, expected_sex)
                ]
                expected_corrections.extend(
                    (
                        2,
                        position,
                        None,
                        tail_fields[position],
                        "вставка",
                        "",
                        expected_line_2[position - 1],
                    )
                    for position in range(45 - missing_tail, 45)
                )
                self.assertEqual(
                    [
                        (
                            item["line"],
                            item["position"],
                            item["source_position"],
                            item["field"],
                            item["operation"],
                            item["source"],
                            item["replacement"],
                        )
                        for item in report["corrections"]
                    ],
                    expected_corrections,
                )

    def test_records_only_missing_kazakh_number_initial_without_shifting_dates(
        self,
    ) -> None:
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        raw_line_1 = "P<KAZNURLANOV<<AZAMAT".ljust(44, "<")
        full_line_2 = "N123456785KAZ9001011M3501014900101300123<<82"

        for length in (44, 39):
            with self.subTest(length=length):
                raw_line_2 = full_line_2[1:length]
                recognition = SimpleNamespace(
                    lines=(raw_line_1, raw_line_2),
                    detected_regions_count=2,
                    selected_regions_count=2,
                )
                with (
                    patch(
                        "app.modules.v1.passport.pipeline.prepare_document_image",
                        return_value=PreparedImage(image=image, metadata={}),
                    ),
                    patch(
                        "app.modules.v1.passport.pipeline.recognize_raw_passport_mrz",
                        return_value=recognition,
                    ),
                ):
                    result = process_passport_image(image, ocr=object())

                self.assertEqual(
                    result["mrz"], [raw_line_1, full_line_2[:length]]
                )
                self.assertEqual(result["raw_mrz"], [raw_line_1, raw_line_2])
                self.assertIsNone(
                    result["diagnostics"]["lower_line_tail_restoration"]
                )
                self.assertEqual(
                    result["diagnostics"]["normalization"],
                    {
                        "corrections_count": 1,
                        "replacements_count": 0,
                        "insertions_count": 1,
                        "issues": [],
                        "corrections": [
                            {
                                "line": 2,
                                "position": 1,
                                "source_position": None,
                                "field": "Номер паспорта",
                                "operation": "вставка",
                                "source": "",
                                "replacement": "N",
                                "reason": (
                                    "Восстановление пропущенной начальной "
                                    "буквы "
                                    "номера паспорта Казахстана"
                                ),
                            }
                        ],
                    },
                )

    def test_replays_corrections_after_insertions_in_both_kazakh_lines(
        self,
    ) -> None:
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        name = "NURLANOV<<AZAMAT".ljust(39, "<")
        raw_line_1 = "<K42" + name
        expected_line_1 = "P<KAZ" + name
        full_raw_line_2 = "N123456785K429001011H3501014900101300123<<82"
        expected_line_2 = "N123456785KAZ9001011M3501014900101300123<<82"
        tail_fields = {
            41: "Дополнительные данные",
            42: "Дополнительные данные",
            43: "Контрольная цифра дополнительных данных",
            44: "Общая контрольная цифра",
        }

        for missing_tail in range(5):
            with self.subTest(missing_tail=missing_tail):
                raw_line_2 = full_raw_line_2[1 : 44 - missing_tail]
                recognition = SimpleNamespace(
                    lines=(raw_line_1, raw_line_2),
                    detected_regions_count=2,
                    selected_regions_count=2,
                )
                with (
                    patch(
                        "app.modules.v1.passport.pipeline.prepare_document_image",
                        return_value=PreparedImage(image=image, metadata={}),
                    ),
                    patch(
                        "app.modules.v1.passport.pipeline.recognize_raw_passport_mrz",
                        return_value=recognition,
                    ),
                ):
                    result = process_passport_image(image, ocr=object())

                self.assertEqual(
                    result["mrz"], [expected_line_1, expected_line_2]
                )
                self.assertEqual(result["raw_mrz"], [raw_line_1, raw_line_2])
                expected_tail_diagnostics = (
                    {
                        "original_length": 43 - missing_tail,
                        "added_fillers": max(missing_tail - 2, 0),
                        "generated_check_digit_positions": (
                            [44] if missing_tail == 1 else [43, 44]
                        ),
                    }
                    if missing_tail
                    else None
                )
                self.assertEqual(
                    result["diagnostics"]["lower_line_tail_restoration"],
                    expected_tail_diagnostics,
                )
                report = result["diagnostics"]["normalization"]
                self.assertEqual(report["replacements_count"], 5)
                self.assertEqual(report["insertions_count"], 2 + missing_tail)
                self.assertEqual(report["corrections_count"], 7 + missing_tail)
                expected_corrections = [
                    (1, 1, None, "Тип документа", "вставка", "", "P"),
                    (1, 4, 3, "Страна выдачи", "замена", "4", "A"),
                    (1, 5, 4, "Страна выдачи", "замена", "2", "Z"),
                    (2, 1, None, "Номер паспорта", "вставка", "", "N"),
                    (2, 12, 11, "Гражданство", "замена", "4", "A"),
                    (2, 13, 12, "Гражданство", "замена", "2", "Z"),
                    (2, 21, 20, "Пол", "замена", "H", "M"),
                ]
                expected_corrections.extend(
                    (
                        2,
                        position,
                        None,
                        tail_fields[position],
                        "вставка",
                        "",
                        expected_line_2[position - 1],
                    )
                    for position in range(45 - missing_tail, 45)
                )
                self.assertEqual(
                    [
                        (
                            item["line"],
                            item["position"],
                            item["source_position"],
                            item["field"],
                            item["operation"],
                            item["source"],
                            item["replacement"],
                        )
                        for item in report["corrections"]
                    ],
                    expected_corrections,
                )
                replay = [list(line) for line in result["raw_mrz"]]
                for item in report["corrections"]:
                    line_index, index = item["line"] - 1, item["position"] - 1
                    if item["operation"] == "вставка":
                        replay[line_index].insert(index, item["replacement"])
                    else:
                        self.assertEqual(
                            result["raw_mrz"][line_index][
                                item["source_position"] - 1
                            ],
                            item["source"],
                        )
                        self.assertEqual(
                            replay[line_index][index], item["source"]
                        )
                        replay[line_index][index] = item["replacement"]
                self.assertEqual(
                    ["".join(line) for line in replay], result["mrz"]
                )

    def test_normalizes_only_ambiguous_russian_internal_name_digits(
        self,
    ) -> None:
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        name = "0152346789<<TEST"
        normalized_name = "OIS2346789<<TEST"
        raw_line_2 = "1235678909RUS9001011M<<<<<<<4100620770120<40"
        cases = list(
            product(
                ("PNRUS", "NRUS"),
                (name, name.ljust(39, "<"), name.ljust(42, "<")),
            )
        )

        for header, name_field in cases:
            with self.subTest(header=header, name_length=len(name_field)):
                raw_line_1 = header + name_field
                recognition = SimpleNamespace(
                    lines=(raw_line_1, raw_line_2),
                    detected_regions_count=2,
                    selected_regions_count=2,
                )
                with (
                    patch(
                        "app.modules.v1.passport.pipeline.prepare_document_image",
                        return_value=PreparedImage(image=image, metadata={}),
                    ),
                    patch(
                        "app.modules.v1.passport.pipeline.recognize_raw_passport_mrz",
                        return_value=recognition,
                    ),
                ):
                    result = process_passport_image(image, ocr=object())

                normalized_name_field = (
                    normalized_name + name_field[len(name) :]
                )
                expected_line_1 = ("PNRUS" + normalized_name_field).ljust(
                    44, "<"
                )
                self.assertEqual(result["mrz"], [expected_line_1, raw_line_2])
                self.assertEqual(result["raw_mrz"], [raw_line_1, raw_line_2])
                expected_corrections = (
                    [
                        (
                            1,
                            1,
                            None,
                            "Тип документа",
                            "вставка",
                            "",
                            "P",
                            (
                                "Восстановление пропущенного символа "
                                "типа документа"
                            ),
                        )
                    ]
                    if header == "NRUS"
                    else []
                )
                expected_corrections.extend(
                    (
                        1,
                        6 + index,
                        len(header) + index + 1,
                        "Фамилия и имена",
                        "замена",
                        source,
                        replacement,
                        "Замена цифры на похожую букву",
                    )
                    for index, (source, replacement) in enumerate(
                        zip("015", "OIS", strict=True)
                    )
                )
                expected_corrections.extend(
                    (
                        1,
                        position,
                        None,
                        "Фамилия и имена",
                        "вставка",
                        "",
                        "<",
                        "Дополнение строки до 44 символов",
                    )
                    for position in range(6 + len(name_field), 45)
                )
                report = result["diagnostics"]["normalization"]
                self.assertEqual(report["replacements_count"], 3)
                self.assertEqual(
                    report["insertions_count"],
                    sum(item[4] == "вставка" for item in expected_corrections),
                )
                self.assertEqual(
                    report["corrections_count"], len(expected_corrections)
                )
                self.assertEqual(report["issues"], [])
                self.assertEqual(
                    [
                        (
                            item["line"],
                            item["position"],
                            item["source_position"],
                            item["field"],
                            item["operation"],
                            item["source"],
                            item["replacement"],
                            item["reason"],
                        )
                        for item in report["corrections"]
                    ],
                    expected_corrections,
                )

    def test_normalizes_russian_internal_lower_line_and_reports_each_change(
        self,
    ) -> None:
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        name_field = "IVANOV<<IVAN<IVANOVI3".ljust(39, "<")
        expected_lower = "4511234569RUS9001158M<<<<<<<5150620770001<88"
        full_raw_lower = "4511234569RU59001158H<<<<<<<5150620770001<88"

        for header, missing_tail in product(
            ("PNRUS", "NRUS", "P<RUS"), range(4)
        ):
            with self.subTest(header=header, missing_tail=missing_tail):
                raw_upper = header + name_field
                raw_lower = full_raw_lower[: 44 - missing_tail]
                recognition = SimpleNamespace(
                    lines=(raw_upper, raw_lower),
                    detected_regions_count=2,
                    selected_regions_count=2,
                )
                with (
                    patch(
                        "app.modules.v1.passport.pipeline.prepare_document_image",
                        return_value=PreparedImage(image=image, metadata={}),
                    ),
                    patch(
                        "app.modules.v1.passport.pipeline.recognize_raw_passport_mrz",
                        return_value=recognition,
                    ),
                ):
                    result = process_passport_image(image, ocr=object())

                self.assertEqual(result["raw_mrz"], [raw_upper, raw_lower])
                report = result["diagnostics"]["normalization"]
                tail = result["diagnostics"]["lower_line_tail_restoration"]
                self.assertEqual(report["issues"], [])
                if header == "P<RUS":
                    self.assertEqual(
                        result["mrz"],
                        [
                            raw_upper.replace("3", "E"),
                            expected_lower[: 44 - missing_tail],
                        ],
                    )
                    self.assertEqual(report["corrections_count"], 3)
                    self.assertEqual(report["replacements_count"], 3)
                    self.assertEqual(report["insertions_count"], 0)
                    self.assertEqual(
                        [item["line"] for item in report["corrections"]],
                        [1, 2, 2],
                    )
                    self.assertIsNone(tail)
                    continue

                self.assertEqual(
                    result["mrz"], ["PNRUS" + name_field, expected_lower]
                )
                expected_changes = [
                    (13, 13, "Гражданство", "замена", "5", "S"),
                    (21, 21, "Пол", "замена", "H", "M"),
                ]
                tail_fields = {
                    42: "Дополнительные данные",
                    43: "Контрольная цифра дополнительных данных",
                    44: "Общая контрольная цифра",
                }
                expected_changes.extend(
                    (
                        position,
                        None,
                        tail_fields[position],
                        "вставка",
                        "",
                        expected_lower[position - 1],
                    )
                    for position in range(len(raw_lower) + 1, 45)
                )
                self.assertEqual(
                    [
                        (
                            item["position"],
                            item["source_position"],
                            item["field"],
                            item["operation"],
                            item["source"],
                            item["replacement"],
                        )
                        for item in report["corrections"]
                        if item["line"] == 2
                    ],
                    expected_changes,
                )
                self.assertEqual(report["replacements_count"], 2)
                self.assertEqual(
                    report["insertions_count"],
                    missing_tail + (header == "NRUS"),
                )
                self.assertEqual(
                    report["corrections_count"], 2 + report["insertions_count"]
                )
                if missing_tail:
                    self.assertEqual(
                        tail,
                        {
                            "original_length": len(raw_lower),
                            "added_fillers": max(missing_tail - 2, 0),
                            "generated_check_digit_positions": (
                                [44] if missing_tail == 1 else [43, 44]
                            ),
                        },
                    )
                else:
                    self.assertIsNone(tail)

    def test_preserves_invalid_russian_lower_line_without_corrections(
        self,
    ) -> None:
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        name = "IVANOV<<IVAN<IVANOVI3"
        lower = "4511234569RU59001158H<<<<<<<5150620770001<"
        full_lower = "4511234569RUS9001158M<<<<<<<5150620770001<88"
        cases = {
            "missing_first_digit": lower[1:],
            "missing_first_two_digits": lower[2:],
            "extra_leading_45_chars": "0" + full_lower,
            "extra_leading_and_lost_tail_44_chars": "0" + full_lower[:-1],
            "unknown_letter_in_number": "X" + lower[1:],
        }
        self.assertEqual(len(cases["extra_leading_45_chars"]), 45)
        self.assertEqual(
            len(cases["extra_leading_and_lost_tail_44_chars"]), 44
        )
        for header, name_field, (case, raw_lower) in product(
            ("PNRUS", "NRUS", "P<RUS"),
            (name, name.ljust(39, "<")),
            cases.items(),
        ):
            with self.subTest(
                header=header, name_length=len(name_field), case=case
            ):
                raw_upper = header + name_field
                recognition = SimpleNamespace(
                    lines=(raw_upper, raw_lower),
                    detected_regions_count=2,
                    selected_regions_count=2,
                )
                with (
                    patch(
                        "app.modules.v1.passport.pipeline.prepare_document_image",
                        return_value=PreparedImage(image=image, metadata={}),
                    ),
                    patch(
                        "app.modules.v1.passport.pipeline.recognize_raw_passport_mrz",
                        return_value=recognition,
                    ),
                ):
                    result = process_passport_image(image, ocr=object())

                expected_upper = (
                    ("PNRUS" + name_field).ljust(44, "<")
                    if header != "P<RUS"
                    else raw_upper.replace("3", "E").ljust(44, "<")
                )
                self.assertEqual(result["raw_mrz"], [raw_upper, raw_lower])
                self.assertEqual(result["mrz"], [expected_upper, raw_lower])
                self.assertIsNone(
                    result["diagnostics"]["lower_line_tail_restoration"]
                )
                report = result["diagnostics"]["normalization"]
                self.assertEqual(
                    report["issues"],
                    [
                        {
                            "line": 2,
                            "code": "invalid_lower_line_structure",
                            "message": "Нарушена структура нижней строки MRZ",
                        }
                    ],
                )
                self.assertTrue(
                    all(item["line"] == 1 for item in report["corrections"])
                )
                expected_replacements = int(header == "P<RUS")
                self.assertEqual(
                    report["replacements_count"], expected_replacements
                )
                expected_insertions = len(expected_upper) - len(raw_upper)
                self.assertEqual(
                    report["insertions_count"], expected_insertions
                )
                self.assertEqual(
                    report["corrections_count"],
                    expected_insertions + expected_replacements,
                )
                self.assertEqual(
                    len(report["corrections"]), report["corrections_count"]
                )

    def test_normalizes_only_russian_international_upper_line(
        self,
    ) -> None:
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        name = "0123456789<<TEST"
        normalized_name = "OIZEASGTBG<<TEST"
        raw_lower = "511234569RU59001158H<<<<<<<5150620770001<88"

        for header, name_field in product(
            ("P<RUS", "<RUS"), (name, name.ljust(39, "<"), name.ljust(42, "<"))
        ):
            with self.subTest(header=header, name_length=len(name_field)):
                raw_upper = header + name_field
                recognition = SimpleNamespace(
                    lines=(raw_upper, raw_lower),
                    detected_regions_count=2,
                    selected_regions_count=2,
                )
                with (
                    patch(
                        "app.modules.v1.passport.pipeline.prepare_document_image",
                        return_value=PreparedImage(image=image, metadata={}),
                    ),
                    patch(
                        "app.modules.v1.passport.pipeline.recognize_raw_passport_mrz",
                        return_value=recognition,
                    ),
                ):
                    result = process_passport_image(image, ocr=object())

                expected_upper = (
                    "P<RUS" + normalized_name + name_field[len(name) :]
                ).ljust(44, "<")
                self.assertEqual(result["mrz"], [expected_upper, raw_lower])
                self.assertEqual(result["raw_mrz"], [raw_upper, raw_lower])
                self.assertIsNone(
                    result["diagnostics"]["lower_line_tail_restoration"]
                )
                expected_corrections = (
                    [(1, 1, None, "Тип документа", "вставка", "", "P")]
                    if header == "<RUS"
                    else []
                )
                expected_corrections.extend(
                    (
                        1,
                        6 + index,
                        len(header) + index + 1,
                        "Фамилия и имена",
                        "замена",
                        source,
                        replacement,
                    )
                    for index, (source, replacement) in enumerate(
                        zip("0123456789", "OIZEASGTBG", strict=True)
                    )
                )
                expected_corrections.extend(
                    (1, position, None, "Фамилия и имена", "вставка", "", "<")
                    for position in range(6 + len(name_field), 45)
                )
                report = result["diagnostics"]["normalization"]
                self.assertEqual(
                    report["issues"],
                    [
                        {
                            "line": 2,
                            "code": "invalid_lower_line_structure",
                            "message": "Нарушена структура нижней строки MRZ",
                        }
                    ],
                )
                self.assertEqual(report["replacements_count"], 10)
                self.assertEqual(
                    report["insertions_count"], len(expected_corrections) - 10
                )
                self.assertEqual(
                    report["corrections_count"], len(expected_corrections)
                )
                self.assertEqual(
                    [
                        (
                            item["line"],
                            item["position"],
                            item["source_position"],
                            item["field"],
                            item["operation"],
                            item["source"],
                            item["replacement"],
                        )
                        for item in report["corrections"]
                    ],
                    expected_corrections,
                )

    def test_normalizes_russian_foreign_lower_fields_and_restores_empty_tail(
        self,
    ) -> None:
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        name_field = "IVANOV<<IVAN".ljust(39, "<")
        full_line_2 = "7512345672RUS9001158M3501151<<<<<<<<<<<<<<08"
        check_fields = {
            43: "Контрольная цифра дополнительных данных",
            44: "Общая контрольная цифра",
        }

        for header, (raw_sex, expected_sex), missing_tail in product(
            ("P<RUS", "<RUS"), (("H", "M"), ("E", "F")), (0, 1, 2, 5, 16)
        ):
            with self.subTest(
                header=header, sex=raw_sex, missing_tail=missing_tail
            ):
                raw_upper = header + name_field
                raw_lower = (
                    full_line_2[:10]
                    + "RU5"
                    + full_line_2[13:20]
                    + raw_sex
                    + full_line_2[21:]
                )[: 44 - missing_tail]
                expected_lower = (
                    full_line_2[:20] + expected_sex + full_line_2[21:]
                )
                recognition = SimpleNamespace(
                    lines=(raw_upper, raw_lower),
                    detected_regions_count=2,
                    selected_regions_count=2,
                )
                with (
                    patch(
                        "app.modules.v1.passport.pipeline.prepare_document_image",
                        return_value=PreparedImage(image=image, metadata={}),
                    ),
                    patch(
                        "app.modules.v1.passport.pipeline.recognize_raw_passport_mrz",
                        return_value=recognition,
                    ),
                ):
                    result = process_passport_image(image, ocr=object())

                self.assertEqual(
                    result["mrz"], ["P<RUS" + name_field, expected_lower]
                )
                self.assertEqual(result["raw_mrz"], [raw_upper, raw_lower])
                expected_tail = (
                    {
                        "original_length": 44 - missing_tail,
                        "added_fillers": max(missing_tail - 2, 0),
                        "generated_check_digit_positions": (
                            [44] if missing_tail == 1 else [43, 44]
                        ),
                    }
                    if missing_tail
                    else None
                )
                self.assertEqual(
                    result["diagnostics"]["lower_line_tail_restoration"],
                    expected_tail,
                )
                expected_corrections = (
                    [(1, 1, None, "Тип документа", "вставка", "", "P")]
                    if header == "<RUS"
                    else []
                )
                expected_corrections.extend(
                    [
                        (2, 13, 13, "Гражданство", "замена", "5", "S"),
                        (2, 21, 21, "Пол", "замена", raw_sex, expected_sex),
                    ]
                )
                expected_corrections.extend(
                    (
                        2,
                        position,
                        None,
                        check_fields.get(position, "Дополнительные данные"),
                        "вставка",
                        "",
                        expected_lower[position - 1],
                    )
                    for position in range(45 - missing_tail, 45)
                )
                report = result["diagnostics"]["normalization"]
                self.assertEqual(report["issues"], [])
                self.assertEqual(report["replacements_count"], 2)
                self.assertEqual(
                    report["insertions_count"],
                    missing_tail + (header == "<RUS"),
                )
                self.assertEqual(
                    report["corrections_count"], len(expected_corrections)
                )
                self.assertEqual(
                    [
                        (
                            item["line"],
                            item["position"],
                            item["source_position"],
                            item["field"],
                            item["operation"],
                            item["source"],
                            item["replacement"],
                        )
                        for item in report["corrections"]
                    ],
                    expected_corrections,
                )

    def _process_recognized_lines(self, upper: str, lower: str):
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        recognition = SimpleNamespace(
            lines=(upper, lower),
            detected_regions_count=2,
            selected_regions_count=2,
        )
        with (
            patch(
                "app.modules.v1.passport.pipeline.prepare_document_image",
                return_value=PreparedImage(image=image, metadata={}),
            ),
            patch(
                "app.modules.v1.passport.pipeline.recognize_raw_passport_mrz",
                return_value=recognition,
            ),
        ):
            return process_passport_image(image, ocr=object())

    def test_normalizes_tajik_lines_after_header_recovery_and_reports_changes(
        self,
    ) -> None:
        name_field = "1VAN0V<<AR4M".ljust(37, "<")
        full_lower = "4012345676TJK9001158M35011513500012345678<00"
        tail_fields = {
            42: "Дополнительные данные",
            43: "Контрольная цифра дополнительных данных",
            44: "Общая контрольная цифра",
        }

        header_cases = (
            ("PPTJK", "PPTJK"),
            ("PTJK", "PPTJK"),
            ("PP7JK", "PPTJK"),
            ("P7JK", "PPTJK"),
            ("P<TJK", "P<TJK"),
            ("<TJK", "P<TJK"),
            ("P<7JK", "P<TJK"),
            ("<7JK", "P<TJK"),
        )
        for (header, normalized_header), (
            raw_sex,
            expected_sex,
        ), missing_tail in product(
            header_cases, (("H", "M"), ("E", "F")), range(4)
        ):
            with self.subTest(
                header=header, sex=raw_sex, missing_tail=missing_tail
            ):
                raw_upper = header + name_field
                expected_upper = (normalized_header + "IVANOV<<ARAM").ljust(
                    44, "<"
                )
                raw_lower = (
                    full_lower[:10]
                    + "7JK"
                    + full_lower[13:20]
                    + raw_sex
                    + full_lower[21:]
                )[: 44 - missing_tail]
                expected_lower = (
                    full_lower[:20] + expected_sex + full_lower[21:]
                )
                result = self._process_recognized_lines(raw_upper, raw_lower)
                self.assertEqual(
                    result["mrz"], [expected_upper, expected_lower]
                )
                self.assertEqual(result["raw_mrz"], [raw_upper, raw_lower])
                self.assertEqual(
                    result["diagnostics"]["lower_line_tail_restoration"],
                    {
                        "original_length": 44 - missing_tail,
                        "added_fillers": max(missing_tail - 2, 0),
                        "generated_check_digit_positions": (
                            [44] if missing_tail == 1 else [43, 44]
                        ),
                    }
                    if missing_tail
                    else None,
                )
                expected_corrections = (
                    [(1, 1, None, "Тип документа", "вставка", "", "P")]
                    if len(header) == 4
                    else []
                )
                if "7" in header:
                    expected_corrections.append(
                        (
                            1,
                            3,
                            header.index("7") + 1,
                            "Страна выдачи",
                            "замена",
                            "7",
                            "T",
                        )
                    )
                expected_corrections.extend(
                    (
                        1,
                        index + 6,
                        len(header) + index + 1,
                        "Фамилия и имена",
                        "замена",
                        source,
                        replacement,
                    )
                    for index, source, replacement in (
                        (0, "1", "I"),
                        (4, "0", "O"),
                        (10, "4", "A"),
                    )
                )
                expected_corrections.extend(
                    [
                        (1, 43, None, "Фамилия и имена", "вставка", "", "<"),
                        (1, 44, None, "Фамилия и имена", "вставка", "", "<"),
                        (2, 11, 11, "Гражданство", "замена", "7", "T"),
                        (2, 21, 21, "Пол", "замена", raw_sex, expected_sex),
                    ]
                )
                expected_corrections.extend(
                    (
                        2,
                        position,
                        None,
                        tail_fields[position],
                        "вставка",
                        "",
                        expected_lower[position - 1],
                    )
                    for position in range(45 - missing_tail, 45)
                )
                report = result["diagnostics"]["normalization"]
                self.assertEqual(report["issues"], [])
                self.assertEqual(
                    report["replacements_count"], 5 + ("7" in header)
                )
                self.assertEqual(
                    report["insertions_count"],
                    2 + (len(header) == 4) + missing_tail,
                )
                self.assertEqual(
                    report["corrections_count"], len(expected_corrections)
                )
                self.assertEqual(
                    [
                        (
                            item["line"],
                            item["position"],
                            item["source_position"],
                            item["field"],
                            item["operation"],
                            item["source"],
                            item["replacement"],
                        )
                        for item in report["corrections"]
                    ],
                    expected_corrections,
                )

    def test_tajik_guard_preserves_invalid_lower_for_both_prefixes(
        self,
    ) -> None:
        full_lower = "40123456767JK9001158H35011513500012345678<00"
        cases = [
            ("missing_digit", full_lower[1:]),
            ("extra_digit", "0" + full_lower),
            ("unknown_letter_in_number", "X" + full_lower[1:]),
        ]
        for header, (case, raw_lower) in product(("PPTJK", "P<TJK"), cases):
            with self.subTest(header=header, case=case):
                raw_upper = header + "TEST0V<<TEST"
                result = self._process_recognized_lines(raw_upper, raw_lower)
                expected_upper = (header + "TESTOV<<TEST").ljust(44, "<")
                self.assertEqual(result["mrz"], [expected_upper, raw_lower])
                self.assertEqual(result["raw_mrz"], [raw_upper, raw_lower])
                self.assertIsNone(
                    result["diagnostics"]["lower_line_tail_restoration"]
                )
                report = result["diagnostics"]["normalization"]
                self.assertEqual(
                    report["issues"],
                    [
                        {
                            "line": 2,
                            "code": "invalid_lower_line_structure",
                            "message": "Нарушена структура нижней строки MRZ",
                        }
                    ],
                )
                self.assertTrue(
                    all(item["line"] == 1 for item in report["corrections"])
                )
                self.assertEqual(report["replacements_count"], 1)
                self.assertEqual(
                    report["insertions_count"],
                    len(expected_upper) - len(raw_upper),
                )
                self.assertEqual(
                    report["corrections_count"],
                    report["insertions_count"] + report["replacements_count"],
                )

    def test_normalizes_uzbek_lower_fields_and_final_check_digits(
        self,
    ) -> None:
        name_field = "TESTOV<<IGOR".ljust(39, "<")
        full_lower = "AA12345678UZB9001158M35011511234567890123456"
        check_fields = {
            43: "Контрольная цифра дополнительных данных",
            44: "Общая контрольная цифра",
        }
        nationality_cases = (
            ("UZB", ()),
            ("UZ8", ((13, "8", "B"),)),
            ("U2B", ((12, "2", "Z"),)),
            ("U28", ((12, "2", "Z"), (13, "8", "B"))),
            ("U08", ((12, "0", "Z"), (13, "8", "B"))),
        )
        for header, (nationality, nationality_changes), (
            raw_sex,
            expected_sex,
        ), missing_tail in product(
            ("P<UZB", "<UZB"),
            nationality_cases,
            (("N", "M"), ("H", "M"), ("E", "F")),
            range(3),
        ):
            with self.subTest(
                header=header,
                nationality=nationality,
                sex=raw_sex,
                missing_tail=missing_tail,
            ):
                raw_upper = header + name_field
                raw_lower = (
                    "44"
                    + full_lower[2:10]
                    + nationality
                    + full_lower[13:20]
                    + raw_sex
                    + full_lower[21:]
                )[: 44 - missing_tail]
                expected_lower = (
                    full_lower[:20] + expected_sex + full_lower[21:]
                )
                result = self._process_recognized_lines(raw_upper, raw_lower)
                self.assertEqual(
                    result["mrz"], ["P<UZB" + name_field, expected_lower]
                )
                self.assertEqual(result["raw_mrz"], [raw_upper, raw_lower])
                self.assertEqual(
                    result["diagnostics"]["lower_line_tail_restoration"],
                    {
                        "original_length": 44 - missing_tail,
                        "added_fillers": 0,
                        "generated_check_digit_positions": [44]
                        if missing_tail == 1
                        else [43, 44],
                    }
                    if missing_tail
                    else None,
                )
                expected_corrections = (
                    [(1, 1, None, "Тип документа", "вставка", "", "P")]
                    if header == "<UZB"
                    else []
                )
                expected_corrections.extend(
                    [
                        (2, 1, 1, "Номер паспорта", "замена", "4", "A"),
                        (2, 2, 2, "Номер паспорта", "замена", "4", "A"),
                    ]
                )
                expected_corrections.extend(
                    (
                        2,
                        position,
                        position,
                        "Гражданство",
                        "замена",
                        source,
                        replacement,
                    )
                    for position, source, replacement in nationality_changes
                )
                expected_corrections.append(
                    (2, 21, 21, "Пол", "замена", raw_sex, expected_sex)
                )
                expected_corrections.extend(
                    (
                        2,
                        position,
                        None,
                        check_fields[position],
                        "вставка",
                        "",
                        expected_lower[position - 1],
                    )
                    for position in range(45 - missing_tail, 45)
                )
                report = result["diagnostics"]["normalization"]
                self.assertEqual(report["issues"], [])
                self.assertEqual(
                    report["replacements_count"], 3 + len(nationality_changes)
                )
                self.assertEqual(
                    report["insertions_count"],
                    missing_tail + (header == "<UZB"),
                )
                self.assertEqual(
                    report["corrections_count"], len(expected_corrections)
                )
                self.assertEqual(
                    [
                        (
                            item["line"],
                            item["position"],
                            item["source_position"],
                            item["field"],
                            item["operation"],
                            item["source"],
                            item["replacement"],
                        )
                        for item in report["corrections"]
                    ],
                    expected_corrections,
                )

    def test_uzbek_guard_preserves_raw_lower_and_reports_structure_issue(
        self,
    ) -> None:
        raw_upper = "P<UZ8TEST0V<<TEST"
        full_lower = "4412345678UZB9001158H35011511234567890123456"
        for raw_lower in (
            full_lower[1:],
            "4" + full_lower,
            full_lower[:3] + "X" + full_lower[4:],
        ):
            with self.subTest(lower=raw_lower):
                result = self._process_recognized_lines(raw_upper, raw_lower)
                expected_upper = "P<UZBTESTOV<<TEST".ljust(44, "<")
                self.assertEqual(result["mrz"], [expected_upper, raw_lower])
                self.assertEqual(result["raw_mrz"], [raw_upper, raw_lower])
                self.assertIsNone(
                    result["diagnostics"]["lower_line_tail_restoration"]
                )
                report = result["diagnostics"]["normalization"]
                self.assertEqual(
                    report["issues"],
                    [
                        {
                            "line": 2,
                            "code": "invalid_lower_line_structure",
                            "message": "Нарушена структура нижней строки MRZ",
                        }
                    ],
                )
                self.assertTrue(
                    all(item["line"] == 1 for item in report["corrections"])
                )
                self.assertEqual(report["replacements_count"], 2)
                self.assertEqual(
                    report["insertions_count"], 44 - len(raw_upper)
                )
                self.assertEqual(
                    report["corrections_count"], len(report["corrections"])
                )

    def test_uzbek_unrecognized_nationality_does_not_block_lower_normalization(
        self,
    ) -> None:
        raw_upper = "P<UZ8TEST0V<<TEST"
        full_lower = "4412345678UZB9001158H35011511234567890123456"
        raw_lower = (full_lower[:10] + "UXX" + full_lower[13:])[:42]
        expected_lower = "AA12345678UXX9001158M35011511234567890123456"

        result = self._process_recognized_lines(raw_upper, raw_lower)

        expected_upper = "P<UZBTESTOV<<TEST".ljust(44, "<")
        self.assertEqual(result["mrz"], [expected_upper, expected_lower])
        self.assertEqual(result["raw_mrz"], [raw_upper, raw_lower])
        self.assertEqual(
            result["diagnostics"]["lower_line_tail_restoration"],
            {
                "original_length": 42,
                "added_fillers": 0,
                "generated_check_digit_positions": [43, 44],
            },
        )
        report = result["diagnostics"]["normalization"]
        self.assertEqual(
            report["issues"],
            [
                {
                    "line": 2,
                    "code": "unrecognized_nationality",
                    "message": (
                        "Код гражданства не сопоставлен с поддерживаемым "
                        "значением"
                    ),
                }
            ],
        )
        self.assertEqual(
            [
                (
                    item["position"],
                    item["operation"],
                    item["source"],
                    item["replacement"],
                )
                for item in report["corrections"]
                if item["line"] == 2
            ],
            [
                (1, "замена", "4", "A"),
                (2, "замена", "4", "A"),
                (21, "замена", "H", "M"),
                (43, "вставка", "", "5"),
                (44, "вставка", "", "6"),
            ],
        )

    def test_normalizes_uzbek_upper_line_and_preserves_lower_line(
        self,
    ) -> None:
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        polygons = [
            np.array([[10, y], [290, y], [290, y + 20], [10, y + 20]])
            for y in (10, 60, 110, 160)
        ]
        detector = FakeModel([{"dt_polys": polygons, "input_img": image}])
        raw_line_1 = "P<UZ8TESTOV<<IG0R<<<<<<<<<<<<<<<<<<<<<<<<<<"
        raw_line_2 = "AA12345678UZB9001158M35011511234567890123456"
        recognizer = FakeModel(
            [
                {"rec_text": raw_line_1, "rec_score": 0.91},
                {"rec_text": raw_line_2, "rec_score": 0.92},
                {"rec_text": "RUSSIA", "rec_score": 0.9},
                {"rec_text": "POC", "rec_score": 0.9},
            ]
        )
        pipeline = SimpleNamespace(
            text_det_model=detector,
            text_rec_model=recognizer,
        )
        ocr = SimpleNamespace(
            paddlex_pipeline=SimpleNamespace(_pipeline=pipeline)
        )

        with patch(
            "app.modules.v1.passport.pipeline.prepare_document_image",
            return_value=PreparedImage(image=image, metadata={}),
        ):
            result = process_passport_image(image, ocr=ocr)

        normalized_line_1 = "P<UZBTESTOV<<IGOR".ljust(44, "<")
        self.assertEqual(result["mrz"], [normalized_line_1, raw_line_2])
        self.assertEqual(result["raw_mrz"], [raw_line_1, raw_line_2])
        report = result["diagnostics"]["normalization"]
        self.assertEqual(report["issues"], [])
        self.assertEqual(report["replacements_count"], 2)
        self.assertEqual(report["insertions_count"], 44 - len(raw_line_1))
        expected_corrections = [
            (1, 5, 5, "Страна выдачи", "замена", "8", "B"),
            (1, 16, 16, "Фамилия и имена", "замена", "0", "O"),
        ]
        expected_corrections.extend(
            (1, position, None, "Фамилия и имена", "вставка", "", "<")
            for position in range(len(raw_line_1) + 1, 45)
        )
        self.assertEqual(
            report["corrections_count"], len(expected_corrections)
        )
        self.assertEqual(
            [
                (
                    item["line"],
                    item["position"],
                    item["source_position"],
                    item["field"],
                    item["operation"],
                    item["source"],
                    item["replacement"],
                )
                for item in report["corrections"]
            ],
            expected_corrections,
        )
        self.assertEqual(
            result["diagnostics"]["header_normalization"]["country"],
            "UZB",
        )
        self.assertEqual(result["diagnostics"]["detected_regions_count"], 4)
        self.assertEqual(result["diagnostics"]["selected_regions_count"], 4)
        self.assertEqual(len(recognizer.inputs), 4)


if __name__ == "__main__":
    unittest.main()
