from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from app.core.images.preprocessing import PreparedImage
from app.modules.v1.passport.parsing import parse_normalized_passport_mrz
from app.modules.v1.passport.pipeline import (
    normalize_passport_mrz,
    normalize_passport_mrz_from_text_result,
    process_passport_image,
    process_passport_image_with_mrz_scanner,
)

# Synthetic fixtures shared with the country normalizer regressions. The last
# value is the shortest lower line whose missing suffix is recoverable.
PASSPORTS = (
    ("P<ARM", "AA12345678ARM9001158M3001156<<<<<<<<<<<<<<06", 28),
    ("P<KAZ", "N123456785KAZ9001011M3501014900101300123<<82", 40),
    ("PNRUS", "4511234569RUS9001158M<<<<<<<5150620770001<88", 41),
    ("P<RUS", "7512345672RUS9001158M3501151<<<<<<<<<<<<<<08", 28),
    ("P<TJK", "4012345676TJK9001158M35011513500012345678<00", 41),
    ("PPTJK", "4012345676TJK9001158M35011513500012345678<00", 41),
    ("P<UZB", "AA12345678UZB9001158M35011511234567890123456", 42),
)
NUMERIC_LOOKALIKES = str.maketrans("012345678", "OIZEASGTB")
NUMERIC_REASON = "Замена буквы на похожую цифру"
NATIONALITY_REASON = (
    "Нормализация кода гражданства по списку поддерживаемых значений"
)


def upper_line(header: str) -> str:
    return (header + "TESTOV<<TEST").ljust(44, "<")


def noisy_lower_line(
    header: str, lower: str, *, lowercase: bool = False
) -> str:
    """Make noisy OCR input while preserving alphabetic MRZ fields."""
    number_start = (
        2 if header in {"P<ARM", "P<UZB"} else int(header == "P<KAZ")
    )
    numeric_positions = set(range(number_start, 10))
    numeric_positions.update(range(13, 20))
    numeric_positions.update(range(21, 28))
    numeric_positions.update(range(42, 44))
    if header != "P<ARM":
        numeric_positions.update(range(28, 42))
    characters = list(lower)
    for index in numeric_positions:
        if index < len(characters):
            value = characters[index].translate(NUMERIC_LOOKALIKES)
            characters[index] = value.lower() if lowercase else value
    return "".join(characters)


def assert_journal_replays_raw(result: dict) -> None:
    report = result["diagnostics"]["normalization"]
    replayed = [list(line) for line in result["raw_mrz"]]
    for correction in report["corrections"]:
        line_index = correction["line"] - 1
        index = correction["position"] - 1
        if correction["operation"] == "вставка":
            assert correction["source_position"] is None
            assert correction["source"] == ""
            replayed[line_index].insert(index, correction["replacement"])
        else:
            source_index = correction["source_position"] - 1
            assert (
                result["raw_mrz"][line_index][source_index]
                == correction["source"]
            )
            assert replayed[line_index][index] == correction["source"]
            replayed[line_index][index] = correction["replacement"]
    assert ["".join(line) for line in replayed] == result["mrz"]
    assert report["corrections_count"] == len(report["corrections"])
    assert report["corrections_count"] == (
        report["replacements_count"] + report["insertions_count"]
    )


@pytest.mark.parametrize("header,lower,_minimum", PASSPORTS)
@pytest.mark.parametrize("lowercase", (False, True))
def test_full_numeric_fields_normalize_and_report_original_symbols(
    header: str,
    lower: str,
    _minimum: int,
    lowercase: bool,
) -> None:
    upper = upper_line(header)
    noisy = noisy_lower_line(header, lower, lowercase=lowercase)
    canonical_noisy = noisy.upper()
    result = normalize_passport_mrz((upper, noisy))

    assert result["mrz"] == [upper, lower]
    assert result["raw_mrz"] == [upper, canonical_noisy]
    assert result["diagnostics"]["lower_line_tail_restoration"] is None
    report = result["diagnostics"]["normalization"]
    assert report["issues"] == []
    assert report["insertions_count"] == 0
    assert [
        (
            item["line"],
            item["position"],
            item["source_position"],
            item["source"],
            item["replacement"],
            item["reason"],
        )
        for item in report["corrections"]
    ] == [
        (2, index + 1, index + 1, source, digit, NUMERIC_REASON)
        for index, (source, digit) in enumerate(
            zip(canonical_noisy, lower, strict=True)
        )
        if source != digit
    ]
    assert_journal_replays_raw(result)


@pytest.mark.parametrize("header,lower,_minimum", PASSPORTS)
def test_optional_data_is_numeric_except_for_armenia(
    header: str,
    lower: str,
    _minimum: int,
) -> None:
    optional = "OIZEASGTB<9OIZ"
    for value in (optional, optional.lower()):
        raw = lower[:28] + value + "99"
        expected_optional = (
            value.upper() if header == "P<ARM" else "012345678<9012"
        )
        expected = lower[:28] + expected_optional + "99"
        result = normalize_passport_mrz((upper_line(header), raw))
        assert result["mrz"][1] == expected
        assert result["diagnostics"]["lower_line_tail_restoration"] is None
        assert_journal_replays_raw(result)


@pytest.mark.parametrize("header,lower,minimum", PASSPORTS)
def test_corrects_fields_before_restoring_each_identifiable_suffix(
    header: str,
    lower: str,
    minimum: int,
) -> None:
    upper = upper_line(header)
    noisy = noisy_lower_line(header, lower, lowercase=True)
    for length in range(minimum, 44):
        result = normalize_passport_mrz((upper, noisy[:length]))
        assert result["mrz"] == [upper, lower], (header, length)
        assert result["diagnostics"]["lower_line_tail_restoration"] == {
            "original_length": length,
            "added_fillers": max(42 - length, 0),
            "generated_check_digit_positions": [44]
            if length == 43
            else [43, 44],
        }
        report = result["diagnostics"]["normalization"]
        assert report["issues"] == []
        assert report["insertions_count"] == 44 - length
        if length == 43:
            observed = next(
                item
                for item in report["corrections"]
                if item["position"] == 43
            )
            assert observed["operation"] == "замена"
            assert observed["source"] == noisy[42].upper()
            assert observed["replacement"] == lower[42]
            assert observed["reason"] == NUMERIC_REASON
        assert_journal_replays_raw(result)


@pytest.mark.parametrize("header,lower,minimum", PASSPORTS)
def test_numeric_corrections_do_not_invent_missing_required_data(
    header: str,
    lower: str,
    minimum: int,
) -> None:
    truncated = lower[: minimum - 1]
    raw = noisy_lower_line(header, truncated)
    result = normalize_passport_mrz((upper_line(header), raw))
    assert result["mrz"][1] == truncated
    assert result["diagnostics"]["lower_line_tail_restoration"] is None
    assert result["diagnostics"]["normalization"]["insertions_count"] == 0
    assert_journal_replays_raw(result)


@pytest.mark.parametrize("header,lower,_minimum", PASSPORTS)
def test_wrong_observed_optional_check_blocks_final_check_generation(
    header: str,
    lower: str,
    _minimum: int,
) -> None:
    wrong_digit = str((int(lower[42]) + 1) % 9)
    expected = lower[:42] + wrong_digit
    noisy_prefix = noisy_lower_line(header, lower[:42])
    for observed in (
        wrong_digit,
        wrong_digit.translate(NUMERIC_LOOKALIKES).lower(),
    ):
        result = normalize_passport_mrz(
            (upper_line(header), noisy_prefix + observed)
        )
        assert result["mrz"][1] == expected
        assert result["diagnostics"]["lower_line_tail_restoration"] is None
        assert result["diagnostics"]["normalization"]["insertions_count"] == 0
        assert_journal_replays_raw(result)


@pytest.mark.parametrize("header,lower,_minimum", PASSPORTS)
def test_existing_numeric_check_digits_are_never_recalculated(
    header: str,
    lower: str,
    _minimum: int,
) -> None:
    damaged = list(lower)
    for position in (9, 19, 27, 42, 43):
        if damaged[position].isdigit():
            damaged[position] = str((int(damaged[position]) + 1) % 10)
    raw = "".join(damaged)
    result = normalize_passport_mrz((upper_line(header), raw))
    assert result["mrz"][1] == raw
    assert result["diagnostics"]["normalization"]["corrections"] == []


@pytest.mark.parametrize("header,lower,_minimum", PASSPORTS)
def test_numeric_map_does_not_replace_country_sex_or_number_prefix_rules(
    header: str,
    lower: str,
    _minimum: int,
) -> None:
    countries = {
        "ARM": "4RM",
        "KAZ": "K42",
        "RUS": "RU5",
        "TJK": "7JK",
        "UZB": "U28",
    }
    raw = list(noisy_lower_line(header, lower))
    raw[10:13] = countries[lower[10:13]]
    raw[20] = "E"
    expected = lower[:20] + "F" + lower[21:]
    if header in {"P<ARM", "P<UZB"}:
        raw[:2] = "01"
        expected = "OI" + expected[2:]
    elif header == "P<KAZ":
        raw[0] = "4"
    result = normalize_passport_mrz((upper_line(header), "".join(raw)))
    assert result["mrz"][1] == expected
    assert result["diagnostics"]["normalization"]["issues"] == []
    assert_journal_replays_raw(result)


@pytest.mark.parametrize(
    ("header", "lower", "raw_nationality", "nationality"),
    (
        (PASSPORTS[0][0], PASSPORTS[0][1], "ARX", "ARM"),
        (PASSPORTS[1][0], PASSPORTS[1][1], "KAX", "KAZ"),
        (PASSPORTS[2][0], PASSPORTS[2][1], "RUB", "RUS"),
        (PASSPORTS[4][0], PASSPORTS[4][1], "TJX", "TJK"),
        (PASSPORTS[6][0], PASSPORTS[6][1], "UZX", "UZB"),
    ),
)
def test_fuzzy_normalizes_each_allowlisted_nationality(
    header: str,
    lower: str,
    raw_nationality: str,
    nationality: str,
) -> None:
    raw_lower = lower[:10] + raw_nationality + lower[13:]
    result = normalize_passport_mrz((upper_line(header), raw_lower))

    assert result["mrz"] == [upper_line(header), lower]
    report = result["diagnostics"]["normalization"]
    assert report["issues"] == []
    assert report["corrections_count"] == 1
    assert report["replacements_count"] == 1
    assert report["insertions_count"] == 0
    assert [
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
    ] == [
        (
            2,
            13,
            13,
            "Гражданство",
            "замена",
            raw_nationality[2],
            nationality[2],
            NATIONALITY_REASON,
        )
    ]
    assert_journal_replays_raw(result)


@pytest.mark.parametrize("header,lower,_minimum", PASSPORTS)
def test_supported_nationality_is_independent_of_issuing_country(
    header: str,
    lower: str,
    _minimum: int,
) -> None:
    nationality = "KAZ" if lower[10:13] == "ARM" else "ARM"
    expected_lower = lower[:10] + nationality + lower[13:]

    result = normalize_passport_mrz((upper_line(header), expected_lower))

    assert result["mrz"] == [upper_line(header), expected_lower]
    report = result["diagnostics"]["normalization"]
    assert report["issues"] == []
    assert report["corrections"] == []
    parsed = parse_normalized_passport_mrz(result["mrz"])
    assert parsed["nationality"] == nationality
    assert parsed["passport_number"] is not None
    assert parsed["mrz_valid"] is True


def test_fuzzy_header_and_nationality_are_both_recorded_in_journal() -> None:
    header, lower, _minimum = PASSPORTS[0]
    raw_upper = upper_line(header)[:4] + "X" + upper_line(header)[5:]
    raw_lower = lower[:12] + "X" + lower[13:]

    result = normalize_passport_mrz((raw_upper, raw_lower))

    assert result["mrz"] == [upper_line(header), lower]
    report = result["diagnostics"]["normalization"]
    assert report["issues"] == []
    assert [
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
    ] == [
        (1, 5, 5, "Страна выдачи", "замена", "X", "M"),
        (2, 13, 13, "Гражданство", "замена", "X", "M"),
    ]
    assert_journal_replays_raw(result)


@pytest.mark.parametrize(
    ("header", "lower"),
    (
        (PASSPORTS[2][0], PASSPORTS[2][1]),
        (PASSPORTS[3][0], PASSPORTS[3][1]),
    ),
)
def test_russian_numeric_fields_continue_after_rub_matches_rus(
    header: str,
    lower: str,
) -> None:
    raw = list(lower)
    raw[2] = "I"
    raw[10:13] = "RUB"
    raw[14] = "O"
    raw_lower = "".join(raw)

    result = normalize_passport_mrz((upper_line(header), raw_lower))

    assert result["mrz"] == [upper_line(header), lower]
    report = result["diagnostics"]["normalization"]
    assert report["issues"] == []
    assert [
        (
            item["position"],
            item["source_position"],
            item["field"],
            item["source"],
            item["replacement"],
            item["reason"],
        )
        for item in report["corrections"]
        if item["line"] == 2
    ] == [
        (3, 3, "Номер паспорта", "I", "1", NUMERIC_REASON),
        (13, 13, "Гражданство", "B", "S", NATIONALITY_REASON),
        (15, 15, "Дата рождения", "O", "0", NUMERIC_REASON),
    ]
    parsed = parse_normalized_passport_mrz(result["mrz"])
    assert parsed["nationality"] == "RUS"
    assert parsed["passport_number"] is not None
    assert parsed["date_of_birth"] is not None
    assert_journal_replays_raw(result)


@pytest.mark.parametrize(
    ("header", "lower"),
    (
        (PASSPORTS[2][0], PASSPORTS[2][1]),
        (PASSPORTS[3][0], PASSPORTS[3][1]),
    ),
)
def test_unrecognized_nationality_does_not_block_numeric_normalization(
    header: str,
    lower: str,
) -> None:
    raw = list(lower)
    raw[2] = "I"
    raw[10:13] = "RXX"
    raw[14] = "O"
    raw_lower = "".join(raw)

    result = normalize_passport_mrz((upper_line(header), raw_lower))

    expected_lower = lower[:10] + "RXX" + lower[13:]
    assert result["mrz"] == [upper_line(header), expected_lower]
    report = result["diagnostics"]["normalization"]
    assert report["issues"] == [
        {
            "line": 2,
            "code": "unrecognized_nationality",
            "message": (
                "Код гражданства не сопоставлен с поддерживаемым значением"
            ),
        }
    ]
    assert [
        (
            item["position"],
            item["field"],
            item["source"],
            item["replacement"],
            item["reason"],
        )
        for item in report["corrections"]
        if item["line"] == 2
    ] == [
        (3, "Номер паспорта", "I", "1", NUMERIC_REASON),
        (15, "Дата рождения", "O", "0", NUMERIC_REASON),
    ]
    parsed = parse_normalized_passport_mrz(result["mrz"])
    assert parsed["nationality"] is None
    assert parsed["passport_number"] is None
    assert parsed["mrz_valid"] is None
    assert_journal_replays_raw(result)


@pytest.mark.parametrize(
    "raw_lower",
    (
        "01000000074RU",
        "01000000074RUS900101IM3501014<<<<<<<<<<<<<<0",
    ),
)
def test_shifted_russian_number_is_not_reinterpreted_as_arm_nationality(
    raw_lower: str,
) -> None:
    upper = upper_line("P<RUS")

    result = normalize_passport_mrz((upper, raw_lower))

    assert result["raw_mrz"] == [upper, raw_lower]
    assert result["mrz"] == [upper, raw_lower]
    assert result["diagnostics"]["lower_line_tail_restoration"] is None
    report = result["diagnostics"]["normalization"]
    assert report["issues"] == [
        {
            "line": 2,
            "code": "invalid_lower_line_structure",
            "message": "Нарушена структура нижней строки MRZ",
        }
    ]
    assert report["corrections"] == []
    assert report["corrections_count"] == 0
    assert report["replacements_count"] == 0
    assert report["insertions_count"] == 0


def test_digit_then_fuzzy_nationality_uses_semantic_journal_reason() -> None:
    header, lower, _minimum = PASSPORTS[2]
    raw_lower = lower[:10] + "RU8" + lower[13:]

    result = normalize_passport_mrz((upper_line(header), raw_lower))

    assert result["mrz"] == [upper_line(header), lower]
    nationality_correction = next(
        item
        for item in result["diagnostics"]["normalization"]["corrections"]
        if item["line"] == 2 and item["field"] == "Гражданство"
    )
    assert (
        nationality_correction["position"],
        nationality_correction["source"],
        nationality_correction["replacement"],
        nationality_correction["reason"],
    ) == (13, "8", "S", NATIONALITY_REASON)
    assert_journal_replays_raw(result)


def test_kazakh_aligned_prefix_is_replaced_with_foreign_nationality() -> None:
    header, lower, _minimum = PASSPORTS[1]
    raw_lower = "4" + lower[1:10] + "RUS" + lower[13:]
    expected_lower = "N" + raw_lower[1:]

    result = normalize_passport_mrz((upper_line(header), raw_lower))

    assert result["mrz"] == [upper_line(header), expected_lower]
    assert len(result["mrz"][1]) == 44
    report = result["diagnostics"]["normalization"]
    assert report["issues"] == []
    assert report["insertions_count"] == 0
    assert report["replacements_count"] == 1
    assert [
        (
            item["position"],
            item["source_position"],
            item["operation"],
            item["source"],
            item["replacement"],
        )
        for item in report["corrections"]
        if item["line"] == 2
    ] == [(1, 1, "замена", "4", "N")]
    assert_journal_replays_raw(result)


@pytest.mark.parametrize("length", (40, 42, 43, 44))
def test_kazakh_missing_n_keeps_numeric_journal_source_positions(
    length: int,
) -> None:
    header, lower, _minimum = PASSPORTS[1]
    raw_upper = "<K42" + upper_line(header)[5:]
    raw_lower = noisy_lower_line(header, lower, lowercase=True)[1:length]
    result = normalize_passport_mrz((raw_upper, raw_lower))
    assert result["mrz"] == [upper_line(header), lower]
    report = result["diagnostics"]["normalization"]
    assert report["issues"] == []
    initial = next(item for item in report["corrections"] if item["line"] == 2)
    assert (
        initial["position"],
        initial["source_position"],
        initial["replacement"],
    ) == (1, None, "N")
    replacements = [
        item
        for item in report["corrections"]
        if item["line"] == 2 and item["operation"] == "замена"
    ]
    assert replacements
    for correction in replacements:
        assert correction["source_position"] == correction["position"] - 1
        assert correction["reason"] == NUMERIC_REASON
    assert_journal_replays_raw(result)


@pytest.mark.parametrize("header,lower,_minimum", PASSPORTS)
def test_all_engine_entrypoints_share_numeric_normalization(
    header: str,
    lower: str,
    _minimum: int,
) -> None:
    image = np.zeros((20, 50, 3), dtype=np.uint8)
    raw_lines = (
        upper_line(header),
        noisy_lower_line(header, lower, lowercase=True),
    )
    recognition = SimpleNamespace(
        lines=raw_lines, detected_regions_count=2, selected_regions_count=2
    )
    expected = normalize_passport_mrz(raw_lines)
    with (
        patch(
            "app.modules.v1.passport.pipeline.prepare_document_image",
            return_value=PreparedImage(image=image, metadata={}),
        ),
        patch(
            "app.modules.v1.passport.pipeline.recognize_raw_passport_mrz",
            return_value=recognition,
        ),
        patch(
            "app.modules.v1.passport.pipeline.recognize_mrz",
            return_value=recognition,
        ),
    ):
        results = [
            process_passport_image(image, ocr=object()),
            process_passport_image_with_mrz_scanner(image, scanner=object()),
        ]
    results.extend(
        normalize_passport_mrz_from_text_result(
            {"items": [{"text": "\n".join(raw_lines)}]},
            engine=engine,
        )
        for engine in ("qwen3-vl-4b", "qwen3-vl-8b", "russian-ocr")
    )
    assert expected["mrz"] == [upper_line(header), lower]
    for result in results:
        assert result["mrz"] == expected["mrz"]
        assert result["raw_mrz"] == expected["raw_mrz"]
        assert (
            result["diagnostics"]["normalization"]
            == expected["diagnostics"]["normalization"]
        )
