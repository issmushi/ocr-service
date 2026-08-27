from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.modules.v1.idcard.pipeline import normalize_idcard_mrz
from app.modules.v1.idcard.profiles import IdcardMrzFormatError

KAZ_TD1 = (
    "IDKAZ1234567897900115300001<<<",
    "9001158M3501151KAZ<<<<<<<<<<<0",
    "TESTOV<<TEST<<<<<<<<<<<<<<<<<<",
)
KAZ_TD2 = (
    "IDKAZTESTOV<<TEST<<<<<<<<<<<<<<<<<<<",
    "0123456784KAZ9001158M3501151300001<6",
)
KGZ_TD1 = (
    "IDKGZID1234567390011530000001<",
    "9001158M3501151KGZ<<<<<<<<<<<2",
    "TESTOV<<TEST<<<<<<<<<<<<<<<<<<",
)


def _changed(value: str, replacements: dict[int, str]) -> str:
    chars = list(value)
    for index, replacement in replacements.items():
        chars[index] = replacement
    return "".join(chars)


@pytest.mark.parametrize("lines", (KAZ_TD1, KAZ_TD2, KGZ_TD1))
def test_valid_mrz_is_unchanged_and_has_an_empty_report(lines):
    result = normalize_idcard_mrz(lines)
    assert result["mrz"] == list(lines)
    assert result["diagnostics"]["normalization"] == {
        "corrections_count": 0,
        "replacements_count": 0,
        "insertions_count": 0,
        "corrections": [],
        "issues": [],
    }


@pytest.mark.parametrize("lines", (KAZ_TD1, KAZ_TD2, KGZ_TD1))
def test_compacts_multiline_ocr_and_selects_longest_rows_in_source_order(
    lines,
):
    text = "\n".join(" \t".join(line.lower()) for line in lines)
    result = normalize_idcard_mrz(["короткий шум", text, "RUSSIA", "RF"])
    assert result["mrz"] == list(lines)


def test_kazakhstan_td1_corrects_only_the_corresponding_fields():
    dirty = [
        _changed(KAZ_TD1[0], {0: "1", 3: "4", 5: "I", 14: "T", 17: "O"}),
        _changed(
            KAZ_TD1[1],
            {1: "O", 6: "B", 7: "N", 8: "E", 14: "I", 16: "4", 29: "O"},
        ),
        _changed(KAZ_TD1[2], {4: "0"}),
    ]
    result = normalize_idcard_mrz(dirty)
    assert result["mrz"] == list(KAZ_TD1)
    report = result["diagnostics"]["normalization"]
    assert report["corrections_count"] == 13
    assert report["insertions_count"] == 0
    assert {item["field"] for item in report["corrections"]} == {
        "Тип документа",
        "Страна выдачи",
        "Номер документа",
        "ИИН",
        "Контрольная цифра номера документа",
        "Дата рождения",
        "Контрольная цифра даты рождения",
        "Пол",
        "Дата окончания срока действия",
        "Контрольная цифра срока действия",
        "Гражданство",
        "Общая контрольная цифра",
        "Фамилия и имена",
    }
    assert all(
        item["source_position"] == item["position"]
        for item in report["corrections"]
    )


def test_kazakhstan_td2_corrects_numeric_suffix_and_preserves_leading_zero():
    dirty = [
        _changed(KAZ_TD2[0], {3: "4", 9: "0"}),
        _changed(
            KAZ_TD2[1],
            {
                0: "O",
                9: "A",
                11: "4",
                14: "O",
                20: "H",
                21: "E",
                28: "E",
                35: "G",
            },
        ),
    ]
    result = normalize_idcard_mrz(dirty)
    assert result["mrz"] == list(KAZ_TD2)
    assert result["mrz"][1].startswith("0")
    corrections = result["diagnostics"]["normalization"]["corrections"]
    assert any(item["field"] == "ИИН" for item in corrections)


def test_kyrgyzstan_corrects_alphabetic_prefix_numeric_pin_and_country():
    dirty = [
        _changed(KGZ_TD1[0], {4: "2", 5: "1", 7: "I", 14: "E", 16: "O"}),
        _changed(
            KGZ_TD1[1],
            {1: "O", 6: "B", 7: "H", 8: "E", 14: "I", 17: "2", 29: "Z"},
        ),
        _changed(KGZ_TD1[2], {4: "0"}),
    ]
    result = normalize_idcard_mrz(dirty)
    assert result["mrz"] == list(KGZ_TD1)
    assert result["mrz"][0].startswith("IDKGZID")
    corrections = result["diagnostics"]["normalization"]["corrections"]
    assert any(item["field"] == "ПИН" for item in corrections)


@pytest.mark.parametrize(
    "prefix,expected,inserted",
    (
        ("DKAZ", "IDKAZ", 1),
        ("IKAZ", "IDKAZ", 2),
        ("DKGZ", "IDKGZ", 1),
        ("<KGZ", "I<KGZ", 1),
    ),
)
def test_restored_header_retains_every_number_character_and_source_position(
    prefix, expected, inserted
):
    base = KAZ_TD1 if expected.endswith("KAZ") else KGZ_TD1
    result = normalize_idcard_mrz([prefix + base[0][5:], *base[1:]])
    assert result["mrz"][0] == expected + base[0][5:]
    correction = result["diagnostics"]["normalization"]["corrections"][0]
    assert correction["position"] == inserted
    assert correction["operation"] == "вставка"
    assert correction["source_position"] is None


def test_header_insertion_keeps_later_replacement_source_positions():
    first = "DKGZ1D" + KGZ_TD1[0][7:]
    result = normalize_idcard_mrz([first, *KGZ_TD1[1:]])
    corrections = result["diagnostics"]["normalization"]["corrections"]
    assert corrections[1]["position"] == 6
    assert corrections[1]["source_position"] == 5
    assert corrections[1]["source"] == "1"
    assert corrections[1]["replacement"] == "I"


@pytest.mark.parametrize("prefix", ("IKGZ", "IDKXZ", "IDUSA", "PNRUS"))
def test_unknown_or_ambiguous_header_does_not_choose_a_country(prefix):
    with pytest.raises(IdcardMrzFormatError):
        normalize_idcard_mrz([prefix + KGZ_TD1[0][5:], *KGZ_TD1[1:]])


def test_nationality_is_corrected_independently_from_issuing_country():
    lower = KAZ_TD1[1][:15] + "KG2" + KAZ_TD1[1][18:]
    result = normalize_idcard_mrz([KAZ_TD1[0], lower, KAZ_TD1[2]])
    assert result["mrz"][0].startswith("IDKAZ")
    assert result["mrz"][1][15:18] == "KGZ"


def test_ambiguous_nationality_is_not_selected_arbitrarily():
    lower = KAZ_TD1[1][:15] + "KXZ" + KAZ_TD1[1][18:]
    result = normalize_idcard_mrz([KAZ_TD1[0], lower, KAZ_TD1[2]])
    assert result["mrz"][1][15:18] == "KXZ"


@pytest.mark.parametrize("length", range(27, 31))
def test_kazakhstan_first_row_restores_only_fillers_after_complete_iin(length):
    result = normalize_idcard_mrz([KAZ_TD1[0][:length], *KAZ_TD1[1:]])
    assert result["mrz"] == list(KAZ_TD1)


@pytest.mark.parametrize("length", (29, 30))
def test_kyrgyzstan_first_row_restores_only_filler_after_complete_pin(length):
    result = normalize_idcard_mrz([KGZ_TD1[0][:length], *KGZ_TD1[1:]])
    assert result["mrz"] == list(KGZ_TD1)


@pytest.mark.parametrize("lines,first_length", ((KAZ_TD1, 26), (KGZ_TD1, 28)))
def test_missing_personal_number_digit_is_not_replaced_with_fillers(
    lines, first_length
):
    result = normalize_idcard_mrz(
        [lines[0][:first_length], lines[1][:18], lines[2]]
    )
    assert result["mrz"][0] == lines[0][:first_length]
    assert result["mrz"][1] == lines[1][:18]


@pytest.mark.parametrize("lines", (KAZ_TD1, KGZ_TD1))
@pytest.mark.parametrize("length", range(18, 30))
def test_td1_second_row_restores_tail_and_only_missing_final_check_digit(
    lines, length
):
    result = normalize_idcard_mrz([lines[0], lines[1][:length], lines[2]])
    assert result["mrz"] == list(lines)
    correction = result["diagnostics"]["normalization"]["corrections"][-1]
    assert correction["field"] == "Общая контрольная цифра"
    assert correction["position"] == 30
    assert correction["source_position"] is None


@pytest.mark.parametrize("length", (34, 35))
def test_td2_second_row_restores_tail_after_complete_iin_suffix(length):
    assert normalize_idcard_mrz([KAZ_TD2[0], KAZ_TD2[1][:length]])[
        "mrz"
    ] == list(KAZ_TD2)


def test_td2_does_not_restore_missing_iin_digits():
    result = normalize_idcard_mrz([KAZ_TD2[0], KAZ_TD2[1][:33]])
    assert result["mrz"][1] == KAZ_TD2[1][:33]


@pytest.mark.parametrize("lines", (KAZ_TD1, KAZ_TD2, KGZ_TD1))
def test_existing_wrong_final_digit_is_preserved(lines):
    dirty = list(lines)
    dirty[1] = dirty[1][:-1] + "9"
    assert normalize_idcard_mrz(dirty)["mrz"] == dirty


@pytest.mark.parametrize("lines", (KAZ_TD1, KAZ_TD2, KGZ_TD1))
def test_disabled_normalization_only_compacts_and_selects_rows(lines):
    dirty = list(lines)
    dirty[1] = dirty[1][:-3]
    dirty[-1] = dirty[-1].replace("O", "0").lower()
    result = normalize_idcard_mrz(["short", *dirty], normalize=False)
    assert result["mrz"] == [line.upper() for line in dirty]
    assert result["raw_mrz"] == result["mrz"]
    assert result["diagnostics"]["normalization"]["corrections_count"] == 0


def test_disabled_normalization_does_not_repair_a_header():
    with pytest.raises(IdcardMrzFormatError):
        normalize_idcard_mrz(
            ["1DKAZ" + KAZ_TD1[0][5:], *KAZ_TD1[1:]], normalize=False
        )


def test_missing_third_td1_row_does_not_select_td2():
    with pytest.raises(IdcardMrzFormatError, match="Неполная MRZ"):
        normalize_idcard_mrz(KAZ_TD1[:2])


def test_td2_name_role_takes_priority_over_numeric_lookalike_letters():
    first = "IDKAZTESTTESTI<<NAME".ljust(36, "<")
    result = normalize_idcard_mrz([first, KAZ_TD2[1]])
    assert result["diagnostics"]["format"] == "TD2"
    assert result["mrz"] == [first, KAZ_TD2[1]]


def test_single_td2_row_is_rejected_before_response_serialization():
    with pytest.raises(IdcardMrzFormatError, match="Неполная MRZ"):
        normalize_idcard_mrz(KAZ_TD2[:1])


@pytest.mark.parametrize("normalize", (True, False))
def test_unknown_td1_number_character_does_not_switch_to_td2(normalize):
    first = "IDKAZ123X567897900101300123<<<"
    with pytest.raises(IdcardMrzFormatError):
        normalize_idcard_mrz([first, *KAZ_TD1[1:]], normalize=normalize)


def test_local_idcard_references_are_unchanged_and_idempotent():
    dataset = Path(__file__).resolve().parents[1] / "dataset"
    paths = sorted(dataset.glob("*/id_card/*/mrz.json"))
    if not paths:
        pytest.skip("Local document reference dataset is not present")
    for path in paths:
        if path.parents[2].name not in {"kaz", "kgz"}:
            continue
        lines = json.loads(path.read_text())["mrz"]
        result = normalize_idcard_mrz(lines)
        assert result["mrz"] == lines, str(path)
        assert (
            normalize_idcard_mrz(result["mrz"])["diagnostics"][
                "normalization"
            ]["corrections_count"]
            == 0
        )
