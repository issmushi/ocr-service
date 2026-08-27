from __future__ import annotations

import pytest

from app.modules.v1.idcard.parsing import parse_idcard_mrz
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


@pytest.mark.parametrize(
    "lines,expected",
    (
        (
            KAZ_TD1,
            {
                "issuing_country": "KAZ",
                "document_number": "123456789",
                "surname": "TESTOV",
                "name": "TEST",
                "personal_number": "900115300001",
                "date_of_birth": "15.01.1990",
                "expiry_date": "15.01.2035",
            },
        ),
        (
            KAZ_TD2,
            {
                "issuing_country": "KAZ",
                "document_number": "012345678",
                "surname": "TESTOV",
                "name": "TEST",
                "personal_number": "900115300001",
                "date_of_birth": "15.01.1990",
                "expiry_date": "15.01.2035",
            },
        ),
        (
            KGZ_TD1,
            {
                "issuing_country": "KGZ",
                "document_number": "ID1234567",
                "surname": "TESTOV",
                "name": "TEST",
                "personal_number": "90011530000001",
                "date_of_birth": "15.01.1990",
                "expiry_date": "15.01.2035",
            },
        ),
    ),
)
def test_supported_country_parsers_return_common_public_fields(
    lines, expected
):
    data = parse_idcard_mrz(lines)
    assert {key: data[key] for key in expected} == expected
    assert data["document_type"] == "identity_card"
    assert data["document_code"] == "ID"
    assert data["nationality"] == expected["issuing_country"]
    assert data["sex"] == "МУЖ."
    assert data["mrz_valid"] is True
    assert all(
        data[field] is None
        for field in (
            "document_series",
            "patronymic",
            "issue_date",
            "department_code",
        )
    )


def test_kyrgyzstan_i_filler_type_is_supported():
    data = parse_idcard_mrz(["I<KGZ" + KGZ_TD1[0][5:], *KGZ_TD1[1:]])
    assert data["document_code"] == "I"
    assert data["issuing_country"] == "KGZ"
    assert data["mrz_valid"] is True


@pytest.mark.parametrize("lines", (KAZ_TD1, KAZ_TD2, KGZ_TD1))
def test_wrong_observed_check_digit_yields_false(lines):
    dirty = list(lines)
    dirty[1] = dirty[1][:-1] + "9"
    assert parse_idcard_mrz(dirty)["mrz_valid"] is False


@pytest.mark.parametrize("prefix", ("IDUSA", "PNRUS", "P<KAZ", "1DKAZ"))
def test_unknown_header_never_falls_back_to_another_document_parser(prefix):
    with pytest.raises(IdcardMrzFormatError):
        parse_idcard_mrz([prefix + KAZ_TD1[0][5:], *KAZ_TD1[1:]])


def test_incomplete_fields_are_none_without_legacy_recovery():
    data = parse_idcard_mrz([KAZ_TD1[0][:26], KAZ_TD1[1][:10], KAZ_TD1[2]])
    assert data["document_number"] == "123456789"
    assert data["personal_number"] is None
    assert data["date_of_birth"] == "15.01.1990"
    assert data["expiry_date"] is None
    assert data["nationality"] is None
    assert data["mrz_valid"] is None


def test_partial_td2_iin_is_not_invented():
    data = parse_idcard_mrz([KAZ_TD2[0], KAZ_TD2[1][:33]])
    assert data["personal_number"] is None
    assert data["document_number"] == "012345678"
    assert data["mrz_valid"] is None


def test_parser_does_not_normalize_raw_numeric_characters():
    first = KAZ_TD1[0][:5] + "I" + KAZ_TD1[0][6:]
    data = parse_idcard_mrz([first, *KAZ_TD1[1:]])
    assert data["document_number"] is None
    assert data["mrz_valid"] is False


def test_kyrgyzstan_single_letter_patronymic_is_preserved():
    data = parse_idcard_mrz([*KGZ_TD1[:2], "TESTOV<<TEST<A".ljust(30, "<")])
    assert data["patronymic"] == "A"


def test_names_keep_single_fillers_inside_a_compound_surname():
    data = parse_idcard_mrz(
        [*KAZ_TD1[:2], "TEST<SURNAME<<TEST".ljust(30, "<")]
    )
    assert data["surname"] == "TEST SURNAME"
    assert data["name"] == "TEST"


def test_no_sex_marker_is_exposed_as_none():
    second = KGZ_TD1[1][:7] + "<" + KGZ_TD1[1][8:]
    data = parse_idcard_mrz([KGZ_TD1[0], second, KGZ_TD1[2]])
    assert data["sex"] is None
    assert data["mrz_valid"] is True
