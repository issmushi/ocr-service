from __future__ import annotations

import pytest

from app.modules.v1.passport.normalization.header import (
    SUPPORTED_PASSPORT_HEADERS,
)
from app.modules.v1.passport.parsing import (
    PASSPORT_MRZ_PARSERS,
    parse_normalized_passport_mrz,
)

LOWER_DERIVED_FIELDS = (
    "sex",
    "date_of_birth",
    "passport_series",
    "passport_number",
    "nationality",
    "issue_date",
    "expiry_date",
    "department_code",
    "personal_number",
    "mrz_valid",
)

SUPPORTED_PARSER_CASES = (
    (
        "PNRUSTESTOV<<TEST<TESTOVI3".ljust(44, "<"),
        "4511234569RUS9001158M<<<<<<<5150620770001<88",
        {
            "passport_type": "internal",
            "document_code": "PN",
            "issuing_country": "RUS",
            "surname": "ТЕСТОВ",
            "name": "ТЕСТ",
            "patronymic": "ТЕСТОВИЧ",
        },
    ),
    (
        "P<RUSIVANOV<<IVAN".ljust(44, "<"),
        "7512345672RUS9001011M3501014<<<<<<<<<<<<<<02",
        {
            "passport_type": "international",
            "document_code": "P",
            "issuing_country": "RUS",
            "surname": "IVANOV",
            "name": "IVAN",
            "patronymic": None,
        },
    ),
    (
        "P<ARMPETROSYAN<<ANI".ljust(44, "<"),
        "AM12345674ARM9001011F3501014<<<<<<<<<<<<<<08",
        {
            "passport_type": "international",
            "document_code": "P",
            "issuing_country": "ARM",
            "surname": "PETROSYAN",
            "name": "ANI",
            "patronymic": None,
        },
    ),
    (
        "P<KAZNURLANOV<<AZAMAT".ljust(44, "<"),
        "N123456785KAZ9001011M3501014900101300123<<82",
        {
            "passport_type": "international",
            "document_code": "P",
            "issuing_country": "KAZ",
            "surname": "NURLANOV",
            "name": "AZAMAT",
            "patronymic": None,
        },
    ),
    (
        "P<TJKRAHMONOV<<FARID".ljust(44, "<"),
        "4012345676TJK9001158M35011513500012345678<00",
        {
            "passport_type": "international",
            "document_code": "P",
            "issuing_country": "TJK",
            "surname": "RAHMONOV",
            "name": "FARID",
            "patronymic": None,
        },
    ),
    (
        "PPTJKSAIDOVA<<ZUHRO".ljust(44, "<"),
        "4012345676TJK9001158M35011513500012345678<00",
        {
            "passport_type": "international",
            "document_code": "PP",
            "issuing_country": "TJK",
            "surname": "SAIDOVA",
            "name": "ZUHRO",
            "patronymic": None,
        },
    ),
    (
        "P<UZBKARIMOV<<AZIZ<ANVAROVICH".ljust(44, "<"),
        "AA12345678UZB9001011M35010143900101300123450",
        {
            "passport_type": "international",
            "document_code": "P",
            "issuing_country": "UZB",
            "surname": "KARIMOV",
            "name": "AZIZ",
            "patronymic": "ANVAROVICH",
        },
    ),
)


def test_header_normalizer_and_parser_support_the_same_headers() -> None:
    assert set(SUPPORTED_PASSPORT_HEADERS) == set(PASSPORT_MRZ_PARSERS)


@pytest.mark.parametrize(
    ("mrz", "expected"),
    [
        (
            [
                "PNRUSTESTOV<<TEST<TESTOVI3<<<<<<<<<<<<<<<<<<",
                "4511234569RUS9001158M<<<<<<<5150620770001<88",
            ],
            {
                "passport_type": "internal",
                "document_code": "PN",
                "issuing_country": "RUS",
                "surname": "ТЕСТОВ",
                "name": "ТЕСТ",
                "patronymic": "ТЕСТОВИЧ",
                "sex": "МУЖ.",
                "date_of_birth": "15.01.1990",
                "passport_series": "45 15",
                "passport_number": "123456",
                "nationality": "RUS",
                "issue_date": "20.06.2015",
                "expiry_date": None,
                "department_code": "770-001",
                "personal_number": None,
                "mrz_valid": True,
            },
        ),
        (
            [
                "P<RUSIVANOV<<IVAN<<<<<<<<<<<<<<<<<<<<<<<<<<<",
                "7512345672RUS9001011M3501014<<<<<<<<<<<<<<02",
            ],
            {
                "passport_type": "international",
                "document_code": "P",
                "issuing_country": "RUS",
                "surname": "IVANOV",
                "name": "IVAN",
                "patronymic": None,
                "sex": "МУЖ.",
                "date_of_birth": "01.01.1990",
                "passport_series": None,
                "passport_number": "751234567",
                "nationality": "RUS",
                "issue_date": "01.01.2025",
                "expiry_date": "01.01.2035",
                "department_code": None,
                "personal_number": None,
                "mrz_valid": True,
            },
        ),
        (
            [
                "P<ARMPETROSYAN<<ANI<<<<<<<<<<<<<<<<<<<<<<<<<",
                "AM12345674ARM9001011F3501014<<<<<<<<<<<<<<08",
            ],
            {
                "passport_type": "international",
                "document_code": "P",
                "issuing_country": "ARM",
                "surname": "PETROSYAN",
                "name": "ANI",
                "patronymic": None,
                "sex": "ЖЕН.",
                "date_of_birth": "01.01.1990",
                "passport_series": None,
                "passport_number": "AM1234567",
                "nationality": "ARM",
                "issue_date": "01.01.2025",
                "expiry_date": "01.01.2035",
                "department_code": None,
                "personal_number": None,
                "mrz_valid": True,
            },
        ),
        (
            [
                "P<KAZNURLANOV<<AZAMAT<<<<<<<<<<<<<<<<<<<<<<<",
                "N123456785KAZ9001011M3501014900101300123<<82",
            ],
            {
                "passport_type": "international",
                "document_code": "P",
                "issuing_country": "KAZ",
                "surname": "NURLANOV",
                "name": "AZAMAT",
                "patronymic": None,
                "sex": "МУЖ.",
                "date_of_birth": "01.01.1990",
                "passport_series": None,
                "passport_number": "N12345678",
                "nationality": "KAZ",
                "issue_date": "02.01.2025",
                "expiry_date": "01.01.2035",
                "department_code": None,
                "personal_number": "900101300123",
                "mrz_valid": True,
            },
        ),
        (
            [
                "PPTJKSAIDOVA<<ZUHRO<<<<<<<<<<<<<<<<<<<<<<<<<",
                "4012345676TJK9001158M35011513500012345678<00",
            ],
            {
                "passport_type": "international",
                "document_code": "PP",
                "issuing_country": "TJK",
                "surname": "SAIDOVA",
                "name": "ZUHRO",
                "patronymic": None,
                "sex": "МУЖ.",
                "date_of_birth": "15.01.1990",
                "passport_series": None,
                "passport_number": "401234567",
                "nationality": "TJK",
                "issue_date": "16.01.2025",
                "expiry_date": "15.01.2035",
                "department_code": None,
                "personal_number": "3500012345678",
                "mrz_valid": True,
            },
        ),
        (
            [
                "P<UZBKARIMOV<<AZIZ<ANVAROVICH<<<<<<<<<<<<<<<",
                "AA12345678UZB9001011M35010143900101300123450",
            ],
            {
                "passport_type": "international",
                "document_code": "P",
                "issuing_country": "UZB",
                "surname": "KARIMOV",
                "name": "AZIZ",
                "patronymic": "ANVAROVICH",
                "sex": "МУЖ.",
                "date_of_birth": "01.01.1990",
                "passport_series": None,
                "passport_number": "AA1234567",
                "nationality": "UZB",
                "issue_date": "02.01.2025",
                "expiry_date": "01.01.2035",
                "department_code": None,
                "personal_number": "39001013001234",
                "mrz_valid": True,
            },
        ),
    ],
)
def test_parses_supported_normalized_passport_mrz(
    mrz: list[str],
    expected: dict[str, object],
) -> None:
    assert parse_normalized_passport_mrz(mrz) == expected


@pytest.mark.parametrize(
    ("upper", "lower", "expected_header"),
    SUPPORTED_PARSER_CASES,
    ids=("PNRUS", "P<RUS", "P<ARM", "P<KAZ", "P<TJK", "PPTJK", "P<UZB"),
)
def test_parser_returns_supported_nationality_independently_of_issuer(
    upper: str,
    lower: str,
    expected_header: dict[str, object],
) -> None:
    original_nationality = lower[10:13]
    nationality = "KAZ" if original_nationality == "ARM" else "ARM"
    modified_lower = lower[:10] + nationality + lower[13:]

    result = parse_normalized_passport_mrz([upper, modified_lower])

    assert result["issuing_country"] == expected_header["issuing_country"]
    assert result["nationality"] == nationality
    assert result["passport_number"] is not None
    assert result["date_of_birth"] is not None
    assert result["mrz_valid"] is True


def test_keeps_compound_surname_in_the_surname_field() -> None:
    result = parse_normalized_passport_mrz(
        [
            "P<RUSVAN<DER<MEER<<ANNA<MARIA<<<<<<<<<<<<<<<<<<",
            "7512345672RUS9001011F3501014<<<<<<<<<<<<<<02",
        ]
    )

    assert result["surname"] == "VAN DER MEER"
    assert result["name"] == "ANNA"
    assert result["patronymic"] == "MARIA"


def test_calculates_uzbek_issue_date_from_valid_synthetic_mrz() -> None:
    result = parse_normalized_passport_mrz(
        [
            "P<UZBTESTOV<<TEST<TESTOVICH".ljust(44, "<"),
            "AA12345678UZB9001158M36010179001153000000124",
        ]
    )

    assert result["expiry_date"] == "01.01.2036"
    assert result["issue_date"] == "02.01.2026"
    assert result["mrz_valid"] is True


def test_calculates_kazakh_issue_date_from_valid_synthetic_mrz() -> None:
    result = parse_normalized_passport_mrz(
        [
            "P<KAZTESTOV<<TEST".ljust(44, "<"),
            "N123456785KAZ9001158M3601017900115300001<<06",
        ]
    )

    assert result["expiry_date"] == "01.01.2036"
    assert result["issue_date"] == "02.01.2026"
    assert result["mrz_valid"] is True


def test_parses_standard_tajik_document_code() -> None:
    result = parse_normalized_passport_mrz(
        [
            "P<TJKRAHMONOV<<FARID<<<<<<<<<<<<<<<<<<<<<<<<",
            "4012345676TJK9001158M35011513500012345678<00",
        ]
    )

    assert result["document_code"] == "P"
    assert result["issuing_country"] == "TJK"
    assert result["passport_number"] == "401234567"
    assert result["personal_number"] == "3500012345678"
    assert result["mrz_valid"] is True


def test_invalid_check_digit_keeps_fields_and_marks_mrz_invalid() -> None:
    result = parse_normalized_passport_mrz(
        [
            "P<RUSIVANOV<<IVAN<<<<<<<<<<<<<<<<<<<<<<<<<<<",
            "7512345670RUS9001011M3501014<<<<<<<<<<<<<<02",
        ]
    )

    assert result["passport_number"] == "751234567"
    assert result["expiry_date"] == "01.01.2035"
    assert result["mrz_valid"] is False


@pytest.mark.parametrize(
    ("upper", "lower"),
    (
        (
            "PNRUSTESTOV<<TEST<TESTOVI3".ljust(44, "<"),
            "4511234569RUS9001158<<<<<<<<5150620770001<88",
        ),
        (
            "P<RUSIVANOV<<IVAN".ljust(44, "<"),
            "7512345672RUS9001011<3501014<<<<<<<<<<<<<<02",
        ),
    ),
    ids=("PNRUS", "P<RUS"),
)
def test_unspecified_sex_is_returned_as_null(
    upper: str,
    lower: str,
) -> None:
    result = parse_normalized_passport_mrz([upper, lower])

    assert result["sex"] is None
    assert result["mrz_valid"] is True


def test_incomplete_lower_line_preserves_header_and_name_fields() -> None:
    result = parse_normalized_passport_mrz(
        ["P<KAZNURLANOV<<AZAMAT".ljust(44, "<"), "N12345678KAZ"]
    )

    assert result["passport_type"] == "international"
    assert result["issuing_country"] == "KAZ"
    assert result["surname"] == "NURLANOV"
    assert result["name"] == "AZAMAT"
    assert result["mrz_valid"] is None


def _damaged_lower_line(lower: str, damage: str) -> str:
    if damage == "short":
        return lower[:27]
    if damage == "overlong":
        return lower + "<"
    if damage == "invalid_symbol":
        return lower[:13] + "?" + lower[14:]
    raise AssertionError(f"Unknown damage case: {damage}")


@pytest.mark.parametrize(
    ("upper", "lower", "expected_header"),
    SUPPORTED_PARSER_CASES,
    ids=("PNRUS", "P<RUS", "P<ARM", "P<KAZ", "P<TJK", "PPTJK", "P<UZB"),
)
@pytest.mark.parametrize("damage", ("short", "overlong", "invalid_symbol"))
def test_invalid_lower_line_keeps_only_fields_from_exact_supported_header(
    upper: str,
    lower: str,
    expected_header: dict[str, object],
    damage: str,
) -> None:
    result = parse_normalized_passport_mrz(
        [upper, _damaged_lower_line(lower, damage)]
    )

    for field, expected in expected_header.items():
        assert result[field] == expected
    for field in LOWER_DERIVED_FIELDS:
        assert result[field] is None


@pytest.mark.parametrize(
    ("upper", "lower"),
    (
        (
            "PNRUSTESTOV<<TEST<TESTOVI3".ljust(44, "<"),
            "7512345672RUS9001011M3501014<<<<<<<<<<<<<<02",
        ),
        (
            "P<RUSIVANOV<<IVAN".ljust(44, "<"),
            "4511234569RUS9001158M<<<<<<<5150620770001<88",
        ),
    ),
    ids=(
        "internal_header_with_international_lower",
        "international_header_with_internal_lower",
    ),
)
def test_does_not_parse_lower_line_with_another_passport_format(
    upper: str,
    lower: str,
) -> None:
    result = parse_normalized_passport_mrz([upper, lower])

    for field in LOWER_DERIVED_FIELDS:
        assert result[field] is None


def test_overlong_international_line_avoids_internal_number_slicing() -> None:
    result = parse_normalized_passport_mrz(
        [
            "P<RUSTESTOV<<TEST".ljust(44, "<"),
            "7512345672RUS9001011M3501014<<<<<<<<<<<<<<02<",
        ]
    )

    assert result["passport_type"] == "international"
    assert result["issuing_country"] == "RUS"
    assert result["surname"] == "TESTOV"
    assert result["name"] == "TEST"
    for field in LOWER_DERIVED_FIELDS:
        assert result[field] is None


def test_unknown_header_returns_completely_empty_passport_data() -> None:
    result = parse_normalized_passport_mrz(
        [
            "P<XXXUNKNOWN<<PERSON".ljust(44, "<"),
            "7512345672XXX9001011M3501014<<<<<<<<<<<<<<02",
        ]
    )

    assert result
    assert all(value is None for value in result.values())
