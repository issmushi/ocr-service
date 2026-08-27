from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date
from typing import Any

from app.modules.v1.idcard.profiles import (
    KAZ_TD1,
    KAZ_TD2,
    KGZ_TD1,
    SUPPORTED_IDCARD_COUNTRIES,
    IdcardMrzFormatError,
    IdcardMrzProfile,
    identify_idcard_profile,
)
from app.modules.v1.mrz import (
    parse_birth_date,
    parse_expiry_date,
    validate_mrz_check_digit,
)


def _field(line: str, start: int, end: int, pattern: str) -> str | None:
    value = line[start:end]
    return (
        value
        if len(value) == end - start and re.fullmatch(pattern, value)
        else None
    )


def _date(value: str | None, *, expiry: bool = False) -> str | None:
    parsed = (parse_expiry_date if expiry else parse_birth_date)(value or "")
    return date.fromisoformat(parsed).strftime("%d.%m.%Y") if parsed else None


def _names(value: str) -> tuple[str | None, str | None, str | None]:
    if re.fullmatch(r"[A-Z<]+", value) is None:
        return None, None, None
    surname, separator, given_names = value.partition("<<")
    if not separator:
        return None, None, None
    given_parts = [part for part in given_names.split("<") if part]
    return (
        " ".join(part for part in surname.split("<") if part) or None,
        given_parts[0] if given_parts else None,
        " ".join(given_parts[1:]) or None,
    )


def _sex(value: str | None) -> str | None:
    return {"M": "МУЖ.", "F": "ЖЕН."}.get(value)


def _country(value: str | None) -> str | None:
    return value if value in SUPPORTED_IDCARD_COUNTRIES else None


def _base_data(first: str) -> dict[str, Any]:
    return {
        "document_type": "identity_card",
        "document_code": first[:2].rstrip("<"),
        "issuing_country": first[2:5],
        "surname": None,
        "name": None,
        "patronymic": None,
        "sex": None,
        "date_of_birth": None,
        "document_series": None,
        "document_number": None,
        "nationality": None,
        "issue_date": None,
        "expiry_date": None,
        "department_code": None,
        "personal_number": None,
        "mrz_valid": None,
    }


def _complete(lines: Sequence[str], profile: IdcardMrzProfile) -> bool:
    return len(lines) == profile.line_count and all(
        len(line) == profile.line_length
        and re.fullmatch(r"[A-Z0-9<]+", line) is not None
        for line in lines
    )


def _parse_td1(
    lines: Sequence[str], profile: IdcardMrzProfile
) -> dict[str, Any]:
    first = lines[0]
    second = lines[1] if len(lines) > 1 else ""
    third = lines[2] if len(lines) > 2 else ""
    data = _base_data(first)
    number_pattern = (
        r"[A-Z]{2}[0-9]{7}" if profile.country == "KGZ" else r"[0-9]{9}"
    )
    personal_end = 15 + profile.personal_number_length
    personal = _field(first, 15, personal_end, r"[0-9]+")
    if first[personal_end:30].strip("<"):
        personal = None
    birth = _field(second, 0, 6, r"[0-9]{6}")
    expiry = _field(second, 8, 14, r"[0-9]{6}")
    sex = _field(second, 7, 8, r"[MF<]")
    surname, name, patronymic = _names(third)
    data.update(
        {
            "surname": surname,
            "name": name,
            "patronymic": patronymic,
            "document_number": _field(first, 5, 14, number_pattern),
            "personal_number": personal,
            "date_of_birth": _date(birth),
            "expiry_date": _date(expiry, expiry=True),
            "sex": _sex(sex),
            "nationality": _country(_field(second, 15, 18, r"[A-Z]{3}")),
        }
    )
    if _complete(lines, profile):
        composite = first[5:30] + second[:7] + second[8:15] + second[18:29]
        data["mrz_valid"] = all(
            (
                data["document_number"] is not None,
                data["personal_number"] is not None,
                data["date_of_birth"] is not None,
                data["expiry_date"] is not None,
                data["nationality"] is not None,
                sex is not None,
                not second[18:29].strip("<"),
                validate_mrz_check_digit(first[5:14], first[14]),
                validate_mrz_check_digit(second[:6], second[6]),
                validate_mrz_check_digit(second[8:14], second[14]),
                validate_mrz_check_digit(composite, second[29]),
            )
        )
    return data


def _parse_kazakhstan_td1(lines: Sequence[str]) -> dict[str, Any]:
    return _parse_td1(lines, KAZ_TD1)


def _parse_kyrgyzstan_td1(lines: Sequence[str]) -> dict[str, Any]:
    return _parse_td1(lines, KGZ_TD1)


def _parse_kazakhstan_td2(lines: Sequence[str]) -> dict[str, Any]:
    first = lines[0]
    second = lines[1] if len(lines) > 1 else ""
    data = _base_data(first)
    surname, name, patronymic = _names(first[5:])
    birth = _field(second, 13, 19, r"[0-9]{6}")
    expiry = _field(second, 21, 27, r"[0-9]{6}")
    personal_suffix = _field(second, 28, 34, r"[0-9]{6}")
    if second[34:35].strip("<"):
        personal_suffix = None
    sex = _field(second, 20, 21, r"[MF<]")
    data.update(
        {
            "surname": surname,
            "name": name,
            "patronymic": patronymic,
            "document_number": _field(second, 0, 9, r"[0-9]{9}"),
            "personal_number": birth + personal_suffix
            if birth and personal_suffix
            else None,
            "date_of_birth": _date(birth),
            "expiry_date": _date(expiry, expiry=True),
            "sex": _sex(sex),
            "nationality": _country(_field(second, 10, 13, r"[A-Z]{3}")),
        }
    )
    if _complete(lines, KAZ_TD2):
        composite = second[:10] + second[13:20] + second[21:35]
        data["mrz_valid"] = all(
            (
                data["document_number"] is not None,
                data["personal_number"] is not None,
                data["date_of_birth"] is not None,
                data["expiry_date"] is not None,
                data["nationality"] is not None,
                sex is not None,
                validate_mrz_check_digit(second[:9], second[9]),
                validate_mrz_check_digit(second[13:19], second[19]),
                validate_mrz_check_digit(second[21:27], second[27]),
                validate_mrz_check_digit(composite, second[35]),
            )
        )
    return data


IDCARD_MRZ_PARSERS = {
    ("KAZ", "TD1"): _parse_kazakhstan_td1,
    ("KAZ", "TD2"): _parse_kazakhstan_td2,
    ("KGZ", "TD1"): _parse_kyrgyzstan_td1,
}


def parse_idcard_mrz(lines: Sequence[str]) -> dict[str, Any]:
    """Read only the fields of an explicitly supported ID-card profile."""
    if not lines or not all(isinstance(line, str) for line in lines):
        raise IdcardMrzFormatError("Неизвестный формат документа")
    profile = identify_idcard_profile(lines[0])
    return IDCARD_MRZ_PARSERS[(profile.country, profile.format)](lines)
