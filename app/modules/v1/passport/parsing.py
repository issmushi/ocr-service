from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from app.modules.v1.mrz import (
    parse_birth_date,
    parse_expiry_date,
    validate_mrz_check_digit,
)
from app.modules.v1.passport.fields import (
    normalize_date,
    normalize_number,
    normalize_series,
    normalize_sex,
)
from app.modules.v1.passport.normalization.common import (
    is_supported_mrz_country_code,
)

MRZ_LINE_LENGTH = 44
MRZ_CHARACTERS = re.compile(r"[A-Z0-9<]{44}")

QWEN_PASSPORT_FIELD_KEYS = frozenset(
    {
        "passport_type",
        "document_code",
        "issuing_country",
        "surname",
        "name",
        "patronymic",
        "sex",
        "date_of_birth",
        "passport_series",
        "passport_number",
        "nationality",
        "issue_date",
        "expiry_date",
        "department_code",
        "personal_number",
    }
)
QWEN_RECOGNIZED_FIELD_KEYS = frozenset(
    {
        "surname",
        "name",
        "patronymic",
        "sex",
        "date_of_birth",
        "passport_series",
        "passport_number",
        "issue_date",
        "department_code",
    }
)

RUSSIAN_MRZ_TO_CYRILLIC = {
    "A": "А",
    "B": "Б",
    "V": "В",
    "G": "Г",
    "D": "Д",
    "E": "Е",
    "2": "Ё",
    "J": "Ж",
    "Z": "З",
    "I": "И",
    "Q": "Й",
    "K": "К",
    "L": "Л",
    "M": "М",
    "N": "Н",
    "O": "О",
    "P": "П",
    "R": "Р",
    "S": "С",
    "T": "Т",
    "U": "У",
    "F": "Ф",
    "H": "Х",
    "C": "Ц",
    "3": "Ч",
    "4": "Ш",
    "W": "Щ",
    "X": "Ъ",
    "Y": "Ы",
    "9": "Ь",
    "6": "Э",
    "7": "Ю",
    "8": "Я",
}


class PassportVlmResultError(ValueError):
    """Raised when Qwen does not return a passport-shaped JSON object."""


def empty_passport_mrz_data() -> dict[str, Any]:
    """Return every public passport field when MRZ parsing is incomplete."""
    return {
        "passport_type": None,
        "document_code": None,
        "issuing_country": None,
        "surname": None,
        "name": None,
        "patronymic": None,
        "sex": None,
        "date_of_birth": None,
        "passport_series": None,
        "passport_number": None,
        "nationality": None,
        "issue_date": None,
        "expiry_date": None,
        "department_code": None,
        "personal_number": None,
        "mrz_valid": None,
    }


def _text_or_none(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _country_code(value: Any) -> str | None:
    text = _text_or_none(value)
    if text is None:
        return None
    code = re.sub(r"[^A-Za-z]", "", text).upper()
    return code if len(code) == 3 else None


def _public_date(value: Any) -> str | None:
    text = _text_or_none(value)
    if text is None:
        return None
    for pattern, order in (
        (r"(\d{4})[.\-/](\d{2})[.\-/](\d{2})", "ymd"),
        (r"(\d{2})[.\-/](\d{2})[.\-/](\d{4})", "dmy"),
    ):
        match = re.fullmatch(pattern, text)
        if match is None:
            continue
        first, second, third = map(int, match.groups())
        year, month, day = (
            (first, second, third)
            if order == "ymd"
            else (third, second, first)
        )
        try:
            parsed = date(year, month, day)
        except ValueError:
            return None
        return parsed.strftime("%d.%m.%Y")
    return None


def _department_code(value: Any) -> str | None:
    text = _text_or_none(value)
    if text is None:
        return None
    digits = re.sub(r"\D", "", text)
    return f"{digits[:3]}-{digits[3:]}" if len(digits) == 6 else None


def _internal_passport_identity(
    series_value: Any,
    number_value: Any,
) -> tuple[str | None, str | None]:
    """Recover the fixed 4+6 digit split of an internal Russian passport."""
    series_digits = re.sub(r"\D", "", str(series_value or ""))
    number_digits = re.sub(r"\D", "", str(number_value or ""))
    combined = series_digits + number_digits
    if len(combined) == 10:
        series_digits, number_digits = combined[:4], combined[4:]

    series = (
        normalize_series(series_digits) if len(series_digits) == 4 else None
    )
    number = number_digits if len(number_digits) == 6 else None
    return series, number


def normalize_visual_passport_data(values: Any) -> dict[str, Any]:
    """Normalize visual OCR fields to the public passport contract."""
    data = {**empty_passport_mrz_data(), "Issue_organization_ru": None}
    if not isinstance(values, dict):
        return data

    nested = values.get("passport")
    if isinstance(nested, dict):
        values = nested

    passport_type = _text_or_none(values.get("passport_type"))
    if passport_type:
        passport_type = passport_type.lower()
    if passport_type not in {"internal", "international"}:
        passport_type = None

    passport_series, passport_number = _internal_passport_identity(
        values.get("passport_series"),
        values.get("passport_number"),
    )

    sex = normalize_sex(values.get("sex"))
    if sex not in {"МУЖ.", "ЖЕН."}:
        sex = None

    document_code = _text_or_none(values.get("document_code"))
    data.update(
        {
            "passport_type": passport_type,
            "document_code": document_code.upper() if document_code else None,
            "issuing_country": _country_code(values.get("issuing_country")),
            "surname": _text_or_none(values.get("surname")),
            "name": _text_or_none(values.get("name")),
            "patronymic": _text_or_none(values.get("patronymic")),
            "sex": sex,
            "date_of_birth": _public_date(values.get("date_of_birth")),
            "passport_series": passport_series,
            "passport_number": passport_number,
            "nationality": _country_code(values.get("nationality")),
            "issue_date": _public_date(values.get("issue_date")),
            "expiry_date": _public_date(values.get("expiry_date")),
            "department_code": _department_code(values.get("department_code")),
            "Issue_organization_ru": _text_or_none(
                values.get("Issue_organization_ru")
            ),
            "personal_number": normalize_number(values.get("personal_number")),
            # Visual OCR cannot validate MRZ control digits.
            "mrz_valid": None,
        }
    )
    return data


def normalize_visual_international_passport_data(
    values: Any,
) -> dict[str, Any]:
    """Map visual foreign-passport fields without an internal number split."""
    if not isinstance(values, dict):
        values = {}

    sex = normalize_sex(values.get("sex"))
    if sex not in {"МУЖ.", "ЖЕН."}:
        sex = None

    return {
        "passport_type": "international",
        "document_code": "P",
        "issuing_country": "RUS",
        "surname": _text_or_none(values.get("surname")),
        "name": _text_or_none(values.get("name")),
        "patronymic": _text_or_none(values.get("patronymic")),
        "surname_latin": _text_or_none(values.get("surname_latin")),
        "name_latin": _text_or_none(values.get("name_latin")),
        "patronymic_latin": _text_or_none(values.get("patronymic_latin")),
        "sex": sex,
        "date_of_birth": _public_date(values.get("date_of_birth")),
        "birth_place_ru": _text_or_none(values.get("birth_place_ru")),
        "birth_place_en": _text_or_none(values.get("birth_place_en")),
        "passport_number": normalize_number(values.get("passport_number")),
        "nationality": "RUS",
        "issue_date": _public_date(values.get("issue_date")),
        "expiry_date": _public_date(values.get("expiry_date")),
        "Issue_organization_ru": _text_or_none(
            values.get("Issue_organization_ru")
        ),
        "issue_organization_code": _text_or_none(
            values.get("issue_organization_code")
        ),
    }


def passport_data_from_vlm_result(result: Any) -> dict[str, Any]:
    """Select the most useful passport-shaped JSON object from
    a Qwen result.
    """
    if not isinstance(result, dict):
        raise PassportVlmResultError("Qwen result must be an object")
    items = result.get("items")
    if not isinstance(items, list):
        raise PassportVlmResultError("Qwen result must contain an items list")

    decoder = json.JSONDecoder()
    best_candidate: dict[str, Any] | None = None
    best_score = (-1, -1)
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            continue
        text = item["text"]
        for index, character in enumerate(text):
            if character != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue

            present_fields = QWEN_PASSPORT_FIELD_KEYS.intersection(parsed)
            present_recognized_fields = (
                QWEN_RECOGNIZED_FIELD_KEYS.intersection(parsed)
            )
            if not present_recognized_fields or any(
                parsed[field] is not None
                and not isinstance(parsed[field], str)
                for field in present_fields
            ):
                continue

            normalized_candidate = normalize_visual_passport_data(parsed)
            recognized_fields = sum(
                normalized_candidate[field] is not None
                for field in QWEN_RECOGNIZED_FIELD_KEYS
            )
            score = (recognized_fields, len(present_fields))
            if score > best_score:
                best_candidate = normalized_candidate
                best_score = score

    if best_candidate is None:
        raise PassportVlmResultError(
            "Qwen did not return a passport-shaped JSON object"
        )
    return best_candidate


LowerLineParser = Callable[[str], dict[str, Any] | None]


@dataclass(frozen=True)
class PassportMrzParser:
    """Parser selected by one exact normalized passport header."""

    passport_type: str
    document_code: str
    issuing_country: str
    lower_line_parser: LowerLineParser
    cyrillic_names: bool = False


def _parse_names(
    upper_mrz: str,
    *,
    header: str,
    cyrillic: bool,
) -> tuple[str | None, str | None, str | None]:
    """Read names only from the name zone of a known passport header."""
    name_zone = upper_mrz[len(header) :]
    allowed_pattern = r"[A-Z0-9<]*" if cyrillic else r"[A-Z<]*"
    if re.fullmatch(allowed_pattern, name_zone) is None:
        return None, None, None

    if cyrillic:
        parts = [part for part in re.split(r"<+", name_zone) if part]
        translated = [
            "".join(RUSSIAN_MRZ_TO_CYRILLIC.get(char, char) for char in part)
            for part in parts[:3]
        ]
        return (
            translated[0] if len(translated) > 0 else None,
            translated[1] if len(translated) > 1 else None,
            translated[2] if len(translated) > 2 else None,
        )

    surname_zone, separator, given_names_zone = name_zone.partition("<<")
    if not separator:
        return None, None, None
    surname_parts = [part for part in surname_zone.split("<") if part]
    given_names = [part for part in given_names_zone.split("<") if part]
    return (
        " ".join(surname_parts) or None,
        given_names[0] if given_names else None,
        " ".join(given_names[1:]) or None,
    )


def _is_complete_lower_line(lower_mrz: str) -> bool:
    return (
        len(lower_mrz) == MRZ_LINE_LENGTH
        and MRZ_CHARACTERS.fullmatch(lower_mrz) is not None
    )


def _public_mrz_sex(value: str) -> str | None:
    return normalize_sex(value) if value in {"M", "F"} else None


def _subtract_ten_years(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = date.fromisoformat(value)
    try:
        return parsed.replace(year=parsed.year - 10).isoformat()
    except ValueError:
        return parsed.replace(year=parsed.year - 10, day=28).isoformat()


def _subtract_ten_years_and_add_one_day(value: str | None) -> str | None:
    shifted = _subtract_ten_years(value)
    if shifted is None:
        return None
    return (date.fromisoformat(shifted) + timedelta(days=1)).isoformat()


def _international_checks(lower_mrz: str) -> tuple[bool, ...]:
    composite = lower_mrz[0:10] + lower_mrz[13:20] + lower_mrz[21:43]
    return (
        validate_mrz_check_digit(lower_mrz[0:9], lower_mrz[9]),
        validate_mrz_check_digit(lower_mrz[13:19], lower_mrz[19]),
        validate_mrz_check_digit(lower_mrz[21:27], lower_mrz[27]),
        validate_mrz_check_digit(lower_mrz[28:42], lower_mrz[42]),
        validate_mrz_check_digit(composite, lower_mrz[43]),
    )


def _internal_russian_checks(lower_mrz: str) -> tuple[bool, ...]:
    composite = (
        lower_mrz[0:10]
        + lower_mrz[13:20]
        + lower_mrz[21:28]
        + lower_mrz[28:43]
    )
    return (
        validate_mrz_check_digit(lower_mrz[0:9], lower_mrz[9]),
        validate_mrz_check_digit(lower_mrz[13:19], lower_mrz[19]),
        lower_mrz[21:28] == "<<<<<<<",
        validate_mrz_check_digit(lower_mrz[28:42], lower_mrz[42]),
        validate_mrz_check_digit(composite, lower_mrz[43]),
    )


def _parse_russian_internal_lower_line(
    lower_mrz: str,
) -> dict[str, Any] | None:
    """Parse only the lower line paired with the exact PNRUS header."""
    if not _is_complete_lower_line(lower_mrz):
        return None
    document_number = lower_mrz[0:9]
    nationality = lower_mrz[10:13]
    optional_data = lower_mrz[28:42]
    if (
        not is_supported_mrz_country_code(nationality)
        or re.fullmatch(r"[0-9]{9}", document_number) is None
        or re.fullmatch(r"[0-9]{6}", lower_mrz[13:19]) is None
        or lower_mrz[20] not in {"M", "F", "<"}
        or lower_mrz[21:28] != "<<<<<<<"
        or re.fullmatch(r"[0-9]{13}<", optional_data) is None
    ):
        return None

    checks = _internal_russian_checks(lower_mrz)
    return {
        "sex": _public_mrz_sex(lower_mrz[20]),
        "date_of_birth": normalize_date(parse_birth_date(lower_mrz[13:19])),
        "passport_series": normalize_series(
            document_number[:3] + optional_data[0]
        ),
        "passport_number": normalize_number(document_number[3:9]),
        "nationality": nationality,
        "issue_date": normalize_date(parse_birth_date(optional_data[1:7])),
        "expiry_date": None,
        "department_code": (f"{optional_data[7:10]}-{optional_data[10:13]}"),
        "personal_number": None,
        "mrz_valid": all(checks),
    }


def _parse_international_lower_line(
    lower_mrz: str,
    *,
    number_pattern: str,
    optional_pattern: str,
    personal_number_length: int | None,
    issue_date_calculator: Callable[[str | None], str | None],
) -> dict[str, Any] | None:
    """Read fixed lower-line positions after a country parser was selected."""
    if not _is_complete_lower_line(lower_mrz):
        return None

    document_number = lower_mrz[0:9]
    nationality = lower_mrz[10:13]
    optional_data = lower_mrz[28:42]
    if (
        not is_supported_mrz_country_code(nationality)
        or re.fullmatch(number_pattern, document_number) is None
        or re.fullmatch(r"[0-9]{6}", lower_mrz[13:19]) is None
        or lower_mrz[20] not in {"M", "F", "<"}
        or re.fullmatch(r"[0-9]{6}", lower_mrz[21:27]) is None
        or re.fullmatch(optional_pattern, optional_data) is None
    ):
        return None

    birth_date = parse_birth_date(lower_mrz[13:19])
    expiry_date = parse_expiry_date(lower_mrz[21:27])
    personal_number = None
    if personal_number_length is not None:
        candidate = optional_data[:personal_number_length]
        if len(candidate) == personal_number_length and candidate.isdigit():
            personal_number = candidate

    checks = _international_checks(lower_mrz)
    return {
        "sex": _public_mrz_sex(lower_mrz[20]),
        "date_of_birth": normalize_date(birth_date),
        "passport_series": None,
        "passport_number": normalize_number(document_number),
        "nationality": nationality,
        "issue_date": normalize_date(issue_date_calculator(expiry_date)),
        "expiry_date": normalize_date(expiry_date),
        "department_code": None,
        "personal_number": personal_number,
        "mrz_valid": all(checks),
    }


def _parse_russian_international_lower_line(
    lower_mrz: str,
) -> dict[str, Any] | None:
    return _parse_international_lower_line(
        lower_mrz,
        number_pattern=r"[0-9]{9}",
        optional_pattern=r"[0-9<]{14}",
        personal_number_length=None,
        issue_date_calculator=_subtract_ten_years,
    )


def _parse_armenian_lower_line(lower_mrz: str) -> dict[str, Any] | None:
    return _parse_international_lower_line(
        lower_mrz,
        number_pattern=r"[A-Z]{2}[0-9]{7}",
        optional_pattern=r"[A-Z0-9<]{14}",
        personal_number_length=None,
        issue_date_calculator=_subtract_ten_years,
    )


def _parse_kazakh_lower_line(lower_mrz: str) -> dict[str, Any] | None:
    return _parse_international_lower_line(
        lower_mrz,
        number_pattern=r"N[0-9]{8}",
        optional_pattern=r"[0-9]{12}<<",
        personal_number_length=12,
        issue_date_calculator=_subtract_ten_years_and_add_one_day,
    )


def _parse_tajik_lower_line(lower_mrz: str) -> dict[str, Any] | None:
    return _parse_international_lower_line(
        lower_mrz,
        number_pattern=r"[0-9]{9}",
        optional_pattern=r"[0-9]{13}<",
        personal_number_length=13,
        issue_date_calculator=_subtract_ten_years_and_add_one_day,
    )


def _parse_standard_tajik_lower_line(lower_mrz: str) -> dict[str, Any] | None:
    """Parser selected only for P<TJK."""
    return _parse_tajik_lower_line(lower_mrz)


def _parse_pp_tajik_lower_line(lower_mrz: str) -> dict[str, Any] | None:
    """Parser selected only for PPTJK."""
    return _parse_tajik_lower_line(lower_mrz)


def _parse_uzbek_lower_line(lower_mrz: str) -> dict[str, Any] | None:
    return _parse_international_lower_line(
        lower_mrz,
        number_pattern=r"[A-Z]{2}[0-9]{7}",
        optional_pattern=r"[0-9]{14}",
        personal_number_length=14,
        issue_date_calculator=_subtract_ten_years_and_add_one_day,
    )


PASSPORT_MRZ_PARSERS: dict[str, PassportMrzParser] = {
    "PNRUS": PassportMrzParser(
        passport_type="internal",
        document_code="PN",
        issuing_country="RUS",
        lower_line_parser=_parse_russian_internal_lower_line,
        cyrillic_names=True,
    ),
    "P<RUS": PassportMrzParser(
        passport_type="international",
        document_code="P",
        issuing_country="RUS",
        lower_line_parser=_parse_russian_international_lower_line,
    ),
    "P<ARM": PassportMrzParser(
        passport_type="international",
        document_code="P",
        issuing_country="ARM",
        lower_line_parser=_parse_armenian_lower_line,
    ),
    "P<KAZ": PassportMrzParser(
        passport_type="international",
        document_code="P",
        issuing_country="KAZ",
        lower_line_parser=_parse_kazakh_lower_line,
    ),
    "P<TJK": PassportMrzParser(
        passport_type="international",
        document_code="P",
        issuing_country="TJK",
        lower_line_parser=_parse_standard_tajik_lower_line,
    ),
    "PPTJK": PassportMrzParser(
        passport_type="international",
        document_code="PP",
        issuing_country="TJK",
        lower_line_parser=_parse_pp_tajik_lower_line,
    ),
    "P<UZB": PassportMrzParser(
        passport_type="international",
        document_code="P",
        issuing_country="UZB",
        lower_line_parser=_parse_uzbek_lower_line,
    ),
}


def parse_normalized_passport_mrz(lines: Sequence[str]) -> dict[str, Any]:
    """Parse MRZ only through the parser registered for its exact header."""
    data = empty_passport_mrz_data()
    if len(lines) != 2 or not all(isinstance(line, str) for line in lines):
        return data

    upper_mrz, lower_mrz = lines
    header = upper_mrz[:5]
    parser = PASSPORT_MRZ_PARSERS.get(header)
    if parser is None:
        return data

    surname, name, patronymic = _parse_names(
        upper_mrz,
        header=header,
        cyrillic=parser.cyrillic_names,
    )
    data.update(
        {
            "passport_type": parser.passport_type,
            "document_code": parser.document_code,
            "issuing_country": parser.issuing_country,
            "surname": surname,
            "name": name,
            "patronymic": patronymic,
        }
    )

    lower_data = parser.lower_line_parser(lower_mrz)
    if lower_data is not None:
        data.update(lower_data)
    return data
