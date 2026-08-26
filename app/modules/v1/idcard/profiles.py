from __future__ import annotations

import re
from dataclasses import dataclass

from app.modules.v1.passport.normalization.common import (
    MRZ_DIGIT_TO_LETTER,
    MRZ_LETTER_TO_DIGIT,
    nearest_mrz_candidates,
)
from app.modules.v1.passport.normalization.header import (
    HEADER_FILLER_LOOKALIKES,
)

SUPPORTED_IDCARD_HEADERS = ("IDKAZ", "IDKGZ", "I<KGZ")
SUPPORTED_IDCARD_COUNTRIES = ("ARM", "KAZ", "KGZ", "RUS", "TJK", "UZB")


class IdcardMrzFormatError(ValueError):
    """The OCR rows do not identify a supported ID-card MRZ profile."""


@dataclass(frozen=True)
class IdcardMrzProfile:
    country: str
    format: str
    line_count: int
    line_length: int
    personal_number_length: int


KAZ_TD1 = IdcardMrzProfile("KAZ", "TD1", 3, 30, 12)
KAZ_TD2 = IdcardMrzProfile("KAZ", "TD2", 2, 36, 12)
KGZ_TD1 = IdcardMrzProfile("KGZ", "TD1", 3, 30, 14)


def normalize_idcard_country(value: str) -> str:
    value = value.translate(MRZ_DIGIT_TO_LETTER)
    matches = nearest_mrz_candidates(value, SUPPORTED_IDCARD_COUNTRIES)
    return matches[0] if len(matches) == 1 else value


def normalize_idcard_header(line: str) -> tuple[str, int | None]:
    """Return a repaired first row and its inserted prefix position, if any."""
    window = "".join(
        HEADER_FILLER_LOOKALIKES.get(char, char) for char in line[:5]
    )
    window = window.translate(MRZ_DIGIT_TO_LETTER)
    matches = nearest_mrz_candidates(window, SUPPORTED_IDCARD_HEADERS)
    if len(matches) == 1:
        return matches[0] + line[5:], None
    if len(matches) > 1:
        raise IdcardMrzFormatError("Неоднозначный формат документа")

    missing_prefixes = {
        "DKAZ": ("IDKAZ", 0),
        "IKAZ": ("IDKAZ", 1),
        "DKGZ": ("IDKGZ", 0),
        "<KGZ": ("I<KGZ", 0),
    }
    if window[:4] in missing_prefixes:
        header, inserted_position = missing_prefixes[window[:4]]
        return header + line[4:], inserted_position
    # IKGZ cannot tell us whether the missing character was D or <.
    raise IdcardMrzFormatError("Неизвестный формат документа")


def identify_idcard_profile(line: str) -> IdcardMrzProfile:
    """Dispatch by an exact header and the role of the first row's data."""
    header = line[:5]
    if header in {"IDKGZ", "I<KGZ"}:
        return KGZ_TD1
    if header != "IDKAZ":
        raise IdcardMrzFormatError("Неизвестный формат документа")

    payload = line[5:]
    _surname, separator, given_names = payload.partition("<<")
    if (
        separator
        and given_names.strip("<")
        and re.fullmatch(r"[A-Z0-9<]+", payload)
    ):
        return KAZ_TD2
    number = payload[:9].translate(MRZ_LETTER_TO_DIGIT)
    if number and re.fullmatch(r"[0-9]{1,9}", number):
        return KAZ_TD1
    raise IdcardMrzFormatError("Неизвестный формат документа")
