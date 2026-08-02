from __future__ import annotations

import re

from app.modules.v1.passport.normalization.common import (
    MRZ_DIGIT_TO_LETTER,
    MRZ_LETTER_TO_DIGIT,
    MRZ_SEX_LOOKALIKES,
    complete_td3_mrz_lower_line_tail,
    is_supported_mrz_country_code,
    normalize_mrz_numeric_fields,
    normalize_supported_mrz_country_code,
    normalize_td3_mrz_upper_line,
)

MRZ_NATIONALITY_START = 10
MRZ_NATIONALITY_END = 13
MRZ_SEX_POSITION = 20
MRZ_OPTIONAL_DATA_START = 28
MRZ_OPTIONAL_DATA_END = 42
MRZ_PERSONAL_NUMBER_END = 40
MRZ_LOWER_LINE_LENGTH = 44
MRZ_LOWER_LINE_PREFIX_PATTERN = re.compile(
    r"N[0-9]{9}[A-Z<]{3}[0-9]{7}[MF<][0-9]{7}"
)
MRZ_OPTIONAL_DATA_PATTERN = re.compile(r"[0-9]{12}<{0,2}")


def normalize_kazakh_mrz_upper_line(mrz: str) -> str:
    """Replace OCR digit lookalikes and pad the normalized upper line."""
    return normalize_td3_mrz_upper_line(mrz)


def is_kazakh_mrz_number_prefix_missing(mrz: str) -> bool:
    """Identify a missing N before an exact supported shifted nationality."""
    shifted_nationality = mrz[9:12].upper().translate(MRZ_DIGIT_TO_LETTER)
    return (
        len(mrz) <= MRZ_LOWER_LINE_LENGTH - 1
        and re.fullmatch(
            r"[0-9]{9}",
            mrz[:9].translate(MRZ_LETTER_TO_DIGIT),
        )
        is not None
        and is_supported_mrz_country_code(shifted_nationality)
    )


def normalize_kazakh_mrz_lower_line(mrz: str) -> str:
    """Normalize OCR lookalikes and restore the tail after a complete IIN."""
    if not mrz:
        return mrz
    if is_kazakh_mrz_number_prefix_missing(mrz):
        mrz = "N" + mrz
    mrz = normalize_mrz_numeric_fields(mrz, number_prefix_length=1)
    nationality = normalize_supported_mrz_country_code(
        mrz[MRZ_NATIONALITY_START:MRZ_NATIONALITY_END]
    )
    sex = mrz[MRZ_SEX_POSITION : MRZ_SEX_POSITION + 1].translate(
        MRZ_SEX_LOOKALIKES
    )
    normalized_mrz = (
        "N"
        + mrz[1:MRZ_NATIONALITY_START]
        + nationality
        + mrz[MRZ_NATIONALITY_END:MRZ_SEX_POSITION]
        + sex
        + mrz[MRZ_SEX_POSITION + 1 :]
    )
    return _restore_lower_line_tail(normalized_mrz)


def _restore_lower_line_tail(mrz: str) -> str:
    """Restore trailing fillers/check digits only when all IIN
    digits remain.
    """
    if not MRZ_PERSONAL_NUMBER_END <= len(mrz) < MRZ_LOWER_LINE_LENGTH:
        return mrz
    if (
        MRZ_LOWER_LINE_PREFIX_PATTERN.fullmatch(mrz[:MRZ_OPTIONAL_DATA_START])
        is None
    ):
        return mrz
    if (
        MRZ_OPTIONAL_DATA_PATTERN.fullmatch(
            mrz[MRZ_OPTIONAL_DATA_START:MRZ_OPTIONAL_DATA_END]
        )
        is None
    ):
        return mrz
    return complete_td3_mrz_lower_line_tail(mrz)
