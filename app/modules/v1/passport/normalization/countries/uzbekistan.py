from __future__ import annotations

import re

from app.modules.v1.passport.normalization.common import (
    MRZ_DIGIT_TO_LETTER,
    MRZ_LETTER_TO_DIGIT,
    MRZ_SEX_LOOKALIKES,
    complete_td3_mrz_lower_line_tail,
    normalize_mrz_numeric_fields,
    normalize_supported_mrz_country_code,
    normalize_td3_mrz_upper_line,
)

PASSPORT_NUMBER_PREFIX_LENGTH = 2
MRZ_NATIONALITY_START = 10
MRZ_NATIONALITY_END = 13
MRZ_SEX_POSITION = 20
MRZ_OPTIONAL_DATA_START = 28
MRZ_OPTIONAL_DATA_END = 42
MRZ_LOWER_LINE_LENGTH = 44
MRZ_NUMBER_WITH_CHECK_PATTERN = re.compile(r"[A-Z]{2}[0-9]{8}")
MRZ_LOWER_LINE_PREFIX_PATTERN = re.compile(
    r"[A-Z]{2}[0-9]{8}[A-Z]{3}[0-9]{7}[MF<][0-9]{7}"
)
MRZ_OPTIONAL_DATA_PATTERN = re.compile(r"[0-9]{14}")


def normalize_uzbek_mrz_upper_line(mrz: str) -> str:
    """Replace OCR digit lookalikes and pad the normalized upper line."""
    return normalize_td3_mrz_upper_line(mrz)


def has_valid_uzbek_mrz_lower_line_prefix(mrz: str) -> bool:
    """Check number and fixed-field alignment without
    interpreting nationality.
    """
    number_with_check = mrz[:PASSPORT_NUMBER_PREFIX_LENGTH].translate(
        MRZ_DIGIT_TO_LETTER
    ) + mrz[PASSPORT_NUMBER_PREFIX_LENGTH:MRZ_NATIONALITY_START].translate(
        MRZ_LETTER_TO_DIGIT
    )
    return (
        MRZ_NATIONALITY_END <= len(mrz) <= MRZ_LOWER_LINE_LENGTH
        and MRZ_NUMBER_WITH_CHECK_PATTERN.fullmatch(number_with_check)
        is not None
        and (
            len(mrz) < 21
            or re.fullmatch(r"[A-Z<]", mrz[MRZ_SEX_POSITION].upper())
            is not None
        )
    )


def normalize_uzbek_mrz_lower_line(mrz: str) -> str:
    """Normalize number letters, citizenship and sex, restoring tail checks."""
    if not has_valid_uzbek_mrz_lower_line_prefix(mrz):
        return mrz
    mrz = normalize_mrz_numeric_fields(
        mrz,
        number_prefix_length=PASSPORT_NUMBER_PREFIX_LENGTH,
    )
    prefix = mrz[:PASSPORT_NUMBER_PREFIX_LENGTH].translate(MRZ_DIGIT_TO_LETTER)
    nationality = normalize_supported_mrz_country_code(
        mrz[MRZ_NATIONALITY_START:MRZ_NATIONALITY_END]
    )
    sex = mrz[MRZ_SEX_POSITION : MRZ_SEX_POSITION + 1].translate(
        MRZ_SEX_LOOKALIKES
    )
    normalized_mrz = (
        prefix
        + mrz[PASSPORT_NUMBER_PREFIX_LENGTH:MRZ_NATIONALITY_START]
        + nationality
        + mrz[MRZ_NATIONALITY_END:MRZ_SEX_POSITION]
        + sex
        + mrz[MRZ_SEX_POSITION + 1 :]
    )
    return _restore_lower_line_tail(normalized_mrz)


def _restore_lower_line_tail(mrz: str) -> str:
    """Restore checks only when all fourteen additional-data digits remain."""
    if not MRZ_OPTIONAL_DATA_END <= len(mrz) < MRZ_LOWER_LINE_LENGTH:
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
