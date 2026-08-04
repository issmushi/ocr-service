from __future__ import annotations

import re

from app.modules.v1.passport.normalization.common import (
    MRZ_SEX_LOOKALIKES,
    MRZ_UPPER_LINE_LENGTH,
    complete_td3_mrz_lower_line_tail,
    has_valid_numeric_mrz_lower_line_prefix,
    normalize_mrz_numeric_fields,
    normalize_supported_mrz_country_code,
)

MRZ_NATIONALITY_START = 10
MRZ_NATIONALITY_END = 13
MRZ_SEX_POSITION = 20
MRZ_OPTIONAL_DATA_START = 28
MRZ_OPTIONAL_DATA_END = 42
MRZ_REQUIRED_DATA_END = 41
MRZ_LOWER_LINE_LENGTH = 44
MRZ_NAME_START = 5
MRZ_NAME_OCR_DIGITS = str.maketrans({"0": "O", "1": "I", "5": "S"})
MRZ_LOWER_LINE_PREFIX_PATTERN = re.compile(
    r"[0-9]{10}[A-Z<]{3}[0-9]{7}[MF<]<{7}"
)
MRZ_OPTIONAL_DATA_PATTERN = re.compile(r"[0-9]{13}<?")


def normalize_russian_internal_mrz_upper_line(mrz: str) -> str:
    """Correct unused OCR digits in names and preserve MRZ transliteration."""
    name_end = min(len(mrz), MRZ_UPPER_LINE_LENGTH)
    normalized = (
        mrz[:MRZ_NAME_START]
        + mrz[MRZ_NAME_START:name_end].translate(MRZ_NAME_OCR_DIGITS)
        + mrz[name_end:]
    )
    return normalized.ljust(MRZ_UPPER_LINE_LENGTH, "<")


def has_valid_russian_internal_mrz_lower_line_prefix(mrz: str) -> bool:
    """Require aligned fixed fields without interpreting nationality."""
    return has_valid_numeric_mrz_lower_line_prefix(mrz)


def normalize_russian_internal_mrz_lower_line(mrz: str) -> str:
    """Normalize citizenship and sex, then restore a missing
    check-digit tail.
    """
    if not has_valid_russian_internal_mrz_lower_line_prefix(mrz):
        return mrz
    mrz = normalize_mrz_numeric_fields(mrz)
    nationality = normalize_supported_mrz_country_code(
        mrz[MRZ_NATIONALITY_START:MRZ_NATIONALITY_END]
    )
    sex = mrz[MRZ_SEX_POSITION : MRZ_SEX_POSITION + 1].translate(
        MRZ_SEX_LOOKALIKES
    )
    normalized_mrz = (
        mrz[:MRZ_NATIONALITY_START]
        + nationality
        + mrz[MRZ_NATIONALITY_END:MRZ_SEX_POSITION]
        + sex
        + mrz[MRZ_SEX_POSITION + 1 :]
    )
    return _restore_lower_line_tail(normalized_mrz)


def _restore_lower_line_tail(mrz: str) -> str:
    """Require the series digit, issue date and department
    code before padding.
    """
    if not MRZ_REQUIRED_DATA_END <= len(mrz) < MRZ_LOWER_LINE_LENGTH:
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
