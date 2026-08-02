from __future__ import annotations

import re

from app.modules.v1.passport.normalization.common import (
    MRZ_DIGIT_TO_LETTER,
    MRZ_SEX_LOOKALIKES,
    complete_td3_mrz_lower_line_tail,
    normalize_mrz_numeric_fields,
    normalize_supported_mrz_country_code,
    normalize_td3_mrz_upper_line,
)

MRZ_LOWER_LINE_LENGTH = 44
PASSPORT_NUMBER_PREFIX_LENGTH = 2
MRZ_NATIONALITY_START = 10
MRZ_NATIONALITY_END = 13
MRZ_SEX_POSITION = 20
MRZ_OPTIONAL_DATA_START = 28
MRZ_OPTIONAL_DATA_END = 42
MRZ_LOWER_LINE_PREFIX_PATTERN = re.compile(
    r"[A-Z]{2}[0-9]{8}[A-Z<]{3}[0-9]{7}[MF<][0-9]{7}"
)


def normalize_armenian_mrz_upper_line(mrz: str) -> str:
    """Replace OCR digit lookalikes and pad the normalized upper line."""
    return normalize_td3_mrz_upper_line(mrz)


def normalize_armenian_mrz_lower_line(mrz: str) -> str:
    """Normalize OCR lookalikes and restore an identifiable missing tail."""
    mrz = normalize_mrz_numeric_fields(
        mrz,
        number_prefix_length=PASSPORT_NUMBER_PREFIX_LENGTH,
        numeric_optional_data=False,
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
    """Reconstruct trailing fillers/check digits without
    guessing field values.
    """
    if not MRZ_OPTIONAL_DATA_START <= len(mrz) < MRZ_LOWER_LINE_LENGTH:
        return mrz
    if (
        MRZ_LOWER_LINE_PREFIX_PATTERN.fullmatch(mrz[:MRZ_OPTIONAL_DATA_START])
        is None
    ):
        return mrz

    optional_data = mrz[MRZ_OPTIONAL_DATA_START:MRZ_OPTIONAL_DATA_END]
    if re.fullmatch(r"[A-Z0-9<]*", optional_data) is None:
        return mrz
    if len(mrz) < MRZ_OPTIONAL_DATA_END and optional_data.strip("<"):
        # A partial personal number cannot be recovered by adding fillers.
        return mrz

    return complete_td3_mrz_lower_line_tail(mrz)
