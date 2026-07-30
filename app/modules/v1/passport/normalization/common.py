from __future__ import annotations

import re
from collections.abc import Iterable

from app.modules.v1.mrz import calculate_mrz_check_digit

MRZ_UPPER_LINE_LENGTH = 44
MRZ_DIGIT_TO_LETTER = str.maketrans(
    {
        "0": "O",
        "1": "I",
        "2": "Z",
        "3": "E",
        "4": "A",
        "5": "S",
        "6": "G",
        "7": "T",
        "8": "B",
        "9": "G",
    }
)
MRZ_SEX_LOOKALIKES = str.maketrans({"N": "M", "H": "M", "E": "F"})
_MRZ_LETTER_TO_DIGIT = {
    "O": "0",
    "I": "1",
    "Z": "2",
    "E": "3",
    "A": "4",
    "S": "5",
    # Both 6 and 9 become G in the alphabetic map; choose 6 explicitly.
    "G": "6",
    "T": "7",
    "B": "8",
}
MRZ_LETTER_TO_DIGIT = str.maketrans(
    {
        **_MRZ_LETTER_TO_DIGIT,
        **{
            letter.lower(): digit
            for letter, digit in _MRZ_LETTER_TO_DIGIT.items()
        },
    }
)

SUPPORTED_MRZ_COUNTRY_CODES = ("ARM", "KAZ", "RUS", "TJK", "UZB")

# A mismatch still counts as one changed character. These weights only decide
# between equally close candidates; visually common OCR substitutions win over
# an arbitrary replacement.
_LOW_COST_MRZ_SUBSTITUTIONS = frozenset(
    {
        frozenset(("<", "C")),
        frozenset(("B", "S")),
        frozenset(("E", "F")),
        frozenset(("H", "M")),
        frozenset(("I", "L")),
        frozenset(("M", "N")),
        frozenset(("O", "Q")),
    }
)


def nearest_mrz_candidates(
    value: str,
    candidates: Iterable[str],
    *,
    max_mismatches: int = 1,
) -> tuple[str, ...]:
    """Return the uniquely best MRZ candidates within a substitution radius.

    Candidate strings must have the same length as ``value``. Insertions and
    deletions are deliberately excluded because they would move fixed MRZ
    fields. The result contains multiple values only when the best score ties.
    """
    scored: list[tuple[tuple[int, int], str]] = []
    for candidate in candidates:
        if len(candidate) != len(value):
            continue
        substitutions = [
            (source, expected)
            for source, expected in zip(value, candidate, strict=True)
            if source != expected
        ]
        if len(substitutions) > max_mismatches:
            continue
        weight = sum(
            1
            if frozenset((source, expected)) in _LOW_COST_MRZ_SUBSTITUTIONS
            else 2
            for source, expected in substitutions
        )
        scored.append(((len(substitutions), weight), candidate))

    if not scored:
        return ()
    best_score = min(score for score, _candidate in scored)
    return tuple(
        candidate for score, candidate in scored if score == best_score
    )


def normalize_supported_mrz_country_code(value: str) -> str:
    """Normalize one three-letter code to a unique supported code within 1
    edit.
    """
    normalized = value.upper().translate(MRZ_DIGIT_TO_LETTER)
    matches = nearest_mrz_candidates(
        normalized,
        SUPPORTED_MRZ_COUNTRY_CODES,
        max_mismatches=1,
    )
    return matches[0] if len(matches) == 1 else normalized


def is_supported_mrz_country_code(value: str) -> bool:
    return value in SUPPORTED_MRZ_COUNTRY_CODES


def normalize_mrz_numeric_fields(
    mrz: str,
    *,
    number_prefix_length: int = 0,
    numeric_optional_data: bool = True,
) -> str:
    """Correct numeric fields without moving symbols or changing fillers."""
    ranges = (
        (number_prefix_length, 10),
        (13, 20),
        (21, 42 if numeric_optional_data else 28),
        (42, 44),
    )
    characters = list(mrz)
    for start, end in ranges:
        for index in range(start, min(end, len(characters))):
            characters[index] = characters[index].translate(
                MRZ_LETTER_TO_DIGIT
            )
    return "".join(characters)


def has_valid_numeric_mrz_lower_line_prefix(mrz: str) -> bool:
    """Check fixed-field alignment without using nationality as an anchor."""
    return (
        13 <= len(mrz) <= 44
        and not _looks_like_extra_leading_character_shift(mrz)
        and re.fullmatch(
            r"[0-9]{10}",
            mrz[:10].translate(MRZ_LETTER_TO_DIGIT),
        )
        is not None
        and (
            len(mrz) < 21
            or re.fullmatch(r"[A-Z<]", mrz[20].upper()) is not None
        )
    )


def _looks_like_extra_leading_character_shift(mrz: str) -> bool:
    """Detect an alternate fixed-field alignment starting at position two.

    This prevents a shifted number check digit from becoming the first letter
    of a seemingly valid nationality, for example ``4RU -> ARU -> ARM``.
    A two-letter country prefix is enough at the end of a truncated row.
    """
    shifted_number = mrz[1:11].translate(MRZ_LETTER_TO_DIGIT)
    if re.fullmatch(r"[0-9]{10}", shifted_number) is None:
        return False

    shifted_country = mrz[11:14].upper().translate(MRZ_DIGIT_TO_LETTER)
    if len(shifted_country) < 2:
        return False
    if len(shifted_country) == 2:
        return any(
            country.startswith(shifted_country)
            for country in SUPPORTED_MRZ_COUNTRY_CODES
        )
    return is_supported_mrz_country_code(shifted_country)


def normalize_td3_mrz_upper_line(mrz: str) -> str:
    """Replace OCR digit lookalikes and pad the normalized upper line."""
    return mrz.translate(MRZ_DIGIT_TO_LETTER).ljust(MRZ_UPPER_LINE_LENGTH, "<")


def complete_td3_mrz_lower_line_tail(mrz: str) -> str:
    """Complete a tail after country rules establish that its data survived."""
    if not 28 <= len(mrz) < 44:
        return mrz

    data = mrz[:42].ljust(42, "<")
    optional_check_digit = calculate_mrz_check_digit(data[28:42])
    if len(mrz) == 43:
        observed_check_digit = mrz[42]
        empty_field_filler = observed_check_digit == "<" and not data[
            28:42
        ].strip("<")
        if (
            observed_check_digit != optional_check_digit
            and not empty_field_filler
        ):
            return mrz
        # Preserve the check digit that OCR actually read, including '<'.
        data += observed_check_digit
    else:
        data += optional_check_digit

    composite = data[:10] + data[13:20] + data[21:43]
    return data + calculate_mrz_check_digit(composite)
