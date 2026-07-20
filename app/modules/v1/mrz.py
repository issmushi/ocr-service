from __future__ import annotations

from datetime import date

MRZ_WEIGHTS = (7, 3, 1)


def parse_birth_date(
    value: str, current_year: int | None = None
) -> str | None:
    """Parse an MRZ birth date in YYMMDD format into ISO date format."""
    if len(value) != 6 or not value.isdigit():
        return None

    current_year = date.today().year if current_year is None else current_year
    current_year_suffix = current_year % 100
    year_suffix = int(value[:2])
    year = (
        2000 + year_suffix
        if year_suffix <= current_year_suffix
        else 1900 + year_suffix
    )

    try:
        return date(year, int(value[2:4]), int(value[4:6])).isoformat()
    except ValueError:
        return None


def parse_expiry_date(
    value: str, current_year: int | None = None
) -> str | None:
    """Parse an MRZ expiry date in YYMMDD format into ISO date format."""
    if len(value) != 6 or not value.isdigit():
        return None

    current_year = date.today().year if current_year is None else current_year
    year_suffix = int(value[:2])
    year = (current_year // 100) * 100 + year_suffix
    if year > current_year + 20:
        year -= 100

    try:
        return date(year, int(value[2:4]), int(value[4:6])).isoformat()
    except ValueError:
        return None


def mrz_character_value(character: str) -> int:
    """Return the numeric value used by the ICAO MRZ check-digit algorithm."""
    if character == "<":
        return 0
    if character.isdigit():
        return int(character)
    if "A" <= character <= "Z":
        return ord(character) - ord("A") + 10
    raise ValueError(f"Unsupported MRZ character: {character!r}")


def calculate_mrz_check_digit(value: str) -> str:
    """Calculate an MRZ check digit with the repeating 7-3-1 weights."""
    total = sum(
        mrz_character_value(character) * MRZ_WEIGHTS[index % len(MRZ_WEIGHTS)]
        for index, character in enumerate(value)
    )
    return str(total % 10)


def validate_mrz_check_digit(value: str, check_digit: str) -> bool:
    """Validate one numeric MRZ check digit."""
    return (
        check_digit.isdigit()
        and calculate_mrz_check_digit(value) == check_digit
    )
