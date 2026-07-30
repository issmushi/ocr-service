from __future__ import annotations

from app.modules.v1.passport.normalization.common import (
    MRZ_DIGIT_TO_LETTER,
    nearest_mrz_candidates,
)
from app.modules.v1.passport.normalization.countries import (
    PassportHeaderPolicy,
    get_passport_header_policy,
    get_supported_passport_headers,
)
from app.modules.v1.passport.normalization.models import (
    HeaderCorrection,
    NormalizedPassportHeader,
)

MRZ_HEADER_LENGTH = 5
MISSING_CHARACTER_HEADER_LENGTH = 4
HEADER_FILLER_LOOKALIKES = {
    "‹": "<",
    "〈": "<",
    "＜": "<",
    "«": "<",
    "»": "<",
}
SUPPORTED_PASSPORT_HEADERS = get_supported_passport_headers()


class PassportHeaderNormalizationError(RuntimeError):
    """Base error for passport MRZ header normalization."""


class PassportCountryNotFoundError(PassportHeaderNormalizationError):
    """Raised when no supported passport country exists in the header."""


class PassportDocumentCodeError(PassportHeaderNormalizationError):
    """Raised when a passport prefix cannot be recovered safely."""


class AmbiguousPassportHeaderError(PassportDocumentCodeError):
    """Raised when the first five characters allow multiple repairs."""


def _correction(
    position: int | None,
    source: str,
    replacement: str,
    reason: str,
) -> HeaderCorrection:
    return HeaderCorrection(
        position=position,
        source=source,
        replacement=replacement,
        reason=reason,
    )


def _normalize_header_window(
    raw_mrz: str,
) -> tuple[list[str], list[HeaderCorrection]]:
    window = list(raw_mrz[:MRZ_HEADER_LENGTH])
    corrections: list[HeaderCorrection] = []

    for position, source in enumerate(window):
        replacement = source.upper()
        reason = "uppercase_header_letter"
        if replacement in HEADER_FILLER_LOOKALIKES:
            replacement = HEADER_FILLER_LOOKALIKES[replacement]
            reason = "normalize_header_filler"
        digit_replacement = replacement.translate(MRZ_DIGIT_TO_LETTER)
        if digit_replacement != replacement:
            replacement = digit_replacement
            reason = "normalize_header_digit_to_letter"

        if replacement != source:
            corrections.append(
                _correction(position, source, replacement, reason)
            )
            window[position] = replacement

    return window, corrections


def _policy_for_country(country: str) -> PassportHeaderPolicy:
    policy = get_passport_header_policy(country)
    if policy is None:
        raise PassportCountryNotFoundError(
            "Supported passport country was not found in MRZ header"
        )
    return policy


def _normalize_complete_header(
    raw_mrz: str,
    window: list[str],
    corrections: list[HeaderCorrection],
    normalized_header: str,
) -> NormalizedPassportHeader:
    country = normalized_header[2:]
    policy = _policy_for_country(country)
    passport_prefix = normalized_header[:2]
    if passport_prefix not in policy.allowed_prefixes:
        raise PassportDocumentCodeError(
            f"Unsupported passport prefix for country {country}"
        )

    for position, (source, replacement) in enumerate(
        zip(window, normalized_header, strict=True)
    ):
        if source == replacement:
            continue
        corrections.append(
            _correction(
                position,
                source,
                replacement,
                "repair_passport_header_by_nearest_match",
            )
        )
    return NormalizedPassportHeader(
        raw_mrz=raw_mrz,
        normalized_mrz=f"{normalized_header}{raw_mrz[MRZ_HEADER_LENGTH:]}",
        raw_window=raw_mrz[:MRZ_HEADER_LENGTH],
        normalized_header=normalized_header,
        passport_prefix=passport_prefix,
        document_code=passport_prefix.rstrip("<"),
        country=country,
        consumed_characters=MRZ_HEADER_LENGTH,
        restored_missing_character=False,
        corrections=tuple(corrections),
    )


def _prefix_for_missing_character(
    remaining_character: str,
    policy: PassportHeaderPolicy,
) -> tuple[str, int, str]:
    country = policy.country
    if remaining_character == "<" and "P<" in policy.allowed_prefixes:
        return "P<", 0, "P"
    if (
        remaining_character == "N"
        and country == "RUS"
        and "PN" in policy.allowed_prefixes
    ):
        return "PN", 0, "P"
    if (
        remaining_character == "P"
        and country == "TJK"
        and "PP" in policy.allowed_prefixes
    ):
        return "PP", 0, "P"
    if remaining_character == "P" and policy.allowed_prefixes == frozenset(
        {"P<"}
    ):
        return "P<", 1, "<"
    if remaining_character == "P" and len(policy.allowed_prefixes) > 1:
        raise AmbiguousPassportHeaderError(
            f"Ambiguous missing passport prefix character for {country}"
        )
    raise PassportDocumentCodeError(
        f"Cannot restore passport prefix for country {country}"
    )


def _normalize_header_with_missing_character(
    raw_mrz: str,
    window: list[str],
    corrections: list[HeaderCorrection],
    country: str,
) -> NormalizedPassportHeader:
    policy = _policy_for_country(country)
    passport_prefix, position, inserted_character = (
        _prefix_for_missing_character(window[0], policy)
    )
    relevant_corrections = [
        correction
        for correction in corrections
        if correction.position is None
        or correction.position < MISSING_CHARACTER_HEADER_LENGTH
    ]
    relevant_corrections.append(
        _correction(
            position,
            "",
            inserted_character,
            "restore_missing_passport_header_character",
        )
    )
    normalized_header = f"{passport_prefix}{country}"
    return NormalizedPassportHeader(
        raw_mrz=raw_mrz,
        normalized_mrz=(
            f"{normalized_header}{raw_mrz[MISSING_CHARACTER_HEADER_LENGTH:]}"
        ),
        raw_window=raw_mrz[:MRZ_HEADER_LENGTH],
        normalized_header=normalized_header,
        passport_prefix=passport_prefix,
        document_code=passport_prefix.rstrip("<"),
        country=country,
        consumed_characters=MISSING_CHARACTER_HEADER_LENGTH,
        restored_missing_character=True,
        corrections=tuple(relevant_corrections),
    )


def normalize_passport_mrz_header(raw_mrz: str) -> NormalizedPassportHeader:
    """Normalize and recover only the passport MRZ header window."""
    if not isinstance(raw_mrz, str):
        raise TypeError("raw_mrz must be a string")
    if len(raw_mrz) < MISSING_CHARACTER_HEADER_LENGTH:
        raise PassportCountryNotFoundError(
            "Passport MRZ header contains fewer than four characters"
        )

    window, corrections = _normalize_header_window(raw_mrz)
    if len(window) >= MRZ_HEADER_LENGTH:
        full_header_matches = nearest_mrz_candidates(
            "".join(window[:MRZ_HEADER_LENGTH]),
            SUPPORTED_PASSPORT_HEADERS,
            max_mismatches=1,
        )
        if len(full_header_matches) == 1:
            return _normalize_complete_header(
                raw_mrz,
                window,
                corrections,
                full_header_matches[0],
            )
        if len(full_header_matches) > 1:
            raise AmbiguousPassportHeaderError(
                "Passport MRZ header has "
                "multiple equally close supported forms"
            )

    missing_character_country = "".join(window[1:4])
    if get_passport_header_policy(missing_character_country) is not None:
        return _normalize_header_with_missing_character(
            raw_mrz,
            window,
            corrections,
            missing_character_country,
        )

    raise PassportCountryNotFoundError(
        "Supported passport country was not found in MRZ header"
    )
