from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PassportHeaderPolicy:
    country: str
    allowed_prefixes: frozenset[str]


PASSPORT_HEADER_POLICIES = {
    "ARM": PassportHeaderPolicy("ARM", frozenset({"P<"})),
    "KAZ": PassportHeaderPolicy("KAZ", frozenset({"P<"})),
    "RUS": PassportHeaderPolicy("RUS", frozenset({"P<", "PN"})),
    "TJK": PassportHeaderPolicy("TJK", frozenset({"P<", "PP"})),
    "UZB": PassportHeaderPolicy("UZB", frozenset({"P<"})),
}


def get_supported_passport_headers() -> tuple[str, ...]:
    """Return every complete passport header accepted by the V1 pipeline."""
    return tuple(
        f"{prefix}{country}"
        for country, policy in PASSPORT_HEADER_POLICIES.items()
        for prefix in sorted(policy.allowed_prefixes)
    )


def get_passport_header_policy(
    country: str,
) -> PassportHeaderPolicy | None:
    return PASSPORT_HEADER_POLICIES.get(country)
