from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HeaderCorrection:
    position: int | None
    source: str
    replacement: str
    reason: str


@dataclass(frozen=True, slots=True)
class NormalizedPassportHeader:
    raw_mrz: str
    normalized_mrz: str
    raw_window: str
    normalized_header: str
    passport_prefix: str
    document_code: str
    country: str
    consumed_characters: int
    restored_missing_character: bool
    corrections: tuple[HeaderCorrection, ...]
