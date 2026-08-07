from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from app.core.images.preprocessing import (
    decode_image_bytes,
    prepare_document_image,
)
from app.modules.v1.passport.normalization import (
    PassportHeaderNormalizationError,
    normalize_passport_mrz_header,
)
from app.modules.v1.passport.normalization.common import (
    is_supported_mrz_country_code,
)
from app.modules.v1.passport.normalization.countries.armenia import (
    normalize_armenian_mrz_lower_line,
    normalize_armenian_mrz_upper_line,
)
from app.modules.v1.passport.normalization.countries.kazakhstan import (
    is_kazakh_mrz_number_prefix_missing,
    normalize_kazakh_mrz_lower_line,
    normalize_kazakh_mrz_upper_line,
)
from app.modules.v1.passport.normalization.countries.russian_foreign import (
    has_valid_russian_foreign_mrz_lower_line_prefix,
    normalize_russian_foreign_mrz_lower_line,
    normalize_russian_foreign_mrz_upper_line,
)
from app.modules.v1.passport.normalization.countries.russian_internal import (
    has_valid_russian_internal_mrz_lower_line_prefix,
    normalize_russian_internal_mrz_lower_line,
    normalize_russian_internal_mrz_upper_line,
)
from app.modules.v1.passport.normalization.countries.tajikistan import (
    has_valid_tajik_mrz_lower_line_prefix,
    normalize_tajik_mrz_lower_line,
    normalize_tajik_mrz_upper_line,
)
from app.modules.v1.passport.normalization.countries.uzbekistan import (
    has_valid_uzbek_mrz_lower_line_prefix,
    normalize_uzbek_mrz_lower_line,
    normalize_uzbek_mrz_upper_line,
)
from app.modules.v1.passport.normalization.diagnostics import (
    MrzNormalizationIssue,
    build_mrz_normalization_report,
)
from app.modules.v1.passport.recognition import (
    recognize_raw_passport_mrz,
    select_longest_passport_mrz_lines,
)
from app.ocr_engine.mrz.factory import create_mrz_scanner
from app.ocr_engine.mrz.recognizer import recognize_mrz
from app.ocr_engine.paddle.factory import create_mrz_ocr

if TYPE_CHECKING:
    from paddleocr import PaddleOCR


def process_passport_image(
    image: np.ndarray,
    ocr: PaddleOCR | None = None,
    *,
    normalize: bool = True,
) -> dict[str, Any]:
    """Run passport OCR, normalize the header and apply
    country-specific rules.
    """
    ocr = create_mrz_ocr() if ocr is None else ocr
    prepared = prepare_document_image(image)
    recognition = recognize_raw_passport_mrz(prepared.image, ocr)
    result = normalize_passport_mrz(recognition.lines, normalize=normalize)
    result["diagnostics"].update(
        {
            "detected_regions_count": recognition.detected_regions_count,
            "selected_regions_count": recognition.selected_regions_count,
            "preprocessing": prepared.metadata,
            "engine": "paddle",
        }
    )
    return result


def process_passport_image_with_mrz_scanner(
    image: np.ndarray,
    scanner: Any = None,
    *,
    normalize: bool = True,
) -> dict[str, Any]:
    """Recognize raw MRZ with MRZ Scanner and apply shared passport rules."""
    scanner = create_mrz_scanner() if scanner is None else scanner
    # The scanner finds and warps its own MRZ region; a page crop can remove
    # it.
    recognition = recognize_mrz(image, scanner)
    result = normalize_passport_mrz(recognition.lines, normalize=normalize)
    result["diagnostics"].update(
        {
            "detected_regions_count": recognition.detected_regions_count,
            "selected_regions_count": recognition.detected_regions_count,
            "preprocessing": {"source": "original"},
            "engine": "mrz",
        }
    )
    return result


def normalize_passport_mrz(
    lines: Sequence[str],
    *,
    normalize: bool = True,
) -> dict[str, Any]:
    """Select two OCR rows and optionally apply the matching country rules."""
    raw_mrz = list(select_longest_passport_mrz_lines(lines))
    normalized_mrz = raw_mrz
    header_normalization: dict[str, Any] | None = None
    lower_line_tail_restoration: dict[str, Any] | None = None
    lower_line_prefix_inserted = False
    lower_line_structure_valid: bool | None = None
    normalization_issues: list[MrzNormalizationIssue] = []
    normalized_header = None
    if normalize and len(raw_mrz) == 2:
        normalized_header = normalize_passport_mrz_header(raw_mrz[0])
        normalized_mrz = [normalized_header.normalized_mrz, raw_mrz[1]]
        if normalized_header.country == "ARM":
            lower_line_structure_valid = True
            normalized_mrz[0] = normalize_armenian_mrz_upper_line(
                normalized_mrz[0]
            )
            normalized_mrz[1] = normalize_armenian_mrz_lower_line(
                normalized_mrz[1]
            )
        elif normalized_header.country == "KAZ":
            lower_line_structure_valid = True
            normalized_mrz[0] = normalize_kazakh_mrz_upper_line(
                normalized_mrz[0]
            )
            if normalized_header.passport_prefix == "P<":
                lower_line_prefix_inserted = (
                    is_kazakh_mrz_number_prefix_missing(raw_mrz[1])
                )
                normalized_mrz[1] = normalize_kazakh_mrz_lower_line(
                    normalized_mrz[1]
                )
        elif normalized_header.normalized_header == "PNRUS":
            normalized_mrz[0] = normalize_russian_internal_mrz_upper_line(
                normalized_mrz[0]
            )
            lower_line_structure_valid = (
                has_valid_russian_internal_mrz_lower_line_prefix(raw_mrz[1])
            )
            if lower_line_structure_valid:
                normalized_mrz[1] = normalize_russian_internal_mrz_lower_line(
                    normalized_mrz[1]
                )
            else:
                normalization_issues.append(
                    {
                        "line": 2,
                        "code": "invalid_lower_line_structure",
                        "message": "Нарушена структура нижней строки MRZ",
                    }
                )
        elif normalized_header.normalized_header == "P<RUS":
            normalized_mrz[0] = normalize_russian_foreign_mrz_upper_line(
                normalized_mrz[0]
            )
            lower_line_structure_valid = (
                has_valid_russian_foreign_mrz_lower_line_prefix(raw_mrz[1])
            )
            if lower_line_structure_valid:
                normalized_mrz[1] = normalize_russian_foreign_mrz_lower_line(
                    normalized_mrz[1]
                )
            else:
                normalization_issues.append(
                    {
                        "line": 2,
                        "code": "invalid_lower_line_structure",
                        "message": "Нарушена структура нижней строки MRZ",
                    }
                )
        elif normalized_header.normalized_header in {"P<TJK", "PPTJK"}:
            normalized_mrz[0] = normalize_tajik_mrz_upper_line(
                normalized_mrz[0]
            )
            lower_line_structure_valid = has_valid_tajik_mrz_lower_line_prefix(
                raw_mrz[1]
            )
            if lower_line_structure_valid:
                normalized_mrz[1] = normalize_tajik_mrz_lower_line(
                    normalized_mrz[1]
                )
            else:
                normalization_issues.append(
                    {
                        "line": 2,
                        "code": "invalid_lower_line_structure",
                        "message": "Нарушена структура нижней строки MRZ",
                    }
                )
        elif normalized_header.normalized_header == "P<UZB":
            normalized_mrz[0] = normalize_uzbek_mrz_upper_line(
                normalized_mrz[0]
            )
            lower_line_structure_valid = has_valid_uzbek_mrz_lower_line_prefix(
                raw_mrz[1]
            )
            if lower_line_structure_valid:
                normalized_mrz[1] = normalize_uzbek_mrz_lower_line(
                    normalized_mrz[1]
                )
            else:
                normalization_issues.append(
                    {
                        "line": 2,
                        "code": "invalid_lower_line_structure",
                        "message": "Нарушена структура нижней строки MRZ",
                    }
                )
        if (
            lower_line_structure_valid
            and len(normalized_mrz[1]) >= 13
            and not is_supported_mrz_country_code(normalized_mrz[1][10:13])
        ):
            normalization_issues.append(
                {
                    "line": 2,
                    "code": "unrecognized_nationality",
                    "message": (
                        "Код гражданства не "
                        "сопоставлен с поддерживаемым значением"
                    ),
                }
            )
        original_length = len(raw_mrz[1])
        aligned_length = original_length + int(lower_line_prefix_inserted)
        if len(normalized_mrz[1]) > aligned_length:
            lower_line_tail_restoration = {
                "original_length": original_length,
                "added_fillers": max(42 - aligned_length, 0),
                "generated_check_digit_positions": [
                    position
                    for position in (43, 44)
                    if position > aligned_length
                ],
            }
        header_normalization = {
            "raw_window": normalized_header.raw_window,
            "normalized_header": normalized_header.normalized_header,
            "passport_prefix": normalized_header.passport_prefix,
            "document_code": normalized_header.document_code,
            "country": normalized_header.country,
            "consumed_characters": normalized_header.consumed_characters,
            "restored_missing_character": (
                normalized_header.restored_missing_character
            ),
            "corrections": [
                {
                    "position": correction.position,
                    "source": correction.source,
                    "replacement": correction.replacement,
                    "reason": correction.reason,
                }
                for correction in normalized_header.corrections
            ],
        }
    return {
        "mrz": normalized_mrz,
        "raw_mrz": raw_mrz,
        "diagnostics": {
            "header_normalization": header_normalization,
            "lower_line_tail_restoration": lower_line_tail_restoration,
            "normalization": build_mrz_normalization_report(
                raw_mrz,
                normalized_mrz,
                header=normalized_header,
                lower_line_prefix_inserted=lower_line_prefix_inserted,
                issues=normalization_issues,
            ),
        },
    }


def normalize_passport_mrz_from_text_result(
    result: Any,
    *,
    engine: str,
    normalize: bool = True,
) -> dict[str, Any]:
    """Extract one passport MRZ pair from a general text OCR response."""
    if not isinstance(result, dict) or not isinstance(
        result.get("items"), list
    ):
        return {"mrz": []}

    raw_lines: list[str] = []
    for item in result["items"]:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            return {"mrz": []}
        raw_lines.append(item["text"])

    try:
        normalized = normalize_passport_mrz(raw_lines, normalize=normalize)
    except PassportHeaderNormalizationError:
        return {"mrz": []}
    if len(normalized["mrz"]) != 2:
        return {"mrz": []}
    normalized["diagnostics"].update(
        {
            "engine": engine,
            "detected_regions_count": sum(
                len(text.splitlines()) for text in raw_lines
            ),
            "selected_regions_count": 2,
            "preprocessing": {"source": "original"},
        }
    )
    return normalized


def process_passport_bytes(
    content: bytes,
    ocr: PaddleOCR | None = None,
    *,
    normalize: bool = True,
) -> dict[str, Any]:
    image = decode_image_bytes(content)
    return process_passport_image(image, ocr=ocr, normalize=normalize)


def process_passport_bytes_with_mrz_scanner(
    content: bytes,
    scanner: Any = None,
    *,
    normalize: bool = True,
) -> dict[str, Any]:
    image = decode_image_bytes(content)
    return process_passport_image_with_mrz_scanner(
        image,
        scanner=scanner,
        normalize=normalize,
    )
