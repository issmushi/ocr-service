from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from app.core.images.preprocessing import PreparedImage
from app.modules.v1.passport.pipeline import (
    normalize_passport_mrz_from_text_result,
    process_passport_image,
    process_passport_image_with_mrz_scanner,
)
from app.modules.v1.passport.recognition import (
    select_longest_passport_mrz_lines,
)

POLYGON = [[0, 0], [100, 0], [100, 30], [0, 30]]
ARMENIAN_UPPER = "P<ARMHAYRAPETYAN<<ARAM".ljust(44, "<")
ARMENIAN_LOWER = "AA12345678ARM9001158M3001156<<<<<<<<<<<<<<06"


def _lowercase_with_whitespace(value: str) -> str:
    return (
        f" \t{value[:11].lower()}\u00a0{value[11:29].lower()}"
        f"\u2003{value[29:].lower()} "
    )


NOISY_TEXT_LINES = (
    _lowercase_with_whitespace(ARMENIAN_UPPER),
    " n \t" * 43,
    _lowercase_with_whitespace(ARMENIAN_LOWER),
    "z" * 42,
)


def test_longest_line_selection_compacts_and_preserves_source_order():
    assert select_longest_passport_mrz_lines(
        (
            "ignored\n \t\u00a0\nab\tcd\u00a0ef",
            " x\u2003x x x x x x x ",
            "qwertyuiop",
        )
    ) == ("XXXXXXXX", "QWERTYUIOP")


def test_longest_line_selection_is_stable_at_the_cutoff():
    assert select_longest_passport_mrz_lines(
        ("aaaaaa", "bbbbbb", "cccccc")
    ) == ("AAAAAA", "BBBBBB")


def test_both_engines_share_normalization_and_preserve_raw_characters():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    raw_lines = (
        "<UZ8TEST0V<<TEST",
        "4412345678U289001158H350115112345678901234",
    )
    expected = [
        "P<UZBTESTOV<<TEST".ljust(44, "<"),
        "AA12345678UZB9001158M35011511234567890123456",
    ]
    scanner = Mock(
        return_value={"mrz_polygon": POLYGON, "mrz_texts": raw_lines}
    )
    paddle_recognition = SimpleNamespace(
        lines=raw_lines,
        detected_regions_count=5,
        selected_regions_count=2,
    )
    with (
        patch(
            "app.modules.v1.passport.pipeline.prepare_document_image",
            return_value=PreparedImage(
                image=image, metadata={"prepared": True}
            ),
        ) as prepare,
        patch(
            "app.modules.v1.passport.pipeline.recognize_raw_passport_mrz",
            return_value=paddle_recognition,
        ) as paddle_recognize,
    ):
        paddle_result = process_passport_image(image, ocr=object())
        paddle_recognize.reset_mock()
        prepare.reset_mock()
        mrz_result = process_passport_image_with_mrz_scanner(
            image, scanner=scanner
        )
        paddle_recognize.assert_not_called()
        prepare.assert_not_called()

    assert paddle_result["mrz"] == mrz_result["mrz"] == expected
    assert mrz_result["raw_mrz"] == list(raw_lines)
    assert (
        mrz_result["diagnostics"]["normalization"]
        == paddle_result["diagnostics"]["normalization"]
    )
    assert mrz_result["diagnostics"]["normalization"]["corrections_count"] > 0
    assert mrz_result["diagnostics"]["engine"] == "mrz"
    assert mrz_result["diagnostics"]["preprocessing"] == {"source": "original"}
    assert mrz_result["diagnostics"]["detected_regions_count"] == 1
    scanner.assert_called_once_with(
        image, do_center_crop=False, do_postprocess=False
    )


def test_mrz_scanner_selects_longest_rows_instead_of_requiring_exactly_two():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    scanner = Mock(
        return_value={"mrz_polygon": POLYGON, "mrz_texts": NOISY_TEXT_LINES}
    )

    result = process_passport_image_with_mrz_scanner(image, scanner=scanner)

    assert result["raw_mrz"] == [ARMENIAN_UPPER, ARMENIAN_LOWER]
    assert result["mrz"] == [ARMENIAN_UPPER, ARMENIAN_LOWER]
    assert result["diagnostics"]["engine"] == "mrz"


def test_paddle_selects_longest_rows_instead_of_the_last_two():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    regions = tuple(SimpleNamespace(box=POLYGON) for _ in NOISY_TEXT_LINES)
    detection = SimpleNamespace(image=image, regions=regions)
    recognized = tuple(SimpleNamespace(text=text) for text in NOISY_TEXT_LINES)

    with (
        patch(
            "app.modules.v1.passport.pipeline.prepare_document_image",
            return_value=PreparedImage(image=image, metadata={}),
        ),
        patch(
            "app.modules.v1.passport.recognition.detect_text_regions",
            return_value=detection,
        ),
        patch(
            "app.modules.v1.passport.recognition.select_passport_mrz_regions",
            return_value=regions,
        ),
        patch(
            "app.modules.v1.passport.recognition.crop_regions",
            return_value=[image] * len(regions),
        ),
        patch(
            "app.modules.v1.passport.recognition.recognize_text_crops",
            return_value=recognized,
        ),
    ):
        result = process_passport_image(image, ocr=object())

    assert result["raw_mrz"] == [ARMENIAN_UPPER, ARMENIAN_LOWER]
    assert result["mrz"] == [ARMENIAN_UPPER, ARMENIAN_LOWER]
    assert result["diagnostics"]["engine"] == "paddle"


@pytest.mark.parametrize(
    "engine",
    ("mrz", "qwen3-vl-4b", "qwen3-vl-8b"),
)
def test_general_text_adapter_selects_longest_non_adjacent_rows(engine: str):
    result = normalize_passport_mrz_from_text_result(
        {"items": [{"text": text} for text in NOISY_TEXT_LINES]},
        engine=engine,
    )

    assert result["raw_mrz"] == [ARMENIAN_UPPER, ARMENIAN_LOWER]
    assert result["mrz"] == [ARMENIAN_UPPER, ARMENIAN_LOWER]
    assert result["diagnostics"]["engine"] == engine
    assert result["diagnostics"]["detected_regions_count"] == 4
    assert result["diagnostics"]["selected_regions_count"] == 2


def test_general_text_adapter_rejects_invalid_shape_and_unknown_header():
    for raw in (
        None,
        {},
        {"items": None},
        {"items": [{"text": 123}]},
        {
            "items": [
                {
                    "text": (
                        "I<UTO123456789012345678901234\n"
                        "123456789012345678901234567890"
                    )
                }
            ]
        },
    ):
        assert normalize_passport_mrz_from_text_result(
            raw, engine="qwen3-vl-4b"
        ) == {"mrz": []}
