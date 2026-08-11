from __future__ import annotations

from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from app.modules.v1.crop.text_pipeline import (
    process_mrz_crop_text_bytes_with_mrz_scanner,
)
from app.modules.v1.passport.inference_worker import (
    mrz_scanner_inference_worker,
)


def encoded_image():
    image = np.full((100, 220, 3), (19, 87, 213), dtype=np.uint8)
    image[12:31, 8:180] = (221, 91, 11)
    image[60:79, 8:180] = (31, 203, 97)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return image, encoded.tobytes()


def test_scanner_returns_raw_lines_without_postprocessing():
    image, content = encoded_image()
    texts = [" p<4RMTEST<<N4ME< ", "ABO123I<O<  ", ""]
    scanner = Mock(
        return_value={
            "mrz_polygon": [[8, 12], [180, 12], [180, 79], [8, 79]],
            "mrz_texts": texts,
        }
    )
    with patch(
        "app.modules.v1.passport.pipeline.normalize_passport_mrz"
    ) as normalize:
        result = process_mrz_crop_text_bytes_with_mrz_scanner(
            content, scanner=scanner
        )
    normalize.assert_not_called()
    assert scanner.call_count == 1
    np.testing.assert_array_equal(scanner.call_args.args[0], image)
    assert scanner.call_args.kwargs == {
        "do_center_crop": False,
        "do_postprocess": False,
    }
    assert result == {
        "items": [
            {
                "text": text,
                "confidence": None,
                "box": None,
                "detection_confidence": None,
            }
            for text in texts
        ],
        "image_width": 220,
        "image_height": 100,
    }


def test_scanner_without_detection_returns_empty_items():
    _, content = encoded_image()
    scanner = Mock(return_value=None)
    assert process_mrz_crop_text_bytes_with_mrz_scanner(
        content, scanner=scanner
    ) == {
        "items": [],
        "image_width": 220,
        "image_height": 100,
    }


def test_invalid_image_is_rejected_before_scanner_runs():
    scanner = Mock()
    with pytest.raises(ValueError, match="Cannot read image"):
        process_mrz_crop_text_bytes_with_mrz_scanner(
            b"invalid-image", scanner=scanner
        )
    scanner.assert_not_called()


def test_scanner_worker_reuses_engine_and_keeps_passport_operation_separate():
    scanner, connection = object(), object()
    expected = {"items": [], "image_width": 12, "image_height": 4}

    def serve(connection_arg, initialize, dispatch):
        assert connection_arg is connection
        engine = initialize()
        for _ in range(2):
            assert (
                dispatch(engine, "mrz_crop_text", (b"mrz-crop",), {})
                is expected
            )
        dispatch(engine, "passport_v1", (b"passport",), {})
        with pytest.raises(
            ValueError, match="Unsupported MRZ Scanner operation"
        ):
            dispatch(engine, "unknown", (b"mrz-crop",), {})

    with (
        patch(
            "app.modules.v1.passport.inference_worker.serve_inference_worker",
            side_effect=serve,
        ),
        patch(
            "app.ocr_engine.mrz.factory.create_mrz_scanner",
            return_value=scanner,
        ) as create,
        patch(
            "app.modules.v1.crop.text_pipeline."
            "process_mrz_crop_text_bytes_with_mrz_scanner",
            return_value=expected,
        ) as raw,
        patch(
            "app.modules.v1.passport.pipeline."
            "process_passport_bytes_with_mrz_scanner"
        ) as passport,
    ):
        mrz_scanner_inference_worker(connection)
    create.assert_called_once_with()
    assert raw.call_count == 2
    raw.assert_called_with(b"mrz-crop", scanner=scanner)
    passport.assert_called_once_with(b"passport", scanner=scanner)
