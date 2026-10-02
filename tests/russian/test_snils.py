from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

from app.modules.russian.snils import recognize_snils_bytes
from tests.russian.test_engine_adapter import image_bytes


def test_snils_adapter_returns_lossless_detected_date_and_number_crops():
    birth_crop = np.full((18, 84, 3), (14, 95, 218), dtype=np.uint8)
    number_crop = np.full((12, 94, 3), (200, 11, 90), dtype=np.uint8)
    result = SimpleNamespace(
        doctype="SNILS_1996",
        ocr={
            "Last_name_ru": "ПРИМЕРОВ",
            "Birth_date": "2 ЯНВАРЯ 1990",
            "Licence_number": "456-789-123 00",
        },
        text_fields=(
            [
                [0, 0, 84, 18, 0.9, 10, "Birth_date"],
                [0, 18, 94, 30, 0.8, 5, "Licence_number"],
                [0, 30, 12, 42, 0.9, 1, "Last_name_ru"],
            ],
            [birth_crop, number_crop, np.zeros((12, 12, 3), dtype=np.uint8)],
        ),
    )
    pipeline = Mock(process_img=Mock(return_value=result))

    actual = recognize_snils_bytes(image_bytes(), pipeline=pipeline)

    assert actual["doctype"] == "SNILS_1996"
    assert actual["ocr"] == result.ocr
    assert set(actual["crops"]) == {"Birth_date", "Licence_number"}
    for name, expected in (
        ("Birth_date", birth_crop),
        ("Licence_number", number_crop),
    ):
        encoded = actual["crops"][name]
        assert encoded.startswith(b"\x89PNG\r\n\x1a\n")
        np.testing.assert_array_equal(
            np.asarray(Image.open(BytesIO(encoded))), expected
        )
    pipeline.process_img.assert_called_once()


def test_snils_adapter_omits_unavailable_date_crops_without_guessing():
    pipeline = Mock(
        process_img=Mock(
            return_value=SimpleNamespace(
                doctype="SNILS_1996",
                ocr={"Birth_date": "неразборчиво"},
                text_fields=([], []),
            )
        )
    )

    actual = recognize_snils_bytes(image_bytes(), pipeline=pipeline)

    assert actual["ocr"]["Birth_date"] == "неразборчиво"
    assert actual["crops"] == {}


@pytest.mark.parametrize("content", [b"", b"not-an-image"])
def test_snils_adapter_rejects_unreadable_image_before_inference(content):
    pipeline = Mock()

    with pytest.raises(ValueError, match="Cannot read image"):
        recognize_snils_bytes(content, pipeline=pipeline)

    pipeline.process_img.assert_not_called()


def test_russian_worker_dispatches_snils_with_its_loaded_pipeline():
    from app.modules.russian.inference_worker import russian_inference_worker

    connection = Mock()
    pipeline = Mock()
    pipeline.process_img.return_value = SimpleNamespace(
        doctype="SNILS_1996",
        ocr={"Birth_date": "2 ЯНВАРЯ 1990"},
        text_fields=([], []),
    )
    with (
        patch(
            "app.ocr_engine.russian.factory.create_pipeline",
            return_value=pipeline,
        ),
        patch(
            "app.modules.russian.inference_worker.serve_inference_worker"
        ) as serve,
    ):
        russian_inference_worker(connection, "cpu")
        _, initialize, dispatch = serve.call_args.args
        assert initialize() is pipeline
        result = dispatch(pipeline, "snils", (image_bytes(),), {})

    assert result["doctype"] == "SNILS_1996"
    assert result["ocr"]["Birth_date"] == "2 ЯНВАРЯ 1990"
    pipeline.process_img.assert_called_once()
