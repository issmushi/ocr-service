"""Russian OCR worker keeps licence fields and date patches together."""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image

from app.modules.russian.driver_license import recognize_driver_license_bytes
from tests.russian.test_engine_adapter import image_bytes


def test_driver_license_adapter_returns_best_lossless_date_crop():
    low = np.full((10, 20, 3), 10, dtype=np.uint8)
    high = np.full((10, 20, 3), 210, dtype=np.uint8)
    result = SimpleNamespace(
        doctype="DL_2020",
        ocr={"Birth_date": "03.02.198S", "Licence_number": "11 22 345678"},
        text_fields=(
            [
                [0, 0, 20, 10, 0.4, 1, "Birth_date"],
                [0, 0, 20, 10, 0.9, 1, "Birth_date"],
                [0, 0, 20, 10, 0.9, 1, "Licence_number"],
            ],
            [low, high, low],
        ),
    )
    pipeline = Mock(process_img=Mock(return_value=result))

    actual = recognize_driver_license_bytes(image_bytes(), pipeline=pipeline)

    assert actual["doctype"] == "DL_2020"
    assert actual["ocr"] == result.ocr
    assert set(actual["crops"]) == {"Birth_date"}
    assert np.array_equal(
        np.asarray(Image.open(BytesIO(actual["crops"]["Birth_date"]))), high
    )
