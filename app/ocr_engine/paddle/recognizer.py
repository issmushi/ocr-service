from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import cv2
import numpy as np

from app.ocr_engine.contracts import RecognizedText
from app.ocr_engine.paddle.factory import direct_ocr_pipeline

if TYPE_CHECKING:
    from paddleocr import PaddleOCR


def recognize_text_crops(
    ocr: PaddleOCR,
    crops: Sequence[np.ndarray],
    *,
    use_textline_orientation: bool = False,
) -> list[RecognizedText]:
    """Recognize a caller-selected batch of text crops."""
    if not crops:
        return []

    pipeline = direct_ocr_pipeline(ocr)
    prepared_crops = list(crops)
    if use_textline_orientation:
        orientation_model = getattr(
            pipeline, "textline_orientation_model", None
        )
        if orientation_model is None:
            raise RuntimeError(
                "PaddleOCR textline orientation model is unavailable"
            )
        orientation_results = list(orientation_model.predict(prepared_crops))
        if len(orientation_results) != len(prepared_crops):
            raise RuntimeError(
                "PaddleOCR orientation result count does not match text crops"
            )

        angles = []
        for result in orientation_results:
            try:
                # PaddleX uses class 0 for upright and class 1 for 180 degrees.
                angle = int(
                    np.asarray(result["class_ids"], dtype=np.int64).ravel()[0]
                )
            except (
                KeyError,
                IndexError,
                TypeError,
                ValueError,
                OverflowError,
            ) as error:
                raise RuntimeError(
                    "PaddleOCR returned invalid textline orientation"
                ) from error
            if angle not in (0, 1):
                raise RuntimeError(
                    "PaddleOCR returned invalid textline orientation"
                )
            angles.append(angle)
        prepared_crops = [
            cv2.rotate(crop, cv2.ROTATE_180) if angle == 1 else crop
            for crop, angle in zip(prepared_crops, angles)
        ]

    raw_results = list(pipeline.text_rec_model.predict(prepared_crops))
    if len(raw_results) != len(prepared_crops):
        raise RuntimeError(
            "PaddleOCR recognition result count does not match text crops"
        )
    return [
        RecognizedText(
            text=str(result.get("rec_text") or "").strip(),
            confidence=float(result.get("rec_score") or 0.0),
        )
        for result in raw_results
    ]
