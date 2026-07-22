from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np

from app.ocr_engine.contracts import DocumentOrientation
from app.ocr_engine.paddle.factory import direct_ocr_pipeline

if TYPE_CHECKING:
    from paddleocr import PaddleOCR


def _orientation_prediction(result: object) -> DocumentOrientation:
    message = "PaddleOCR returned invalid document orientation"
    try:
        if not isinstance(result, Mapping):
            raise ValueError("Result must be a mapping")
        labels = np.asarray(result["label_names"], dtype=object)
        scores = np.asarray(result["scores"], dtype=object)
        if (
            labels.ndim != 1
            or scores.ndim != 1
            or not len(labels)
            or len(labels) != len(scores)
        ):
            raise ValueError(
                "Labels and scores must be matching nonempty vectors"
            )

        angles, probabilities = [], []
        for label, score in zip(labels, scores):
            if isinstance(label, str) and label in {"0", "90", "180", "270"}:
                angle = int(label)
            elif (
                isinstance(label, Integral)
                and not isinstance(label, bool)
                and label in (0, 90, 180, 270)
            ):
                angle = int(label)
            else:
                raise ValueError("Invalid orientation label")
            if isinstance(score, bool) or not isinstance(score, Real):
                raise ValueError("Invalid orientation score")
            probability = float(score)
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Invalid orientation score")
            angles.append(angle)
            probabilities.append(probability)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(message) from error

    # PaddleX doc_preprocessor passes label_names[0] to rotate_image(), whose
    # positive angle uses OpenCV's counterclockwise getRotationMatrix2D().
    # Our public value is the equivalent clockwise correction, not that label.
    return DocumentOrientation(
        rotation_degrees=(-angles[0]) % 360,
        confidence=probabilities[0],
    )


def predict_document_orientation(
    image: np.ndarray,
    *,
    ocr: PaddleOCR,
) -> DocumentOrientation:
    """Run only the document classifier already owned by this OCR instance."""
    pipeline = direct_ocr_pipeline(ocr)
    preprocessor = getattr(pipeline, "doc_preprocessor_pipeline", None)
    model = getattr(preprocessor, "doc_ori_classify_model", None)
    if model is None:
        internal = getattr(preprocessor, "_pipeline", None)
        model = getattr(internal, "doc_ori_classify_model", None)
    predict = getattr(model, "predict", None)
    if not callable(predict):
        raise RuntimeError(
            "PaddleOCR document orientation model is unavailable"
        )

    try:
        predictions = list(predict(image))
    except TypeError as error:
        raise RuntimeError(
            "PaddleOCR returned invalid document orientation"
        ) from error
    if len(predictions) != 1:
        raise RuntimeError(
            "PaddleOCR document orientation result count must be one"
        )
    return _orientation_prediction(predictions[0])
