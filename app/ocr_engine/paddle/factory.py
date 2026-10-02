from __future__ import annotations

import contextlib
import os
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from paddleocr import PaddleOCR, TextRecognition

CYRILLIC_TEXT_MODEL = "cyrillic_PP-OCRv5_mobile_rec"


def create_cyrillic_text_recognizer() -> TextRecognition:
    """Load only the experimental Cyrillic text recognizer, always on CPU."""
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="No ccache found.*",
                category=UserWarning,
            )
            with (
                contextlib.redirect_stdout(devnull),
                contextlib.redirect_stderr(devnull),
            ):
                from paddleocr import TextRecognition as TextRecognitionClass

                return TextRecognitionClass(
                    model_name=CYRILLIC_TEXT_MODEL,
                    device="cpu",
                    enable_mkldnn=False,
                )


def create_paddle_ocr_quietly(kwargs: dict[str, Any]) -> PaddleOCR:
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="No ccache found.*",
                category=UserWarning,
            )
            with (
                contextlib.redirect_stdout(devnull),
                contextlib.redirect_stderr(devnull),
            ):
                from paddleocr import PaddleOCR as PaddleOCRClass

                return PaddleOCRClass(**kwargs)


def create_ocr(device: str | None = None) -> PaddleOCR:
    """Create the general Cyrillic and Latin PaddleOCR model pair."""
    current_kwargs: dict[str, Any] = {
        "lang": "ru",
        "use_textline_orientation": True,
    }
    legacy_kwargs: dict[str, Any] = {
        "lang": "ru",
        "use_angle_cls": False,
        "show_log": True,
    }

    if device:
        current_kwargs["device"] = device
        legacy_kwargs["device"] = device
        if device == "cpu":
            current_kwargs["enable_mkldnn"] = False
            legacy_kwargs["enable_mkldnn"] = False
            legacy_kwargs["cpu_threads"] = 1

    for candidate_kwargs in (
        current_kwargs,
        legacy_kwargs,
        {"lang": "ru", "use_textline_orientation": True},
        {"lang": "ru", "use_angle_cls": False, "show_log": True},
    ):
        try:
            return create_paddle_ocr_quietly(candidate_kwargs)
        except TypeError:
            continue

    fallback_kwargs: dict[str, Any] = {
        "lang": "ru",
        "use_angle_cls": False,
    }
    if device:
        fallback_kwargs["device"] = device
    return create_paddle_ocr_quietly(fallback_kwargs)


def create_mrz_ocr(device: str | None = None) -> PaddleOCR:
    """Create the lightweight Latin detector and recognizer used for MRZ."""
    return create_latin_ocr(device=device, use_textline_orientation=False)


def create_latin_ocr(
    device: str | None = None,
    *,
    use_textline_orientation: bool,
) -> PaddleOCR:
    current_kwargs: dict[str, Any] = {
        "lang": "en",
        "text_detection_model_name": "PP-OCRv5_mobile_det",
        "text_recognition_model_name": "latin_PP-OCRv5_mobile_rec",
        "use_doc_orientation_classify": True,
        "use_doc_unwarping": False,
        "use_textline_orientation": use_textline_orientation,
        "text_det_limit_side_len": 1600,
        "text_det_limit_type": "max",
    }
    if device:
        current_kwargs["device"] = device
        if device == "cpu":
            current_kwargs["enable_mkldnn"] = False

    try:
        return create_paddle_ocr_quietly(current_kwargs)
    except TypeError:
        if use_textline_orientation:
            raise
        legacy_kwargs: dict[str, Any] = {
            "lang": "en",
            "use_angle_cls": False,
            "show_log": False,
        }
        if device:
            legacy_kwargs["device"] = device
            if device == "cpu":
                legacy_kwargs["enable_mkldnn"] = False
                legacy_kwargs["cpu_threads"] = 1
        return create_paddle_ocr_quietly(legacy_kwargs)


def direct_ocr_pipeline(ocr: PaddleOCR) -> Any:
    """Return Paddle's detector/recognizer pair without the full pipeline."""
    paddlex_pipeline = getattr(ocr, "paddlex_pipeline", None)
    pipeline = getattr(paddlex_pipeline, "_pipeline", None)
    if (
        pipeline is None
        or getattr(pipeline, "text_det_model", None) is None
        or getattr(pipeline, "text_rec_model", None) is None
    ):
        raise RuntimeError(
            "Installed PaddleOCR version does not expose direct text models"
        )
    return pipeline
