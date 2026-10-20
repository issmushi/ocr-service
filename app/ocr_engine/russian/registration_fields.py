"""Find registration lines and stamps without running text recognition."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from app.ocr_engine.russian.vendor.document_processing.geometry import (
    corners,
)
from app.ocr_engine.russian.vendor.document_processing.pipeline_modules import (  # noqa: E501
    AddressLinesDetector,
)

STAMP_LABELS = {
    "Stamp_registered": "Штамп регистрации",
    "Stamp_deregistered": "Штамп снятия с учёта",
    "Stamp_unclear": "Неопределённый штамп",
}


def _quad(box: list[float]) -> list[list[float]]:
    width, height = max(1, round(box[2])), max(1, round(box[3]))
    geometry = AddressLinesDetector.crop_geometry(box)
    return geometry.to_input(corners(0, 0, width, height)).tolist()


def detect_registration_fields(
    image: np.ndarray, pipeline: Any, *, padding: int = 4
) -> dict[str, Any]:
    if isinstance(padding, bool) or not isinstance(padding, int):
        raise ValueError("Padding must be an integer")
    if not 0 <= padding <= 64:
        raise ValueError("Padding must be between 0 and 64")

    started = perf_counter()
    result = pipeline.process_img(
        image, ocr=False, find_text_fields=False, check_quality=False
    )
    document = result.img_with_fixed_perspective
    metadata = result.meta_results
    timings = {"document": perf_counter() - started}
    fields = []
    if str(result.doctype or "").startswith("INTPASSPORTADDR"):
        started = perf_counter()
        pipeline._ensure_address_models()
        timings["model_loading"] = perf_counter() - started
        for detector, kind in (
            (pipeline.address_lines, "address_line"),
            (pipeline.registration_stamps, "registration_stamp"),
        ):
            started = perf_counter()
            boxes = detector.predict(document)[detector.model_name]["obbox"]
            for box in boxes:
                rectangle = [float(value) for value in box[:5]]
                if (
                    not np.isfinite(rectangle).all()
                    or rectangle[2] <= 0
                    or rectangle[3] <= 0
                ):
                    continue
                padded = rectangle.copy()
                padded[2] += 2 * padding
                padded[3] += 2 * padding
                patch = AddressLinesDetector.crop_rotated(document, padded)
                fields.append(
                    {
                        "kind": kind,
                        "name": (
                            "Address"
                            if kind == "address_line"
                            else "Registration_stamp"
                        ),
                        "label": (
                            "Строка адреса"
                            if kind == "address_line"
                            else STAMP_LABELS.get(str(box[7]), "Штамп")
                        ),
                        "detector_label": str(box[7]),
                        "confidence": float(box[5]),
                        "quad": _quad(rectangle),
                        "crop_quad": _quad(padded),
                        "rotation_degrees": float(np.degrees(rectangle[4])),
                        "image": patch,
                    }
                )
            timings[kind] = perf_counter() - started
    return {
        "document_image": document,
        "document_type": result.doctype or "NONE",
        "document_detected": bool(
            result.documents or (metadata.get("DocDetector") or {}).get("segm")
        ),
        "rotation_degrees": int(result.angle),
        "orientation_confidence": float(metadata["Angle90"]["confidence"]),
        "fields": fields,
        "timings": timings,
    }
