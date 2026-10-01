from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, TypedDict

import cv2
import numpy as np

from app.ocr_engine.russian.factory import create_pipeline
from app.ocr_engine.russian.recognizer import decode_image_bytes


class FieldCrop(TypedDict):
    name: str
    bbox: list[int]
    detection_bbox: list[int]
    confidence: float
    image: np.ndarray
    rotation_degrees: int


class PassportFieldDetection(TypedDict):
    document_image: np.ndarray
    fields: list[FieldCrop]
    rotation_degrees: int
    orientation_confidence: float
    document_detected: bool
    original_width: int
    original_height: int
    timings: dict[str, float]


def detect_passport_fields(
    image: np.ndarray | str | Path,
    pipeline: Any | None = None,
    *,
    device: str = "cpu",
    crop_document: bool = True,
    padding: int = 4,
) -> PassportFieldDetection:
    """Detect named fields without word detection or text recognition.

    Images are RGB. Both bounding boxes use pixel coordinates in the returned
    oriented document, with exclusive right/bottom edges. ``bbox`` includes
    padding; ``detection_bbox`` is the unpadded detection clipped to the image.
    Rotations are counterclockwise; a rotated crop retains its document bbox.

    The existing stage modules resize their own model inputs and map detections
    back to the supplied image. Avoid Pipeline.__call__: it downscales the
    source and invokes the word detector even with OCR disabled.
    """
    if not isinstance(padding, int) or isinstance(padding, bool):
        raise ValueError("Field padding must be an integer between 0 and 64")
    if not 0 <= padding <= 64:
        raise ValueError("Field padding must be an integer between 0 and 64")
    if isinstance(image, (str, Path)):
        image = decode_image_bytes(Path(image).read_bytes())
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or not image.size
    ):
        raise ValueError("Expected a nonempty RGB uint8 image")

    original_height, original_width = image.shape[:2]
    if pipeline is None:
        pipeline = create_pipeline(device=device)

    timings: dict[str, float] = {}
    started = perf_counter()
    angle = pipeline.angle90.predict_transform(image)["Angle90"]
    document_image = angle["warped_img"]
    timings["orientation"] = perf_counter() - started

    document_detected = False
    if crop_document:
        started = perf_counter()
        document = pipeline.doc_detector.predict_transform(document_image)[
            "DocDetector"
        ]
        document_image = document["warped_img"]
        segments = document.get("segm")
        document_detected = segments is not None and len(segments) > 0
        timings["document"] = perf_counter() - started

    started = perf_counter()
    detections = pipeline.text_fields.predict(document_image)[
        "TextFieldsDetector"
    ]["bbox"]
    height, width = document_image.shape[:2]
    fields: list[FieldCrop] = []
    for detection in detections:
        # Detectors can return boxes that extend beyond the image boundary.
        # Clip before padding so an off-image/empty detection cannot yield an
        # unrelated patch, and negative indices never wrap around the image.
        x1, y1, x2, y2 = map(int, detection[:4])
        x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
        y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            continue
        detection_bbox = [x1, y1, x2, y2]
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(width, x2 + padding), min(height, y2 + padding)
        crop = document_image[y1:y2, x1:x2].copy()
        name = str(detection[-1])
        rotation_degrees = 0
        detected_width = detection_bbox[2] - detection_bbox[0]
        detected_height = detection_bbox[3] - detection_bbox[1]
        if name == "Licence_number" and detected_height > detected_width:
            crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
            rotation_degrees = 90
        fields.append(
            {
                "name": name,
                "bbox": [x1, y1, x2, y2],
                "detection_bbox": detection_bbox,
                "confidence": float(detection[4]),
                "image": crop,
                "rotation_degrees": rotation_degrees,
            }
        )
    timings["fields"] = perf_counter() - started
    timings["total"] = sum(timings.values())

    return {
        "document_image": document_image,
        "fields": fields,
        "rotation_degrees": int(angle["angle"]),
        "orientation_confidence": float(angle["confidence"]),
        "document_detected": document_detected,
        "original_width": original_width,
        "original_height": original_height,
        "timings": timings,
    }
