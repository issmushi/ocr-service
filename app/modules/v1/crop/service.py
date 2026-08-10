from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from app.core.config import settings
from app.core.errors import AppError, ErrorCode
from app.core.images.decoding import InvalidImageError, decode_image
from app.core.images.geometry import order_quad_points
from app.core.images.mrz_detection import find_mrz_bbox
from app.core.images.orientation import rotate_document
from app.core.images.preprocessing import find_document_quad, warp_document
from app.core.images.validation import validate_image_dimensions
from app.modules.paddle.runtime import paddle_runtime
from app.ocr_engine.contracts import DocumentOrientation

MIN_ORIENTATION_CONFIDENCE = 0.80


@dataclass(frozen=True)
class _PreparedDocument:
    original: np.ndarray
    cropped: np.ndarray
    quad: np.ndarray | None
    started: float


def _jpeg_bytes(image: np.ndarray) -> bytes:
    encoded, buffer = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 94],
    )
    if not encoded:
        raise AppError(
            ErrorCode.INTERNAL_ERROR, "Не удалось подготовить изображение"
        )
    return buffer.tobytes()


def _jpeg_data_url(image: np.ndarray) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(
        _jpeg_bytes(image),
    ).decode("ascii")


def _prepare_document(content: bytes) -> _PreparedDocument:
    started = perf_counter()
    try:
        with decode_image(content) as decoded:
            validate_image_dimensions(*decoded.size)
            image = cv2.cvtColor(np.asarray(decoded), cv2.COLOR_RGB2BGR)
    except InvalidImageError as error:
        raise AppError(
            ErrorCode.IMAGE_READ_ERROR,
            "Изображение невозможно прочитать",
        ) from error

    document_quad = find_document_quad(image)
    cropped = (
        warp_document(image, document_quad)
        if document_quad is not None
        else image
    )
    return _PreparedDocument(image, cropped, document_quad, started)


def _has_content(image: np.ndarray) -> bool:
    # A missing external border is normal for a scan that already fills the
    # frame. Only a virtually uniform image can skip orientation/text search.
    return float(cv2.meanStdDev(image)[1].max()) >= 2.0


def _rotated_points(
    points: np.ndarray, width: int, height: int, rotation: int
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    x, y = points[:, 0], points[:, 1]
    if rotation == 90:
        return np.column_stack((height - 1 - y, x)).astype(np.float32)
    if rotation == 180:
        return np.column_stack((width - 1 - x, height - 1 - y)).astype(
            np.float32
        )
    if rotation == 270:
        return np.column_stack((y, width - 1 - x)).astype(np.float32)
    return points.copy()


def _bbox_quad(bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.array(
        [[x1, y1], [x2 - 1, y1], [x2 - 1, y2 - 1], [x1, y2 - 1]], np.float32
    )


def _refine_document(
    prepared: _PreparedDocument,
    rotation: int,
    cropped: np.ndarray,
    mrz_bbox: tuple[int, int, int, int] | None,
) -> tuple[np.ndarray, np.ndarray | None, tuple[int, int, int, int] | None]:
    """Reject borders that cut known MRZ and prefer the complete data page."""
    original = prepared.original
    height, width = original.shape[:2]
    upright = rotate_document(original, rotation)
    source_bbox = (
        find_mrz_bbox(upright) if prepared.quad is not None else mrz_bbox
    )
    region = _bbox_quad(source_bbox) if source_bbox is not None else None
    if region is None and mrz_bbox is not None and prepared.quad is not None:
        # MRZ may be too small in the full photo but visible after
        # rectification.
        # Map its four corners back to the upright source before ranking
        # borders.
        crop_height, crop_width = cropped.shape[:2]
        upright_quad = order_quad_points(
            _rotated_points(prepared.quad, width, height, rotation)
        )
        matrix = cv2.getPerspectiveTransform(
            _bbox_quad((0, 0, crop_width, crop_height)),
            upright_quad,
        )
        region = cv2.perspectiveTransform(_bbox_quad(mrz_bbox)[None], matrix)[
            0
        ]
    if region is None:
        return cropped, prepared.quad, mrz_bbox

    selected = find_document_quad(
        upright, required_region=region, prefer_landscape=True
    )
    if selected is None:
        # A complete visible MRZ is stronger evidence than an internal printed
        # separator. Keep an already framed scan instead of losing its fields.
        if source_bbox is not None:
            return upright, None, source_bbox
        return cropped, prepared.quad, mrz_bbox
    upright_height, upright_width = upright.shape[:2]
    source_quad = order_quad_points(
        _rotated_points(
            selected,
            upright_width,
            upright_height,
            (-rotation) % 360,
        )
    )
    if prepared.quad is not None and np.allclose(
        source_quad, prepared.quad, atol=1.0
    ):
        refined, source_quad, refined_bbox = cropped, prepared.quad, mrz_bbox
    else:
        refined = rotate_document(
            warp_document(original, source_quad), rotation
        )
        refined_bbox = find_mrz_bbox(refined)
    # Rectification can join or weaken end glyphs. Preserve the MRZ that was
    # already observed instead of letting a second detection truncate it.
    refined_height, refined_width = refined.shape[:2]
    matrix = cv2.getPerspectiveTransform(
        order_quad_points(selected),
        _bbox_quad((0, 0, refined_width, refined_height)),
    )
    projected = cv2.perspectiveTransform(region[None], matrix)[0]
    lower = np.floor(projected.min(axis=0) + 1e-3).astype(int)
    upper = np.ceil(projected.max(axis=0) - 1e-3).astype(int) + 1
    known_bbox = (
        max(0, int(lower[0])),
        max(0, int(lower[1])),
        min(refined_width, int(upper[0])),
        min(refined_height, int(upper[1])),
    )
    if refined_bbox is None:
        refined_bbox = known_bbox
    else:
        refined_bbox = (
            min(refined_bbox[0], known_bbox[0]),
            min(refined_bbox[1], known_bbox[1]),
            max(refined_bbox[2], known_bbox[2]),
            max(refined_bbox[3], known_bbox[3]),
        )
    return refined, source_quad, refined_bbox


def _finish_document(
    prepared: _PreparedDocument,
    orientation: DocumentOrientation | None,
) -> dict[str, Any]:
    image, cropped, document_quad = (
        prepared.original,
        prepared.cropped,
        prepared.quad,
    )
    orientation_detected = (
        orientation is not None
        and orientation.confidence >= MIN_ORIENTATION_CONFIDENCE
    )
    document_rotation = (
        orientation.rotation_degrees if orientation_detected else 0
    )
    if orientation_detected:
        cropped = rotate_document(cropped, document_rotation)
    height, width = image.shape[:2]
    mrz_bbox = None
    if _has_content(image):
        mrz_bbox = find_mrz_bbox(cropped)
        cropped, document_quad, mrz_bbox = _refine_document(
            prepared,
            document_rotation,
            cropped,
            mrz_bbox,
        )
    cropped_height, cropped_width = cropped.shape[:2]
    mrz_image = None
    mrz_width = mrz_height = None
    if mrz_bbox is not None:
        x1, y1, x2, y2 = mrz_bbox
        mrz_image = _jpeg_data_url(cropped[y1:y2, x1:x2])
        mrz_width, mrz_height = x2 - x1, y2 - y1
    original_url = _jpeg_data_url(image)
    cropped_url = original_url if cropped is image else _jpeg_data_url(cropped)
    return {
        "document_detected": document_quad is not None,
        "original_image": original_url,
        "cropped_image": cropped_url,
        "document_quad": (
            document_quad.astype(float).round(2).tolist()
            if document_quad is not None
            else None
        ),
        "original_width": width,
        "original_height": height,
        "cropped_width": cropped_width,
        "cropped_height": cropped_height,
        "orientation_detected": orientation_detected,
        "document_rotation": document_rotation,
        "orientation_confidence": orientation.confidence
        if orientation is not None
        else None,
        "mrz_detected": mrz_bbox is not None,
        "mrz_image": mrz_image,
        "mrz_bbox": list(mrz_bbox) if mrz_bbox is not None else None,
        "mrz_width": mrz_width,
        "mrz_height": mrz_height,
        "duration_seconds": round(perf_counter() - prepared.started, 4),
    }


def crop_document_bytes(
    content: bytes,
    *,
    orientation_predictor: Callable[[np.ndarray], DocumentOrientation]
    | None = None,
) -> dict[str, Any]:
    """In-memory pipeline; inject an orientation engine for
    offline processing.
    """
    prepared = _prepare_document(content)
    orientation = (
        orientation_predictor(prepared.cropped)
        if _has_content(prepared.original)
        and orientation_predictor is not None
        else None
    )
    return _finish_document(prepared, orientation)


def _orientation_input(image: np.ndarray) -> bytes:
    # Keep the crop lossless while transferring it to the existing model
    # process.
    encoded, buffer = cv2.imencode(
        ".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1]
    )
    if not encoded:
        raise AppError(
            ErrorCode.INTERNAL_ERROR,
            "Не удалось подготовить документ для выравнивания",
        )
    return buffer.tobytes()


async def crop_document(content: bytes) -> dict[str, Any]:
    """Run the production crop pipeline in memory without saving a capture."""
    prepared = await asyncio.to_thread(_prepare_document, content)
    orientation = await _predict_document_orientation(prepared)
    return await asyncio.to_thread(_finish_document, prepared, orientation)


async def _predict_document_orientation(
    prepared: _PreparedDocument,
) -> DocumentOrientation | None:
    orientation = None
    if _has_content(prepared.original):
        orientation_input = await asyncio.to_thread(
            _orientation_input, prepared.cropped
        )
        prediction = await paddle_runtime.run(
            "document_orientation",
            orientation_input,
            timeout_seconds=settings.paddle_ocr_document_hard_timeout_seconds,
        )
        orientation = DocumentOrientation(
            rotation_degrees=prediction["rotation_degrees"],
            confidence=prediction["confidence"],
        )
    return orientation


async def crop_document_image_bytes(content: bytes) -> bytes:
    """Crop and orient the document without searching for or extracting MRZ."""
    prepared = await asyncio.to_thread(_prepare_document, content)
    orientation = await _predict_document_orientation(prepared)
    cropped = prepared.cropped
    if (
        orientation is not None
        and orientation.confidence >= MIN_ORIENTATION_CONFIDENCE
    ):
        cropped = rotate_document(cropped, orientation.rotation_degrees)
    return await asyncio.to_thread(_jpeg_bytes, cropped)


async def crop_mrz_bytes(content: bytes) -> bytes:
    """Return only the detected MRZ JPEG using the shared crop pipeline."""
    result = await crop_document(content)
    mrz_image = result.get("mrz_image")
    if not result.get("mrz_detected") or not isinstance(mrz_image, str):
        raise AppError(
            ErrorCode.OCR_FAILED, "Не удалось найти MRZ на изображении"
        )
    try:
        prefix, encoded = mrz_image.split(",", 1)
        if prefix != "data:image/jpeg;base64":
            raise ValueError("Unexpected MRZ image format")
        crop = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as error:
        raise AppError(
            ErrorCode.INTERNAL_ERROR,
            "Не удалось подготовить найденную область MRZ",
        ) from error
    if not crop:
        raise AppError(
            ErrorCode.INTERNAL_ERROR,
            "Не удалось подготовить найденную область MRZ",
        )
    return crop
