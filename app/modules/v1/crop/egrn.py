from __future__ import annotations

import asyncio
import base64
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from fastapi import UploadFile

from app.core.config import settings
from app.core.errors import AppError, ErrorCode, api_success_response
from app.core.images.decoding import InvalidImageError, decode_image
from app.core.images.printed_content import crop_printed_content
from app.core.images.uploads import prepare_upload_stream
from app.core.images.validation import (
    validate_image_dimensions,
    validate_image_extension,
)
from app.core.model_availability import ensure_ocr_model_enabled
from app.modules.v1.crop.egrn_layout import FIELD_LABELS, locate_egrn_fields
from app.modules.v1.crop.egrn_schemas import EgrnOcrModel
from app.modules.v1.crop.field_ocr import recognize_field_previews
from app.modules.v1.crop.field_schemas import FieldOcrModel


def _prepare_egrn(content: bytes) -> np.ndarray:
    try:
        image = decode_image(content)
    except InvalidImageError as error:
        raise AppError(
            ErrorCode.IMAGE_READ_ERROR, "Изображение невозможно прочитать"
        ) from error
    validate_image_dimensions(*image.size)
    source = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    try:
        return crop_printed_content(source)
    except ValueError as error:
        raise AppError(
            ErrorCode.OCR_FAILED,
            "Не удалось найти печатное содержимое выписки на изображении",
        ) from error


def _png(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise AppError(
            ErrorCode.INTERNAL_ERROR, "Не удалось сохранить вырезку выписки"
        )
    return encoded.tobytes()


async def crop_egrn_bytes(content: bytes) -> bytes:
    return await asyncio.to_thread(lambda: _png(_prepare_egrn(content)))


def _preview(image: np.ndarray) -> dict[str, Any]:
    return {
        "image": "data:image/png;base64,"
        + base64.b64encode(_png(image)).decode("ascii"),
        "width": image.shape[1],
        "height": image.shape[0],
    }


def _crop_egrn_fields(content: bytes, include_page: bool) -> dict[str, Any]:
    started = perf_counter()
    document = _prepare_egrn(content)
    boxes = locate_egrn_fields(document)
    if not boxes:
        raise AppError(
            ErrorCode.OCR_FAILED,
            "Не найдены поля первой страницы выписки Госуслуг: "
            "ожидаются верхняя таблица из двух строк и таблица характеристик",
        )
    fields = []
    for name, label in FIELD_LABELS.items():
        if name not in boxes:
            continue
        x1, y1, x2, y2 = boxes[name]
        fields.append(
            {
                "name": name,
                "label": label,
                "bbox": [x1, y1, x2, y2],
                **_preview(document[y1:y2, x1:x2]),
            }
        )
    return {
        "document": _preview(document) if include_page else None,
        "fields": fields,
        "missing_fields": [name for name in FIELD_LABELS if name not in boxes],
        "meta": {
            "engine": "opencv",
            "model": "egrn-gosuslugi-first-page",
            "duration_seconds": perf_counter() - started,
        },
    }


async def read_egrn_fields_upload(
    image: UploadFile, *, ocr_model: EgrnOcrModel, include_page: bool
) -> dict[str, Any]:
    try:
        if ocr_model != EgrnOcrModel.NONE:
            ensure_ocr_model_enabled("paddle")
        validate_image_extension(image.filename or "image.jpg")
        await prepare_upload_stream(
            image, max_bytes=settings.ocr_max_image_size_bytes
        )
        content = await image.read()
    finally:
        await image.close()
    started = perf_counter()
    data = await asyncio.to_thread(_crop_egrn_fields, content, include_page)
    data["meta"]["ocr"] = await recognize_field_previews(
        data["fields"], FieldOcrModel(ocr_model.value), split_lines=True
    )
    data["meta"]["duration_seconds"] = perf_counter() - started
    return data


async def crop_egrn_fields_upload(
    image: UploadFile, *, ocr_model: EgrnOcrModel = EgrnOcrModel.NONE
) -> dict[str, Any]:
    data = await read_egrn_fields_upload(
        image, ocr_model=ocr_model, include_page=True
    )
    return api_success_response(data)
