from __future__ import annotations

import asyncio

import cv2
import numpy as np

from app.core.errors import AppError, ErrorCode
from app.core.images.decoding import InvalidImageError, decode_image
from app.core.images.printed_content import crop_printed_content
from app.core.images.validation import validate_image_dimensions


def _crop_egrn(content: bytes) -> bytes:
    try:
        image = decode_image(content)
    except InvalidImageError as error:
        raise AppError(
            ErrorCode.IMAGE_READ_ERROR, "Изображение невозможно прочитать"
        ) from error
    validate_image_dimensions(*image.size)
    source = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    try:
        cropped = crop_printed_content(source)
    except ValueError as error:
        raise AppError(
            ErrorCode.OCR_FAILED,
            "Не удалось найти печатное содержимое выписки на изображении",
        ) from error
    success, encoded = cv2.imencode(".png", cropped)
    if not success:
        raise AppError(
            ErrorCode.INTERNAL_ERROR, "Не удалось сохранить вырезку выписки"
        )
    return encoded.tobytes()


async def crop_egrn_bytes(content: bytes) -> bytes:
    return await asyncio.to_thread(_crop_egrn, content)
