from __future__ import annotations

import asyncio
import io
import logging
import warnings
from collections.abc import Sequence
from time import perf_counter

import cv2
import numpy as np
import qrcode
from fastapi import Response, UploadFile
from PIL import Image

from app.core.config import settings
from app.core.errors import AppError, ErrorCode, api_success_response
from app.modules.qr.crypto import (
    TOKEN_ALGORITHM,
    decrypt_json_payload,
    encrypt_json_payload,
)
from app.modules.qr.schemas import QRGenerateRequest
from app.modules.qr.selection import QRImageCandidate, prepare_qr_candidates

logger = logging.getLogger(__name__)
QR_BOX_SIZE = 10
QR_BORDER = 4
QR_FORMAT = "passport"
LEGACY_QR_FORMAT_VERSION = 1
QR_FORMAT_VERSION = 2
PASSPORT_FIELDS = (
    "surname",
    "name",
    "patronymic",
    "sex",
    "date_of_birth",
    "passport_series",
    "passport_number",
)
QR_FRAME_RETRY_CODES = frozenset(
    {
        ErrorCode.IMAGE_READ_ERROR,
        ErrorCode.QR_NOT_FOUND,
        ErrorCode.QR_INVALID_PAYLOAD,
    }
)


def clean_passport_data(
    payload: QRGenerateRequest,
) -> dict[str, str | None]:
    data = payload.model_dump()
    cleaned: dict[str, str | None] = {}
    for field in PASSPORT_FIELDS:
        value = data.get(field)
        text = str(value).strip() if value is not None else None
        cleaned[field] = text or None

    required_fields = (
        "surname",
        "name",
        "sex",
        "date_of_birth",
        "passport_series",
        "passport_number",
    )
    if any(not cleaned[field] for field in required_fields):
        raise AppError(
            ErrorCode.EMPTY_PAYLOAD,
            "Не заполнены обязательные паспортные поля",
        )
    return cleaned


def encode_passport_qr(payload: QRGenerateRequest) -> str:
    passport = clean_passport_data(payload)
    return encrypt_json_payload(
        {
            "t": "p",
            "v": QR_FORMAT_VERSION,
            "d": [passport[field] for field in PASSPORT_FIELDS],
        }
    )


def render_qr_png(
    text: str,
    *,
    error_correction: int,
    mask_pattern: int,
) -> bytes:
    qr = qrcode.QRCode(
        version=None,
        error_correction=error_correction,
        box_size=QR_BOX_SIZE,
        border=QR_BORDER,
        mask_pattern=mask_pattern,
    )
    qr.add_data(text)
    qr.make(fit=True)

    image = qr.make_image(fill_color="black", back_color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def qr_png_contains_value(content: bytes, expected: str) -> bool:
    image = cv2.imdecode(
        np.frombuffer(content, dtype=np.uint8),
        cv2.IMREAD_GRAYSCALE,
    )
    if image is None:
        return False
    value, _points, _straight = cv2.QRCodeDetector().detectAndDecode(image)
    return value == expected


def generate_qr_png(payload: QRGenerateRequest) -> bytes:
    started_at = perf_counter()
    text = encode_passport_qr(payload)
    correction_levels = (
        ("M", qrcode.constants.ERROR_CORRECT_M),
        ("L", qrcode.constants.ERROR_CORRECT_L),
    )
    for correction_name, correction_level in correction_levels:
        for mask_pattern in range(8):
            result = render_qr_png(
                text,
                error_correction=correction_level,
                mask_pattern=mask_pattern,
            )
            if not qr_png_contains_value(result, text):
                continue
            logger.info(
                "QR generated payload_size=%s image_bytes=%s correction=%s "
                "mask_pattern=%s duration_ms=%s",
                len(text.encode("utf-8")),
                len(result),
                correction_name,
                mask_pattern,
                int((perf_counter() - started_at) * 1000),
            )
            return result

    logger.error(
        "QR generation self-check failed payload_size=%s",
        len(text.encode("utf-8")),
    )
    raise AppError(
        ErrorCode.INTERNAL_ERROR,
        "Не удалось сформировать QR-код",
    )


async def generate_qr(payload: QRGenerateRequest) -> Response:
    content = await asyncio.to_thread(generate_qr_png, payload)
    return Response(content=content, media_type="image/png")


def passport_from_qr_value(
    value: str,
) -> tuple[dict[str, str | None], str] | None:
    try:
        parsed, key_id = decrypt_json_payload(value)
    except AppError as error:
        if error.code == ErrorCode.QR_INVALID_PAYLOAD:
            return None
        raise

    if not isinstance(parsed, dict):
        return None

    passport: dict[str, object]
    compact_data = parsed.get("d")
    if (
        parsed.get("t") == "p"
        and parsed.get("v") == QR_FORMAT_VERSION
        and isinstance(compact_data, list)
        and len(compact_data) == len(PASSPORT_FIELDS)
    ):
        passport = dict(zip(PASSPORT_FIELDS, compact_data, strict=True))
    else:
        legacy_passport = parsed.get("passport")
        if (
            parsed.get("type") != QR_FORMAT
            or parsed.get("version") != LEGACY_QR_FORMAT_VERSION
            or not isinstance(legacy_passport, dict)
        ):
            return None
        passport = legacy_passport

    result: dict[str, str | None] = {}
    for field in PASSPORT_FIELDS:
        item = passport.get(field)
        text = str(item).strip() if item is not None else None
        result[field] = text or None

    required_fields = (
        "surname",
        "name",
        "sex",
        "date_of_birth",
        "passport_series",
        "passport_number",
    )
    if any(not result[field] for field in required_fields):
        return None
    return result, key_id


def first_passport_payload(
    values: list[str],
) -> tuple[dict[str, str | None], str]:
    for value in values:
        passport = passport_from_qr_value(value)
        if passport is not None:
            return passport
    raise AppError(
        ErrorCode.QR_INVALID_PAYLOAD,
        "QR-код найден, но не содержит паспортные данные",
    )


def validate_image_content(content: bytes) -> None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as image:
                width, height = image.size
                if (
                    width <= 0
                    or height <= 0
                    or width > settings.ocr_max_image_width
                    or height > settings.ocr_max_image_height
                    or width * height > settings.ocr_max_image_pixels
                ):
                    raise AppError(
                        ErrorCode.IMAGE_TOO_LARGE,
                        "Разрешение изображения превышает допустимый лимит",
                    )
                image.verify()
    except AppError:
        raise
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise AppError(
            ErrorCode.IMAGE_TOO_LARGE,
            "Разрешение изображения превышает допустимый лимит",
        ) from error
    except Exception as error:
        raise AppError(
            ErrorCode.IMAGE_READ_ERROR,
            "Изображение невозможно прочитать",
        ) from error


def decode_qr_content(
    content: bytes,
    *,
    filename: str | None,
    content_type: str | None,
) -> dict[str, object]:
    started_at = perf_counter()
    validate_image_content(content)
    image = cv2.imdecode(
        np.frombuffer(content, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise AppError(
            ErrorCode.IMAGE_READ_ERROR,
            "Изображение невозможно прочитать",
        )

    detector = cv2.QRCodeDetector()
    ok, decoded_info, _points, _straight = detector.detectAndDecodeMulti(image)
    values = [value for value in decoded_info if value] if ok else []
    if not values:
        value, _points, _straight = detector.detectAndDecode(image)
        if value:
            values = [value]
    if not values:
        raise AppError(ErrorCode.QR_NOT_FOUND, "QR-код не найден")

    passport, key_id = first_passport_payload(values)
    logger.info(
        "QR decoded filename=%s size=%s count=%s duration_ms=%s",
        filename,
        len(content),
        len(values),
        int((perf_counter() - started_at) * 1000),
    )
    return api_success_response(
        {
            "passport": passport,
            "meta": {
                "format": QR_FORMAT,
                "version": QR_FORMAT_VERSION,
                "encrypted": True,
                "algorithm": TOKEN_ALGORITHM,
                "key_id": key_id,
                "qr_count": len(values),
            },
            "filename": filename,
            "content_type": content_type,
        }
    )


async def decode_candidate(candidate: QRImageCandidate) -> dict[str, object]:
    candidate.image.file.seek(0)
    content = await asyncio.to_thread(candidate.image.file.read)
    return await asyncio.to_thread(
        decode_qr_content,
        content,
        filename=candidate.image.filename,
        content_type=candidate.image.content_type,
    )


def final_qr_frame_error(errors: Sequence[AppError]) -> AppError:
    for code in (
        ErrorCode.QR_INVALID_PAYLOAD,
        ErrorCode.IMAGE_READ_ERROR,
        ErrorCode.QR_NOT_FOUND,
    ):
        for error in errors:
            if error.code == code:
                return error
    return AppError(ErrorCode.QR_NOT_FOUND, "QR-код не найден")


async def decode_qr(
    images: Sequence[UploadFile] | None,
) -> dict[str, object]:
    candidates = await prepare_qr_candidates(images)
    logger.info(
        "QR frames ranked candidate_count=%s order=%s",
        len(candidates),
        [candidate.index + 1 for candidate in candidates],
    )

    errors: list[AppError] = []
    for candidate in candidates:
        try:
            return await decode_candidate(candidate)
        except AppError as error:
            if error.code not in QR_FRAME_RETRY_CODES:
                raise
            errors.append(error)
            logger.info(
                "QR frame rejected frame_index=%s code=%s",
                candidate.index + 1,
                error.code.value,
            )
    raise final_qr_frame_error(errors)
