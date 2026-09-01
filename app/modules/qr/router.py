from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Response, UploadFile

from app.core.api_schemas import (
    COMMON_ERROR_RESPONSES,
    PAYLOAD_ERROR_RESPONSE,
    QR_DECODE_ERROR_RESPONSES,
    QR_DECODE_FAILED_RESPONSE,
    QR_DECODE_RESPONSE,
    QR_SERVICE_UNAVAILABLE_RESPONSE,
    ApiResponseSchema,
)
from app.modules.qr.schemas import QRGenerateRequest
from app.modules.qr.service import decode_qr, generate_qr

router = APIRouter(tags=["QR"])


@router.post(
    "/qr/generate",
    response_class=Response,
    summary="Сгенерировать QR-код",
    description=(
        "Принимает паспортные данные, шифрует их внутри приложения "
        "алгоритмом AES-256-GCM и возвращает PNG-изображение. "
        "Ключ шифрования хранится только на backend."
    ),
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "PNG-изображение QR-кода.",
        },
        400: PAYLOAD_ERROR_RESPONSE,
        503: QR_SERVICE_UNAVAILABLE_RESPONSE,
        500: COMMON_ERROR_RESPONSES[500],
    },
)
async def generate_qr_code(payload: QRGenerateRequest) -> Response:
    return await generate_qr(payload)


@router.post(
    "/qr/decode",
    response_model=ApiResponseSchema,
    summary="Распознать QR-код на изображении",
    description=(
        "Принимает от одного до трех изображений "
        "повторением multipart/form-data поля `image`, "
        "ранжирует кадры по качеству и возвращает "
        "расшифрованные данные первого успешно проверенного QR-кода."
    ),
    responses={
        200: QR_DECODE_RESPONSE,
        **QR_DECODE_ERROR_RESPONSES,
        422: QR_DECODE_FAILED_RESPONSE,
        503: QR_SERVICE_UNAVAILABLE_RESPONSE,
    },
)
async def decode_qr_code(
    image: list[UploadFile] | None = File(
        default=None,
        description=(
            "От одного до трех кадров с QR-кодом. Для каждого файла "
            "повторите поле image."
        ),
    ),
) -> dict[str, Any]:
    return await decode_qr(image)
