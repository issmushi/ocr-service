from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import APIRouter, File, Response, UploadFile

from app.core.api_schemas import (
    COMMON_ERROR_RESPONSES,
    OCR_ENGINE_UNAVAILABLE_RESPONSE,
    OCR_FAILED_RESPONSE,
    OCR_TIMEOUT_RESPONSE,
)
from app.core.config import settings
from app.core.images.uploads import prepare_upload_stream
from app.core.images.validation import validate_image_extension
from app.modules.v1.crop.service import (
    crop_document_image_bytes,
    crop_mrz_bytes,
)

router = APIRouter(tags=["Crop"])

ERROR_RESPONSES = {
    400: COMMON_ERROR_RESPONSES[400],
    413: COMMON_ERROR_RESPONSES[413],
    422: OCR_FAILED_RESPONSE,
    500: COMMON_ERROR_RESPONSES[500],
    503: OCR_ENGINE_UNAVAILABLE_RESPONSE,
    504: OCR_TIMEOUT_RESPONSE,
}


async def _crop_upload(
    image: UploadFile,
    operation: Callable[[bytes], Awaitable[bytes]],
) -> Response:
    filename = image.filename or "image.jpg"
    try:
        validate_image_extension(filename)
        await prepare_upload_stream(
            image, max_bytes=settings.ocr_max_image_size_bytes
        )
        content = await image.read()
    finally:
        await image.close()
    result = await operation(content)
    return Response(
        content=result,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@router.post(
    "/crop/document",
    response_class=Response,
    summary="Обрезать и выровнять документ",
    description=(
        "Находит границы документа, исправляет перспективу и ориентацию, "
        "затем возвращает обрезанное изображение в формате JPEG. Если внешняя "
        "граница не найдена, возвращает полный выровненный кадр."
    ),
    responses={
        200: {
            "content": {"image/jpeg": {}},
            "description": "Обрезанный документ.",
        },
        **ERROR_RESPONSES,
    },
)
async def crop_document(
    image: UploadFile = File(..., description="Фото документа или скан."),
) -> Response:
    return await _crop_upload(image, crop_document_image_bytes)


@router.post(
    "/crop/mrz",
    response_class=Response,
    summary="Обрезать MRZ",
    description=(
        "Находит и выравнивает документ, выделяет область MRZ и возвращает её "
        "в формате JPEG. Если MRZ не найдена, возвращает OCR_FAILED."
    ),
    responses={
        200: {"content": {"image/jpeg": {}}, "description": "Обрезанная MRZ."},
        **ERROR_RESPONSES,
    },
)
async def crop_mrz(
    image: UploadFile = File(
        ..., description="Фото документа или скан с MRZ."
    ),
) -> Response:
    return await _crop_upload(image, crop_mrz_bytes)
