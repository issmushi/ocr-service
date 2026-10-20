from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter, File, Form, Response, UploadFile

from app.core.api_schemas import (
    COMMON_ERROR_RESPONSES,
    OCR_ENGINE_UNAVAILABLE_RESPONSE,
    OCR_FAILED_RESPONSE,
    OCR_TIMEOUT_RESPONSE,
)
from app.core.config import settings
from app.core.images.uploads import prepare_upload_stream
from app.core.images.validation import validate_image_extension
from app.modules.v1.crop.field_schemas import (
    FieldOcrModel,
    PassportFieldCropResponse,
)
from app.modules.v1.crop.passport_fields import crop_passport_fields_upload
from app.modules.v1.crop.registration_fields import (
    crop_registration_fields_upload,
)
from app.modules.v1.crop.registration_schemas import RegistrationCropResponse
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


@router.post(
    "/crop/document-fields",
    response_model=PassportFieldCropResponse,
    summary="Найти и вырезать поля документа",
    description=(
        "Эксперимент для третьей страницы внутреннего паспорта РФ. "
        "Использует детектор Russian OCR, исправляет ориентацию, "
        "при crop_document=true выравнивает документ, затем вырезает поля. "
        "При ocr_model=none возвращает только вырезки. Можно отдельно "
        "прочитать каждое текстовое поле через Paddle eslav v5, "
        "Paddle Cyrillic v5 или Russian OCR accurate на CPU, либо Qwen "
        "в текущем режиме VLM-сервера. Текст возвращается без нормализации, "
        "фото и подпись пропускаются. Ошибки OCR указаны в отдельных полях; "
        "success=true означает, что вырезки получены. "
        "Возвращает обзор документа в JPEG, вырезки в PNG без потерь, "
        "координаты и уверенность детектора. Вертикальная серия и номер "
        "поворачиваются горизонтально. Если поля не найдены, fields пустой. "
        "Изображения не сохраняются."
    ),
    responses=ERROR_RESPONSES,
)
async def crop_document_fields(
    response: Response,
    image: UploadFile = File(
        ..., description="Фото третьей страницы внутреннего паспорта РФ."
    ),
    crop_document: bool = Form(
        True, description="Найти границы и исправить перспективу документа."
    ),
    padding: int = Form(
        4, ge=0, le=64, description="Отступ вокруг каждого поля в пикселях."
    ),
    ocr_model: FieldOcrModel = Form(
        FieldOcrModel.NONE,
        description="Модель чтения вырезанных полей; none — только вырезки.",
    ),
) -> dict[str, Any]:
    response.headers["Cache-Control"] = "no-store"
    return await crop_passport_fields_upload(
        image,
        crop_document=crop_document,
        padding=padding,
        ocr_model=ocr_model,
    )


@router.post(
    "/crop/registration-fields-test",
    response_model=RegistrationCropResponse,
    summary="Тест: вырезать строки адреса и штампы прописки",
    description=(
        "Находит и выравнивает страницу или разворот прописки, затем отдельно "
        "вырезает адресные строки и штампы в PNG. Координаты относятся к "
        "возвращённому обзору документа. "
        "При ocr_model=none текст не читается. "
        "Можно передать эти же PNG выбранной OCR-модели; результат будет "
        "в ocr каждого фрагмента без нормализации. Возвращаются и печатные, "
        "и рукописные области; точность чтения зависит от модели. "
        "Для другого типа документа или при отсутствии областей fields "
        "пустой. Изображения не сохраняются."
    ),
    responses=ERROR_RESPONSES,
)
async def crop_registration_fields_test(
    response: Response,
    image: UploadFile = File(
        ..., description="Фото страницы или разворота с пропиской."
    ),
    padding: int = Form(
        4, ge=0, le=64, description="Отступ вокруг вырезки в пикселях."
    ),
    ocr_model: FieldOcrModel = Form(
        FieldOcrModel.NONE,
        description="Модель чтения вырезок; none — только детекция и обрезка.",
    ),
) -> dict[str, Any]:
    response.headers["Cache-Control"] = "no-store"
    return await crop_registration_fields_upload(
        image, padding=padding, ocr_model=ocr_model
    )


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
