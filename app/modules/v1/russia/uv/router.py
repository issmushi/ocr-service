"""Single-image UV experiment, available directly from Swagger."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, File, UploadFile

from app.core.api_schemas import COMMON_ERROR_RESPONSES
from app.core.config import settings
from app.core.errors import api_success_response
from app.core.images.uploads import prepare_upload_stream
from app.core.images.validation import validate_image_extension
from app.modules.v1.russia.uv.analysis import analyze_uv
from app.modules.v1.russia.uv.schemas import UvResponseSchema

router = APIRouter(tags=["Russia"])


@router.post(
    "/passport/uv-test",
    response_model=UvResponseSchema,
    summary="Проверить UV-признаки паспорта РФ (тест)",
    description=(
        "Загрузите цветной UV-скан страницы с фотографией целиком, "
        "почти без внешнего фона. Проверяются качество, цветная полоса, "
        "свечение фона и зоны фотографии по экспериментальным порогам. "
        "Тип документа, версия бланка и защитный рисунок не распознаются. "
        "uv_check_passed: true — признаки найдены, false — есть отличия, "
        "null — оценка невозможна. is_authentic всегда null: подлинность "
        "этим тестом не устанавливается. Файл не сохраняется."
    ),
    responses={
        400: COMMON_ERROR_RESPONSES[400],
        413: COMMON_ERROR_RESPONSES[413],
        500: COMMON_ERROR_RESPONSES[500],
    },
)
async def check_passport_uv(
    image: UploadFile = File(
        ..., description="Цветной UV-снимок одной страницы паспорта РФ."
    ),
) -> dict[str, Any]:
    validate_image_extension(image.filename or "image.png")
    await prepare_upload_stream(
        image, max_bytes=settings.ocr_max_image_size_bytes
    )
    content = await image.read()
    result = await asyncio.to_thread(analyze_uv, content)
    return api_success_response(result.model_dump(mode="json"))
