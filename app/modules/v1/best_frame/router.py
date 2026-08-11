from __future__ import annotations

from fastapi import APIRouter, File, Response, UploadFile

from app.core.api_schemas import (
    COMMON_ERROR_RESPONSES,
    VALIDATION_ERROR_RESPONSE,
)
from app.core.config import settings
from app.modules.v1.best_frame.service import select_best_frame_uploads

router = APIRouter(tags=["Best frame"])


@router.post(
    "/best-frame",
    response_class=Response,
    summary="Выбрать лучший кадр документа",
    description=(
        f"Принимает от одного до {settings.ocr_max_images_per_request} "
        "изображений "
        "повторением поля image. Оценивает "
        "полноту документа, резкость, блики, "
        "освещение, перспективу, контраст, "
        "разрешение и шум. Возвращает исходный "
        "файл с максимальным баллом качества. При равенстве баллов сравнивает "
        "резкость, затем разрешение, затем предпочитает первый кадр. "
        "X-Selected-Index содержит индекс выбранного файла, начиная с нуля; "
        "X-Quality-Score — итоговую оценку от 0 до 1."
    ),
    responses={
        200: {
            "description": "Лучший исходный кадр в его собственном формате.",
            "content": {
                media_type: {"schema": {"type": "string", "format": "binary"}}
                for media_type in (
                    "image/jpeg",
                    "image/png",
                    "image/webp",
                    "image/bmp",
                    "image/tiff",
                )
            },
            "headers": {
                "X-Selected-Index": {
                    "description": (
                        "Индекс кадра в переданном списке, начиная с нуля."
                    ),
                    "schema": {"type": "integer", "minimum": 0},
                },
                "X-Quality-Score": {
                    "description": "Общая оценка качества выбранного кадра.",
                    "schema": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
        400: COMMON_ERROR_RESPONSES[400],
        413: COMMON_ERROR_RESPONSES[413],
        422: VALIDATION_ERROR_RESPONSE,
        500: COMMON_ERROR_RESPONSES[500],
    },
)
async def select_best_frame(
    image: list[UploadFile] = File(
        ...,
        description=(
            "Фотографии одного документа. Для "
            "каждого файла повторите поле image."
        ),
    ),
) -> Response:
    return await select_best_frame_uploads(image)
