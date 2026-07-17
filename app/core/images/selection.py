from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from fastapi import UploadFile

from app.core.config import settings
from app.core.errors import AppError, ErrorCode
from app.core.images.quality import (
    ImageQualityAssessment,
    assess_image_quality,
)
from app.core.images.uploads import StoredImage, store_image_upload
from app.core.infrastructure.storage import delete_upload


@dataclass(frozen=True)
class SelectedImage:
    image: StoredImage
    index: int
    assessment: ImageQualityAssessment
    candidate_count: int


def select_best_stored_image(images: Sequence[StoredImage]) -> SelectedImage:
    if not images:
        raise AppError(ErrorCode.NO_IMAGE, "В запросе нет поля image")

    try:
        assessments = [assess_image_quality(image.path) for image in images]
    except Exception as error:
        raise AppError(
            ErrorCode.IMAGE_READ_ERROR,
            "Изображение невозможно проанализировать",
        ) from error

    selected_index = max(
        range(len(images)),
        key=lambda index: (
            assessments[index].score,
            assessments[index].component_scores["sharpness"],
            assessments[index].component_scores["resolution"],
            -index,
        ),
    )
    return SelectedImage(
        image=images[selected_index],
        index=selected_index,
        assessment=assessments[selected_index],
        candidate_count=len(images),
    )


async def store_and_select_image_uploads(
    images: Sequence[UploadFile] | None,
    job_id: str,
    default_filename: str,
) -> SelectedImage:
    candidates = list(images or [])
    if not candidates:
        raise AppError(ErrorCode.NO_IMAGE, "В запросе нет поля image")
    if len(candidates) > settings.ocr_max_images_per_request:
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            (
                "Можно передать не более "
                f"{settings.ocr_max_images_per_request} изображений"
            ),
        )

    stored_images: list[StoredImage] = []
    try:
        for index, image in enumerate(candidates, start=1):
            stored_images.append(
                await store_image_upload(
                    image,
                    f"{job_id}-frame-{index}",
                    default_filename,
                )
            )
        selected = await asyncio.to_thread(
            select_best_stored_image,
            stored_images,
        )
    except BaseException:
        for stored_image in stored_images:
            delete_upload(str(stored_image.path))
        raise

    for index, stored_image in enumerate(stored_images):
        if index != selected.index:
            delete_upload(str(stored_image.path))
    return selected
