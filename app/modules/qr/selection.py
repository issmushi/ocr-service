from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from fastapi import UploadFile

from app.core.config import settings
from app.core.errors import AppError, ErrorCode
from app.core.images.quality import (
    ImageQualityAssessment,
    qr_quality_score,
)
from app.core.images.quality import (
    assess_qr_image_quality as assess_image_quality,
)
from app.core.images.uploads import prepare_upload_stream


@dataclass(frozen=True)
class QRImageCandidate:
    image: UploadFile
    index: int
    size: int
    quality_score: float
    assessment: ImageQualityAssessment


async def prepare_qr_candidates(
    images: Sequence[UploadFile] | None,
) -> list[QRImageCandidate]:
    uploads = list(images or [])
    if not uploads:
        raise AppError(ErrorCode.NO_IMAGE, "В запросе нет поля image")
    if len(uploads) > settings.qr_max_images_per_request:
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            (
                "Можно передать не более "
                f"{settings.qr_max_images_per_request} изображений"
            ),
        )

    candidates: list[QRImageCandidate] = []
    for index, image in enumerate(uploads):
        size = await prepare_upload_stream(
            image,
            max_bytes=settings.qr_max_image_bytes,
        )
        try:
            assessment = await asyncio.to_thread(
                assess_image_quality,
                image.file,
            )
        except AppError:
            raise
        except Exception as error:
            raise AppError(
                ErrorCode.IMAGE_READ_ERROR,
                "Изображение невозможно прочитать",
            ) from error

        candidates.append(
            QRImageCandidate(
                image=image,
                index=index,
                size=size,
                quality_score=qr_quality_score(assessment),
                assessment=assessment,
            )
        )

    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.quality_score,
            candidate.assessment.component_scores["sharpness"],
            candidate.assessment.component_scores["resolution"],
            -candidate.index,
        ),
        reverse=True,
    )
