from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile

from app.core.config import settings
from app.core.errors import AppError, ErrorCode
from app.core.images.validation import (
    validate_image_extension,
    validate_image_file,
)
from app.core.infrastructure.storage import (
    UploadSizeExceeded,
    delete_upload,
    save_upload,
    upload_size,
)


@dataclass(frozen=True)
class StoredImage:
    path: Path
    filename: str
    content_type: str
    suffix: str
    size: int


async def store_image_upload(
    image: UploadFile,
    job_id: str,
    default_filename: str,
) -> StoredImage:
    filename = image.filename or default_filename
    suffix = validate_image_extension(filename)
    image_path: Path | None = None

    try:
        image_path, size = await save_upload(
            image,
            job_id,
            suffix,
            settings.ocr_max_image_size_bytes,
        )
        if size == 0:
            raise AppError(
                ErrorCode.EMPTY_IMAGE,
                "Передано пустое изображение",
            )
        validate_image_file(image_path)
    except UploadSizeExceeded as error:
        raise AppError(
            ErrorCode.IMAGE_TOO_LARGE,
            (
                "Изображение превышает лимит "
                f"{settings.ocr_max_image_size_mb} МБ"
            ),
        ) from error
    except Exception:
        if image_path is not None:
            delete_upload(str(image_path))
        raise

    assert image_path is not None
    return StoredImage(
        path=image_path,
        filename=filename,
        content_type=image.content_type or "application/octet-stream",
        suffix=suffix,
        size=size,
    )


async def prepare_upload_stream(
    image: UploadFile,
    *,
    max_bytes: int,
) -> int:
    size = await asyncio.to_thread(upload_size, image.file)
    if size == 0:
        raise AppError(ErrorCode.EMPTY_IMAGE, "Передано пустое изображение")
    if size > max_bytes:
        raise AppError(
            ErrorCode.IMAGE_TOO_LARGE,
            "Изображение превышает допустимый размер",
        )
    return size
