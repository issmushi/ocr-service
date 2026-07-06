from __future__ import annotations

import warnings
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from PIL import Image

from app.core.config import settings
from app.core.errors import AppError, ErrorCode

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def validate_image_extension(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        raise AppError(
            ErrorCode.UNSUPPORTED_FORMAT,
            (
                "Неподдерживаемый формат изображения: "
                f"{suffix or 'без расширения'}"
            ),
        )
    return suffix


def validate_image_dimensions(
    width: int,
    height: int,
    *,
    max_width: int | None = None,
    max_height: int | None = None,
    max_pixels: int | None = None,
) -> None:
    width_limit = (
        settings.ocr_max_image_width if max_width is None else max_width
    )
    height_limit = (
        settings.ocr_max_image_height if max_height is None else max_height
    )
    pixel_limit = (
        settings.ocr_max_image_pixels if max_pixels is None else max_pixels
    )

    if (
        width <= 0
        or height <= 0
        or width > width_limit
        or height > height_limit
        or width * height > pixel_limit
    ):
        raise AppError(
            ErrorCode.IMAGE_TOO_LARGE,
            "Разрешение изображения превышает допустимый лимит",
        )


def validate_image_file(image_path: str | Path) -> None:
    _validate_image_source(Path(image_path))


def validate_image_bytes(content: bytes) -> None:
    _validate_image_source(BytesIO(content))


def _validate_image_source(source: Path | BinaryIO) -> None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(source) as image:
                validate_image_dimensions(*image.size)
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
