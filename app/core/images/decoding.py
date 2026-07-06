from __future__ import annotations

import io
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError


class InvalidImageError(ValueError):
    pass


def decode_image(content: bytes) -> Image.Image:
    """Decode one frame, apply EXIF orientation, and composite alpha
    on white.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as source:
                if getattr(source, "n_frames", 1) != 1:
                    raise InvalidImageError("Only a single image is supported")
                source.load()
                image = ImageOps.exif_transpose(source)
                if (
                    image.mode in {"RGBA", "LA"}
                    or "transparency" in image.info
                ):
                    rgba = image.convert("RGBA")
                    background = Image.new("RGBA", rgba.size, "white")
                    image = Image.alpha_composite(background, rgba)
                return image.convert("RGB")
    except (
        OSError,
        ValueError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise InvalidImageError("Cannot read image") from exc
