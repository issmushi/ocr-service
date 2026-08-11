from __future__ import annotations

import asyncio
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from fastapi import Response, UploadFile
from PIL import Image

from app.core.images.selection import (
    SelectedImage,
    store_and_select_image_uploads,
)
from app.core.infrastructure.storage import delete_upload


def _read_selected_image(path: Path) -> tuple[bytes, str]:
    content = path.read_bytes()
    with Image.open(BytesIO(content)) as image:
        media_type = Image.MIME.get(image.format, "application/octet-stream")
    return content, media_type


async def select_best_frame_uploads(
    images: Sequence[UploadFile] | None,
) -> Response:
    """Use the shared quality ranking and return the original winning image."""
    uploads = list(images or [])
    selected: SelectedImage | None = None
    try:
        selected = await store_and_select_image_uploads(
            uploads,
            str(uuid4()),
            "image.jpg",
        )
        content, media_type = await asyncio.to_thread(
            _read_selected_image, selected.image.path
        )
        return Response(
            content=content,
            media_type=media_type,
            headers={
                "Cache-Control": "no-store",
                "X-Selected-Index": str(selected.index),
                "X-Quality-Score": str(selected.assessment.score),
            },
        )
    finally:
        try:
            if selected is not None:
                delete_upload(str(selected.image.path))
        finally:
            await asyncio.gather(*(upload.close() for upload in uploads))
