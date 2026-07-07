from __future__ import annotations

import asyncio
from pathlib import Path
from typing import BinaryIO

from fastapi import UploadFile

from app.core.config import settings

UPLOAD_CHUNK_SIZE = 1024 * 1024


class UploadSizeExceeded(Exception):
    def __init__(self, size: int, limit: int) -> None:
        self.size = size
        self.limit = limit
        super().__init__(f"Upload size {size} exceeds limit {limit}")


def upload_path(job_id: str, suffix: str) -> Path:
    settings.storage_path.mkdir(parents=True, exist_ok=True)
    safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    return settings.storage_path / f"{job_id}{safe_suffix or '.jpg'}"


def copy_upload_to_path(
    source: BinaryIO,
    destination: Path,
    max_bytes: int,
    chunk_size: int = UPLOAD_CHUNK_SIZE,
) -> int:
    total_size = 0
    source.seek(0)
    try:
        with destination.open("wb") as output:
            while chunk := source.read(chunk_size):
                total_size += len(chunk)
                if total_size > max_bytes:
                    raise UploadSizeExceeded(total_size, max_bytes)
                output.write(chunk)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return total_size


def upload_size(source: BinaryIO) -> int:
    source.seek(0, 2)
    size = source.tell()
    source.seek(0)
    return size


async def save_upload(
    upload: UploadFile,
    job_id: str,
    suffix: str,
    max_bytes: int,
) -> tuple[Path, int]:
    destination = upload_path(job_id, suffix)
    copy_task = asyncio.create_task(
        asyncio.to_thread(
            copy_upload_to_path,
            upload.file,
            destination,
            max_bytes,
        )
    )
    try:
        size = await asyncio.shield(copy_task)
    except asyncio.CancelledError:
        await asyncio.gather(copy_task, return_exceptions=True)
        destination.unlink(missing_ok=True)
        raise
    return destination, size


def delete_upload(image_path: str) -> None:
    Path(image_path).unlink(missing_ok=True)
