from __future__ import annotations

from multiprocessing.connection import Connection
from typing import Any

from app.core.infrastructure.inference import serve_inference_worker


def russian_inference_worker(
    connection: Connection,
    device: str,
) -> None:
    """Own a RussianDocsOCR pipeline in a serial, killable worker."""
    from app.modules.russian.passport import recognize_passport_text_bytes
    from app.ocr_engine.russian.factory import create_pipeline

    def initialize() -> Any:
        return create_pipeline(device=device)

    def dispatch(
        pipeline: Any,
        operation: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if operation != "passport_text":
            raise ValueError(
                f"Unsupported RussianDocsOCR operation: {operation}"
            )
        # The former HTTP worker needed a temporary file suffix. Decoding the
        # exact bytes in memory uses the same RGB pixels without a temp file.
        kwargs.pop("suffix", None)
        return recognize_passport_text_bytes(
            *args, pipeline=pipeline, **kwargs
        )

    serve_inference_worker(connection, initialize, dispatch)
