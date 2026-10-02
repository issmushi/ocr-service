from __future__ import annotations

from multiprocessing.connection import Connection
from typing import Any

from app.core.infrastructure.inference import serve_inference_worker


def russian_inference_worker(
    connection: Connection,
    device: str,
) -> None:
    """Own a RussianDocsOCR pipeline in a serial, killable worker."""
    from app.modules.russian.field_ocr import recognize_field_text_bytes
    from app.modules.russian.fields import detect_passport_fields_bytes
    from app.modules.russian.passport import recognize_passport_text_bytes
    from app.modules.russian.snils import recognize_snils_bytes
    from app.ocr_engine.russian.factory import create_pipeline
    from app.ocr_engine.russian.field_recognizer import create_cpu_field_reader

    field_reader = None

    def initialize() -> Any:
        return create_pipeline(device=device)

    def dispatch(
        pipeline: Any,
        operation: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        nonlocal field_reader
        if operation == "field_text_accurate":
            if field_reader is None:
                field_reader = create_cpu_field_reader(pipeline, device=device)
            return recognize_field_text_bytes(
                *args, reader=field_reader, **kwargs
            )
        if operation == "passport_fields":
            return detect_passport_fields_bytes(
                *args, pipeline=pipeline, **kwargs
            )
        if operation == "snils":
            return recognize_snils_bytes(*args, pipeline=pipeline, **kwargs)
        if operation not in {"passport_text", "passport_international"}:
            raise ValueError(
                f"Unsupported RussianDocsOCR operation: {operation}"
            )
        # The former HTTP worker needed a temporary file suffix. Decoding the
        # exact bytes in memory uses the same RGB pixels without a temp file.
        kwargs.pop("suffix", None)
        return recognize_passport_text_bytes(
            *args,
            pipeline=pipeline,
            passport_kind=(
                "international"
                if operation == "passport_international"
                else "internal"
            ),
            **kwargs,
        )

    serve_inference_worker(connection, initialize, dispatch)
