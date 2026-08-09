from __future__ import annotations

from multiprocessing.connection import Connection
from typing import Any

from app.core.infrastructure.inference import serve_inference_worker


def mrz_scanner_inference_worker(connection: Connection) -> None:
    """Load MRZ Scanner only inside its dedicated inference process."""

    def initialize() -> Any:
        from app.ocr_engine.mrz.factory import create_mrz_scanner

        return create_mrz_scanner()

    def dispatch(
        engine: Any,
        operation: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if operation == "mrz_crop_text":
            from app.modules.v1.crop.text_pipeline import (
                process_mrz_crop_text_bytes_with_mrz_scanner,
            )

            return process_mrz_crop_text_bytes_with_mrz_scanner(
                *args,
                scanner=engine,
                **kwargs,
            )

        if operation != "passport_v1":
            raise ValueError(f"Unsupported MRZ Scanner operation: {operation}")

        from app.modules.v1.passport.pipeline import (
            process_passport_bytes_with_mrz_scanner,
        )

        return process_passport_bytes_with_mrz_scanner(
            *args, scanner=engine, **kwargs
        )

    serve_inference_worker(connection, initialize, dispatch)
