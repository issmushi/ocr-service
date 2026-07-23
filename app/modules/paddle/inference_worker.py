from __future__ import annotations

from multiprocessing.connection import Connection
from typing import Any

from app.core.infrastructure.inference import serve_inference_worker


def paddle_inference_worker(connection: Connection, device: str) -> None:
    """Own the PaddleOCR model and execute commands in an isolated process."""

    from app.core.images.preprocessing import decode_image_bytes
    from app.modules.v1.full.paddle import recognize_full_bytes
    from app.modules.v1.idcard.recognition import process_idcard_text_bytes
    from app.modules.v1.passport.pipeline import process_passport_bytes
    from app.ocr_engine.paddle.factory import (
        create_mrz_ocr,
        create_ocr,
    )
    from app.ocr_engine.paddle.orientation import predict_document_orientation

    def initialize() -> Any:
        return {
            "general": create_ocr(device=device),
            "mrz": create_mrz_ocr(device=device),
        }

    def dispatch(
        engine: Any,
        operation: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if operation == "document_orientation":
            orientation = predict_document_orientation(
                decode_image_bytes(*args, **kwargs),
                ocr=engine["mrz"],
            )
            return {
                "rotation_degrees": orientation.rotation_degrees,
                "confidence": orientation.confidence,
            }

        if operation == "passport_v1":
            return process_passport_bytes(
                *args,
                ocr=engine["mrz"],
                **kwargs,
            )

        if operation == "idcard_v1_text":
            return process_idcard_text_bytes(
                *args, ocr=engine["mrz"], **kwargs
            )

        if operation == "full":
            return recognize_full_bytes(*args, ocr=engine["general"], **kwargs)
        raise ValueError(f"Unsupported PaddleOCR operation: {operation}")

    serve_inference_worker(connection, initialize, dispatch)
