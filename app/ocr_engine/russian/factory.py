from __future__ import annotations

from typing import Any

from app.ocr_engine.russian.assets import verify_models


def create_pipeline(device: str = "cpu") -> Any:
    """Load the embedded engine only when its dedicated worker initializes."""
    verify_models()
    from app.ocr_engine.russian.vendor.document_processing import Pipeline

    return Pipeline(
        device=device,
        model_format="ONNX",
        ocr="accurate",
        page_registration=False,
    )
