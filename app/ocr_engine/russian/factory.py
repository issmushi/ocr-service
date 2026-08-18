from __future__ import annotations

from importlib import import_module
from typing import Any

from app.ocr_engine.russian.assets import verify_models


def create_pipeline(device: str = "cpu") -> Any:
    """Load the embedded engine only when its dedicated worker initializes."""
    verify_models()
    from app.ocr_engine.russian.vendor.document_processing import Pipeline

    # DocType imports this at its first prediction. Pay the cold sklearn/scipy
    # import cost during worker startup, outside the per-document hard timeout.
    import_module("sklearn.neighbors")
    return Pipeline(device=device)
