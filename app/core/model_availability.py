from __future__ import annotations

from app.core.config import settings
from app.core.errors import AppError, ErrorCode


def ensure_ocr_model_enabled(model: str) -> None:
    """Reject a model that was not selected by the installed edition."""
    if settings.is_model_enabled(model):
        return
    raise AppError(
        ErrorCode.OCR_ENGINE_UNAVAILABLE,
        f"Модель {model} отключена текущей конфигурацией OCR_MODELS",
    )
