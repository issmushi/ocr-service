from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response, status

from app.core.api_schemas import (
    HEALTH_OK_RESPONSE,
    HEALTH_UNAVAILABLE_RESPONSE,
    HealthResponseSchema,
)
from app.modules.health.service import get_health_status

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponseSchema,
    summary="Проверить состояние OCR-стека",
    description=(
        "Параллельно проверяет API, PostgreSQL, RabbitMQ, workers, "
        "PaddleOCR, RussianDocsOCR, MRZ Scanner, Qwen VLM и QR. "
        "Возвращает HTTP 503, "
        "если хотя бы один компонент недоступен."
    ),
    responses={
        200: HEALTH_OK_RESPONSE,
        503: HEALTH_UNAVAILABLE_RESPONSE,
    },
)
async def health(response: Response) -> dict[str, Any]:
    result = await get_health_status()
    if result["success"] is not True:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result
