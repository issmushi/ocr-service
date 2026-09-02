from __future__ import annotations

import asyncio
import logging
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx
from sqlalchemy import text

from app.core.config import settings
from app.core.db.session import async_engine
from app.core.errors import (
    ErrorCode,
    api_error_response,
    api_success_response,
)
from app.core.infrastructure.queue import get_queue_status
from app.modules.paddle import paddle_runtime
from app.modules.qr.crypto import (
    TOKEN_ALGORITHM,
    active_encryption_key_id,
)
from app.modules.russian.runtime import russian_runtime
from app.modules.v1.passport.runtime import mrz_scanner_runtime

PACKAGE_NAME = "ocr-service"
SERVICE_NAME = "ocr-service"

logger = logging.getLogger(__name__)


def get_app_version() -> str:
    for directory in Path(__file__).resolve().parents:
        pyproject_path = directory / "pyproject.toml"
        if pyproject_path.is_file():
            with pyproject_path.open("rb") as pyproject_file:
                return tomllib.load(pyproject_file)["project"]["version"]

    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return "unknown"


def elapsed_ms(started_at: float) -> int:
    return max(0, round((perf_counter() - started_at) * 1000))


def healthy_component(
    latency_ms: int,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "ok",
        "latency_ms": latency_ms,
        "details": details or {},
        "error": None,
    }


def failed_component(
    latency_ms: int,
    message: str,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "error",
        "latency_ms": latency_ms,
        "details": details or {},
        "error": message,
    }


def disabled_component(engine: str) -> dict[str, Any]:
    return healthy_component(
        0,
        details={"engine": engine, "status": "disabled"},
    )


def model_enabled(model: str) -> bool:
    enabled = getattr(settings, "enabled_ocr_models", None)
    return True if enabled is None else model in enabled


async def probe_database() -> dict[str, Any]:
    started_at = perf_counter()

    async def execute_probe() -> None:
        async with async_engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

    try:
        await asyncio.wait_for(
            execute_probe(),
            timeout=settings.health_check_timeout_seconds,
        )
        return healthy_component(elapsed_ms(started_at))
    except Exception as error:
        logger.warning(
            "Health database probe failed error=%s",
            error,
        )
        return failed_component(
            elapsed_ms(started_at),
            "PostgreSQL недоступен",
        )


async def probe_queue() -> dict[str, Any]:
    started_at = perf_counter()
    try:
        queue_status = await asyncio.wait_for(
            get_queue_status(),
            timeout=settings.health_check_timeout_seconds,
        )
        return healthy_component(
            elapsed_ms(started_at),
            details=queue_status,
        )
    except Exception as error:
        logger.warning(
            "Health RabbitMQ probe failed error=%s",
            error,
        )
        return failed_component(
            elapsed_ms(started_at),
            "RabbitMQ недоступен",
            details={
                "queue_name": settings.ocr_queue_name,
                "message_count": None,
                "consumer_count": 0,
                "dead_letter_queue_name": (
                    settings.ocr_dead_letter_queue_name
                ),
                "dead_letter_message_count": None,
            },
        )


async def probe_paddle_runtime() -> dict[str, Any]:
    started_at = perf_counter()
    runtime_status = paddle_runtime.status
    ready = paddle_runtime.ready
    busy = paddle_runtime.busy
    if runtime_status == "starting":
        state = "starting"
    elif runtime_status in {"ready", "busy"} and ready:
        state = "busy" if busy else "ready"
    else:
        state = "down"
    details = {
        "engine": "paddleocr",
        "status": state,
        "busy": busy,
        "ready": ready,
        "worker_pid": paddle_runtime.worker_pid,
        "device": settings.ocr_device,
    }
    if state in {"ready", "busy"}:
        return healthy_component(elapsed_ms(started_at), details=details)
    return failed_component(
        elapsed_ms(started_at),
        (
            "PaddleOCR запускается"
            if state == "starting"
            else "PaddleOCR недоступен"
        ),
        details=details,
    )


async def probe_russian_runtime() -> dict[str, Any]:
    started_at = perf_counter()
    runtime_status = russian_runtime.status
    ready = russian_runtime.ready
    busy = russian_runtime.busy
    # The executor also reports busy during initialization, and a request lock
    # can remain held while a worker is unavailable. Neither means it is ready.
    if runtime_status == "starting":
        state = "starting"
    elif runtime_status in {"ready", "busy"} and (ready or busy):
        state = "busy" if busy else "ready"
    else:
        state = "down"
    details = {
        "engine": "russian-docs-ocr",
        "status": state,
        "busy": busy,
        "ready": ready,
        "worker_pid": russian_runtime.worker_pid,
        "device": settings.russian_ocr_device,
        "model_format": "ONNX",
    }
    if state in {"ready", "busy"}:
        return healthy_component(elapsed_ms(started_at), details=details)
    return failed_component(
        elapsed_ms(started_at),
        (
            "RussianDocsOCR запускается"
            if state == "starting"
            else "RussianDocsOCR недоступен"
        ),
        details=details,
    )


async def probe_mrz_runtime() -> dict[str, Any]:
    started_at = perf_counter()
    runtime_status = mrz_scanner_runtime.status
    ready = mrz_scanner_runtime.ready
    busy = mrz_scanner_runtime.busy
    if runtime_status == "stopped":
        # MRZ Scanner is intentionally loaded on its first request.
        state = "idle"
    elif runtime_status == "starting":
        state = "starting"
    elif runtime_status in {"ready", "busy"} and ready:
        state = "busy" if busy else "ready"
    else:
        state = "down"
    details = {
        "engine": "mrz-scanner",
        "status": state,
        "busy": busy,
        "ready": ready,
        "worker_pid": mrz_scanner_runtime.worker_pid,
    }
    if state in {"idle", "ready", "busy"}:
        return healthy_component(elapsed_ms(started_at), details=details)
    return failed_component(
        elapsed_ms(started_at),
        (
            "MRZ Scanner запускается"
            if state == "starting"
            else "MRZ Scanner недоступен"
        ),
        details=details,
    )


def vlm_details(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or {}
    raw_models = payload.get("models")
    models = (
        [model for model in raw_models if isinstance(model, str)]
        if isinstance(raw_models, list)
        else []
    )
    active_model = payload.get("active_model")
    return {
        "engine": "qwen3-vl",
        "status": (
            payload.get("status")
            if isinstance(payload.get("status"), str)
            else "down"
        ),
        "models": models,
        "active_model": active_model
        if isinstance(active_model, str)
        else None,
    }


async def probe_vlm_service() -> dict[str, Any]:
    started_at = perf_counter()
    try:
        async with httpx.AsyncClient(
            timeout=settings.health_check_timeout_seconds,
            trust_env=False,
        ) as client:
            response = await client.get(
                f"{settings.vlm_ocr_url.rstrip('/')}/health"
            )
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("VLM health response is not an object")
        details = vlm_details(payload)
        if response.status_code == 200 and details["status"] == "ready":
            return healthy_component(
                elapsed_ms(started_at),
                details=details,
            )
        return failed_component(
            elapsed_ms(started_at),
            "Qwen VLM недоступен",
            details=details,
        )
    except Exception as error:
        logger.warning("Health Qwen VLM probe failed error=%s", error)
        return failed_component(
            elapsed_ms(started_at),
            "Qwen VLM недоступен",
            details=vlm_details(),
        )


async def probe_qr_module() -> dict[str, Any]:
    started_at = perf_counter()
    try:
        key_id = active_encryption_key_id()
    except Exception:
        logger.warning("Health QR encryption configuration check failed")
        return failed_component(
            elapsed_ms(started_at),
            "QR encryption не настроен",
        )
    return healthy_component(
        elapsed_ms(started_at),
        details={
            "engine": "opencv-qrcode",
            "encryption": TOKEN_ALGORITHM,
            "key_id": key_id,
        },
    )


def worker_component(queue_component: dict[str, Any]) -> dict[str, Any]:
    queue_details = queue_component.get("details")
    queue_details = queue_details if isinstance(queue_details, dict) else {}
    active_workers = int(queue_details.get("consumer_count") or 0)
    details = {
        "configured_workers": settings.ocr_workers,
        "active_workers": active_workers,
    }
    if queue_component.get("status") != "ok":
        return failed_component(
            0,
            ("Невозможно проверить workers: RabbitMQ недоступен"),
            details=details,
        )
    if active_workers != 1:
        return failed_component(
            0,
            (f"Активных workers: {active_workers} (должен быть ровно один)"),
            details=details,
        )
    return healthy_component(0, details=details)


async def get_health_status() -> dict[str, Any]:
    app_version = get_app_version()
    paddle_probe = (
        probe_paddle_runtime()
        if model_enabled("paddle")
        else asyncio.sleep(0, result=disabled_component("paddleocr"))
    )
    russian_probe = (
        probe_russian_runtime()
        if model_enabled("russian-ocr")
        else asyncio.sleep(0, result=disabled_component("russian-docs-ocr"))
    )
    mrz_probe = (
        probe_mrz_runtime()
        if model_enabled("mrz")
        else asyncio.sleep(0, result=disabled_component("mrz-scanner"))
    )
    qwen_enabled = model_enabled("qwen3-vl-4b") or model_enabled("qwen3-vl-8b")
    qwen_probe = (
        probe_vlm_service()
        if qwen_enabled
        else asyncio.sleep(0, result=disabled_component("qwen3-vl"))
    )
    (
        database,
        rabbitmq,
        paddle_ocr,
        russian_ocr,
        mrz_scanner,
        qwen_vlm_service,
        qr,
    ) = await asyncio.gather(
        probe_database(),
        probe_queue(),
        paddle_probe,
        russian_probe,
        mrz_probe,
        qwen_probe,
        probe_qr_module(),
    )
    worker = worker_component(rabbitmq)
    active_workers = int(worker.get("details", {}).get("active_workers") or 0)
    components = {
        "api": healthy_component(
            0,
            details={"version": app_version},
        ),
        "postgres": database,
        "rabbitmq": rabbitmq,
        "worker": worker,
        "paddle_ocr": paddle_ocr,
        "russian_ocr": russian_ocr,
        "mrz_scanner": mrz_scanner,
        "qwen_vlm_service": qwen_vlm_service,
        "qr": qr,
    }
    failed_components = [
        name
        for name, component in components.items()
        if component.get("status") != "ok"
    ]
    data = {
        "status": "ok" if not failed_components else "degraded",
        "service": SERVICE_NAME,
        "version": app_version,
        "queue": rabbitmq["status"],
        "workers": settings.ocr_workers,
        "active_workers": active_workers,
        "components": components,
    }

    if not failed_components:
        return api_success_response(data)

    return api_error_response(
        ErrorCode.SERVICE_UNAVAILABLE,
        "Недоступны компоненты: " + ", ".join(failed_components),
        data=data,
    )
