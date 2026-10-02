"""Compose the public API from its HTTP configuration and feature routers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from app.bootstrap.frontend import configure_frontend
from app.bootstrap.http import configure_http
from app.bootstrap.lifecycle import lifespan
from app.core.config import settings
from app.core.logging_config import configure_logging
from app.core.openapi import normalize_binary_file_schemas
from app.modules.changelog import router as changelog_router
from app.modules.health.router import router as health_router
from app.modules.health.service import get_app_version
from app.modules.qr.router import router as qr_router
from app.modules.tasks.router import router as tasks_router
from app.modules.v1.router import router as v1_router


def configure_openapi(app: FastAPI) -> None:
    """Keep the generated schema and its cache local to this application."""

    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema

        schema = get_openapi(
            title=app.title,
            version=app.version,
            routes=app.routes,
        )
        schema["paths"]["/qr/generate"]["post"]["responses"].pop("422", None)
        normalize_binary_file_schemas(schema)
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi


def create_app(*, frontend_dist_dir: str | Path | None = None) -> FastAPI:
    """Build the API backed by this process's shared OCR runtimes."""
    configure_logging(
        service_name="api",
        log_dir=settings.log_path,
        level=settings.api_log_level,
        max_bytes=settings.log_max_bytes,
        backup_count=settings.log_backup_count,
    )
    app = FastAPI(
        title="OCR Service",
        version=get_app_version(),
        lifespan=lifespan,
    )
    configure_http(app)
    app.include_router(qr_router)
    app.include_router(v1_router)
    app.include_router(tasks_router)
    app.include_router(health_router)
    app.include_router(changelog_router)
    configure_frontend(app, frontend_dist_dir)
    configure_openapi(app)
    return app
