"""HTTP middleware, request logging, and public error responses."""

from __future__ import annotations

import logging
import os
from time import perf_counter
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.errors import (
    ERROR_HTTP_STATUS,
    AppError,
    ErrorCode,
    api_error_response,
)
from app.core.http.body_limit import RequestBodyLimitMiddleware

logger = logging.getLogger(__name__)


def configure_http(app: FastAPI) -> None:
    """Register middleware in request logging, CORS, then body-limit order."""
    max_body_size = settings.api_max_request_body_size_bytes
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_body_size=max_body_size,
        path_limits={
            "/qr/decode": settings.api_qr_max_request_body_size_bytes,
            "/v1/ocr/full": max_body_size,
            "/v1/ocr/best-frame": max_body_size,
            "/v1/ocr/passport/mrz": max_body_size,
            "/v1/ocr/idcard/mrz": max_body_size,
            "/v1/ocr/passport/page": max_body_size,
            "/v1/ocr/crop/document": max_body_size,
            "/v1/ocr/crop/mrz": max_body_size,
            "/v1/ocr/crop/passport-fields": max_body_size,
            "/v1/tasks/passport/mrz": max_body_size,
            "/v1/tasks/passport/page": max_body_size,
            "/v1/tasks/idcard/mrz": max_body_size,
            "/v1/tasks/full": max_body_size,
            "/v1/tasks/crop/document": max_body_size,
            "/v1/tasks/crop/mrz": max_body_size,
        },
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Selected-Index", "X-Quality-Score"],
    )
    app.middleware("http")(log_http_requests)
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, internal_error_handler)


async def log_http_requests(request: Request, call_next: Any) -> Any:
    started_at = perf_counter()
    client = request.client.host if request.client else "-"
    logger.debug(
        "HTTP request received method=%s path=%s client=%s pid=%s "
        "content_type=%s content_length=%s",
        request.method,
        request.url.path,
        client,
        os.getpid(),
        request.headers.get("content-type"),
        request.headers.get("content-length"),
    )
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "HTTP request failed method=%s path=%s client=%s pid=%s",
            request.method,
            request.url.path,
            client,
            os.getpid(),
        )
        raise

    duration_ms = int((perf_counter() - started_at) * 1000)
    quiet_success = response.status_code < 400 and (
        request.url.path == "/health"
        or (
            request.method == "GET"
            and request.url.path.startswith("/v1/tasks/")
        )
    )
    log_method = (
        logger.warning
        if response.status_code >= 400
        else logger.debug
        if quiet_success
        else logger.info
    )
    log_method(
        "HTTP request completed method=%s path=%s status=%s duration_ms=%s "
        "client=%s pid=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        client,
        os.getpid(),
    )
    return response


async def summarize_form_request(request: Request) -> dict[str, object]:
    summary: dict[str, object] = {
        "content_type": request.headers.get("content-type"),
        "content_length": request.headers.get("content-length"),
        "fields": [],
        "files": [],
    }
    try:
        form = await request.form()
    except Exception as error:
        summary["form_error"] = str(error)
        return summary

    fields: list[str] = []
    files: list[dict[str, str | None]] = []
    for key, value in form.multi_items():
        filename = getattr(value, "filename", None)
        content_type = getattr(value, "content_type", None)
        if filename is None:
            fields.append(key)
        else:
            files.append(
                {
                    "field": key,
                    "filename": filename,
                    "content_type": content_type,
                }
            )

    summary["fields"] = fields
    summary["files"] = files
    return summary


def validation_error_summary(
    error: RequestValidationError,
) -> list[dict[str, object]]:
    return [
        {
            "type": item.get("type"),
            "loc": item.get("loc"),
            "msg": item.get("msg"),
        }
        for item in error.errors()
    ]


def validation_error_payload(
    request: Request,
    error: RequestValidationError,
) -> tuple[ErrorCode, str]:
    if request.url.path == "/qr/generate":
        return (
            ErrorCode.EMPTY_PAYLOAD,
            "Некорректный запрос: проверьте паспортные поля",
        )

    for item in error.errors():
        location = item.get("loc") or ()
        if tuple(location)[-2:] == ("body", "image"):
            return ErrorCode.NO_IMAGE, "В запросе нет поля image"

    return ErrorCode.VALIDATION_ERROR, "Некорректные параметры запроса"


async def app_error_handler(_request: Any, error: AppError) -> JSONResponse:
    logger.warning(
        "Handled API error code=%s message=%s", error.code.value, error.message
    )
    return JSONResponse(
        status_code=error.status_code,
        content=api_error_response(error.code, error.message),
    )


async def validation_error_handler(
    request: Request, error: RequestValidationError
) -> JSONResponse:
    logger.warning(
        "Request validation failed method=%s path=%s details=%s form=%s",
        request.method,
        request.url.path,
        validation_error_summary(error),
        await summarize_form_request(request),
    )
    code, message = validation_error_payload(request, error)

    return JSONResponse(
        status_code=ERROR_HTTP_STATUS[code],
        content=api_error_response(code, message),
    )


async def internal_error_handler(
    _request: Any, _error: Exception
) -> JSONResponse:
    logger.exception("Unhandled API error")
    return JSONResponse(
        status_code=500,
        content=api_error_response(
            ErrorCode.INTERNAL_ERROR,
            "Непредвиденная ошибка",
        ),
    )
