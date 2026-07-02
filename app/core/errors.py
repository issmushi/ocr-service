from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

INTERNAL_ERROR_MESSAGE = "Непредвиденная ошибка"


class ErrorCode(StrEnum):
    EMPTY_PAYLOAD = "EMPTY_PAYLOAD"
    NO_IMAGE = "NO_IMAGE"
    EMPTY_IMAGE = "EMPTY_IMAGE"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    IMAGE_READ_ERROR = "IMAGE_READ_ERROR"
    OCR_FAILED = "OCR_FAILED"
    OCR_TIMEOUT = "OCR_TIMEOUT"
    QUEUE_FULL = "QUEUE_FULL"
    TASK_NOT_FOUND = "TASK_NOT_FOUND"
    TASK_NOT_READY = "TASK_NOT_READY"
    TASK_CANCELLED = "TASK_CANCELLED"
    OCR_ENGINE_UNAVAILABLE = "OCR_ENGINE_UNAVAILABLE"
    QR_NOT_FOUND = "QR_NOT_FOUND"
    QR_INVALID_PAYLOAD = "QR_INVALID_PAYLOAD"
    QR_SERVICE_UNAVAILABLE = "QR_SERVICE_UNAVAILABLE"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


ERROR_HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.EMPTY_PAYLOAD: 400,
    ErrorCode.NO_IMAGE: 400,
    ErrorCode.EMPTY_IMAGE: 400,
    ErrorCode.UNSUPPORTED_FORMAT: 400,
    ErrorCode.IMAGE_TOO_LARGE: 413,
    ErrorCode.IMAGE_READ_ERROR: 400,
    ErrorCode.OCR_FAILED: 422,
    ErrorCode.OCR_TIMEOUT: 504,
    ErrorCode.QUEUE_FULL: 429,
    ErrorCode.TASK_NOT_FOUND: 404,
    ErrorCode.TASK_NOT_READY: 409,
    ErrorCode.TASK_CANCELLED: 409,
    ErrorCode.OCR_ENGINE_UNAVAILABLE: 503,
    ErrorCode.QR_NOT_FOUND: 422,
    ErrorCode.QR_INVALID_PAYLOAD: 422,
    ErrorCode.QR_SERVICE_UNAVAILABLE: 503,
    ErrorCode.SERVICE_UNAVAILABLE: 503,
    ErrorCode.VALIDATION_ERROR: 422,
    ErrorCode.INTERNAL_ERROR: 500,
}


class AppError(Exception):
    def __init__(
        self, code: ErrorCode, message: str, status_code: int | None = None
    ) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code or ERROR_HTTP_STATUS[code]
        super().__init__(message)


def error_payload(code: ErrorCode, message: str) -> dict[str, str]:
    if code == ErrorCode.INTERNAL_ERROR:
        message = INTERNAL_ERROR_MESSAGE
    return {
        "code": code.value,
        "message": message,
    }


def encode_error(code: ErrorCode, message: str) -> str:
    return json.dumps(error_payload(code, message), ensure_ascii=False)


def decode_error(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return _decoded_error_payload(value)

    text = str(value)
    if not text:
        return None

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return error_payload(ErrorCode.INTERNAL_ERROR, INTERNAL_ERROR_MESSAGE)

    if isinstance(parsed, dict):
        return _decoded_error_payload(parsed)

    return error_payload(ErrorCode.INTERNAL_ERROR, INTERNAL_ERROR_MESSAGE)


def _decoded_error_payload(value: dict[str, Any]) -> dict[str, str]:
    try:
        code = ErrorCode(str(value.get("code") or ""))
    except ValueError:
        code = ErrorCode.INTERNAL_ERROR
    message = str(value.get("message") or "")
    return error_payload(code, message)


def api_error_response(
    code: ErrorCode,
    message: str,
    *,
    data: Any = None,
) -> dict[str, Any]:
    return {
        "success": False,
        "data": data,
        "error": error_payload(code, message),
    }


def api_success_response(data: Any) -> dict[str, Any]:
    return {
        "success": True,
        "data": data,
        "error": None,
    }
