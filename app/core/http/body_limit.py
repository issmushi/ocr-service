from __future__ import annotations

import json
import logging
from collections.abc import Mapping

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.errors import ErrorCode, api_error_response

logger = logging.getLogger(__name__)


class RequestBodyLimitExceeded(Exception):
    pass


class RequestBodyLimitMiddleware:
    """Reject oversized request bodies before FastAPI parses multipart data."""

    def __init__(
        self,
        app: ASGIApp,
        max_body_size: int,
        path_limits: Mapping[str, int] | None = None,
    ) -> None:
        self.app = app
        self.max_body_size = max_body_size
        self.path_limits = dict(path_limits or {})

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = str(scope.get("path"))
        if self.path_limits and path not in self.path_limits:
            await self.app(scope, receive, send)
            return
        body_limit = self.path_limits.get(path, self.max_body_size)
        content_length = self._content_length(scope)
        if content_length is not None and content_length > body_limit:
            await self._reject(scope, send, content_length, body_limit)
            return

        received_size = 0

        async def limited_receive() -> Message:
            nonlocal received_size
            message = await receive()
            if message["type"] == "http.request":
                received_size += len(message.get("body", b""))
                if received_size > body_limit:
                    raise RequestBodyLimitExceeded
            return message

        try:
            await self.app(scope, limited_receive, send)
        except RequestBodyLimitExceeded:
            await self._reject(scope, send, received_size, body_limit)

    @staticmethod
    def _content_length(scope: Scope) -> int | None:
        for name, value in scope.get("headers", []):
            if name.lower() != b"content-length":
                continue
            try:
                return int(value)
            except ValueError:
                return None
        return None

    async def _reject(
        self,
        scope: Scope,
        send: Send,
        received_size: int,
        body_limit: int,
    ) -> None:
        logger.warning(
            "Request body rejected: too large method=%s path=%s "
            "size=%s limit=%s",
            scope.get("method"),
            scope.get("path"),
            received_size,
            body_limit,
        )
        payload = api_error_response(
            ErrorCode.IMAGE_TOO_LARGE,
            "Размер запроса превышает допустимый лимит",
        )
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json; charset=utf-8"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"connection", b"close"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
