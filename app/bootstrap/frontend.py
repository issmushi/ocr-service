"""Serve the compiled web interface from the API process."""

from __future__ import annotations

import logging
from pathlib import Path, PurePosixPath

from fastapi import FastAPI
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

logger = logging.getLogger(__name__)

DEFAULT_FRONTEND_DIST_DIR = (
    Path(__file__).resolve().parents[1] / "static" / "ui"
)

UI_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; "
        "img-src 'self' blob: data:; connect-src 'self' http: https:; "
        "font-src 'self'; media-src 'self' blob:; base-uri 'self'; "
        "form-action 'self'; "
        "frame-ancestors 'none'"
    ),
    "Permissions-Policy": "camera=(self), microphone=()",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}


class SpaStaticFiles(StaticFiles):
    """Serve static files and fall back to ``index.html`` for SPA routes."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            response = await super().get_response(path, scope)
        except HTTPException as error:
            if error.status_code != 404 or not self._is_spa_route(path, scope):
                raise
            path = "index.html"
            response = await super().get_response(path, scope)

        for name, value in UI_SECURITY_HEADERS.items():
            response.headers[name] = value
        if PurePosixPath(path).name == "index.html":
            response.headers["Cache-Control"] = "no-cache"
        elif PurePosixPath(path).parts[:1] == ("assets",):
            response.headers["Cache-Control"] = (
                "public, max-age=31536000, immutable"
            )
        else:
            response.headers["Cache-Control"] = "no-cache"
        return response

    @staticmethod
    def _is_spa_route(path: str, scope: Scope) -> bool:
        if scope.get("method") not in {"GET", "HEAD"}:
            return False
        if PurePosixPath(path).suffix:
            return False

        accept = Headers(scope=scope).get("accept", "")
        return not accept or "text/html" in accept or "*/*" in accept


def configure_frontend(
    app: FastAPI,
    dist_dir: str | Path | None = None,
) -> bool:
    """Mount a complete frontend build and keep API startup independent.

    A source checkout may legitimately have no compiled frontend yet. In that
    case the API still starts, ``/ui`` stays a normal 404, and the installer is
    responsible for producing the build before a production start.
    """

    directory = Path(dist_dir or DEFAULT_FRONTEND_DIST_DIR)
    if not directory.is_dir() or not (directory / "index.html").is_file():
        logger.warning(
            "Frontend build is unavailable; /ui is disabled dist_dir=%s",
            directory,
        )
        return False

    app.mount(
        "/ui",
        SpaStaticFiles(directory=directory, html=True),
        name="ui",
    )
    logger.info("Frontend mounted path=/ui dist_dir=%s", directory)
    return True
