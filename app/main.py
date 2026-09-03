"""ASGI entrypoint and command-line server launcher."""

from __future__ import annotations

import logging

import uvicorn

from app.bootstrap.factory import create_app
from app.core.config import settings

logger = logging.getLogger(__name__)
app = create_app()


def run() -> None:
    logger.info(
        "Starting uvicorn API host=%s port=%s",
        settings.host,
        settings.port,
    )
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        workers=1,
        log_config=None,
    )


if __name__ == "__main__":
    run()
