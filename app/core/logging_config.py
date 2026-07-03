from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
LEVEL_COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[1;31m",
}


class ServiceNameFilter(logging.Filter):
    def __init__(self, service_name: str) -> None:
        super().__init__()
        self.service_name = service_name

    def filter(self, record: logging.LogRecord) -> bool:
        record.service_name = self.service_name
        return True


class ServiceConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        level = record.levelname
        level_color = LEVEL_COLORS.get(level, "")
        service_name = getattr(record, "service_name", "api")
        message = record.getMessage()

        formatted = (
            f"{DIM}{timestamp}{RESET} "
            f"{level_color}{BOLD}{level:<8}{RESET} "
            f"{CYAN}{service_name:<7}{RESET} "
            f"{DIM}{record.name}{RESET} "
            f"{message}"
        )

        if record.exc_info:
            formatted = f"{formatted}\n{self.formatException(record.exc_info)}"
        if record.stack_info:
            formatted = f"{formatted}\n{self.formatStack(record.stack_info)}"
        return formatted


def configure_logging(
    *,
    service_name: str,
    log_dir: Path,
    level: str,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s level=%(levelname)s service=%(service_name)s "
        "logger=%(name)s message=%(message)s"
    )
    console_formatter: logging.Formatter
    console_formatter = ServiceConsoleFormatter()
    service_filter = ServiceNameFilter(service_name)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(service_filter)

    file_handler = RotatingFileHandler(
        log_dir / f"{service_name}.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(service_filter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    for logger_name in (
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "httpx",
        "aio_pika",
    ):
        logging.getLogger(logger_name).setLevel(numeric_level)
