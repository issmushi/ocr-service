from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_OCR_MODELS = frozenset(
    {
        "paddle",
        "mrz",
        "russian-ocr",
        "qwen3-vl-4b",
        "qwen3-vl-8b",
    }
)
DEFAULT_OCR_MODELS = ",".join(sorted(SUPPORTED_OCR_MODELS))


def load_env_file() -> Path:
    config_path = Path(__file__).resolve()
    for directory in (Path.cwd(), *config_path.parents):
        env_path = directory / ".env"
        if not env_path.is_file():
            continue

        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key:
                os.environ.setdefault(key, value)
        return env_path.parent

    return Path.cwd()


PROJECT_ROOT = load_env_file()


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    host: str = os.getenv("API_HOST", "127.0.0.1")
    port: int = env_int("API_PORT", 8000)
    russian_ocr_device: str = os.getenv(
        "RUSSIAN_OCR_DEVICE", os.getenv("OCR_DEVICE", "cpu")
    )
    russian_ocr_startup_timeout_seconds: float = env_float(
        "RUSSIAN_OCR_STARTUP_TIMEOUT_SECONDS", 300.0
    )
    russian_ocr_shutdown_timeout_seconds: float = env_float(
        "RUSSIAN_OCR_SHUTDOWN_TIMEOUT_SECONDS", 1.0
    )
    russian_ocr_document_hard_timeout_seconds: float = env_float(
        "RUSSIAN_OCR_DOCUMENT_HARD_TIMEOUT_SECONDS", 18.0
    )
    russian_ocr_full_hard_timeout_seconds: float = env_float(
        "RUSSIAN_OCR_FULL_HARD_TIMEOUT_SECONDS", 38.0
    )
    vlm_ocr_url: str = os.getenv("VLM_OCR_URL", "http://127.0.0.1:8004")
    vlm_ocr_timeout_seconds: float = env_float(
        "VLM_OCR_TIMEOUT_SECONDS", 300.0
    )
    qr_max_image_bytes: int = env_int(
        "QR_MAX_IMAGE_BYTES",
        25 * 1024 * 1024,
    )
    qr_max_images_per_request: int = env_int(
        "QR_MAX_IMAGES_PER_REQUEST",
        3,
    )
    api_qr_max_request_body_size_mb: int = env_int(
        "API_QR_MAX_REQUEST_BODY_SIZE_MB",
        ((qr_max_image_bytes * qr_max_images_per_request) // (1024 * 1024))
        + 1,
    )
    health_check_timeout_seconds: float = env_float(
        "HEALTH_CHECK_TIMEOUT_SECONDS",
        3.0,
    )
    paddle_ocr_startup_timeout_seconds: float = env_float(
        "PADDLE_OCR_STARTUP_TIMEOUT_SECONDS",
        300.0,
    )
    paddle_ocr_shutdown_timeout_seconds: float = env_float(
        "PADDLE_OCR_SHUTDOWN_TIMEOUT_SECONDS",
        1.0,
    )
    paddle_ocr_document_hard_timeout_seconds: float = env_float(
        "PADDLE_OCR_DOCUMENT_HARD_TIMEOUT_SECONDS",
        18.0,
    )
    paddle_ocr_full_hard_timeout_seconds: float = env_float(
        "PADDLE_OCR_FULL_HARD_TIMEOUT_SECONDS",
        38.0,
    )
    mrz_scanner_startup_timeout_seconds: float = env_float(
        "MRZ_SCANNER_STARTUP_TIMEOUT_SECONDS",
        300.0,
    )
    mrz_scanner_shutdown_timeout_seconds: float = env_float(
        "MRZ_SCANNER_SHUTDOWN_TIMEOUT_SECONDS",
        1.0,
    )
    mrz_scanner_document_hard_timeout_seconds: float = env_float(
        "MRZ_SCANNER_DOCUMENT_HARD_TIMEOUT_SECONDS",
        18.0,
    )
    qr_active_key_id: str = os.getenv("QR_ACTIVE_KEY_ID", "primary")
    qr_encryption_keys: str = os.getenv("QR_ENCRYPTION_KEYS", "")
    postgres_host: str = os.getenv("POSTGRES_HOST", "127.0.0.1")
    postgres_port: int = env_int("POSTGRES_PORT", 5432)
    postgres_user: str = os.getenv("POSTGRES_USER", "ocr")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "ocr")
    postgres_db: str = os.getenv("POSTGRES_DB", "ocr")
    rabbitmq_host: str = os.getenv("RABBITMQ_HOST", "127.0.0.1")
    rabbitmq_port: int = env_int("RABBITMQ_PORT", 5672)
    rabbitmq_user: str = os.getenv("RABBITMQ_USER", "guest")
    rabbitmq_password: str = os.getenv("RABBITMQ_PASSWORD", "guest")
    ocr_queue_name: str = os.getenv("OCR_QUEUE_NAME", "ocr.jobs")
    ocr_dead_letter_queue_name: str = os.getenv(
        "OCR_DLQ_NAME",
        f"{os.getenv('OCR_QUEUE_NAME', 'ocr.jobs')}.dlq",
    )
    ocr_storage_dir: str = os.getenv("OCR_STORAGE_DIR", "storage/uploads")
    ocr_save_uploads: bool = env_bool("OCR_SAVE_UPLOADS", False)
    ocr_max_image_size_mb: int = env_int("OCR_MAX_IMAGE_SIZE_MB", 75)
    ocr_max_images_per_request: int = env_int(
        "OCR_MAX_IMAGES_PER_REQUEST",
        3,
    )
    api_max_request_body_size_mb: int = env_int(
        "API_MAX_REQUEST_BODY_SIZE_MB",
        (ocr_max_image_size_mb * ocr_max_images_per_request) + 1,
    )
    ocr_max_image_width: int = env_int("OCR_MAX_IMAGE_WIDTH", 12_000)
    ocr_max_image_height: int = env_int("OCR_MAX_IMAGE_HEIGHT", 12_000)
    ocr_max_image_pixels: int = env_int("OCR_MAX_IMAGE_PIXELS", 50_000_000)
    ocr_quality_max_side: int = env_int("OCR_QUALITY_MAX_SIDE", 1_600)
    ocr_max_attempts: int = env_int("OCR_MAX_ATTEMPTS", 2)
    ocr_retry_delay_seconds: float = env_float("OCR_RETRY_DELAY_SECONDS", 3.0)
    ocr_workers: int = env_int("OCR_WORKERS", 1)
    ocr_worker_startup_timeout_seconds: float = env_float(
        "OCR_WORKER_STARTUP_TIMEOUT_SECONDS",
        30.0,
    )
    ocr_worker_lease_seconds: float = env_float(
        "OCR_WORKER_LEASE_SECONDS",
        30.0,
    )
    ocr_worker_heartbeat_seconds: float = env_float(
        "OCR_WORKER_HEARTBEAT_SECONDS",
        5.0,
    )
    ocr_lease_recovery_interval_seconds: float = env_float(
        "OCR_LEASE_RECOVERY_INTERVAL_SECONDS",
        10.0,
    )
    ocr_queued_watchdog_interval_seconds: float = env_float(
        "OCR_QUEUED_WATCHDOG_INTERVAL_SECONDS",
        10.0,
    )
    ocr_queued_watchdog_stale_seconds: float = env_float(
        "OCR_QUEUED_WATCHDOG_STALE_SECONDS",
        60.0,
    )
    ocr_device: str = os.getenv("OCR_DEVICE", "cpu")
    ocr_models: str = os.getenv("OCR_MODELS", DEFAULT_OCR_MODELS)
    ocr_queue_max_size: int = env_int("OCR_QUEUE_MAX_SIZE", 50)
    ocr_outbox_poll_interval_seconds: float = env_float(
        "OCR_OUTBOX_POLL_INTERVAL_SECONDS",
        1.0,
    )
    ocr_outbox_batch_size: int = env_int("OCR_OUTBOX_BATCH_SIZE", 20)
    ocr_outbox_publish_timeout_seconds: float = env_float(
        "OCR_OUTBOX_PUBLISH_TIMEOUT_SECONDS",
        5.0,
    )
    ocr_outbox_max_retry_delay_seconds: float = env_float(
        "OCR_OUTBOX_MAX_RETRY_DELAY_SECONDS",
        60.0,
    )
    ocr_result_ttl_seconds: float = env_float(
        "OCR_RESULT_TTL_SECONDS", 86400.0
    )
    ocr_cleanup_interval_seconds: float = env_float(
        "OCR_CLEANUP_INTERVAL_SECONDS", 300.0
    )
    api_log_level: str = os.getenv("API_LOG_LEVEL", "INFO")
    log_dir: str = os.getenv("LOG_DIR", "logs")
    log_max_bytes: int = env_int("LOG_MAX_BYTES", 10 * 1024 * 1024)
    log_backup_count: int = env_int("LOG_BACKUP_COUNT", 5)

    def __post_init__(self) -> None:
        if self.ocr_device not in {"cpu", "gpu"}:
            raise ValueError("OCR_DEVICE must be cpu or gpu")
        if self.russian_ocr_device not in {"cpu", "gpu"}:
            raise ValueError("RUSSIAN_OCR_DEVICE must be cpu or gpu")
        if not self.enabled_ocr_models:
            raise ValueError("OCR_MODELS must contain at least one model")
        unknown_models = self.enabled_ocr_models - SUPPORTED_OCR_MODELS
        if unknown_models:
            raise ValueError(
                "OCR_MODELS contains unsupported models: "
                + ", ".join(sorted(unknown_models))
            )
        if self.ocr_workers != 1:
            raise ValueError("OCR_WORKERS must be exactly 1")
        if self.ocr_worker_startup_timeout_seconds <= 0:
            raise ValueError(
                "OCR_WORKER_STARTUP_TIMEOUT_SECONDS must be positive"
            )

        timeout_values = {
            "RUSSIAN_OCR_STARTUP_TIMEOUT_SECONDS": (
                self.russian_ocr_startup_timeout_seconds
            ),
            "RUSSIAN_OCR_SHUTDOWN_TIMEOUT_SECONDS": (
                self.russian_ocr_shutdown_timeout_seconds
            ),
            "RUSSIAN_OCR_DOCUMENT_HARD_TIMEOUT_SECONDS": (
                self.russian_ocr_document_hard_timeout_seconds
            ),
            "RUSSIAN_OCR_FULL_HARD_TIMEOUT_SECONDS": (
                self.russian_ocr_full_hard_timeout_seconds
            ),
            "MRZ_SCANNER_STARTUP_TIMEOUT_SECONDS": (
                self.mrz_scanner_startup_timeout_seconds
            ),
            "MRZ_SCANNER_SHUTDOWN_TIMEOUT_SECONDS": (
                self.mrz_scanner_shutdown_timeout_seconds
            ),
            "MRZ_SCANNER_DOCUMENT_HARD_TIMEOUT_SECONDS": (
                self.mrz_scanner_document_hard_timeout_seconds
            ),
            "PADDLE_OCR_STARTUP_TIMEOUT_SECONDS": (
                self.paddle_ocr_startup_timeout_seconds
            ),
            "PADDLE_OCR_SHUTDOWN_TIMEOUT_SECONDS": (
                self.paddle_ocr_shutdown_timeout_seconds
            ),
            "PADDLE_OCR_DOCUMENT_HARD_TIMEOUT_SECONDS": (
                self.paddle_ocr_document_hard_timeout_seconds
            ),
            "PADDLE_OCR_FULL_HARD_TIMEOUT_SECONDS": (
                self.paddle_ocr_full_hard_timeout_seconds
            ),
        }
        for name, value in timeout_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def ocr_max_image_size_bytes(self) -> int:
        return self.ocr_max_image_size_mb * 1024 * 1024

    @property
    def api_max_request_body_size_bytes(self) -> int:
        return self.api_max_request_body_size_mb * 1024 * 1024

    @property
    def api_qr_max_request_body_size_bytes(self) -> int:
        return self.api_qr_max_request_body_size_mb * 1024 * 1024

    @property
    def postgres_async_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_sync_dsn(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def rabbitmq_url(self) -> str:
        return (
            f"amqp://{self.rabbitmq_user}:{self.rabbitmq_password}"
            f"@{self.rabbitmq_host}:{self.rabbitmq_port}/"
        )

    @property
    def storage_path(self) -> Path:
        path = Path(self.ocr_storage_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def log_path(self) -> Path:
        path = Path(self.log_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def enabled_ocr_models(self) -> frozenset[str]:
        return frozenset(
            model.strip().lower()
            for model in self.ocr_models.split(",")
            if model.strip()
        )

    def is_model_enabled(self, model: str) -> bool:
        return model.lower() in self.enabled_ocr_models


settings = Settings()
