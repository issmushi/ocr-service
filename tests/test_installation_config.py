from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.errors import AppError, ErrorCode
from app.core.model_availability import ensure_ocr_model_enabled
from app.main import app

ROOT = Path(__file__).resolve().parents[1]


def test_env_template_contains_only_current_installer_metadata() -> None:
    lines = (ROOT / ".env.template").read_text(encoding="utf-8").splitlines()
    keys = [
        line.split("=", 1)[0]
        for line in lines
        if line and not line.startswith("#") and "=" in line
    ]

    assert len(keys) == len(set(keys))
    assert {
        "OCR_EDITION",
        "OCR_ACCELERATOR",
        "RUSSIAN_OCR_MODEL_FORMAT",
    }.isdisjoint(keys)
    assert {
        "OCR_MODELS",
        "OCR_RUNTIME_PROFILE",
        "HF_HOME",
        "RUSSIAN_OCR_MODELS_DIR",
    } <= set(keys)

    template = (ROOT / ".env.template").read_text(encoding="utf-8")
    assert "# RUSSIAN_OCR_MODELS_SOURCE_DIR=" in template


def test_settings_parse_enabled_models() -> None:
    settings = Settings(ocr_models=" paddle,MRZ,paddle ")

    assert settings.enabled_ocr_models == frozenset({"paddle", "mrz"})
    assert settings.is_model_enabled("PADDLE")
    assert not settings.is_model_enabled("qwen3-vl-4b")


@pytest.mark.parametrize("models", ["", "unknown", "paddle,unknown"])
def test_settings_reject_invalid_model_lists(models: str) -> None:
    with pytest.raises(ValueError, match="OCR_MODELS"):
        Settings(ocr_models=models)


def test_disabled_model_returns_engine_unavailable() -> None:
    configured = SimpleNamespace(
        is_model_enabled=lambda model: model == "paddle"
    )

    with (
        patch("app.core.model_availability.settings", configured),
        pytest.raises(AppError) as raised,
    ):
        ensure_ocr_model_enabled("qwen3-vl-4b")

    assert raised.value.code == ErrorCode.OCR_ENGINE_UNAVAILABLE
    assert "OCR_MODELS" in raised.value.message


@pytest.mark.parametrize(
    ("path", "form"),
    [
        ("/v1/ocr/passport/mrz", {"model": "qwen3-vl-4b"}),
        ("/v1/ocr/passport/page", {"page": "3", "model": "qwen3-vl-4b"}),
        ("/v1/ocr/idcard/mrz", {"model": "qwen3-vl-4b"}),
        ("/v1/ocr/full", {"model": "qwen3-vl-4b"}),
        ("/v1/tasks/passport/mrz", {"model": "qwen3-vl-4b"}),
        ("/v1/tasks/passport/page", {"page": "3", "model": "qwen3-vl-4b"}),
        ("/v1/tasks/idcard/mrz", {"model": "qwen3-vl-4b"}),
        ("/v1/tasks/full", {"model": "qwen3-vl-4b"}),
    ],
)
def test_disabled_model_is_rejected_by_sync_and_task_endpoints(
    path: str,
    form: dict[str, str],
) -> None:
    configured = SimpleNamespace(
        is_model_enabled=lambda model: model != "qwen3-vl-4b"
    )

    client = TestClient(app)
    try:
        with patch("app.core.model_availability.settings", configured):
            response = client.post(
                path,
                files={"image": ("passport.jpg", b"not-read", "image/jpeg")},
                data=form,
            )
    finally:
        client.close()

    assert response.status_code == 503
    assert response.json()["error"]["code"] == ErrorCode.OCR_ENGINE_UNAVAILABLE
