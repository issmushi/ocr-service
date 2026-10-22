from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.bootstrap.factory import create_app
from app.modules.health.service import get_app_version


def test_changelog_matches_exposed_service_version() -> None:
    with patch("app.bootstrap.factory.configure_logging"):
        app = create_app()

    response = TestClient(app).get("/v1/changelog")

    assert response.status_code == 200
    releases = response.json()["releases"]
    assert releases[0]["version"] == get_app_version() == app.version
    assert releases[0]["date"] == "2026-10-22"
    assert releases[0]["changes"]
    assert "/v1/changelog" not in app.openapi()["paths"]
