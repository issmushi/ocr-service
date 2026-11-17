from __future__ import annotations

import json
from datetime import date
from pathlib import Path
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
    dates = [date.fromisoformat(release["date"]) for release in releases]
    assert dates == sorted(dates, reverse=True)
    assert releases[0]["changes"]
    assert "/v1/changelog" not in app.openapi()["paths"]
    frontend = Path(__file__).resolve().parents[1] / "frontend"
    package = json.loads((frontend / "package.json").read_text())
    lock = json.loads((frontend / "package-lock.json").read_text())
    assert package["version"] == app.version
    assert lock["version"] == lock["packages"][""]["version"] == app.version
