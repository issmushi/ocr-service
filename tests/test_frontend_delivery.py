from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.bootstrap.factory import create_app


def _frontend_build(tmp_path: Path) -> Path:
    dist = tmp_path / "frontend" / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(
        "<!doctype html><html><body>OCR UI</body></html>",
        encoding="utf-8",
    )
    (dist / "assets" / "app.js").write_text(
        "globalThis.ocrUi = true;",
        encoding="utf-8",
    )
    return dist


def _create_test_app(dist: Path):
    with patch("app.bootstrap.factory.configure_logging"):
        return create_app(frontend_dist_dir=dist)


def test_frontend_build_is_served_under_ui_with_spa_fallback(
    tmp_path: Path,
) -> None:
    app = _create_test_app(_frontend_build(tmp_path))
    client = TestClient(app)

    index = client.get("/ui/")
    route = client.get("/ui/docs")
    asset = client.get("/ui/assets/app.js")

    assert index.status_code == 200
    assert index.text == route.text
    assert "OCR UI" in route.text
    assert asset.status_code == 200
    assert asset.text == "globalThis.ocrUi = true;"
    assert index.headers["cache-control"] == "no-cache"
    assert asset.headers["cache-control"] == (
        "public, max-age=31536000, immutable"
    )
    for response in (index, route, asset):
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert response.headers["referrer-policy"] == "no-referrer"
        assert response.headers["permissions-policy"] == (
            "camera=(self), microphone=()"
        )
        assert (
            "frame-ancestors 'none'"
            in response.headers["content-security-policy"]
        )
        assert (
            "media-src 'self' blob:"
            in response.headers["content-security-policy"]
        )


def test_frontend_missing_asset_does_not_fall_back_to_html(
    tmp_path: Path,
) -> None:
    client = TestClient(_create_test_app(_frontend_build(tmp_path)))

    response = client.get("/ui/assets/missing.js")

    assert response.status_code == 404
    assert "OCR UI" not in response.text


def test_api_starts_and_ui_is_404_when_frontend_build_is_absent(
    tmp_path: Path,
) -> None:
    client = TestClient(_create_test_app(tmp_path / "missing-dist"))

    ui_response = client.get("/ui/")
    openapi_response = client.get("/openapi.json")

    assert ui_response.status_code == 404
    assert openapi_response.status_code == 200


def test_frontend_mount_is_not_added_to_openapi(tmp_path: Path) -> None:
    app = _create_test_app(_frontend_build(tmp_path))

    assert "/ui" not in app.openapi()["paths"]
