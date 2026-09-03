from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.bootstrap import http
from app.bootstrap.factory import create_app


class ApplicationFactoryTests(unittest.TestCase):
    def test_openapi_cache_and_routes_belong_to_each_application(self):
        with patch("app.bootstrap.factory.configure_logging"):
            first = create_app()
            second = create_app()

        @first.get("/first-only")
        async def first_only() -> dict[str, bool]:
            return {"ok": True}

        first_schema = first.openapi()
        second_schema = second.openapi()

        self.assertIs(first.openapi(), first_schema)
        self.assertIs(second.openapi(), second_schema)
        self.assertIsNot(first_schema, second_schema)
        self.assertIn("/first-only", first_schema["paths"])
        self.assertNotIn("/first-only", second_schema["paths"])
        self.assertNotIn(
            "422", first_schema["paths"]["/qr/generate"]["post"]["responses"]
        )

    def test_cors_and_logging_wrap_body_size_rejections(self):
        with (
            patch("app.bootstrap.factory.configure_logging"),
            patch.object(http, "settings") as settings,
        ):
            settings.api_max_request_body_size_bytes = 4
            settings.api_qr_max_request_body_size_bytes = 2
            app = create_app()

        with patch.object(http.logger, "warning") as warning:
            response = TestClient(app).post(
                "/v1/ocr/full",
                content=b"12345",
                headers={"Origin": "https://example.test"},
            )

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error"]["code"], "IMAGE_TOO_LARGE")
        self.assertEqual(response.headers["access-control-allow-origin"], "*")
        warning.assert_called_once()
        self.assertEqual(warning.call_args.args[3], 413)


if __name__ == "__main__":
    unittest.main()
