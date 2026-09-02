from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.modules.health.service import (
    failed_component,
    get_health_status,
    healthy_component,
    probe_mrz_runtime,
    probe_paddle_runtime,
    probe_russian_runtime,
    probe_vlm_service,
)


def health_settings(workers: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        ocr_workers=workers,
        ocr_device="cpu",
        russian_ocr_device="cpu",
        health_check_timeout_seconds=0.25,
        vlm_ocr_url="http://vlm.test:8004",
    )


def queue_component(consumers: int = 1) -> dict[str, object]:
    return healthy_component(
        4,
        details={
            "queue_name": "ocr.jobs",
            "message_count": 0,
            "consumer_count": consumers,
        },
    )


class HealthServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_vlm_is_healthy_without_network_probe(self) -> None:
        configured = health_settings()
        configured.enabled_ocr_models = frozenset(
            {"paddle", "mrz", "russian-ocr"}
        )
        vlm_probe = AsyncMock(
            side_effect=AssertionError("disabled VLM must not be probed")
        )
        with (
            patch("app.modules.health.service.settings", configured),
            patch(
                "app.modules.health.service.get_app_version",
                return_value="0.1.0",
            ),
            patch(
                "app.modules.health.service.probe_database",
                new=AsyncMock(return_value=healthy_component(3)),
            ),
            patch(
                "app.modules.health.service.probe_queue",
                new=AsyncMock(return_value=queue_component()),
            ),
            patch(
                "app.modules.health.service.probe_russian_runtime",
                new=AsyncMock(return_value=healthy_component(1)),
            ),
            patch(
                "app.modules.health.service.probe_paddle_runtime",
                new=AsyncMock(return_value=healthy_component(1)),
            ),
            patch(
                "app.modules.health.service.probe_mrz_runtime",
                new=AsyncMock(return_value=healthy_component(1)),
            ),
            patch(
                "app.modules.health.service.probe_vlm_service",
                new=vlm_probe,
            ),
            patch(
                "app.modules.health.service.probe_qr_module",
                new=AsyncMock(return_value=healthy_component(1)),
            ),
        ):
            result = await get_health_status()

        self.assertTrue(result["success"])
        self.assertEqual(
            result["data"]["components"]["qwen_vlm_service"]["details"],
            {"engine": "qwen3-vl", "status": "disabled"},
        )
        vlm_probe.assert_not_awaited()

    async def test_returns_success_when_all_components_are_healthy(
        self,
    ) -> None:
        async def russian_probe() -> dict[str, object]:
            return healthy_component(
                5,
                details={"engine": "russian-docs-ocr"},
            )

        with (
            patch(
                "app.modules.health.service.settings",
                health_settings(),
            ),
            patch(
                "app.modules.health.service.get_app_version",
                return_value="0.1.0",
            ),
            patch(
                "app.modules.health.service.probe_database",
                new=AsyncMock(return_value=healthy_component(3)),
            ),
            patch(
                "app.modules.health.service.probe_queue",
                new=AsyncMock(return_value=queue_component()),
            ),
            patch(
                "app.modules.health.service.probe_russian_runtime",
                new=AsyncMock(side_effect=russian_probe),
            ),
            patch(
                "app.modules.health.service.probe_paddle_runtime",
                new=AsyncMock(return_value=healthy_component(1)),
            ),
            patch(
                "app.modules.health.service.probe_mrz_runtime",
                new=AsyncMock(return_value=healthy_component(1)),
            ),
            patch(
                "app.modules.health.service.probe_vlm_service",
                new=AsyncMock(return_value=healthy_component(1)),
            ),
            patch(
                "app.modules.health.service.probe_qr_module",
                new=AsyncMock(return_value=healthy_component(1)),
            ),
        ):
            result = await get_health_status()

        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["status"], "ok")
        self.assertEqual(result["data"]["service"], "ocr-service")
        self.assertEqual(result["data"]["active_workers"], 1)
        self.assertNotIn("gpu_available", result["data"])
        self.assertEqual(
            set(result["data"]["components"]),
            {
                "api",
                "postgres",
                "rabbitmq",
                "worker",
                "paddle_ocr",
                "russian_ocr",
                "mrz_scanner",
                "qwen_vlm_service",
                "qr",
            },
        )
        self.assertIsNone(result["error"])
        self.assertTrue(
            all(
                component["status"] == "ok"
                for component in result["data"]["components"].values()
            )
        )

    async def test_returns_degraded_when_ocr_service_is_unavailable(
        self,
    ) -> None:
        async def russian_probe() -> dict[str, object]:
            return healthy_component(5)

        with (
            patch(
                "app.modules.health.service.settings",
                health_settings(),
            ),
            patch(
                "app.modules.health.service.get_app_version",
                return_value="0.1.0",
            ),
            patch(
                "app.modules.health.service.probe_database",
                new=AsyncMock(return_value=healthy_component(3)),
            ),
            patch(
                "app.modules.health.service.probe_queue",
                new=AsyncMock(return_value=queue_component()),
            ),
            patch(
                "app.modules.health.service.probe_russian_runtime",
                new=AsyncMock(side_effect=russian_probe),
            ),
            patch(
                "app.modules.health.service.probe_paddle_runtime",
                new=AsyncMock(
                    return_value=failed_component(3000, "PaddleOCR недоступен")
                ),
            ),
            patch(
                "app.modules.health.service.probe_mrz_runtime",
                new=AsyncMock(return_value=healthy_component(1)),
            ),
            patch(
                "app.modules.health.service.probe_vlm_service",
                new=AsyncMock(return_value=healthy_component(1)),
            ),
            patch(
                "app.modules.health.service.probe_qr_module",
                new=AsyncMock(return_value=healthy_component(1)),
            ),
        ):
            result = await get_health_status()

        self.assertFalse(result["success"])
        self.assertEqual(result["data"]["status"], "degraded")
        self.assertEqual(
            result["data"]["components"]["paddle_ocr"]["status"],
            "error",
        )
        self.assertEqual(result["error"]["code"], "SERVICE_UNAVAILABLE")
        self.assertIn("paddle_ocr", result["error"]["message"])

    async def test_reports_missing_worker_as_degraded(self) -> None:
        with (
            patch(
                "app.modules.health.service.settings",
                health_settings(),
            ),
            patch(
                "app.modules.health.service.get_app_version",
                return_value="0.1.0",
            ),
            patch(
                "app.modules.health.service.probe_database",
                new=AsyncMock(return_value=healthy_component(3)),
            ),
            patch(
                "app.modules.health.service.probe_queue",
                new=AsyncMock(return_value=queue_component(consumers=0)),
            ),
            patch(
                "app.modules.health.service.probe_russian_runtime",
                new=AsyncMock(return_value=healthy_component(5)),
            ),
            patch(
                "app.modules.health.service.probe_paddle_runtime",
                new=AsyncMock(return_value=healthy_component(1)),
            ),
            patch(
                "app.modules.health.service.probe_mrz_runtime",
                new=AsyncMock(return_value=healthy_component(1)),
            ),
            patch(
                "app.modules.health.service.probe_vlm_service",
                new=AsyncMock(return_value=healthy_component(1)),
            ),
            patch(
                "app.modules.health.service.probe_qr_module",
                new=AsyncMock(return_value=healthy_component(1)),
            ),
        ):
            result = await get_health_status()

        worker = result["data"]["components"]["worker"]
        self.assertFalse(result["success"])
        self.assertEqual(worker["status"], "error")
        self.assertEqual(worker["details"]["active_workers"], 0)
        self.assertIn("worker", result["error"]["message"])

    async def test_reports_multiple_workers_as_degraded(self) -> None:
        with (
            patch(
                "app.modules.health.service.settings",
                health_settings(),
            ),
            patch(
                "app.modules.health.service.get_app_version",
                return_value="0.1.0",
            ),
            patch(
                "app.modules.health.service.probe_database",
                new=AsyncMock(return_value=healthy_component(3)),
            ),
            patch(
                "app.modules.health.service.probe_queue",
                new=AsyncMock(return_value=queue_component(consumers=2)),
            ),
            patch(
                "app.modules.health.service.probe_russian_runtime",
                new=AsyncMock(return_value=healthy_component(5)),
            ),
            patch(
                "app.modules.health.service.probe_paddle_runtime",
                new=AsyncMock(return_value=healthy_component(1)),
            ),
            patch(
                "app.modules.health.service.probe_mrz_runtime",
                new=AsyncMock(return_value=healthy_component(1)),
            ),
            patch(
                "app.modules.health.service.probe_vlm_service",
                new=AsyncMock(return_value=healthy_component(1)),
            ),
            patch(
                "app.modules.health.service.probe_qr_module",
                new=AsyncMock(return_value=healthy_component(1)),
            ),
        ):
            result = await get_health_status()

        worker = result["data"]["components"]["worker"]
        self.assertFalse(result["success"])
        self.assertEqual(worker["status"], "error")
        self.assertEqual(worker["details"]["active_workers"], 2)


class PaddleRuntimeHealthTests(unittest.IsolatedAsyncioTestCase):
    async def test_starting_runtime_is_not_reported_as_healthy_busy(
        self,
    ) -> None:
        runtime = SimpleNamespace(
            status="starting",
            ready=False,
            busy=True,
            worker_pid=1234,
        )
        with (
            patch("app.modules.health.service.settings", health_settings()),
            patch("app.modules.health.service.paddle_runtime", runtime),
        ):
            result = await probe_paddle_runtime()

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"], "PaddleOCR запускается")
        self.assertEqual(
            result["details"],
            {
                "engine": "paddleocr",
                "status": "starting",
                "busy": True,
                "ready": False,
                "worker_pid": 1234,
                "device": "cpu",
            },
        )

    async def test_ready_runtime_can_be_idle_or_processing(self) -> None:
        for busy, expected_state in ((False, "ready"), (True, "busy")):
            runtime = SimpleNamespace(
                status="ready",
                ready=True,
                busy=busy,
                worker_pid=1234,
            )
            with (
                self.subTest(busy=busy),
                patch(
                    "app.modules.health.service.settings", health_settings()
                ),
                patch("app.modules.health.service.paddle_runtime", runtime),
            ):
                result = await probe_paddle_runtime()

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["details"]["status"], expected_state)


class MrzRuntimeHealthTests(unittest.IsolatedAsyncioTestCase):
    async def test_stopped_lazy_runtime_is_healthy_idle(self) -> None:
        runtime = SimpleNamespace(
            status="stopped",
            ready=False,
            busy=False,
            worker_pid=None,
        )
        with patch("app.modules.health.service.mrz_scanner_runtime", runtime):
            result = await probe_mrz_runtime()

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["error"], None)
        self.assertEqual(
            result["details"],
            {
                "engine": "mrz-scanner",
                "status": "idle",
                "busy": False,
                "ready": False,
                "worker_pid": None,
            },
        )

    async def test_runtime_states_distinguish_ready_starting_and_down(
        self,
    ) -> None:
        cases = (
            ("ready", True, False, "ready", "ok"),
            ("ready", True, True, "busy", "ok"),
            ("starting", False, True, "starting", "error"),
            ("unavailable", False, False, "down", "error"),
        )
        for status, ready, busy, expected_state, expected_health in cases:
            runtime = SimpleNamespace(
                status=status,
                ready=ready,
                busy=busy,
                worker_pid=1234 if status != "unavailable" else None,
            )
            with (
                self.subTest(status=status, busy=busy),
                patch(
                    "app.modules.health.service.mrz_scanner_runtime", runtime
                ),
            ):
                result = await probe_mrz_runtime()

            self.assertEqual(result["status"], expected_health)
            self.assertEqual(result["details"]["status"], expected_state)


class VlmServiceHealthTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def http_client(
        response: object | None = None,
    ) -> tuple[MagicMock, AsyncMock]:
        client = AsyncMock()
        if response is not None:
            client.get.return_value = response
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=client)
        context.__aexit__ = AsyncMock(return_value=False)
        return context, client

    async def test_ready_service_reports_models_and_uses_short_timeout(
        self,
    ) -> None:
        response = SimpleNamespace(
            status_code=200,
            json=lambda: {
                "status": "ready",
                "models": ["qwen3-vl-4b", "qwen3-vl-8b"],
                "active_model": "qwen3-vl-4b",
            },
        )
        context, client = self.http_client(response)
        with (
            patch("app.modules.health.service.settings", health_settings()),
            patch(
                "app.modules.health.service.httpx.AsyncClient",
                return_value=context,
            ) as client_factory,
        ):
            result = await probe_vlm_service()

        self.assertEqual(result["status"], "ok")
        self.assertEqual(
            result["details"],
            {
                "engine": "qwen3-vl",
                "status": "ready",
                "models": ["qwen3-vl-4b", "qwen3-vl-8b"],
                "active_model": "qwen3-vl-4b",
            },
        )
        client_factory.assert_called_once_with(timeout=0.25, trust_env=False)
        client.get.assert_awaited_once_with("http://vlm.test:8004/health")

    async def test_non_ready_service_is_unavailable(self) -> None:
        response = SimpleNamespace(
            status_code=503,
            json=lambda: {
                "status": "stopping",
                "models": ["qwen3-vl-4b", "qwen3-vl-8b"],
                "active_model": None,
            },
        )
        context, _client = self.http_client(response)
        with (
            patch("app.modules.health.service.settings", health_settings()),
            patch(
                "app.modules.health.service.httpx.AsyncClient",
                return_value=context,
            ),
        ):
            result = await probe_vlm_service()

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"], "Qwen VLM недоступен")
        self.assertEqual(result["details"]["status"], "stopping")

    async def test_connection_failure_returns_safe_down_component(
        self,
    ) -> None:
        context, client = self.http_client()
        client.get.side_effect = OSError("private connection details")
        with (
            patch("app.modules.health.service.settings", health_settings()),
            patch(
                "app.modules.health.service.httpx.AsyncClient",
                return_value=context,
            ),
        ):
            result = await probe_vlm_service()

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"], "Qwen VLM недоступен")
        self.assertEqual(
            result["details"],
            {
                "engine": "qwen3-vl",
                "status": "down",
                "models": [],
                "active_model": None,
            },
        )
        self.assertNotIn("private", str(result))


class RussianRuntimeHealthTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_states_include_pid_device_and_format_without_http(
        self,
    ) -> None:
        cases = (
            ("ready", True, False, 1234, "ready", "ok"),
            ("ready", True, True, 1234, "busy", "ok"),
            ("starting", False, True, 1234, "starting", "error"),
            ("stopped", False, False, None, "down", "error"),
            ("unavailable", False, False, None, "down", "error"),
            ("unavailable", False, True, None, "down", "error"),
        )
        for status, ready, busy, pid, expected_state, expected_health in cases:
            runtime = SimpleNamespace(
                status=status, ready=ready, busy=busy, worker_pid=pid
            )
            with (
                self.subTest(status=status, busy=busy),
                patch(
                    "app.modules.health.service.settings", health_settings()
                ),
                patch("app.modules.health.service.russian_runtime", runtime),
                patch("httpx.AsyncClient") as http_client,
            ):
                result = await probe_russian_runtime()
            http_client.assert_not_called()
            self.assertEqual(result["status"], expected_health)
            self.assertEqual(
                result["details"],
                {
                    "engine": "russian-docs-ocr",
                    "status": expected_state,
                    "ready": ready,
                    "busy": busy,
                    "worker_pid": pid,
                    "device": "cpu",
                    "model_format": "ONNX",
                },
            )

    async def test_starting_russian_worker_degrades_component_while_busy(
        self,
    ) -> None:
        runtime = SimpleNamespace(
            status="starting", ready=False, busy=True, worker_pid=1234
        )
        with (
            patch("app.modules.health.service.settings", health_settings()),
            patch(
                "app.modules.health.service.get_app_version",
                return_value="0.1.0",
            ),
            patch(
                "app.modules.health.service.probe_database",
                AsyncMock(return_value=healthy_component(1)),
            ),
            patch(
                "app.modules.health.service.probe_queue",
                AsyncMock(return_value=queue_component()),
            ),
            patch(
                "app.modules.health.service.probe_paddle_runtime",
                AsyncMock(return_value=healthy_component(1)),
            ),
            patch(
                "app.modules.health.service.probe_mrz_runtime",
                AsyncMock(return_value=healthy_component(1)),
            ),
            patch(
                "app.modules.health.service.probe_vlm_service",
                AsyncMock(return_value=healthy_component(1)),
            ),
            patch(
                "app.modules.health.service.probe_qr_module",
                AsyncMock(return_value=healthy_component(1)),
            ),
            patch("app.modules.health.service.russian_runtime", runtime),
            patch("httpx.AsyncClient") as http_client,
        ):
            result = await get_health_status()
        http_client.assert_not_called()
        self.assertFalse(result["success"])
        self.assertEqual(result["data"]["status"], "degraded")
        self.assertEqual(
            result["data"]["components"]["russian_ocr"]["details"]["status"],
            "starting",
        )
        self.assertEqual(result["error"]["code"], "SERVICE_UNAVAILABLE")
        self.assertIn("russian_ocr", result["error"]["message"])


if __name__ == "__main__":
    unittest.main()
