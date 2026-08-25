from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from app.core.errors import AppError, ErrorCode
from app.modules.vlm.client import recognize_vlm_image

CLIENT = "app.modules.vlm.client"
SETTINGS = SimpleNamespace(
    vlm_ocr_url="http://127.0.0.1:8004/", vlm_ocr_timeout_seconds=300.0
)


class VlmClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_multipart_dispatches_model_and_preserves_response(
        self,
    ):
        result = {
            "items": [
                {
                    "text": "  <div>ПАСПОРТ</div>\n\n**ФАМИЛИЯ**  ",
                    "confidence": None,
                    "box": None,
                }
            ],
            "image_width": 120,
            "image_height": 60,
        }
        for model in ("qwen3-vl-4b", "qwen3-vl-8b"):
            with self.subTest(model=model):

                def respond(request):
                    self.assertEqual(request.method, "POST")
                    self.assertEqual(
                        str(request.url), "http://127.0.0.1:8004/recognize"
                    )
                    self.assertTrue(
                        request.headers["content-type"].startswith(
                            "multipart/form-data;"
                        )
                    )
                    self.assertIn(
                        b'name="image"; filename="scan.png"', request.content
                    )
                    self.assertIn(b"Content-Type: image/png", request.content)
                    self.assertIn(b"fake-image-bytes", request.content)
                    self.assertIn(
                        b'name="model"\r\n\r\n' + model.encode() + b"\r\n",
                        request.content,
                    )
                    return httpx.Response(200, json=result)

                client = httpx.AsyncClient(
                    transport=httpx.MockTransport(respond)
                )
                with (
                    patch(f"{CLIENT}.settings", SETTINGS),
                    patch(
                        f"{CLIENT}.httpx.AsyncClient", return_value=client
                    ) as factory,
                ):
                    actual = await recognize_vlm_image(
                        b"fake-image-bytes",
                        model=model,
                        filename="scan.png",
                        content_type="image/png",
                    )
                self.assertEqual(actual, result)
                self.assertTrue(client.is_closed)
                factory.assert_called_once_with(timeout=300.0, trust_env=False)

    async def test_passport_task_is_sent_only_when_requested(self):
        def respond(request):
            self.assertIn(
                b'name="task"\r\n\r\npassport_fields\r\n',
                request.content,
            )
            return httpx.Response(
                200,
                json={"items": [], "image_width": 120, "image_height": 60},
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        with (
            patch(f"{CLIENT}.settings", SETTINGS),
            patch(f"{CLIENT}.httpx.AsyncClient", return_value=client),
        ):
            await recognize_vlm_image(
                b"image",
                model="qwen3-vl-4b",
                task="passport_fields",
            )

    async def test_backend_http_errors_map_to_public_codes_without_details(
        self,
    ):
        for status, code in (
            (400, ErrorCode.IMAGE_READ_ERROR),
            (502, ErrorCode.OCR_ENGINE_UNAVAILABLE),
            (503, ErrorCode.OCR_ENGINE_UNAVAILABLE),
            (504, ErrorCode.OCR_TIMEOUT),
            (422, ErrorCode.OCR_FAILED),
            (500, ErrorCode.OCR_FAILED),
        ):
            with self.subTest(status=status):
                client = httpx.AsyncClient(
                    transport=httpx.MockTransport(
                        lambda request: httpx.Response(
                            status, json={"detail": "private traceback"}
                        )
                    )
                )
                with (
                    patch(f"{CLIENT}.settings", SETTINGS),
                    patch(f"{CLIENT}.httpx.AsyncClient", return_value=client),
                    self.assertRaises(AppError) as caught,
                ):
                    await recognize_vlm_image(b"image", model="qwen3-vl-4b")
                self.assertEqual(caught.exception.code, code)
                self.assertNotIn("private", caught.exception.message)
                self.assertTrue(client.is_closed)

    async def test_transport_timeout_and_connection_errors_are_distinct(self):
        for error_type, code in (
            (httpx.ReadTimeout, ErrorCode.OCR_TIMEOUT),
            (httpx.ConnectError, ErrorCode.OCR_ENGINE_UNAVAILABLE),
        ):
            with self.subTest(error_type=error_type):

                def fail(request):
                    raise error_type(
                        "private network details", request=request
                    )

                client = httpx.AsyncClient(transport=httpx.MockTransport(fail))
                with (
                    patch(f"{CLIENT}.settings", SETTINGS),
                    patch(f"{CLIENT}.httpx.AsyncClient", return_value=client),
                    self.assertRaises(AppError) as caught,
                ):
                    await recognize_vlm_image(b"image", model="qwen3-vl-4b")
                self.assertEqual(caught.exception.code, code)
                self.assertNotIn("private", caught.exception.message)
                self.assertTrue(client.is_closed)

    async def test_invalid_json_and_non_object_result_are_ocr_failures(self):
        for content in (b"not json", b"[]", b"null"):
            with self.subTest(content=content):
                client = httpx.AsyncClient(
                    transport=httpx.MockTransport(
                        lambda request: httpx.Response(200, content=content)
                    )
                )
                with (
                    patch(f"{CLIENT}.settings", SETTINGS),
                    patch(f"{CLIENT}.httpx.AsyncClient", return_value=client),
                    self.assertRaises(AppError) as caught,
                ):
                    await recognize_vlm_image(b"image", model="qwen3-vl-8b")
                self.assertEqual(caught.exception.code, ErrorCode.OCR_FAILED)
