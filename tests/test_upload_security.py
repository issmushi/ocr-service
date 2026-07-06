from __future__ import annotations

import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from app.bootstrap.http import validation_error_summary
from app.core.errors import (
    AppError,
    ErrorCode,
    api_error_response,
    decode_error,
)
from app.core.http.body_limit import RequestBodyLimitMiddleware
from app.core.images.uploads import prepare_upload_stream
from app.core.images.validation import validate_image_dimensions
from app.core.infrastructure.storage import (
    UploadSizeExceeded,
    copy_upload_to_path,
)


class UploadStorageTests(unittest.TestCase):
    def test_copy_upload_writes_in_chunks(self) -> None:
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "image.jpg"

            size = copy_upload_to_path(
                BytesIO(b"123456"),
                destination,
                max_bytes=6,
                chunk_size=2,
            )

            self.assertEqual(size, 6)
            self.assertEqual(destination.read_bytes(), b"123456")

    def test_copy_upload_removes_partial_file_over_limit(self) -> None:
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "image.jpg"

            with self.assertRaises(UploadSizeExceeded):
                copy_upload_to_path(
                    BytesIO(b"123456"),
                    destination,
                    max_bytes=5,
                    chunk_size=2,
                )

            self.assertFalse(destination.exists())


class ImageDimensionTests(unittest.TestCase):
    def test_rejects_excessive_width(self) -> None:
        with self.assertRaises(AppError) as context:
            validate_image_dimensions(
                101,
                10,
                max_width=100,
                max_height=100,
                max_pixels=10_000,
            )

        self.assertEqual(context.exception.code, ErrorCode.IMAGE_TOO_LARGE)

    def test_rejects_excessive_pixel_count(self) -> None:
        with self.assertRaises(AppError) as context:
            validate_image_dimensions(
                100,
                100,
                max_width=100,
                max_height=100,
                max_pixels=9_999,
            )

        self.assertEqual(context.exception.code, ErrorCode.IMAGE_TOO_LARGE)


class RequestBodyLimitTests(unittest.TestCase):
    def create_client(self) -> TestClient:
        app = FastAPI()
        app.add_middleware(
            RequestBodyLimitMiddleware,
            max_body_size=10,
            path_limits={"/upload": 10, "/small": 4},
        )

        @app.post("/{name}")
        async def read_body(name: str, request: Request) -> dict[str, object]:
            return {"name": name, "size": len(await request.body())}

        return TestClient(app)

    def test_rejects_content_length_before_endpoint(self) -> None:
        response = self.create_client().post("/upload", content=b"12345678901")

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error"]["code"], "IMAGE_TOO_LARGE")

    def test_applies_stricter_path_limit(self) -> None:
        response = self.create_client().post("/small", content=b"12345")

        self.assertEqual(response.status_code, 413)

    def test_allows_body_within_limit(self) -> None:
        response = self.create_client().post("/upload", content=b"12345")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["size"], 5)


class ChunkedRequestBodyLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_chunked_body_without_content_length(self) -> None:
        messages = iter(
            [
                {
                    "type": "http.request",
                    "body": b"123",
                    "more_body": True,
                },
                {
                    "type": "http.request",
                    "body": b"456",
                    "more_body": False,
                },
            ]
        )
        sent: list[dict[str, object]] = []

        async def receive() -> dict[str, object]:
            return next(messages)

        async def send(message: dict[str, object]) -> None:
            sent.append(message)

        async def consume_body(scope, receive, send) -> None:
            while True:
                message = await receive()
                if not message.get("more_body"):
                    break

        middleware = RequestBodyLimitMiddleware(
            consume_body,
            max_body_size=5,
            path_limits={"/upload": 5},
        )
        await middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/upload",
                "headers": [],
            },
            receive,
            send,
        )

        response_start = sent[0]
        self.assertEqual(response_start["status"], 413)


class UploadStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_prepare_upload_stream_rejects_oversized_file(self) -> None:
        upload = UploadFile(filename="qr.png", file=BytesIO(b"123456"))

        with self.assertRaises(AppError) as context:
            await prepare_upload_stream(upload, max_bytes=5)

        self.assertEqual(context.exception.code, ErrorCode.IMAGE_TOO_LARGE)

    async def test_prepare_upload_stream_rewinds_file(self) -> None:
        upload = UploadFile(filename="qr.png", file=BytesIO(b"12345"))
        upload.file.seek(3)

        size = await prepare_upload_stream(upload, max_bytes=5)

        self.assertEqual(size, 5)
        self.assertEqual(upload.file.tell(), 0)


class PublicErrorTests(unittest.TestCase):
    def test_internal_error_payload_hides_exception_text(self) -> None:
        response = api_error_response(
            ErrorCode.INTERNAL_ERROR,
            "database password=secret",
        )

        self.assertEqual(response["error"]["message"], "Непредвиденная ошибка")
        self.assertNotIn("secret", str(response))

    def test_legacy_plain_error_is_not_exposed(self) -> None:
        error = decode_error("/private/path: connection failed")

        self.assertEqual(error["code"], "INTERNAL_ERROR")
        self.assertEqual(error["message"], "Непредвиденная ошибка")

    def test_validation_log_summary_omits_input_value(self) -> None:
        error = RequestValidationError(
            [
                {
                    "type": "string_too_long",
                    "loc": ("body", "passport_number"),
                    "msg": "String should have at most 10 characters",
                    "input": "secret-passport-number",
                }
            ]
        )

        details = validation_error_summary(error)

        self.assertNotIn("input", details[0])
        self.assertNotIn("secret-passport-number", str(details))


if __name__ == "__main__":
    unittest.main()
