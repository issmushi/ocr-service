from __future__ import annotations

import asyncio
import base64
import unittest
from unittest.mock import AsyncMock, patch

from app.core.errors import AppError, ErrorCode
from app.modules.tasks.enums import TaskOperation
from app.modules.tasks.execution import execute_task_operation
from app.modules.v1.full.schemas import FullOcrModel
from app.modules.v1.idcard.schemas import IdcardOcrModel
from app.modules.v1.passport.schemas import (
    PassportOcrModel,
    PassportPage,
    PassportPageOcrModel,
)

EXECUTION = "app.modules.tasks.execution"


def job(operation: TaskOperation, options: dict | None = None) -> dict:
    return {
        "mode": operation.value,
        "filename": "document.png",
        "content_type": "image/png",
        "options": options or {},
    }


class TaskExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_options_are_rehydrated_and_upload_metadata_is_preserved(
        self,
    ) -> None:
        cases = (
            (
                TaskOperation.PASSPORT_MRZ,
                "recognize_passport_upload",
                {
                    "model": "mrz",
                    "crop": False,
                    "normalize": False,
                    "normalization": True,
                },
                {
                    "model": PassportOcrModel.MRZ,
                    "crop": False,
                    "normalize": False,
                    "normalization": True,
                },
            ),
            (
                TaskOperation.PASSPORT_PAGE,
                "recognize_passport_page_upload",
                {"page": 3, "model": "russian-ocr", "crop": False},
                {
                    "page": PassportPage.PAGE_3,
                    "model": PassportPageOcrModel.RUSSIAN_OCR,
                    "crop": False,
                },
            ),
            (
                TaskOperation.IDCARD_MRZ,
                "recognize_idcard_upload",
                {
                    "model": "qwen3-vl-8b",
                    "crop": True,
                    "normalize": True,
                    "normalization": False,
                },
                {
                    "model": IdcardOcrModel.QWEN3_VL_8B,
                    "crop": True,
                    "normalize": True,
                    "normalization": False,
                },
            ),
            (
                TaskOperation.FULL_V1,
                "recognize_full_upload",
                {"model": "qwen3-vl-4b"},
                {"model": FullOcrModel.QWEN3_VL_4B},
            ),
        )
        response = {"success": True, "data": None, "error": None}
        for operation, handler_name, options, expected in cases:
            with self.subTest(operation=operation):
                received = []

                async def recognize(upload, **kwargs):
                    received.append(upload)
                    self.assertEqual(upload.filename, "document.png")
                    self.assertEqual(upload.content_type, "image/png")
                    self.assertEqual(upload.size, len(b"original-image"))
                    self.assertEqual(await upload.read(), b"original-image")
                    self.assertEqual(kwargs, expected)
                    self.assertIsInstance(
                        kwargs["model"], type(expected["model"])
                    )
                    return response

                handler = AsyncMock(side_effect=recognize)
                original_options = dict(options)
                with patch(f"{EXECUTION}.{handler_name}", handler):
                    result = await execute_task_operation(
                        job(operation, options), b"original-image"
                    )
                self.assertEqual(
                    result, {"format": "json", "response": response}
                )
                self.assertTrue(received[0].file.closed)
                self.assertEqual(options, original_options)

    async def test_json_payload_has_the_same_schema_serialization_as_http(
        self,
    ) -> None:
        response = {
            "success": True,
            "data": {
                "text": "Текст\n123",
                "meta": {
                    "engine": "qwen3-vl",
                    "model": FullOcrModel.QWEN3_VL_4B,
                    "duration_seconds": "2.5",
                },
                "extra": "must not leak",
            },
            "error": None,
            "debug": "must not leak",
        }
        with patch(
            f"{EXECUTION}.recognize_full_upload",
            AsyncMock(return_value=response),
        ):
            result = await execute_task_operation(
                job(TaskOperation.FULL_V1), b"image"
            )
        self.assertEqual(
            result["response"],
            {
                "success": True,
                "data": {
                    "text": "Текст\n123",
                    "meta": {
                        "engine": "qwen3-vl",
                        "model": "qwen3-vl-4b",
                        "duration_seconds": 2.5,
                    },
                },
                "error": None,
            },
        )

    async def test_crop_outputs_are_persisted_as_exact_jpeg_bytes(
        self,
    ) -> None:
        jpeg = b"\xff\xd8\xffjpeg-result\xff\xd9"
        for operation, name in (
            (TaskOperation.CROP_DOCUMENT, "crop_document_image_bytes"),
            (TaskOperation.CROP_MRZ, "crop_mrz_bytes"),
        ):
            with self.subTest(operation=operation):
                handler = AsyncMock(return_value=jpeg)
                with (
                    patch(f"{EXECUTION}.{name}", handler),
                    patch(
                        f"{EXECUTION}.UploadFile",
                        side_effect=AssertionError("Crop uses bytes directly"),
                    ),
                ):
                    result = await execute_task_operation(
                        job(operation), b"source"
                    )
                handler.assert_awaited_once_with(b"source")
                self.assertEqual(result["format"], "jpeg")
                self.assertEqual(
                    base64.b64decode(result["content_base64"], validate=True),
                    jpeg,
                )

    async def test_upload_closes_on_service_error(self) -> None:
        received = []

        async def fail(upload, **_kwargs):
            received.append(upload)
            raise AppError(ErrorCode.OCR_TIMEOUT, "timeout")

        with patch(f"{EXECUTION}.recognize_full_upload", fail):
            with self.assertRaises(AppError) as error:
                await execute_task_operation(
                    job(TaskOperation.FULL_V1), b"source"
                )
        self.assertEqual(error.exception.code, ErrorCode.OCR_TIMEOUT)
        self.assertTrue(received[0].file.closed)

    async def test_upload_closes_when_execution_is_cancelled(self) -> None:
        received = []
        started = asyncio.Event()

        async def recognize(upload, **_kwargs):
            received.append(upload)
            started.set()
            await asyncio.Event().wait()

        with patch(f"{EXECUTION}.recognize_full_upload", recognize):
            task = asyncio.create_task(
                execute_task_operation(job(TaskOperation.FULL_V1), b"source")
            )
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(received[0].file.closed)

    async def test_unknown_operation_does_not_fall_back_to_full_ocr(
        self,
    ) -> None:
        with patch(
            f"{EXECUTION}.recognize_full_upload", new=AsyncMock()
        ) as handler:
            with self.assertRaises(ValueError):
                await execute_task_operation({"mode": "unknown"}, b"image")
        handler.assert_not_awaited()
