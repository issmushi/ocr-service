from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi import Response
from fastapi.testclient import TestClient

from app.main import app
from app.modules.tasks.enums import JobStatus, TaskOperation
from app.modules.v1.passport.parsing import empty_passport_mrz_data
from app.modules.v1.passport.schemas import PassportOcrModel

JOB_ID = "757c8c4a-e1a5-4ff4-9855-2aa9df95397d"


class RouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_qr_image_batch_is_documented_as_binary_files(self) -> None:
        schema = app.openapi()

        request_schema = schema["paths"]["/qr/decode"]["post"]["requestBody"][
            "content"
        ]["multipart/form-data"]["schema"]
        component_name = request_schema["$ref"].rsplit("/", 1)[-1]
        image_schema = schema["components"]["schemas"][component_name][
            "properties"
        ]["image"]
        array_schema = next(
            candidate
            for candidate in image_schema["anyOf"]
            if candidate.get("type") == "array"
        )

        self.assertEqual(
            array_schema["items"],
            {"type": "string", "format": "binary"},
        )

    def test_openapi_contains_only_the_documented_synthetic_passport(
        self,
    ) -> None:
        schema = app.openapi()
        synthetic_passport = {
            "surname": "ТЕСТОВ",
            "name": "ТЕСТ",
            "patronymic": "ТЕСТОВИЧ",
            "sex": "МУЖ.",
            "date_of_birth": "15.01.1990",
            "passport_series": "45 15",
            "passport_number": "123456",
        }
        request_properties = schema["components"]["schemas"][
            "QRGenerateRequest"
        ]["properties"]

        for field, value in synthetic_passport.items():
            self.assertEqual(request_properties[field]["examples"], [value])

        response_example = schema["paths"]["/qr/decode"]["post"]["responses"][
            "200"
        ]["content"]["application/json"]["examples"]["decoded"]["value"]
        self.assertEqual(
            response_example["data"]["passport"], synthetic_passport
        )

    def test_v1_passport_documents_binary_file_and_model_selection(
        self,
    ) -> None:
        schema = app.openapi()
        request_schema = schema["paths"]["/v1/ocr/passport/mrz"]["post"][
            "requestBody"
        ]["content"]["multipart/form-data"]["schema"]
        component_name = request_schema["$ref"].rsplit("/", 1)[-1]
        body_schema = schema["components"]["schemas"][component_name]
        image_schema = body_schema["properties"]["image"]
        model_schema = body_schema["properties"]["model"]
        crop_schema = body_schema["properties"]["crop"]
        normalization_schema = body_schema["properties"]["normalization"]
        model_component = model_schema["$ref"].rsplit("/", 1)[-1]
        model_enum = schema["components"]["schemas"][model_component]

        self.assertEqual(image_schema["type"], "string")
        self.assertEqual(image_schema["format"], "binary")
        self.assertIn("image", body_schema["required"])
        self.assertNotIn("model", body_schema["required"])
        self.assertNotIn("crop", body_schema["required"])
        self.assertNotIn("normalization", body_schema["required"])
        self.assertEqual(model_schema["default"], "paddle")
        self.assertEqual(crop_schema["type"], "boolean")
        self.assertIs(crop_schema["default"], True)
        self.assertEqual(normalization_schema["type"], "boolean")
        self.assertIs(normalization_schema["default"], True)
        self.assertEqual(model_enum["type"], "string")
        self.assertEqual(
            model_enum["enum"],
            ["paddle", "mrz", "qwen3-vl-4b", "qwen3-vl-8b"],
        )

        response_schema = schema["components"]["schemas"][
            "PassportMrzDataSchema"
        ]
        self.assertTrue(
            {
                "mrz",
                "normalization",
                "passport_type",
                "document_code",
                "issuing_country",
                "surname",
                "name",
                "patronymic",
                "sex",
                "date_of_birth",
                "passport_series",
                "passport_number",
                "nationality",
                "issue_date",
                "expiry_date",
                "department_code",
                "personal_number",
                "mrz_valid",
                "meta",
            }.issubset(response_schema["properties"])
        )
        meta_schema = schema["components"]["schemas"]["PassportOcrMetaSchema"]
        self.assertEqual(
            set(meta_schema["properties"]),
            {"engine", "model", "duration_seconds"},
        )

    def test_recognize_v1_passport_raw_mrz(self) -> None:
        expected = {
            "success": True,
            "data": {
                "mrz": [
                    "P<UTOTESTOV<<TEST<<<<<<<<<<<<<<<<<<<<<<<<<<<",
                    "A123456780UTO9001011M3001019<<<<<<<<<<<<<<",
                ],
                "normalization": None,
                **empty_passport_mrz_data(),
                "meta": {
                    "engine": "paddleocr",
                    "model": "paddle",
                    "duration_seconds": 0.25,
                },
            },
            "error": None,
        }
        recognize_passport_upload = AsyncMock(return_value=expected)
        with patch(
            "app.modules.v1.passport.router.recognize_passport_upload",
            new=recognize_passport_upload,
        ):
            response = self.client.post(
                "/v1/ocr/passport/mrz",
                files={
                    "image": (
                        "passport.jpg",
                        b"fake-image",
                        "image/jpeg",
                    )
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected)
        recognize_passport_upload.assert_awaited_once()
        self.assertEqual(
            recognize_passport_upload.await_args.kwargs,
            {
                "model": PassportOcrModel.PADDLE,
                "crop": True,
                "normalization": True,
            },
        )

    def test_v1_passport_forwards_disabled_crop(self) -> None:
        expected = {
            "success": True,
            "data": {
                "mrz": ["line-1", "line-2"],
                "normalization": None,
                **empty_passport_mrz_data(),
                "meta": {
                    "engine": "qwen3-vl",
                    "model": "qwen3-vl-8b",
                    "duration_seconds": 0.5,
                },
            },
            "error": None,
        }
        recognize_passport_upload = AsyncMock(return_value=expected)
        with patch(
            "app.modules.v1.passport.router.recognize_passport_upload",
            new=recognize_passport_upload,
        ):
            response = self.client.post(
                "/v1/ocr/passport/mrz",
                data={
                    "model": "qwen3-vl-8b",
                    "crop": "false",
                    "normalization": "false",
                },
                files={"image": ("passport.jpg", b"fake-image", "image/jpeg")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected)
        self.assertEqual(
            recognize_passport_upload.await_args.kwargs,
            {
                "model": PassportOcrModel.QWEN3_VL_8B,
                "crop": False,
                "normalization": False,
            },
        )

    def test_v1_passport_rejects_invalid_crop_flag(self) -> None:
        recognize_passport_upload = AsyncMock()
        with patch(
            "app.modules.v1.passport.router.recognize_passport_upload",
            new=recognize_passport_upload,
        ):
            response = self.client.post(
                "/v1/ocr/passport/mrz",
                data={"crop": "sometimes"},
                files={"image": ("passport.jpg", b"fake-image", "image/jpeg")},
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "VALIDATION_ERROR")
        recognize_passport_upload.assert_not_awaited()

    def test_v1_passport_rejects_invalid_normalization_flag(self) -> None:
        recognize_passport_upload = AsyncMock()
        with patch(
            "app.modules.v1.passport.router.recognize_passport_upload",
            new=recognize_passport_upload,
        ):
            response = self.client.post(
                "/v1/ocr/passport/mrz",
                data={"normalization": "sometimes"},
                files={"image": ("passport.jpg", b"fake-image", "image/jpeg")},
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "VALIDATION_ERROR")
        recognize_passport_upload.assert_not_awaited()

    def test_get_task_status(self) -> None:
        job = {
            "id": JOB_ID,
            "mode": TaskOperation.PASSPORT_MRZ.value,
            "status": JobStatus.QUEUED.value,
            "filename": "document.jpg",
            "content_type": "image/jpeg",
            "image_path": "/tmp/document.jpg",
            "result": None,
            "error": None,
            "attempts": 0,
            "max_attempts": 2,
            "created_at": None,
            "updated_at": None,
            "started_at": None,
            "finished_at": None,
        }
        with patch(
            "app.modules.tasks.router.get_job",
            new=AsyncMock(return_value=job),
        ):
            response = self.client.get(f"/v1/tasks/{JOB_ID}")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["data"]["task_id"], JOB_ID)
        self.assertEqual(body["data"]["status"], "queued")
        self.assertEqual(body["data"]["operation"], "passport_mrz")

    def test_get_task_status_invalid_uuid_error(self) -> None:
        response = self.client.get("/v1/tasks/not-a-uuid")

        self.assertEqual(response.status_code, 422)
        body = response.json()
        self.assertFalse(body["success"])
        self.assertEqual(body["error"]["code"], "VALIDATION_ERROR")

    def test_cancel_task(self) -> None:
        job = {
            "id": JOB_ID,
            "mode": TaskOperation.PASSPORT_MRZ.value,
            "status": JobStatus.QUEUED.value,
            "image_path": "/tmp/document.jpg",
        }
        mark_cancelled_mock = AsyncMock(return_value=True)
        with (
            patch(
                "app.modules.tasks.router.get_job",
                new=AsyncMock(
                    side_effect=[job, {**job, "status": "cancelled"}]
                ),
            ),
            patch(
                "app.modules.tasks.router.mark_cancelled",
                new=mark_cancelled_mock,
            ),
            patch("app.modules.tasks.router.delete_upload"),
        ):
            response = self.client.post(f"/v1/tasks/{JOB_ID}/cancel")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        task = body["data"]
        self.assertEqual(task["task_id"], JOB_ID)
        self.assertEqual(task["status"], "cancelled")
        self.assertTrue(task["cancelled"])
        mark_cancelled_mock.assert_awaited_once()
        self.assertEqual(mark_cancelled_mock.await_args.args[1], JOB_ID)
        self.assertIsInstance(mark_cancelled_mock.await_args.args[1], str)

    def test_generate_qr(self) -> None:
        with patch(
            "app.modules.qr.router.generate_qr",
            new=AsyncMock(
                return_value=Response(content=b"png", media_type="image/png")
            ),
        ):
            response = self.client.post(
                "/qr/generate",
                json={
                    "surname": "ТЕСТОВ",
                    "name": "ТЕСТ",
                    "patronymic": "ТЕСТОВИЧ",
                    "sex": "МУЖ.",
                    "date_of_birth": "15.01.1990",
                    "passport_series": "45 15",
                    "passport_number": "123456",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertEqual(response.content, b"png")

    def test_generate_qr_rejects_oversized_text_field(self) -> None:
        response = self.client.post(
            "/qr/generate",
            json={
                "surname": "И" * 101,
                "name": "ТЕСТ",
                "patronymic": "ТЕСТОВИЧ",
                "sex": "МУЖ.",
                "date_of_birth": "15.01.1990",
                "passport_series": "45 15",
                "passport_number": "123456",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "EMPTY_PAYLOAD")

    def test_decode_qr(self) -> None:
        expected = {
            "success": True,
            "data": {
                "passport": {
                    "surname": "ТЕСТОВ",
                    "name": "ТЕСТ",
                    "patronymic": "ТЕСТОВИЧ",
                    "sex": "МУЖ.",
                    "date_of_birth": "15.01.1990",
                    "passport_series": "45 15",
                    "passport_number": "123456",
                },
                "meta": {
                    "format": "passport",
                    "version": 2,
                    "encrypted": True,
                    "algorithm": "A256GCM",
                    "key_id": "primary",
                    "qr_count": 1,
                },
                "filename": "qr.png",
                "content_type": "image/png",
            },
            "error": None,
        }
        decode_qr = AsyncMock(return_value=expected)
        with patch(
            "app.modules.qr.router.decode_qr",
            new=decode_qr,
        ):
            response = self.client.post(
                "/qr/decode",
                files=[
                    (
                        "image",
                        (
                            f"qr-{index}.png",
                            f"fake-image-{index}".encode(),
                            "image/png",
                        ),
                    )
                    for index in range(1, 4)
                ],
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected)
        images = decode_qr.await_args.args[0]
        self.assertEqual(len(images), 3)

    def test_decode_qr_missing_image_error(self) -> None:
        response = self.client.post("/qr/decode")

        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertFalse(body["success"])
        self.assertEqual(body["error"]["code"], "NO_IMAGE")

    def test_decode_qr_rejects_more_than_three_images(self) -> None:
        response = self.client.post(
            "/qr/decode",
            files=[
                (
                    "image",
                    (
                        f"qr-{index}.png",
                        b"fake-image",
                        "image/png",
                    ),
                )
                for index in range(4)
            ],
        )

        self.assertEqual(response.status_code, 422)
        body = response.json()
        self.assertFalse(body["success"])
        self.assertEqual(body["error"]["code"], "VALIDATION_ERROR")

    def test_health_returns_503_for_degraded_stack(self) -> None:
        expected = {
            "success": False,
            "data": {
                "status": "degraded",
                "service": "ocr-service",
                "version": "0.1.0",
                "queue": "error",
                "workers": 1,
                "active_workers": 0,
                "components": {
                    name: {
                        "status": "error",
                        "latency_ms": 1,
                        "details": {},
                        "error": "Недоступен",
                    }
                    for name in (
                        "api",
                        "postgres",
                        "rabbitmq",
                        "worker",
                        "paddle_ocr",
                        "russian_ocr",
                        "mrz_scanner",
                        "qwen_vlm_service",
                        "qr",
                    )
                },
            },
            "error": {
                "code": "SERVICE_UNAVAILABLE",
                "message": "Недоступны компоненты",
            },
        }
        with patch(
            "app.modules.health.router.get_health_status",
            new=AsyncMock(return_value=expected),
        ):
            response = self.client.get("/health")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), expected)

    def test_health_openapi_documents_healthy_and_degraded_responses(
        self,
    ) -> None:
        responses = app.openapi()["paths"]["/health"]["get"]["responses"]

        self.assertIn("200", responses)
        self.assertIn("503", responses)
