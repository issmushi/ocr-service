from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.modules.tasks.enums import JobStatus, TaskOperation

JOB_ID = "757c8c4a-e1a5-4ff4-9855-2aa9df95397d"


def sample_mrz_response() -> dict:
    return {
        "success": True,
        "data": {
            "mrz": [
                "P<ARMTESTOV<<TEST".ljust(44, "<"),
                "AA12345678ARM9001158M3001156<<<<<<<<<<<<<<06",
            ],
            "normalization": None,
            "document_type": "international_passport",
            "document_code": "P",
            "issuing_country": "ARM",
            "surname": "TESTOV",
            "name": "TEST",
            "patronymic": None,
            "sex": "МУЖ.",
            "date_of_birth": "15.01.1990",
            "document_series": None,
            "document_number": "AA1234567",
            "nationality": "ARM",
            "issue_date": None,
            "expiry_date": "15.01.2030",
            "department_code": None,
            "personal_number": None,
            "mrz_valid": True,
            "meta": {
                "engine": "mrz-scanner",
                "model": "mrz",
                "duration_seconds": 0.25,
            },
        },
        "error": None,
    }


class RouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_unified_mrz_openapi_documents_route_and_fields(self) -> None:
        schema = app.openapi()
        endpoint = schema["paths"]["/v1/ocr/mrz"]["post"]
        self.assertEqual(endpoint["tags"], ["MRZ Sync"])
        for old in ("/v1/ocr/passport/mrz", "/v1/ocr/idcard/mrz"):
            self.assertNotIn(old, schema["paths"])
        request = endpoint["requestBody"]["content"]["multipart/form-data"][
            "schema"
        ]
        body_name = request["$ref"].rsplit("/", 1)[-1]
        body = schema["components"]["schemas"][body_name]
        self.assertEqual(
            set(body["properties"]),
            {"image", "normalize", "normalization", "fields"},
        )
        self.assertEqual(body["required"], ["image"])
        image = body["properties"]["image"]
        self.assertEqual(image["type"], "string")
        self.assertEqual(image["format"], "binary")
        for flag in ("normalize", "normalization"):
            self.assertEqual(body["properties"][flag]["type"], "boolean")
            self.assertIs(body["properties"][flag]["default"], True)
        data = schema["components"]["schemas"]["UnifiedMrzDataSchema"]
        fields = data["properties"]
        mrz_schema = next(
            value
            for value in fields["mrz"]["anyOf"]
            if value.get("type") == "array"
        )
        self.assertEqual(mrz_schema["minItems"], 2)
        self.assertEqual(mrz_schema["maxItems"], 3)
        types = next(
            value["enum"]
            for value in fields["document_type"]["anyOf"]
            if "enum" in value
        )
        self.assertEqual(
            set(types),
            {
                "internal_passport",
                "international_passport",
                "identity_card",
            },
        )
        self.assertTrue(
            {
                "mrz",
                "normalization",
                "document_type",
                "document_code",
                "issuing_country",
                "surname",
                "name",
                "patronymic",
                "sex",
                "date_of_birth",
                "document_series",
                "document_number",
                "nationality",
                "issue_date",
                "expiry_date",
                "department_code",
                "personal_number",
                "mrz_valid",
                "meta",
            }.issubset(fields)
        )
        for old in ("passport_type", "passport_series", "passport_number"):
            self.assertNotIn(old, fields)
        normalization_ref = next(
            candidate["$ref"]
            for candidate in fields["normalization"]["anyOf"]
            if "$ref" in candidate
        )
        normalization_name = normalization_ref.rsplit("/", 1)[-1]
        normalization_fields = schema["components"]["schemas"][
            normalization_name
        ]["properties"]
        correction_ref = normalization_fields["corrections"]["items"]["$ref"]
        correction_name = correction_ref.rsplit("/", 1)[-1]
        correction_fields = schema["components"]["schemas"][correction_name][
            "properties"
        ]
        self.assertEqual(correction_fields["line"]["enum"], [1, 2, 3])
        meta_name = fields["meta"]["$ref"].rsplit("/", 1)[-1]
        meta = schema["components"]["schemas"][meta_name]["properties"]
        self.assertEqual(meta["engine"]["const"], "mrz-scanner")
        self.assertEqual(meta["model"]["const"], "mrz")

    def test_unified_mrz_forwards_default_flags(self) -> None:
        expected = sample_mrz_response()
        recognize = AsyncMock(return_value=expected)
        with patch(
            "app.modules.v1.mrz_api.router.recognize_mrz_upload", recognize
        ):
            response = self.client.post(
                "/v1/ocr/mrz",
                files={"image": ("passport.jpg", b"fake-image", "image/jpeg")},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), expected)
        recognize.assert_awaited_once()
        self.assertEqual(
            recognize.await_args.kwargs,
            {"normalize": True, "normalization": True, "fields": ["all"]},
        )

    def test_unified_mrz_forwards_disabled_flags(self) -> None:
        expected = sample_mrz_response()
        recognize = AsyncMock(return_value=expected)
        with patch(
            "app.modules.v1.mrz_api.router.recognize_mrz_upload", recognize
        ):
            response = self.client.post(
                "/v1/ocr/mrz",
                data={
                    "normalize": "false",
                    "normalization": "false",
                },
                files={"image": ("passport.jpg", b"fake-image", "image/jpeg")},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), expected)
        self.assertEqual(
            recognize.await_args.kwargs,
            {"normalize": False, "normalization": False, "fields": ["all"]},
        )

    def test_unified_mrz_rejects_invalid_flags_before_ocr(self) -> None:
        for flag in ("normalize", "normalization"):
            with self.subTest(flag=flag):
                recognize = AsyncMock()
                with patch(
                    "app.modules.v1.mrz_api.router.recognize_mrz_upload",
                    recognize,
                ):
                    response = self.client.post(
                        "/v1/ocr/mrz",
                        data={flag: "sometimes"},
                        files={
                            "image": (
                                "passport.jpg",
                                b"fake-image",
                                "image/jpeg",
                            )
                        },
                    )
                self.assertEqual(response.status_code, 422)
                self.assertEqual(
                    response.json()["error"]["code"], "VALIDATION_ERROR"
                )
                recognize.assert_not_awaited()

    def test_old_mrz_ocr_routes_are_removed(self) -> None:
        for path in ("/v1/ocr/passport/mrz", "/v1/ocr/idcard/mrz"):
            with self.subTest(path=path):
                response = self.client.post(
                    path,
                    files={"image": ("scan.jpg", b"fake-image", "image/jpeg")},
                )
                self.assertEqual(response.status_code, 404)

    def test_get_task_status(self) -> None:
        job = {
            "id": JOB_ID,
            "mode": TaskOperation.MRZ.value,
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
        self.assertEqual(body["data"]["operation"], "mrz")

    def test_get_task_status_invalid_uuid_error(self) -> None:
        response = self.client.get("/v1/tasks/not-a-uuid")

        self.assertEqual(response.status_code, 422)
        body = response.json()
        self.assertFalse(body["success"])
        self.assertEqual(body["error"]["code"], "VALIDATION_ERROR")

    def test_cancel_task(self) -> None:
        job = {
            "id": JOB_ID,
            "mode": TaskOperation.MRZ.value,
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
