from __future__ import annotations

import asyncio
import base64
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient
from PIL import Image

from app.core.errors import AppError, ErrorCode, encode_error
from app.main import app
from app.modules.tasks.creation import create_image_task
from app.modules.tasks.enums import TaskOperation
from app.modules.tasks.exceptions import QueueCapacityExceeded

TASK_ID = "757c8c4a-e1a5-4ff4-9855-2aa9df95397d"
ROUTER = "app.modules.tasks.router"
CREATION = "app.modules.tasks.creation"
CASES = [
    (
        "mrz",
        "mrz",
        {},
        {
            "crop": True,
            "normalize": True,
            "normalization": True,
        },
    ),
    (
        "passport/internal",
        "passport_page",
        {},
        {},
    ),
    (
        "passport/international",
        "passport_international",
        {},
        {},
    ),
    ("full", "full_v1", {}, {"model": "paddle"}),
    ("crop/document", "crop_document", {}, {}),
    ("crop/mrz", "crop_mrz", {}, {}),
]


@pytest.fixture
def client():
    client = TestClient(app)
    yield client
    client.close()


@pytest.fixture
def image_content():
    output = BytesIO()
    Image.new("RGB", (64, 48), "white").save(output, format="PNG")
    return output.getvalue()


@pytest.fixture
def persistence(tmp_path):
    with (
        patch(
            "app.core.infrastructure.storage.settings",
            SimpleNamespace(storage_path=tmp_path),
        ),
        patch(f"{CREATION}.create_job", new_callable=AsyncMock) as create,
    ):
        yield create, tmp_path


def task(**values):
    return {
        "id": TASK_ID,
        "mode": "full_v1",
        "status": "queued",
        "filename": "scan.png",
        "content_type": "image/png",
        "image_path": "/tmp/task.png",
        "result": None,
        "error": None,
        "options": {"model": "paddle"},
        "created_at": datetime(2026, 9, 25, 12, 0, tzinfo=UTC),
        "started_at": None,
        "finished_at": None,
        **values,
    }


def post_task(client, path, content, form=None):
    return client.post(
        f"/v1/tasks/{path}",
        files={"image": ("scan.png", content, "image/png")},
        data=form or {},
    )


@pytest.mark.parametrize(("path", "operation", "form", "options"), CASES)
def test_create_each_operation_persists_one_image_and_exact_sync_defaults(
    client,
    image_content,
    persistence,
    path,
    operation,
    form,
    options,
):
    create, directory = persistence
    response = post_task(client, path, image_content, form)

    assert response.status_code == 202, response.text
    payload = response.json()
    task_id = payload["data"]["task_id"]
    UUID(task_id)
    assert payload == {
        "success": True,
        "data": {"task_id": task_id, "status": "queued"},
        "error": None,
    }
    create.assert_awaited_once()
    kwargs = create.await_args.kwargs
    assert kwargs == {
        "job_id": task_id,
        "mode": operation,
        "options": options,
        "filename": "scan.png",
        "content_type": "image/png",
        "image_path": str(directory / f"{task_id}.png"),
    }
    assert Path(kwargs["image_path"]).read_bytes() == image_content
    assert len(list(directory.iterdir())) == 1


def test_mrz_custom_flags_are_saved_as_booleans(
    client, image_content, persistence
):
    create, _ = persistence
    response = post_task(
        client,
        "mrz",
        image_content,
        {
            "crop": "false",
            "normalize": "false",
            "normalization": "false",
        },
    )

    assert response.status_code == 202
    assert create.await_args.kwargs["options"] == {
        "crop": False,
        "normalize": False,
        "normalization": False,
    }


def test_mrz_task_schema_has_no_model_choice(client):
    schema = client.get("/openapi.json").json()
    request = schema["paths"]["/v1/tasks/mrz"]["post"]["requestBody"]
    body = request["content"]["multipart/form-data"]["schema"]
    fields = schema["components"]["schemas"][body["$ref"].split("/")[-1]][
        "properties"
    ]

    assert set(fields) == {"image", "crop", "normalize", "normalization"}


def test_obsolete_mrz_model_form_is_not_persisted(
    client, image_content, persistence
):
    create, _ = persistence
    response = post_task(
        client, "mrz", image_content, {"model": "qwen3-vl-8b"}
    )

    assert response.status_code == 202
    assert create.await_args.kwargs["options"] == {
        "crop": True,
        "normalize": True,
        "normalization": True,
    }


def test_full_custom_model(client, image_content, persistence):
    create, _ = persistence
    response = post_task(
        client, "full", image_content, {"model": "qwen3-vl-8b"}
    )
    assert response.status_code == 202
    assert create.await_args.kwargs["options"] == {"model": "qwen3-vl-8b"}


def test_internal_passport_task_exposes_only_image_and_ignores_old_options(
    client, image_content, persistence
):
    schema = client.get("/openapi.json").json()
    request = schema["paths"]["/v1/tasks/passport/internal"]["post"][
        "requestBody"
    ]
    body = request["content"]["multipart/form-data"]["schema"]
    fields = schema["components"]["schemas"][body["$ref"].split("/")[-1]][
        "properties"
    ]
    assert set(fields) == {"image"}

    create, _ = persistence
    response = post_task(
        client,
        "passport/internal",
        image_content,
        {"page": "3", "model": "qwen3-vl-8b", "crop": "false"},
    )
    assert response.status_code == 202
    assert create.await_args.kwargs["options"] == {}


def test_international_passport_task_exposes_only_image(
    client, image_content, persistence
):
    schema = client.get("/openapi.json").json()
    request = schema["paths"]["/v1/tasks/passport/international"]["post"][
        "requestBody"
    ]
    body = request["content"]["multipart/form-data"]["schema"]
    fields = schema["components"]["schemas"][body["$ref"].split("/")[-1]][
        "properties"
    ]
    assert set(fields) == {"image"}

    create, _ = persistence
    response = post_task(
        client,
        "passport/international",
        image_content,
        {"model": "qwen3-vl-8b", "crop": "false"},
    )
    assert response.status_code == 202
    assert create.await_args.kwargs["options"] == {}


def test_old_passport_page_task_route_is_removed(
    client, image_content, persistence
):
    create, directory = persistence
    response = post_task(client, "passport/page", image_content)
    assert response.status_code == 404
    create.assert_not_awaited()
    assert list(directory.iterdir()) == []


@pytest.mark.parametrize("path", ["passport/mrz", "idcard/mrz"])
def test_old_mrz_task_routes_are_removed(
    client, image_content, persistence, path
):
    create, directory = persistence
    response = post_task(client, path, image_content)
    assert response.status_code == 404
    create.assert_not_awaited()
    assert list(directory.iterdir()) == []


@pytest.mark.parametrize(
    ("path", "form"),
    [
        ("mrz", {"crop": "sometimes"}),
        ("mrz", {"normalize": "sometimes"}),
        ("full", {"model": "mrz"}),
    ],
)
def test_invalid_options_do_not_store_or_enqueue(
    client, image_content, persistence, path, form
):
    create, directory = persistence
    response = post_task(client, path, image_content, form)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    create.assert_not_awaited()
    assert list(directory.iterdir()) == []


@pytest.mark.parametrize(
    ("content", "code"),
    [(b"", "EMPTY_IMAGE"), (b"not an image", "IMAGE_READ_ERROR")],
)
def test_invalid_image_leaves_no_upload_or_task(
    client, persistence, content, code
):
    create, directory = persistence
    response = post_task(client, "full", content)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == code
    create.assert_not_awaited()
    assert list(directory.iterdir()) == []


def test_queue_full_cleans_upload_and_returns_429(
    client, image_content, persistence
):
    create, directory = persistence
    create.side_effect = QueueCapacityExceeded(5, 5)
    response = post_task(client, "full", image_content)
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "QUEUE_FULL"
    assert list(directory.iterdir()) == []


@pytest.mark.parametrize(
    "failure",
    [None, RuntimeError("database down"), QueueCapacityExceeded(5, 5)],
)
def test_creator_closes_upload_on_success_and_database_errors(
    image_content, persistence, failure
):
    create, directory = persistence
    create.side_effect = failure
    image = UploadFile(file=BytesIO(image_content), filename="scan.png")
    if failure is None:
        asyncio.run(create_image_task(image, operation=TaskOperation.FULL_V1))
        assert len(list(directory.iterdir())) == 1
    else:
        error_class = (
            AppError
            if isinstance(failure, QueueCapacityExceeded)
            else RuntimeError
        )
        with pytest.raises(error_class):
            asyncio.run(
                create_image_task(image, operation=TaskOperation.FULL_V1)
            )
        assert list(directory.iterdir()) == []
    assert image.file.closed


def test_creator_closes_invalid_upload(persistence):
    image = UploadFile(file=BytesIO(b"invalid"), filename="scan.png")
    with pytest.raises(AppError):
        asyncio.run(create_image_task(image, operation=TaskOperation.FULL_V1))
    assert image.file.closed


@pytest.mark.parametrize(
    "status",
    ["queued", "processing", "done", "cancelled", "failed", "timeout"],
)
def test_status_query_succeeds_independently_of_task_outcome(client, status):
    stored = (
        encode_error(ErrorCode.OCR_FAILED, "Не найден текст")
        if status == "failed"
        else None
    )
    with patch(
        f"{ROUTER}.get_job",
        AsyncMock(return_value=task(status=status, error=stored)),
    ):
        response = client.get(f"/v1/tasks/{TASK_ID}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["error"] is None
    assert payload["data"]["task_id"] == TASK_ID
    assert payload["data"]["status"] == status
    assert payload["data"]["operation"] == "full_v1"
    assert payload["data"]["created_at"] == "2026-09-25T12:00:00Z"
    assert payload["data"]["started_at"] is None
    assert payload["data"]["finished_at"] is None
    assert set(payload["data"]) == {
        "task_id",
        "status",
        "operation",
        "created_at",
        "started_at",
        "finished_at",
        "error",
    }
    if status == "failed":
        assert payload["data"]["error"] == {
            "code": "OCR_FAILED",
            "message": "Не найден текст",
        }
    elif status == "timeout":
        assert payload["data"]["error"]["code"] == "OCR_TIMEOUT"
    else:
        assert payload["data"]["error"] is None


def test_done_json_result_is_original_public_response(client):
    original = {
        "success": True,
        "data": {
            "text": "Весь текст\nSecond line",
            "meta": {
                "engine": "paddleocr",
                "model": "paddle",
                "duration_seconds": 0.5,
            },
        },
        "error": None,
    }
    stored = {"format": "json", "response": original}
    with patch(
        f"{ROUTER}.get_job",
        AsyncMock(return_value=task(status="done", result=stored)),
    ):
        response = client.get(f"/v1/tasks/{TASK_ID}/result")
    assert response.status_code == 200
    assert response.json() == original
    assert response.headers["cache-control"] == "no-store"


def test_done_crop_result_returns_exact_jpeg(client, image_content):
    output = BytesIO()
    with Image.open(BytesIO(image_content)) as image:
        image.save(output, format="JPEG")
    jpeg = output.getvalue()
    stored = {
        "format": "jpeg",
        "content_base64": base64.b64encode(jpeg).decode("ascii"),
    }
    with patch(
        f"{ROUTER}.get_job",
        AsyncMock(
            return_value=task(mode="crop_mrz", status="done", result=stored)
        ),
    ):
        response = client.get(f"/v1/tasks/{TASK_ID}/result")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == jpeg
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    ("status", "code"),
    [
        ("queued", "TASK_NOT_READY"),
        ("processing", "TASK_NOT_READY"),
        ("cancelled", "TASK_CANCELLED"),
    ],
)
def test_result_unavailable_until_done(client, status, code):
    with patch(
        f"{ROUTER}.get_job", AsyncMock(return_value=task(status=status))
    ):
        response = client.get(f"/v1/tasks/{TASK_ID}/result")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == code


@pytest.mark.parametrize(
    ("state", "code", "status"),
    [
        ("failed", ErrorCode.OCR_FAILED, 422),
        ("failed", ErrorCode.OCR_ENGINE_UNAVAILABLE, 503),
        ("timeout", ErrorCode.OCR_TIMEOUT, 504),
    ],
)
def test_failed_result_returns_saved_processing_error(
    client, state, code, status
):
    job = task(status=state, error=encode_error(code, "Ошибка обработки"))
    with patch(f"{ROUTER}.get_job", AsyncMock(return_value=job)):
        response = client.get(f"/v1/tasks/{TASK_ID}/result")
    assert response.status_code == status
    assert response.json() == {
        "success": False,
        "data": None,
        "error": {"code": code.value, "message": "Ошибка обработки"},
    }


@pytest.mark.parametrize(
    "result",
    [
        None,
        {},
        {"format": "json", "response": []},
        {"format": "jpeg", "content_base64": "broken"},
        {"format": "jpeg", "content_base64": ""},
    ],
)
def test_corrupt_result_is_an_internal_error(client, result):
    with patch(
        f"{ROUTER}.get_job",
        AsyncMock(return_value=task(status="done", result=result)),
    ):
        response = client.get(f"/v1/tasks/{TASK_ID}/result")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"


@pytest.mark.parametrize(
    ("method", "suffix"),
    [("get", ""), ("get", "/result"), ("post", "/cancel")],
)
def test_unknown_task_returns_task_not_found(client, method, suffix):
    with patch(f"{ROUTER}.get_job", AsyncMock(return_value=None)):
        response = getattr(client, method)(f"/v1/tasks/{TASK_ID}{suffix}")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "TASK_NOT_FOUND"


@pytest.mark.parametrize(
    ("method", "suffix"),
    [("get", ""), ("get", "/result"), ("post", "/cancel")],
)
def test_invalid_task_uuid_is_rejected_before_query(client, method, suffix):
    with patch(f"{ROUTER}.get_job", new_callable=AsyncMock) as get:
        response = getattr(client, method)(f"/v1/tasks/not-a-uuid{suffix}")
    assert response.status_code == 422
    get.assert_not_awaited()


@pytest.mark.parametrize(
    ("initial_status", "started", "delete"),
    [
        ("queued", None, True),
        ("queued", datetime(2026, 9, 25, 12, 1, tzinfo=UTC), False),
        ("processing", datetime(2026, 9, 25, 12, 1, tzinfo=UTC), False),
    ],
)
def test_cancel_preserves_image_if_worker_started_during_request(
    client, initial_status, started, delete
):
    before = task(status=initial_status)
    after = task(status="cancelled", started_at=started)
    with (
        patch(
            f"{ROUTER}.get_job", AsyncMock(side_effect=[before, after])
        ) as get,
        patch(
            f"{ROUTER}.mark_cancelled", AsyncMock(return_value=True)
        ) as cancel,
        patch(f"{ROUTER}.delete_upload") as remove,
        patch(f"{ROUTER}.settings", SimpleNamespace(ocr_save_uploads=False)),
    ):
        response = client.post(f"/v1/tasks/{TASK_ID}/cancel")
    assert response.json() == {
        "success": True,
        "data": {"task_id": TASK_ID, "status": "cancelled", "cancelled": True},
        "error": None,
    }
    assert get.await_count == 2
    cancel.assert_awaited_once()
    assert remove.call_count == int(delete)


@pytest.mark.parametrize("latest_status", ["done", "failed", "cancelled"])
def test_cancel_race_reports_fresh_terminal_status(client, latest_status):
    with (
        patch(
            f"{ROUTER}.get_job",
            AsyncMock(side_effect=[task(), task(status=latest_status)]),
        ),
        patch(f"{ROUTER}.mark_cancelled", AsyncMock(return_value=False)),
        patch(f"{ROUTER}.delete_upload") as remove,
    ):
        response = client.post(f"/v1/tasks/{TASK_ID}/cancel")
    assert response.status_code == 200
    assert response.json()["data"] == {
        "task_id": TASK_ID,
        "status": latest_status,
        "cancelled": False,
    }
    remove.assert_not_called()


@pytest.mark.parametrize("status", ["done", "failed", "timeout", "cancelled"])
def test_terminal_task_cancel_is_noop(client, status):
    with (
        patch(
            f"{ROUTER}.get_job", AsyncMock(return_value=task(status=status))
        ),
        patch(f"{ROUTER}.mark_cancelled", new_callable=AsyncMock) as cancel,
        patch(f"{ROUTER}.delete_upload") as remove,
    ):
        response = client.post(f"/v1/tasks/{TASK_ID}/cancel")
    assert response.json()["data"] == {
        "task_id": TASK_ID,
        "status": status,
        "cancelled": False,
    }
    cancel.assert_not_awaited()
    remove.assert_not_called()


def test_openapi_creation_parameters_match_sync_endpoints():
    schema = app.openapi()

    def body(path):
        ref = schema["paths"][path]["post"]["requestBody"]["content"][
            "multipart/form-data"
        ]["schema"]["$ref"]
        return schema["components"]["schemas"][ref.rsplit("/", 1)[-1]]

    for suffix, _, _, _ in CASES:
        asynchronous = body(f"/v1/tasks/{suffix}")
        synchronous = body(f"/v1/ocr/{suffix}")
        assert asynchronous["required"] == synchronous["required"]
        assert set(asynchronous["properties"]) == set(
            synchronous["properties"]
        )
        for name, actual in asynchronous["properties"].items():
            expected = synchronous["properties"][name]
            for key in ("type", "format", "default", "$ref"):
                assert actual.get(key) == expected.get(key), (
                    suffix,
                    name,
                    key,
                )
        assert (
            "202"
            in schema["paths"][f"/v1/tasks/{suffix}"]["post"]["responses"]
        )
    assert schema["paths"]["/v1/tasks/{task_id}"]["get"]["tags"] == ["Tasks"]
    contents = schema["paths"]["/v1/tasks/{task_id}/result"]["get"][
        "responses"
    ]["200"]["content"]
    assert set(contents) == {"application/json", "image/jpeg"}
