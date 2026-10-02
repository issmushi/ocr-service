from __future__ import annotations

import asyncio
import base64
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.workers.ocr_worker import execute_ocr_job


@pytest.mark.parametrize(
    ("operation", "form", "idcard"),
    [
        ("mrz", {}, False),
        (
            "mrz",
            {
                "crop": "false",
                "normalize": "false",
                "normalization": "false",
            },
            False,
        ),
        ("passport/internal", {}, False),
        ("mrz", {}, True),
        (
            "mrz",
            {
                "crop": "false",
                "normalize": "false",
                "normalization": "false",
            },
            True,
        ),
        ("full", {"model": "paddle"}, False),
        ("crop/document", {}, False),
        ("crop/mrz", {}, False),
    ],
)
def test_task_result_matches_direct_http_response(
    tmp_path, operation, form, idcard
):
    """Exercise upload, options, worker dispatch, and result retrieval."""
    image_buffer = BytesIO()
    Image.new("RGB", (100, 50), "white").save(image_buffer, format="PNG")
    content = image_buffer.getvalue()
    jpeg_buffer = BytesIO()
    Image.new("RGB", (90, 20), "white").save(jpeg_buffer, format="JPEG")
    mrz_jpeg = jpeg_buffer.getvalue()
    stored: dict = {}

    async def persist_task(_session_factory, **values):
        stored.update(values)

    passport_rows = [
        "P<ARMTESTOV<<TEST".ljust(44, "<"),
        "AA12345678ARM9001158M3001156<<<<<<<<<<<<<<06",
    ]
    idcard_rows = [
        "IDKAZ1234567897900115300001<<<",
        "9001158M3501151KAZ<<<<<<<<<<<0",
        "TESTOV<<TEST<<<<<<<<<<<<<<<<<<",
    ]
    raw_text = "\n".join(idcard_rows if idcard else passport_rows)

    async def recognize_mrz(scanner_operation, _content, **options):
        assert scanner_operation == "mrz_crop_text"
        return {"items": [{"text": raw_text}]}

    with ExitStack() as stack:
        client = TestClient(app)
        stack.callback(client.close)
        stack.enter_context(
            patch(
                "app.core.infrastructure.storage.upload_path",
                side_effect=lambda task_id, suffix: (
                    tmp_path / f"{task_id}{suffix}"
                ),
            )
        )
        stack.enter_context(
            patch(
                "app.modules.tasks.creation.create_job",
                new=AsyncMock(side_effect=persist_task),
            )
        )
        for module in ("mrz_api", "passport", "full"):
            stack.enter_context(
                patch(
                    f"app.modules.v1.{module}.service.perf_counter",
                    return_value=10.0,
                )
            )
        stack.enter_context(
            patch(
                "app.modules.v1.full.service.recognize_vlm_image",
                new=AsyncMock(return_value={"items": [{"text": raw_text}]}),
            )
        )
        scanner = stack.enter_context(
            patch(
                "app.modules.v1.passport.runtime.mrz_scanner_runtime.run",
                new=AsyncMock(side_effect=recognize_mrz),
            )
        )
        stack.enter_context(
            patch(
                "app.modules.v1.passport.service.russian_runtime.run",
                new=AsyncMock(
                    return_value={
                        "passport": {
                            "surname": "ТЕСТОВ",
                            "name": "ТЕСТ",
                            "date_of_birth": "15.01.1990",
                            "Issue_organization_ru": "ГУ МВД ПО МОСКВЕ",
                        }
                    }
                ),
            )
        )
        stack.enter_context(
            patch(
                "app.modules.v1.mrz_api.service.crop_mrz_bytes",
                new=AsyncMock(return_value=content),
            )
        )
        stack.enter_context(
            patch(
                "app.modules.v1.full.service.paddle_runtime.run",
                new=AsyncMock(
                    return_value={"text": "Полный текст\nMixed CASE 123"}
                ),
            )
        )
        stack.enter_context(
            patch(
                "app.modules.v1.crop.service.crop_document",
                new=AsyncMock(
                    return_value={
                        "mrz_detected": True,
                        "mrz_image": "data:image/jpeg;base64,"
                        + base64.b64encode(mrz_jpeg).decode("ascii"),
                    }
                ),
            )
        )
        files = {"image": ("scan.png", content, "image/png")}
        direct = client.post(f"/v1/ocr/{operation}", files=files, data=form)
        assert direct.status_code == 200, direct.text
        created = client.post(f"/v1/tasks/{operation}", files=files, data=form)
        assert created.status_code == 202, created.text
        task_id = created.json()["data"]["task_id"]
        assert stored["job_id"] == task_id
        if operation == "mrz":
            assert stored["options"] == {
                key: form.get(key, "true") == "true"
                for key in ("crop", "normalize", "normalization")
            }
        image_path = Path(stored["image_path"])
        assert image_path.read_bytes() == content
        job = {
            **stored,
            "id": task_id,
            "status": "processing",
            "error": None,
            "created_at": None,
            "started_at": None,
            "finished_at": None,
        }
        result = asyncio.run(execute_ocr_job(task_id, job, image_path))
        job.update(status="done", result=result)
        stack.enter_context(
            patch(
                "app.modules.tasks.router.get_job",
                new=AsyncMock(return_value=job),
            )
        )
        completed = client.get(f"/v1/tasks/{task_id}/result")
        if operation == "mrz":
            assert scanner.await_count == 2

    assert completed.status_code == 200, completed.text
    assert completed.headers["content-type"] == direct.headers["content-type"]
    if operation.startswith("crop/"):
        assert completed.content == direct.content
    else:
        assert completed.json() == direct.json()
        if operation == "mrz":
            data = completed.json()["data"]
            assert data["document_type"] == (
                "identity_card" if idcard else "international_passport"
            )
            assert "passport_type" not in data
            assert "passport_number" not in data
            assert data["document_number"] == (
                "123456789" if idcard else "AA1234567"
            )
        if operation == "passport/internal":
            data = completed.json()["data"]
            assert set(data) == {
                "passport_type",
                "surname",
                "name",
                "patronymic",
                "sex",
                "date_of_birth",
                "passport_series",
                "passport_number",
                "issue_date",
                "Issue_organization_ru",
                "department_code",
                "meta",
            }
            assert data["Issue_organization_ru"] == ("ГУ МВД ПО МОСКВЕ")
