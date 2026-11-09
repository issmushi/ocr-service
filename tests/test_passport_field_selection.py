from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.errors import AppError
from app.main import app
from app.modules.russian.passport import INTERNAL_PASSPORT_OCR_FIELDS
from app.modules.v1.passport.field_selection import (
    INTERNAL_FIELDS,
    resolve_internal_fields,
)
from tests.test_passport_v1_text_service import FILES, PATH, SERVICE


@pytest.mark.parametrize(
    "fields",
    [
        ["surname", "issue_date"],
        "surname,issue_date",
        ["surname", "issue_date", "surname"],
    ],
)
def test_selection_reaches_runtime_and_omits_unrequested_fields(fields):
    client = TestClient(app)
    with (
        patch(
            f"{SERVICE}.russian_runtime.run",
            new=AsyncMock(
                return_value={
                    "passport": {"surname": "ТЕСТОВ", "name": "ТЕСТ"},
                    "mrz": ["LINE1", "LINE2"],
                }
            ),
        ) as run,
    ):
        response = client.post(PATH, files=FILES, data={"fields": fields})
    client.close()
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert set(data) == {"surname", "issue_date", "meta"}
    assert data["surname"] == "ТЕСТОВ"
    assert data["issue_date"] is None
    assert run.await_args.kwargs["fields"] == ("surname", "issue_date")


@pytest.mark.parametrize("path", [PATH, "/v1/tasks/passport/internal"])
@pytest.mark.parametrize(
    "fields", ["", "unknown", "[]", ["all", "surname"], "all,name"]
)
def test_invalid_selection_rejected_before_ocr_or_queue(path, fields):
    client = TestClient(app)
    with (
        patch(f"{SERVICE}.russian_runtime.run", new=AsyncMock()) as run,
        patch(
            "app.modules.tasks.router.create_image_task", new=AsyncMock()
        ) as create,
    ):
        response = client.post(path, files=FILES, data={"fields": fields})
    client.close()
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    run.assert_not_awaited()
    create.assert_not_awaited()


@pytest.mark.parametrize("fields", [[], [""], [42], "surname"])
def test_invalid_persisted_fields_are_rejected(fields):
    with pytest.raises(AppError):
        resolve_internal_fields(fields)


def test_all_fields_have_engine_mapping():
    assert set(INTERNAL_FIELDS) == set(INTERNAL_PASSPORT_OCR_FIELDS)
    assert resolve_internal_fields() == resolve_internal_fields(["all"])
