import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.field_selection import resolve_fields
from app.main import app
from app.modules.tasks.execution import execute_task_operation
from app.modules.v1.mrz_api.field_selection import MrzField
from app.modules.v1.passport.international_fields import (
    PassportInternationalField,
)
from app.modules.v1.russia.driver_license.field_selection import (
    DriverLicenseField,
)
from app.modules.v1.russia.egrn.field_selection import EgrnField
from app.modules.v1.russia.snils.field_selection import SnilsField
from tests.test_egrn_fields import synthetic_extract
from tests.test_passport_v1_text_service import FILES, IMAGE_BYTES

CASES = [
    (
        "passport/international",
        "passport_international",
        PassportInternationalField,
        "surname_latin",
        "birth_place_en",
    ),
    ("snils", "snils", SnilsField, "surname", "birth_place"),
    (
        "driver-license",
        "driver_license",
        DriverLicenseField,
        "surname",
        "birth_place_en",
    ),
    ("egrn", "egrn", EgrnField, "application_number", "address"),
    ("mrz", "mrz", MrzField, "surname", "department_code"),
]


def raw_document(operation, *_args, **_kwargs):
    return {
        "doctype": "SNILS_1996" if operation == "snils" else "DL_2011",
        "ocr": {"Last_name_ru": "ТЕСТОВ"},
        "passport": {"surname_latin": "TESTOV"},
        # Ignore unrequested crops even if a worker returns them.
        "crops": {
            "Birth_date": b"birth",
            "Issue_date": b"issue",
            "Licence_number": b"number",
        },
    }


@pytest.mark.parametrize("path,mode,enum,first,missing", CASES)
def test_selected_fields_survive_direct_and_queued_serialization(
    path, mode, enum, first, missing
):
    client = TestClient(app)
    selected = [first, missing]
    raw_egrn = {
        "fields": [
            {
                "name": "application_number",
                "ocr": {"text": "00042", "status": "success", "error": None},
            }
        ],
        "meta": {"duration_seconds": 0},
    }
    rows = [
        "P<ARMTESTOV<<TEST".ljust(44, "<"),
        "AA12345678ARM9001158M3001156<<<<<<<<<<<<<<06",
    ]
    with (
        patch(
            "app.modules.russian.runtime.russian_runtime.run",
            new=AsyncMock(side_effect=raw_document),
        ) as russian,
        patch(
            "app.modules.paddle.runtime.paddle_runtime.run", new=AsyncMock()
        ) as paddle,
        patch("app.modules.v1.passport.service.perf_counter", return_value=10),
        patch(
            "app.modules.v1.russia.snils.service.perf_counter", return_value=10
        ),
        patch(
            "app.modules.v1.russia.driver_license.service.perf_counter",
            return_value=10,
        ),
        patch("app.modules.v1.mrz_api.service.perf_counter", return_value=10),
        patch(
            "app.modules.v1.mrz_api.service.crop_mrz_bytes",
            new=AsyncMock(return_value=IMAGE_BYTES),
        ),
        patch(
            "app.modules.v1.passport.runtime.mrz_scanner_runtime.run",
            new=AsyncMock(return_value={"items": [{"text": "\n".join(rows)}]}),
        ) as scanner,
        patch(
            "app.modules.v1.russia.egrn.service.read_egrn_fields_upload",
            new=AsyncMock(return_value=raw_egrn),
        ) as egrn,
        patch(
            "app.modules.tasks.router.create_image_task",
            new=AsyncMock(
                return_value={
                    "success": True,
                    "data": {
                        "task_id": "757c8c4a-e1a5-4ff4-9855-2aa9df95397d",
                        "status": "queued",
                    },
                    "error": None,
                }
            ),
        ) as create,
    ):
        direct = client.post(
            f"/v1/ocr/{path}", files=FILES, data={"fields": selected}
        )
        assert direct.status_code == 200, direct.text
        data = direct.json()["data"]
        assert set(data) == {*selected, "meta"}
        assert data[first] is not None
        assert data[missing] is None
        created = client.post(
            f"/v1/tasks/{path}", files=FILES, data={"fields": selected}
        )
        assert created.status_code == 202, created.text
        options = create.await_args.kwargs["options"]
        assert options["fields"] == selected
        result = asyncio.run(
            execute_task_operation(
                {
                    "mode": mode,
                    "filename": "synthetic.png",
                    "content_type": "image/png",
                    "options": options,
                },
                IMAGE_BYTES,
            )
        )
        assert result["response"] == direct.json()
        paddle.assert_not_awaited()
        if path in {"snils", "driver-license", "passport/international"}:
            assert russian.await_count == 2
            assert russian.await_args.kwargs["fields"] == tuple(selected)
        elif path == "egrn":
            assert egrn.await_args.kwargs["fields"] == tuple(selected)
        else:
            assert scanner.await_count == 2
            assert "fields" not in scanner.await_args.kwargs
    client.close()


@pytest.mark.parametrize("path,mode,enum,first,missing", CASES)
@pytest.mark.parametrize("prefix", ["ocr", "tasks"])
@pytest.mark.parametrize("value", ["", "not_a_field", "[]", "all,surname"])
def test_invalid_fields_fail_before_execution(
    path, mode, enum, first, missing, prefix, value
):
    client = TestClient(app)
    with (
        patch(
            "app.modules.russian.runtime.russian_runtime.run", new=AsyncMock()
        ) as russian,
        patch(
            "app.modules.tasks.router.create_image_task", new=AsyncMock()
        ) as create,
        patch(
            "app.modules.v1.russia.egrn.service.read_egrn_fields_upload",
            new=AsyncMock(),
        ) as egrn,
        patch(
            "app.modules.v1.mrz_api.service.crop_mrz_bytes", new=AsyncMock()
        ) as crop,
    ):
        response = client.post(
            f"/v1/{prefix}/{path}", files=FILES, data={"fields": value}
        )
        assert response.status_code == 422, response.text
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"
        for mock in (russian, create, egrn, crop):
            mock.assert_not_awaited()
    client.close()


@pytest.mark.parametrize("path,mode,enum,first,missing", CASES)
def test_openapi_choices_match_document_response(
    path, mode, enum, first, missing
):
    spec = app.openapi()
    for prefix in ("ocr", "tasks"):
        op = spec["paths"][f"/v1/{prefix}/{path}"]["post"]
        ref = op["requestBody"]["content"]["multipart/form-data"]["schema"][
            "$ref"
        ]
        field = spec["components"]["schemas"][ref.split("/")[-1]][
            "properties"
        ]["fields"]
        assert field["default"] == ["all"]
        names = spec["components"]["schemas"][
            field["items"]["$ref"].split("/")[-1]
        ]["enum"]
        assert set(names) == {item.value for item in enum}
    response_ref = spec["paths"][f"/v1/ocr/{path}"]["post"]["responses"][
        "200"
    ]["content"]["application/json"]["schema"]["$ref"]
    response = spec["components"]["schemas"][response_ref.split("/")[-1]]
    data_ref = next(
        item["$ref"]
        for item in response["properties"]["data"]["anyOf"]
        if "$ref" in item
    )
    data = spec["components"]["schemas"][data_ref.split("/")[-1]]
    assert set(data["properties"]) == set(resolve_fields(None, enum)) | {
        "meta"
    }
    assert data["required"] == ["meta"]


def test_snils_name_only_works_without_paddle_in_direct_and_task_routes():
    client = TestClient(app)
    with (
        patch.object(
            type(settings),
            "is_model_enabled",
            side_effect=lambda model: model == "russian-ocr",
        ),
        patch(
            "app.modules.russian.runtime.russian_runtime.run",
            new=AsyncMock(side_effect=raw_document),
        ),
        patch(
            "app.modules.paddle.runtime.paddle_runtime.run", new=AsyncMock()
        ) as paddle,
        patch(
            "app.modules.tasks.router.create_image_task",
            new=AsyncMock(
                return_value={
                    "success": True,
                    "data": {
                        "task_id": "757c8c4a-e1a5-4ff4-9855-2aa9df95397d",
                        "status": "queued",
                    },
                    "error": None,
                }
            ),
        ),
    ):
        for prefix, status in (("ocr", 200), ("tasks", 202)):
            result = client.post(
                f"/v1/{prefix}/snils", files=FILES, data={"fields": "surname"}
            )
            assert result.status_code == status, result.text
        paddle.assert_not_awaited()
        assert (
            client.post(
                "/v1/ocr/snils", files=FILES, data={"fields": "date_of_birth"}
            ).status_code
            == 503
        )
    client.close()


@pytest.mark.parametrize(
    "path,doctype", [("snils", "SNILS_1996"), ("driver-license", "DL_2011")]
)
def test_paddle_only_reads_requested_date(path, doctype):
    raw = {
        "doctype": doctype,
        "ocr": {},
        "crops": {
            "Birth_date": b"birth",
            "Issue_date": b"issue",
            "Expiration_date": b"expiry",
            "Licence_number": b"number",
        },
    }
    client = TestClient(app)
    with (
        patch(
            "app.modules.russian.runtime.russian_runtime.run",
            new=AsyncMock(return_value=raw),
        ),
        patch(
            "app.modules.paddle.runtime.paddle_runtime.run",
            new=AsyncMock(
                return_value={"text": "01.01.1990", "confidence": 0.99}
            ),
        ) as paddle,
    ):
        response = client.post(
            f"/v1/ocr/{path}", files=FILES, data={"fields": "date_of_birth"}
        )
    assert response.status_code == 200, response.text
    assert response.json()["data"]["date_of_birth"] == "01.01.1990"
    assert [call.args[1] for call in paddle.await_args_list] == [b"birth"]
    client.close()


def test_egrn_selection_happens_before_paddle():
    image, _ = synthetic_extract()

    async def recognize(fields, *_args, **_kwargs):
        assert [field["name"] for field in fields] == ["address"]
        for field in fields:
            field["ocr"] = {
                "text": "ТЕСТОВЫЙ АДРЕС",
                "status": "success",
                "error": None,
            }
        return {}

    with (
        patch("app.modules.v1.crop.egrn._prepare_egrn", return_value=image),
        patch(
            "app.modules.v1.crop.egrn.recognize_field_previews",
            new=AsyncMock(side_effect=recognize),
        ) as paddle,
    ):
        client = TestClient(app)
        response = client.post(
            "/v1/ocr/egrn", files=FILES, data={"fields": "address"}
        )
        client.close()
    assert response.status_code == 200, response.text
    assert set(response.json()["data"]) == {"address", "meta"}
    assert response.json()["data"]["address"] == "ТЕСТОВЫЙ АДРЕС"
    paddle.assert_awaited_once()


@pytest.mark.parametrize(
    "normalization,selected", [(False, ["normalization"]), (True, ["surname"])]
)
def test_mrz_selection_respects_normalization_flag(normalization, selected):
    rows = [
        "P<ARMTESTOV<<TEST".ljust(44, "<"),
        "AA12345678ARM9001158M3001156<<<<<<<<<<<<<<06",
    ]
    with (
        patch(
            "app.modules.v1.mrz_api.service.crop_mrz_bytes",
            new=AsyncMock(return_value=IMAGE_BYTES),
        ),
        patch(
            "app.modules.v1.passport.runtime.mrz_scanner_runtime.run",
            new=AsyncMock(return_value={"items": [{"text": "\n".join(rows)}]}),
        ),
    ):
        client = TestClient(app)
        response = client.post(
            "/v1/ocr/mrz",
            files=FILES,
            data={
                "fields": selected,
                "normalization": str(normalization).lower(),
            },
        )
        client.close()
    assert response.status_code == 200, response.text
    assert set(response.json()["data"]) == set(selected) | {"meta"}
    if "normalization" in selected:
        assert response.json()["data"]["normalization"] is None
