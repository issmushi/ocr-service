from __future__ import annotations

from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

PATH = "/v1/ocr/idcard/mrz"

KAZ_TD1 = [
    "IDKAZ1234567897900115300001<<<",
    "9001158M3501151KAZ<<<<<<<<<<<0",
    "TESTOV<<TEST<<<<<<<<<<<<<<<<<<",
]
KAZ_TD2 = [
    "IDKAZTESTOV<<TEST<<<<<<<<<<<<<<<<<<<",
    "0123456784KAZ9001158M3501151300001<6",
]
KGZ_TD1 = [
    "IDKGZID1234567390011530000001<",
    "9001158M3501151KGZ<<<<<<<<<<<2",
    "TESTOV<<TEST<<<<<<<<<<<<<<<<<<",
]
KGZ_TD1_I_FILLER = [f"I<{KGZ_TD1[0][2:]}", *KGZ_TD1[1:]]

EMPTY_NORMALIZATION = {
    "corrections_count": 0,
    "replacements_count": 0,
    "insertions_count": 0,
    "corrections": [],
    "issues": [],
}

KAZ_TD1_DOCUMENT = {
    "document_type": "identity_card",
    "document_code": "ID",
    "issuing_country": "KAZ",
    "surname": "TESTOV",
    "name": "TEST",
    "patronymic": None,
    "sex": "МУЖ.",
    "date_of_birth": "15.01.1990",
    "document_series": None,
    "document_number": "123456789",
    "nationality": "KAZ",
    "issue_date": None,
    "expiry_date": "15.01.2035",
    "department_code": None,
    "personal_number": "900115300001",
    "mrz_valid": True,
}
KAZ_TD2_DOCUMENT = {
    "document_type": "identity_card",
    "document_code": "ID",
    "issuing_country": "KAZ",
    "surname": "TESTOV",
    "name": "TEST",
    "patronymic": None,
    "sex": "МУЖ.",
    "date_of_birth": "15.01.1990",
    "document_series": None,
    "document_number": "012345678",
    "nationality": "KAZ",
    "issue_date": None,
    "expiry_date": "15.01.2035",
    "department_code": None,
    "personal_number": "900115300001",
    "mrz_valid": True,
}
KGZ_TD1_DOCUMENT = {
    "document_type": "identity_card",
    "document_code": "ID",
    "issuing_country": "KGZ",
    "surname": "TESTOV",
    "name": "TEST",
    "patronymic": None,
    "sex": "МУЖ.",
    "date_of_birth": "15.01.1990",
    "document_series": None,
    "document_number": "ID1234567",
    "nationality": "KGZ",
    "issue_date": None,
    "expiry_date": "15.01.2035",
    "department_code": None,
    "personal_number": "90011530000001",
    "mrz_valid": True,
}

ENGINE_BY_MODEL = {
    "paddle": "paddleocr",
    "mrz": "mrz-scanner",
    "qwen3-vl-4b": "qwen3-vl",
    "qwen3-vl-8b": "qwen3-vl",
}


def _raw_result(lines: list[str]) -> dict[str, Any]:
    return {"items": [{"text": line} for line in lines]}


def _post_idcard(
    client: TestClient,
    lines: list[str],
    *,
    model: str | None = "paddle",
    crop: bool = False,
    normalize: bool = True,
    normalization: bool = True,
) -> tuple[Any, dict[str, AsyncMock]]:
    raw_result = _raw_result(lines)
    mocks = {
        "crop": AsyncMock(return_value=b"cropped-mrz"),
        "paddle": AsyncMock(return_value=raw_result),
        "mrz": AsyncMock(return_value=raw_result),
        "vlm": AsyncMock(return_value=raw_result),
    }
    with (
        patch(
            "app.modules.v1.idcard.service.crop_mrz_bytes",
            new=mocks["crop"],
        ),
        patch(
            "app.modules.v1.idcard.service.paddle_runtime.run",
            new=mocks["paddle"],
        ),
        patch(
            "app.modules.v1.idcard.service.mrz_scanner_runtime.run",
            new=mocks["mrz"],
        ),
        patch(
            "app.modules.v1.idcard.service.recognize_vlm_image",
            new=mocks["vlm"],
        ),
        patch(
            "app.modules.v1.idcard.service.settings",
            new=replace(
                settings,
                paddle_ocr_document_hard_timeout_seconds=11.0,
                mrz_scanner_document_hard_timeout_seconds=17.0,
            ),
        ),
        patch(
            "app.modules.v1.idcard.service.perf_counter",
            side_effect=[10.0, 10.25],
        ),
    ):
        data = {
            "crop": str(crop).lower(),
            "normalize": str(normalize).lower(),
            "normalization": str(normalization).lower(),
        }
        if model is not None:
            data["model"] = model
        response = client.post(
            PATH,
            data=data,
            files={"image": ("idcard.png", b"fake-image", "image/png")},
        )
    return response, mocks


def _assert_selected_runtime(
    mocks: dict[str, AsyncMock],
    *,
    model: str,
    crop: bool,
) -> None:
    content = b"cropped-mrz" if crop else b"fake-image"
    if crop:
        mocks["crop"].assert_awaited_once_with(b"fake-image")
    else:
        mocks["crop"].assert_not_awaited()

    if model == "paddle":
        mocks["paddle"].assert_awaited_once_with(
            "idcard_v1_text",
            content,
            timeout_seconds=11.0,
        )
    else:
        mocks["paddle"].assert_not_awaited()

    if model == "mrz":
        mocks["mrz"].assert_awaited_once_with(
            "mrz_crop_text",
            content,
            timeout_seconds=17.0,
        )
    else:
        mocks["mrz"].assert_not_awaited()

    if model.startswith("qwen"):
        mocks["vlm"].assert_awaited_once_with(
            content,
            model=model,
            filename="mrz.jpg" if crop else "idcard.png",
            content_type="image/jpeg" if crop else "image/png",
        )
    else:
        mocks["vlm"].assert_not_awaited()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.mark.parametrize("model", tuple(ENGINE_BY_MODEL))
@pytest.mark.parametrize("crop", (False, True))
def test_all_models_and_crop_modes_return_the_same_document(
    client: TestClient,
    model: str,
    crop: bool,
) -> None:
    response, mocks = _post_idcard(
        client,
        KAZ_TD2,
        model=model,
        crop=crop,
    )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "success": True,
        "data": {
            "mrz": KAZ_TD2,
            "normalization": EMPTY_NORMALIZATION,
            **KAZ_TD2_DOCUMENT,
            "meta": {
                "engine": ENGINE_BY_MODEL[model],
                "model": model,
                "duration_seconds": 0.25,
            },
        },
        "error": None,
    }
    _assert_selected_runtime(mocks, model=model, crop=crop)


@pytest.mark.parametrize(
    ("mrz", "expected_document"),
    (
        (KAZ_TD1, KAZ_TD1_DOCUMENT),
        (KAZ_TD2, KAZ_TD2_DOCUMENT),
        (KGZ_TD1, KGZ_TD1_DOCUMENT),
        (
            KGZ_TD1_I_FILLER,
            {**KGZ_TD1_DOCUMENT, "document_code": "I"},
        ),
    ),
)
def test_supported_td1_td2_and_kyrgyz_document_codes(
    client: TestClient,
    mrz: list[str],
    expected_document: dict[str, Any],
) -> None:
    response, _ = _post_idcard(
        client,
        mrz,
        normalization=False,
    )

    assert response.status_code == 200, response.text
    assert response.json()["data"] == {
        "mrz": mrz,
        "normalization": None,
        **expected_document,
        "meta": {
            "engine": "paddleocr",
            "model": "paddle",
            "duration_seconds": 0.25,
        },
    }


CORRECTABLE_KAZ_TD1 = [
    KAZ_TD1[0],
    "9O01158M3501151KAZ<<<<<<<<<<<0",
    "TEST0V<<TEST<<<<<<<<<<<<<<<<<<",
]
CORRECTION_REPORT = {
    "corrections_count": 2,
    "replacements_count": 2,
    "insertions_count": 0,
    "corrections": [
        {
            "line": 2,
            "position": 2,
            "source_position": 2,
            "field": "Дата рождения",
            "operation": "замена",
            "source": "O",
            "replacement": "0",
            "reason": "Замена буквы на похожую цифру",
        },
        {
            "line": 3,
            "position": 5,
            "source_position": 5,
            "field": "Фамилия и имена",
            "operation": "замена",
            "source": "0",
            "replacement": "O",
            "reason": "Замена цифры на похожую букву",
        },
    ],
    "issues": [],
}
LOWERCASE_SPACED_KAZ_TD1 = [
    f" \t{KAZ_TD1[0][:15].lower()}\u00a0{KAZ_TD1[0][15:].lower()} ",
    f" {KAZ_TD1[1][:8].lower()}\u2003{KAZ_TD1[1][8:].lower()} ",
    f" {KAZ_TD1[2][:9].lower()} {KAZ_TD1[2][9:].lower()} ",
]


@pytest.mark.parametrize(
    ("normalize", "normalization", "raw_mrz", "expected_report"),
    (
        (True, True, CORRECTABLE_KAZ_TD1, CORRECTION_REPORT),
        (True, False, CORRECTABLE_KAZ_TD1, None),
        (False, True, LOWERCASE_SPACED_KAZ_TD1, EMPTY_NORMALIZATION),
        (False, False, LOWERCASE_SPACED_KAZ_TD1, None),
    ),
)
def test_normalize_and_normalization_flags_are_independent(
    client: TestClient,
    normalize: bool,
    normalization: bool,
    raw_mrz: list[str],
    expected_report: dict[str, Any] | None,
) -> None:
    response, _ = _post_idcard(
        client,
        raw_mrz,
        normalize=normalize,
        normalization=normalization,
    )

    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["mrz"] == KAZ_TD1
    assert data["normalization"] == expected_report
    assert {key: data[key] for key in KAZ_TD1_DOCUMENT} == KAZ_TD1_DOCUMENT


def test_normalize_false_does_not_repair_header(client: TestClient) -> None:
    raw_mrz = [f"1{KAZ_TD1[0][1:]}", *KAZ_TD1[1:]]
    response, mocks = _post_idcard(
        client,
        raw_mrz,
        normalize=False,
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "OCR_FAILED"
    mocks["paddle"].assert_awaited_once()


def test_normalize_false_keeps_short_line_without_padding(
    client: TestClient,
) -> None:
    raw_mrz = [*KAZ_TD1[:2], KAZ_TD1[2][:-1]]
    response, mocks = _post_idcard(
        client,
        raw_mrz,
        normalize=False,
    )

    assert response.status_code == 200, response.text
    assert response.json()["data"]["mrz"] == raw_mrz
    assert response.json()["data"]["normalization"] == EMPTY_NORMALIZATION
    assert response.json()["data"]["mrz_valid"] is None
    assert len(response.json()["data"]["mrz"][2]) == 29
    mocks["paddle"].assert_awaited_once()


@pytest.mark.parametrize(
    "raw_mrz",
    (
        [
            "IDXXX1234567897900115300001<<<",
            "9001158M3501151XXX<<<<<<<<<<<0",
            KAZ_TD1[2],
        ],
        [
            "IDXXXTESTOV<<TEST<<<<<<<<<<<<<<<<<<<",
            "0123456784XXX9001158M3501151300001<6",
        ],
    ),
    ids=("td1", "td2"),
)
def test_unknown_nonempty_idcard_headers_return_ocr_failed(
    client: TestClient,
    raw_mrz: list[str],
) -> None:
    response, mocks = _post_idcard(client, raw_mrz)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "OCR_FAILED"
    mocks["paddle"].assert_awaited_once()


def test_td1_missing_third_line_returns_ocr_failed(client: TestClient) -> None:
    response, mocks = _post_idcard(client, KAZ_TD1[:2])

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "OCR_FAILED"
    mocks["paddle"].assert_awaited_once()


def test_unknown_model_is_rejected_before_dispatch(client: TestClient) -> None:
    with (
        patch(
            "app.modules.v1.idcard.service.crop_mrz_bytes",
            new_callable=AsyncMock,
        ) as crop_mrz,
        patch(
            "app.modules.v1.idcard.service.paddle_runtime.run",
            new_callable=AsyncMock,
        ) as paddle_run,
        patch(
            "app.modules.v1.idcard.service.mrz_scanner_runtime.run",
            new_callable=AsyncMock,
        ) as mrz_run,
        patch(
            "app.modules.v1.idcard.service.recognize_vlm_image",
            new_callable=AsyncMock,
        ) as vlm_run,
    ):
        response = client.post(
            PATH,
            data={"model": "removed-model"},
            files={"image": ("idcard.jpg", b"fake-image", "image/jpeg")},
        )

    assert response.status_code == 422
    assert response.json() == {
        "success": False,
        "data": None,
        "error": {
            "code": "VALIDATION_ERROR",
            "message": "Некорректные параметры запроса",
        },
    }
    crop_mrz.assert_not_awaited()
    paddle_run.assert_not_awaited()
    mrz_run.assert_not_awaited()
    vlm_run.assert_not_awaited()


def test_openapi_documents_models_flags_and_response_fields() -> None:
    schema = app.openapi()
    request_schema = schema["paths"][PATH]["post"]["requestBody"]["content"][
        "multipart/form-data"
    ]["schema"]
    component_name = request_schema["$ref"].rsplit("/", 1)[-1]
    body_schema = schema["components"]["schemas"][component_name]
    properties = body_schema["properties"]
    model_component = properties["model"]["$ref"].rsplit("/", 1)[-1]

    assert properties["image"]["type"] == "string"
    assert properties["image"]["format"] == "binary"
    assert body_schema["required"] == ["image"]
    assert properties["model"]["default"] == "paddle"
    assert schema["components"]["schemas"][model_component]["enum"] == [
        "paddle",
        "mrz",
        "qwen3-vl-4b",
        "qwen3-vl-8b",
    ]
    for flag in ("crop", "normalize", "normalization"):
        assert properties[flag]["type"] == "boolean"
        assert properties[flag]["default"] is True

    response_properties = schema["components"]["schemas"][
        "IdcardMrzDataSchema"
    ]["properties"]
    assert {
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
    }.issubset(response_properties)


def test_request_defaults_enable_crop_normalization_and_report(
    client: TestClient,
) -> None:
    response, mocks = _post_idcard(
        client,
        KAZ_TD1,
        model=None,
        crop=True,
    )

    assert response.status_code == 200, response.text
    assert response.json()["data"]["normalization"] == EMPTY_NORMALIZATION
    assert response.json()["data"]["meta"] == {
        "engine": "paddleocr",
        "model": "paddle",
        "duration_seconds": 0.25,
    }
    _assert_selected_runtime(mocks, model="paddle", crop=True)
