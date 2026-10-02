from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.images.preprocessing import PreparedImage
from app.main import app
from app.modules.v1.passport.pipeline import (
    normalize_passport_mrz,
    normalize_passport_mrz_from_text_result,
    process_passport_bytes,
    process_passport_bytes_with_mrz_scanner,
)

PATH = "/v1/ocr/passport/mrz"
RAW_MRZ = [
    "P<ARMHAYRAPETYAN<<AR4M<<",
    "44123456784RM9001158N3001156<<<<<<<<<<<<<<",
]
EMPTY_REPORT = {
    "corrections_count": 0,
    "replacements_count": 0,
    "insertions_count": 0,
    "corrections": [],
    "issues": [],
}
TEXT_ADAPTER_ENGINES = ("paddle", "mrz", "qwen3-vl-4b", "qwen3-vl-8b")


def test_openapi_exposes_normalization_switches_without_model_selection():
    schema = app.openapi()
    reference = schema["paths"][PATH]["post"]["requestBody"]["content"][
        "multipart/form-data"
    ]["schema"]["$ref"]
    body = schema["components"]["schemas"][reference.rsplit("/", 1)[-1]]
    assert set(body["properties"]) == {
        "image",
        "crop",
        "normalize",
        "normalization",
    }
    for field in ("normalize", "normalization"):
        assert body["properties"][field]["type"] == "boolean"
        assert body["properties"][field]["default"] is True
        assert field not in body["required"]


def test_invalid_normalize_flag_is_rejected_before_ocr():
    with patch(
        "app.modules.v1.passport.router.recognize_passport_upload",
        AsyncMock(),
    ) as recognize:
        response = TestClient(app).post(
            PATH,
            data={"normalize": "sometimes"},
            files={"image": ("passport.jpg", b"image", "image/jpeg")},
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    recognize.assert_not_awaited()


def test_disabled_normalization_never_calls_header_or_country_rules():
    rows = ["ignored", " \t<armtest0v<<aram<<\n", "  " + RAW_MRZ[1].lower()]
    with patch(
        "app.modules.v1.passport.pipeline.normalize_passport_mrz_header",
        side_effect=AssertionError("normalization must not run"),
    ):
        result = normalize_passport_mrz(rows, normalize=False)

    assert result["mrz"] == ["<ARMTEST0V<<ARAM<<", RAW_MRZ[1]]
    assert result["raw_mrz"] == result["mrz"]
    assert result["diagnostics"] == {
        "header_normalization": None,
        "lower_line_tail_restoration": None,
        "normalization": EMPTY_REPORT,
    }


@pytest.mark.parametrize("engine", TEXT_ADAPTER_ENGINES)
def test_text_adapter_keeps_ocr_characters_with_normalization_disabled(engine):
    result = normalize_passport_mrz_from_text_result(
        {"items": [{"text": "\n".join(RAW_MRZ)}]},
        engine=engine,
        normalize=False,
    )
    assert result["mrz"] == RAW_MRZ
    assert result["diagnostics"]["normalization"] == EMPTY_REPORT


@pytest.mark.parametrize("engine", ("paddle", "mrz"))
def test_bytes_pipelines_propagate_normalization_switch(engine):
    image = np.zeros((60, 300, 3), dtype=np.uint8)
    recognition = SimpleNamespace(
        lines=RAW_MRZ,
        detected_regions_count=2,
        selected_regions_count=2,
    )
    scanner = Mock(
        return_value={
            "mrz_polygon": [[0, 0], [300, 0], [300, 60], [0, 60]],
            "mrz_texts": RAW_MRZ,
        }
    )
    with (
        patch(
            "app.modules.v1.passport.pipeline.decode_image_bytes",
            return_value=image,
        ),
        patch(
            "app.modules.v1.passport.pipeline.prepare_document_image",
            return_value=PreparedImage(image=image, metadata={}),
        ),
        patch(
            "app.modules.v1.passport.pipeline.recognize_raw_passport_mrz",
            return_value=recognition,
        ),
        patch(
            "app.modules.v1.passport.pipeline.normalize_passport_mrz_header",
            side_effect=AssertionError("normalization must not run"),
        ),
    ):
        result = (
            process_passport_bytes(b"image", ocr=object(), normalize=False)
            if engine == "paddle"
            else process_passport_bytes_with_mrz_scanner(
                b"image",
                scanner=scanner,
                normalize=False,
            )
        )
    assert result["mrz"] == RAW_MRZ
    assert result["diagnostics"]["normalization"] == EMPTY_REPORT
    if engine == "mrz":
        scanner.assert_called_once_with(
            image, do_center_crop=False, do_postprocess=False
        )


@pytest.mark.parametrize("crop", (False, True))
@pytest.mark.parametrize("normalize", (False, True))
@pytest.mark.parametrize("normalization", (False, True))
def test_scanner_normalization_and_report_switches_are_independent(
    crop,
    normalize,
    normalization,
):
    raw_text = {"items": [{"text": "\n".join(RAW_MRZ)}]}

    async def run(operation, content, **kwargs):
        if operation == "mrz_crop_text":
            assert "normalize" not in kwargs
            return raw_text
        assert operation == "passport_v1"
        assert kwargs.get("normalize", True) == normalize
        return normalize_passport_mrz(
            RAW_MRZ, normalize=kwargs.get("normalize", True)
        )

    with (
        patch(
            "app.modules.v1.passport.service.crop_mrz_bytes",
            AsyncMock(return_value=b"crop"),
        ) as crop_mrz,
        patch(
            "app.modules.v1.passport.service.mrz_scanner_runtime.run",
            AsyncMock(side_effect=run),
        ) as mrz_run,
    ):
        response = TestClient(app).post(
            PATH,
            data={
                "crop": str(crop).lower(),
                "normalize": str(normalize).lower(),
                "normalization": str(normalization).lower(),
            },
            files={"image": ("passport.jpg", b"image", "image/jpeg")},
        )

    assert response.status_code == 200, response.text
    options = {} if crop or normalize else {"normalize": False}
    mrz_run.assert_awaited_once_with(
        "mrz_crop_text" if crop else "passport_v1",
        b"crop" if crop else b"image",
        timeout_seconds=settings.mrz_scanner_document_hard_timeout_seconds,
        **options,
    )
    if crop:
        crop_mrz.assert_awaited_once_with(b"image")
    else:
        crop_mrz.assert_not_awaited()
    data = response.json()["data"]
    assert data["meta"]["engine"] == "mrz-scanner"
    assert data["meta"]["model"] == "mrz"
    if normalize:
        assert data["mrz"] == [
            "P<ARMHAYRAPETYAN<<ARAM".ljust(44, "<"),
            "AA12345678ARM9001158M3001156<<<<<<<<<<<<<<06",
        ]
        assert data["passport_number"] == "AA1234567"
        if normalization:
            assert data["normalization"]["corrections_count"] > 0
    else:
        assert data["mrz"] == RAW_MRZ
        assert data["name"] is None
        assert data["passport_number"] is None
        assert data["mrz_valid"] is None
        if normalization:
            assert data["normalization"] == EMPTY_REPORT
    if not normalization:
        assert data["normalization"] is None


@pytest.mark.parametrize("crop", (False, True))
def test_disabled_normalization_does_not_repair_unknown_header(crop):
    raw = ["<ARMHAYRAPETYAN<<ARAM".ljust(43, "<"), RAW_MRZ[1]]
    raw_text = {"items": [{"text": "\n".join(raw)}]}

    async def run(operation, content, **kwargs):
        if operation == "mrz_crop_text":
            return raw_text
        return normalize_passport_mrz(
            raw, normalize=kwargs.get("normalize", True)
        )

    with (
        patch(
            "app.modules.v1.passport.service.crop_mrz_bytes",
            AsyncMock(return_value=b"crop"),
        ),
        patch(
            "app.modules.v1.passport.service.mrz_scanner_runtime.run",
            AsyncMock(side_effect=run),
        ),
    ):
        response = TestClient(app).post(
            PATH,
            data={
                "crop": str(crop).lower(),
                "normalize": "false",
            },
            files={"image": ("passport.jpg", b"image", "image/jpeg")},
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "OCR_FAILED"
