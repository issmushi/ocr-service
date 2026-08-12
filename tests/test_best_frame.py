from __future__ import annotations

import asyncio
from io import BytesIO

import cv2
import numpy as np
import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import settings
from app.core.errors import AppError
from app.core.infrastructure import storage
from app.main import app

PATH = "/v1/ocr/best-frame"


@pytest.fixture
def client():
    client = TestClient(app)
    yield client
    client.close()


@pytest.fixture(autouse=True)
def temporary_uploads(monkeypatch, tmp_path):
    def upload_path(task_id, suffix):
        return tmp_path / f"{task_id}{suffix}"

    monkeypatch.setattr(storage, "upload_path", upload_path)
    return tmp_path


@pytest.fixture(scope="module")
def frames():
    sharp = np.full((900, 1400, 3), 45, dtype=np.uint8)
    cv2.rectangle(sharp, (90, 80), (1310, 820), (238, 238, 238), -1)
    cv2.rectangle(sharp, (90, 80), (1310, 820), (20, 20, 20), 5)
    for row in range(180, 740, 70):
        cv2.putText(
            sharp,
            f"DOCUMENT TEXT {row}",
            (180, row),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (25, 25, 25),
            3,
            cv2.LINE_AA,
        )
    blurred = cv2.GaussianBlur(sharp, (31, 31), 0)

    def encode(image, extension):
        valid, encoded = cv2.imencode(extension, image)
        assert valid
        return encoded.tobytes()

    return {
        "sharp": encode(sharp, ".png"),
        "blurred": encode(blurred, ".png"),
        "jpeg": encode(sharp, ".jpg"),
    }


def upload(client, candidates):
    return client.post(
        PATH, files=[("image", candidate) for candidate in candidates]
    )


def assert_headers(response, index, mime):
    assert response.status_code == 200, response.text[:200]
    assert response.headers["content-type"] == mime
    assert response.headers["cache-control"] == "no-store"
    assert int(response.headers["x-selected-index"]) == index
    assert 0 <= float(response.headers["x-quality-score"]) <= 1


def test_real_quality_selection_prefers_sharp_frame_and_returns_original_bytes(
    client,
    frames,
    temporary_uploads,
):
    response = upload(
        client,
        [
            ("blurred.png", frames["blurred"], "image/png"),
            ("sharp.png", frames["sharp"], "image/png"),
        ],
    )

    assert_headers(response, 1, "image/png")
    assert response.content == frames["sharp"]
    assert list(temporary_uploads.iterdir()) == []


@pytest.mark.parametrize(
    ("key", "filename", "mime"),
    [
        ("sharp", "actual-png.jpg", "image/png"),
        ("jpeg", "actual-jpeg.png", "image/jpeg"),
    ],
)
def test_single_frame_mime_comes_from_image_not_filename_or_client_header(
    client,
    frames,
    temporary_uploads,
    key,
    filename,
    mime,
):
    response = upload(
        client, [(filename, frames[key], "application/octet-stream")]
    )

    assert_headers(response, 0, mime)
    assert response.content == frames[key]
    assert list(temporary_uploads.iterdir()) == []


def test_original_jpeg_exif_and_bytes_are_preserved(client, temporary_uploads):
    output = BytesIO()
    exif = Image.Exif()
    exif[274] = 6
    exif[270] = "Keep original image metadata"
    Image.new("RGB", (160, 100), (140, 160, 180)).save(
        output, format="JPEG", exif=exif, quality=83
    )
    original = output.getvalue()

    response = upload(client, [("camera.jpg", original, "image/png")])

    assert_headers(response, 0, "image/jpeg")
    assert response.content == original
    with Image.open(BytesIO(response.content)) as image:
        assert image.size == (160, 100)
        assert image.getexif()[274] == 6
    assert list(temporary_uploads.iterdir()) == []


def test_missing_required_image_is_rejected(client, temporary_uploads):
    response = client.post(PATH)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "NO_IMAGE"
    assert list(temporary_uploads.iterdir()) == []


@pytest.mark.parametrize(
    ("bad_content", "code"),
    [
        (b"", "EMPTY_IMAGE"),
        (b"not an image", "IMAGE_READ_ERROR"),
    ],
)
def test_invalid_later_frame_removes_already_stored_candidates(
    client,
    frames,
    temporary_uploads,
    bad_content,
    code,
):
    response = upload(
        client,
        [
            ("first.png", frames["sharp"], "image/png"),
            ("invalid.png", bad_content, "image/png"),
        ],
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == code
    assert list(temporary_uploads.iterdir()) == []


def test_unsupported_filename_is_rejected_without_leaving_files(
    client, frames, temporary_uploads
):
    response = upload(client, [("document.txt", frames["sharp"], "image/png")])

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "UNSUPPORTED_FORMAT"
    assert list(temporary_uploads.iterdir()) == []


def test_too_many_images_are_rejected_before_any_file_is_written(
    client, frames, temporary_uploads
):
    candidates = [
        (f"frame-{index}.png", frames["sharp"], "image/png")
        for index in range(settings.ocr_max_images_per_request + 1)
    ]
    response = upload(client, candidates)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert list(temporary_uploads.iterdir()) == []


def test_success_closes_all_uploads_and_removes_selected_file(
    frames, temporary_uploads
):
    from app.modules.v1.best_frame.service import select_best_frame_uploads

    uploads = [
        UploadFile(file=BytesIO(frames[key]), filename=f"{key}.png")
        for key in ("blurred", "sharp")
    ]
    response = asyncio.run(select_best_frame_uploads(uploads))

    assert response.body == frames["sharp"]
    assert all(image.file.closed for image in uploads)
    assert list(temporary_uploads.iterdir()) == []


def test_validation_error_closes_every_upload_including_unprocessed_ones(
    frames, temporary_uploads
):
    from app.modules.v1.best_frame.service import select_best_frame_uploads

    uploads = [
        UploadFile(file=BytesIO(content), filename=f"frame-{index}.png")
        for index, content in enumerate(
            (frames["sharp"], b"invalid", frames["blurred"])
        )
    ]
    with pytest.raises(AppError) as error:
        asyncio.run(select_best_frame_uploads(uploads))

    assert error.value.code.value == "IMAGE_READ_ERROR"
    assert all(image.file.closed for image in uploads)
    assert list(temporary_uploads.iterdir()) == []


def test_quality_analysis_failure_cleans_all_candidates(
    client,
    frames,
    temporary_uploads,
    monkeypatch,
):
    def fail_analysis(_path):
        raise RuntimeError("test quality failure")

    monkeypatch.setattr(
        "app.core.images.selection.assess_image_quality", fail_analysis
    )
    response = upload(
        client,
        [
            ("first.png", frames["sharp"], "image/png"),
            ("second.png", frames["blurred"], "image/png"),
        ],
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "IMAGE_READ_ERROR"
    assert list(temporary_uploads.iterdir()) == []


def test_openapi_exposes_only_a_binary_image_list_and_best_frame_tag():
    schema = app.openapi()
    operation = schema["paths"][PATH]["post"]
    assert operation["tags"] == ["Best frame"]
    body_ref = operation["requestBody"]["content"]["multipart/form-data"][
        "schema"
    ]["$ref"]
    body = schema["components"]["schemas"][body_ref.rsplit("/", 1)[-1]]
    assert set(body["properties"]) == {"image"}
    image = body["properties"]["image"]
    array = next(
        (
            option
            for option in image.get("anyOf", [])
            if option.get("type") == "array"
        ),
        image,
    )
    assert array["type"] == "array"
    assert array["items"] == {"type": "string", "format": "binary"}
    assert {"200", "400", "413", "422"} <= set(operation["responses"])
