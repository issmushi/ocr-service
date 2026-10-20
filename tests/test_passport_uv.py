"""Protocol and failure-mode checks, not an authenticity accuracy benchmark."""

from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageFilter

from app.main import app
from app.modules.v1.russia.uv.analysis import analyze_uv

PATH = "/v1/ocr/passport/uv-test"


def encode(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def synthetic_pattern(*, band=True, bright_background=False) -> Image.Image:
    # An invented arrangement of coloured shapes, without personal data,
    # deliberately demonstrates that these heuristics cannot authenticate.
    rng = np.random.default_rng(42)
    rgb = np.empty((700, 1000, 3), dtype=np.uint8)
    rgb[:] = (20, 25, 50)
    if bright_background:
        rgb[:] = (125, 140, 170)
    rgb[110:430, 40:340] = (60, 110, 230)
    if band:
        for row in (430, 450, 470):
            for col in range(20, 970, 15):
                rgb[row : row + 8, col : col + 10] = (160, 220, 80)
    noise = rng.integers(-6, 7, rgb.shape, dtype=np.int16)
    rgb = np.clip(rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb)


@pytest.fixture
def client():
    client = TestClient(app)
    yield client
    client.close()


def test_matching_colours_never_claim_physical_authenticity(client):
    response = client.post(
        PATH,
        files={"image": ("synthetic.png", encode(synthetic_pattern()))},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["success"] is True
    data = result["data"]
    assert data["uv_check_passed"] is True
    assert data["status"] == "consistent"
    assert data["is_authentic"] is None
    assert data["limitations"]
    assert data["meta"]["experimental"] is True
    assert data["meta"]["duration_seconds"] >= 0


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_right_angle_rotation_preserves_result(turns):
    rgb = np.asarray(synthetic_pattern())
    rotated = Image.fromarray(np.rot90(rgb, k=turns))
    data = analyze_uv(encode(rotated))
    assert data.uv_check_passed is True
    assert data.meta.rotation_degrees == turns * 90


@pytest.mark.parametrize("failure", ["band", "background"])
def test_visible_deviation_is_suspicious_not_a_fake_verdict(failure):
    data = analyze_uv(
        encode(
            synthetic_pattern(
                band=failure != "band",
                bright_background=failure == "background",
            )
        )
    )
    assert data.uv_check_passed is False
    assert data.status == "suspicious"
    assert data.is_authentic is None
    check_code = "fluorescent_band" if failure == "band" else "background_glow"
    check = next(check for check in data.checks if check.code == check_code)
    assert check.status == "failed"


@pytest.mark.parametrize(
    "kind",
    [
        "dark",
        "white",
        "gray",
        "small",
        "blurred",
        "wrong_geometry",
    ],
)
def test_unusable_input_is_inconclusive_and_skips_uv_checks(kind):
    image = synthetic_pattern()
    if kind == "dark":
        image = Image.new("RGB", image.size, "black")
    elif kind == "white":
        image = Image.new("RGB", image.size, "white")
    elif kind == "gray":
        image = image.convert("L")
    elif kind == "small":
        image = image.resize((100, 70))
    elif kind == "blurred":
        image = image.filter(ImageFilter.GaussianBlur(radius=15))
    else:
        image = image.resize((1000, 300))
    data = analyze_uv(encode(image))
    assert data.status == "inconclusive"
    assert data.uv_check_passed is None
    assert data.is_authentic is None
    assert all(check.status == "not_checked" for check in data.checks[-3:])


def test_a_solid_green_rectangle_is_not_a_pattern():
    rgb = np.asarray(synthetic_pattern(band=False)).copy()
    rgb[430:480, 20:980] = (160, 220, 80)
    data = analyze_uv(encode(Image.fromarray(rgb)))
    assert data.uv_check_passed is False
    check = next(c for c in data.checks if c.code == "fluorescent_band")
    assert check.metrics["bright_components"] == 0


@pytest.mark.parametrize(
    ("files", "status", "code"),
    [
        ({}, 400, "NO_IMAGE"),
        ({"image": ("empty.png", b"")}, 400, "EMPTY_IMAGE"),
        ({"image": ("bad.png", b"not-an-image")}, 400, "IMAGE_READ_ERROR"),
        ({"image": ("bad.pdf", b"pdf")}, 400, "UNSUPPORTED_FORMAT"),
    ],
)
def test_upload_errors(client, files, status, code):
    response = client.post(PATH, files=files)
    assert response.status_code == status
    assert response.json()["error"]["code"] == code


def test_upload_size_limit(client, monkeypatch):
    monkeypatch.setattr(
        "app.modules.v1.russia.uv.router.settings",
        SimpleNamespace(ocr_max_image_size_bytes=20),
    )
    response = client.post(PATH, files={"image": ("large.png", b"x" * 21)})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "IMAGE_TOO_LARGE"


def test_animation_is_not_silently_reduced_to_one_frame(client):
    output = BytesIO()
    Image.new("RGB", (600, 400), "blue").save(
        output,
        "PNG",
        save_all=True,
        append_images=[Image.new("RGB", (600, 400), "green")],
    )
    response = client.post(
        PATH, files={"image": ("animated.png", output.getvalue())}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "IMAGE_READ_ERROR"


def test_swagger_accepts_one_photo_and_exposes_null_authenticity():
    schema = app.openapi()
    route = schema["paths"][PATH]["post"]
    assert route["tags"] == ["Russia"]
    ref = route["requestBody"]["content"]["multipart/form-data"]["schema"]
    body = schema["components"]["schemas"][ref["$ref"].split("/")[-1]]
    assert set(body["properties"]) == {"image"}
    assert body["required"] == ["image"]
    data = schema["components"]["schemas"]["UvDataSchema"]
    assert data["properties"]["is_authentic"]["type"] == "null"
