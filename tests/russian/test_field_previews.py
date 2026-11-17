from __future__ import annotations

import base64
from io import BytesIO
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from app.modules.russian.fields import detect_passport_fields_bytes
from app.modules.russian.inference_worker import russian_inference_worker

WORKER = "app.modules.russian.fields"


def image_bytes(image: Image.Image | None = None, **kwargs) -> bytes:
    output = BytesIO()
    if image is None:
        image = Image.new("RGB", (80, 60), "white")
    image.save(output, format="PNG", **kwargs)
    return output.getvalue()


def detector_result() -> dict:
    image = np.full((60, 80, 3), (220, 15, 35), dtype=np.uint8)
    return {
        "document_image": image,
        "document_detected": True,
        "rotation_degrees": 0,
        "orientation_confidence": 0.98,
        "fields": [
            {
                "name": "Last_name_ru",
                "bbox": [6, 6, 54, 24],
                "detection_bbox": [10, 10, 50, 20],
                "confidence": 0.95,
                "rotation_degrees": 0,
                "image": image[6:24, 6:54],
            }
        ],
        "timings": {"orientation": 0.01, "document": 0.02, "fields": 0.03},
    }


def test_preview_encoding_preserves_rgb_and_uses_exif_orientation():
    image = Image.new("RGB", (80, 60), "white")
    exif = Image.Exif()
    exif[274] = 6
    pipeline = Mock()
    detected = detector_result()
    # Detailed pixels expose JPEG loss even when dimensions and colors match.
    expected = np.random.default_rng(0).integers(
        0, 256, size=(18, 48, 3), dtype=np.uint8
    )
    detected["fields"][0]["image"] = expected
    with patch(
        f"{WORKER}.detect_passport_fields", return_value=detected
    ) as detect:
        result = detect_passport_fields_bytes(
            image_bytes(image, exif=exif), pipeline, padding=8
        )

    assert detect.call_args.args[0].shape == (80, 60, 3)
    assert detect.call_args.kwargs["pipeline"] is pipeline
    assert detect.call_args.kwargs["padding"] == 8
    encoded = result["fields"][0]["image"].split(",", 1)[1]
    with Image.open(BytesIO(base64.b64decode(encoded))) as preview:
        assert preview.format == "PNG"
        assert preview.size == (48, 18)
        np.testing.assert_array_equal(np.asarray(preview), expected)


def test_worker_reuses_existing_pipeline_for_field_detection():
    with (
        patch(
            "app.modules.russian.inference_worker.serve_inference_worker"
        ) as serve,
        patch(f"{WORKER}.detect_passport_fields_bytes") as detect,
        patch(
            "app.modules.russian.passport.recognize_passport_text_bytes"
        ) as ocr,
    ):
        russian_inference_worker(Mock(), "cpu")
        dispatch = serve.call_args.args[2]
        pipeline = Mock()
        dispatch(
            pipeline,
            "passport_fields",
            (b"image",),
            {"crop_document": False, "padding": 8},
        )

    detect.assert_called_once_with(
        b"image", pipeline=pipeline, crop_document=False, padding=8
    )
    ocr.assert_not_called()
