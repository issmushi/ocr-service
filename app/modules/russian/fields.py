"""Prepare field detector previews inside the existing inference worker."""

from __future__ import annotations

import base64
from io import BytesIO
from time import perf_counter
from typing import Any, Literal

import numpy as np
from PIL import Image

from app.core.images.decoding import decode_image
from app.ocr_engine.russian.fields import detect_passport_fields

FIELD_LABELS = {
    "Face": "Фотография",
    "Signature": "Подпись",
    "Driver_class": "Категория водительского удостоверения",
    "Last_name_ru": "Фамилия",
    "First_name_ru": "Имя",
    "Middle_name_ru": "Отчество",
    "Last_name_en": "Фамилия латиницей",
    "First_name_en": "Имя латиницей",
    "Middle_name_en": "Отчество латиницей",
    "Licence_number": "Серия и номер паспорта",
    "Birth_date": "Дата рождения",
    "Birth_place_ru": "Место рождения",
    "Birth_place_en": "Место рождения латиницей",
    "Sex_ru": "Пол",
    "Sex_en": "Пол латиницей",
    "Issue_date": "Дата выдачи",
    "Expiration_date": "Срок действия",
    "Issue_organisation_code": "Код подразделения",
    "Issue_organization_ru": "Кем выдан",
    "Issue_organization_en": "Кем выдан латиницей",
    "Living_region_ru": "Место жительства",
    "Living_region_en": "Место жительства латиницей",
}


def _preview(
    image: np.ndarray, *, image_format: Literal["JPEG", "PNG"] = "JPEG"
) -> dict[str, Any]:
    output = BytesIO()
    options = (
        {"quality": 94} if image_format == "JPEG" else {"compress_level": 3}
    )
    Image.fromarray(image).save(output, format=image_format, **options)
    height, width = image.shape[:2]
    return {
        "image": f"data:image/{image_format.lower()};base64,"
        + base64.b64encode(output.getvalue()).decode("ascii"),
        "width": width,
        "height": height,
    }


def detect_passport_fields_bytes(
    content: bytes,
    pipeline: Any,
    *,
    crop_document: bool = True,
    padding: int = 4,
) -> dict[str, Any]:
    """Decode once and encode crops before transferring them out of worker."""
    with decode_image(content) as decoded:
        result = detect_passport_fields(
            np.asarray(decoded),
            pipeline=pipeline,
            crop_document=crop_document,
            padding=padding,
        )
    started = perf_counter()
    fields = [
        {
            "name": field["name"],
            "label": FIELD_LABELS.get(field["name"], field["name"]),
            "bbox": field["bbox"],
            "detection_bbox": field["detection_bbox"],
            "confidence": field["confidence"],
            "rotation_degrees": field["rotation_degrees"],
            **_preview(field["image"], image_format="PNG"),
        }
        for field in result["fields"]
    ]
    document = {
        **_preview(result["document_image"]),
        "detected": result["document_detected"],
        "rotation_degrees": result["rotation_degrees"],
        "orientation_confidence": result["orientation_confidence"],
    }
    return {
        "document": document,
        "fields": fields,
        "timings": {
            **result["timings"],
            "encoding": perf_counter() - started,
        },
    }
