"""Read the exact field previews returned by the passport detector."""

from __future__ import annotations

import base64
from math import isfinite
from time import perf_counter
from typing import Any

from app.core.config import settings
from app.core.errors import AppError, ErrorCode, error_payload
from app.modules.paddle.runtime import paddle_runtime
from app.modules.v1.crop.field_schemas import FieldOcrModel
from app.modules.vlm.client import recognize_vlm_image

ENGINE_BY_MODEL = {
    FieldOcrModel.PADDLE: "paddleocr",
    FieldOcrModel.QWEN3_VL_4B: "qwen3-vl",
    FieldOcrModel.QWEN3_VL_8B: "qwen3-vl",
}
NON_TEXT_FIELDS = {"Face", "Signature"}


def _text_result(
    result: dict[str, Any], model: FieldOcrModel
) -> tuple[str, float | None, str | None]:
    if model == FieldOcrModel.PADDLE:
        text = result.get("text")
        confidence = result.get("confidence")
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise AppError(ErrorCode.OCR_FAILED, "Некорректный ответ OCR")
    else:
        items = result.get("items")
        if not isinstance(items, list) or any(
            not isinstance(item, dict) or not isinstance(item.get("text"), str)
            for item in items
        ):
            raise AppError(ErrorCode.OCR_FAILED, "Некорректный ответ OCR")
        text = "\n".join(item["text"] for item in items)
        confidence = None
    if not isinstance(text, str):
        raise AppError(ErrorCode.OCR_FAILED, "Некорректный ответ OCR")
    device = result.get("device")
    return text, confidence, device if isinstance(device, str) else None


async def recognize_field_previews(
    fields: list[dict[str, Any]], model: FieldOcrModel
) -> dict[str, Any] | None:
    """Keep crop order and duplicates; isolate failures to each field.

    Calls are sequential because the local model runtimes serialize inference.
    Every OCR input is the displayed JPEG, including its padding and rotation.
    """
    if model == FieldOcrModel.NONE:
        for field in fields:
            field["ocr"] = None
        return None

    started = perf_counter()
    processed = failed = 0
    engine = ENGINE_BY_MODEL[model]
    for index, field in enumerate(fields):
        ocr: dict[str, Any] = {
            "model": model.value,
            "engine": engine,
            "device": None,
            "status": "skipped",
            "text": None,
            "confidence": None,
            "duration_seconds": 0.0,
            "error": None,
        }
        field["ocr"] = ocr
        if field["name"] in NON_TEXT_FIELDS:
            continue

        field_started = perf_counter()
        processed += 1
        try:
            content = base64.b64decode(field["image"].split(",", 1)[1])
            if model == FieldOcrModel.PADDLE:
                result = await paddle_runtime.run(
                    "field_text",
                    content,
                    timeout_seconds=(
                        settings.paddle_ocr_full_hard_timeout_seconds
                    ),
                )
            else:
                result = await recognize_vlm_image(
                    content,
                    model=model.value,
                    filename=f"field-{index + 1}.jpg",
                    content_type="image/jpeg",
                    task="ocr",
                    report_runtime=True,
                )
            text, confidence, device = _text_result(result, model)
            ocr.update(
                status="success",
                text=text,
                confidence=confidence,
                device=device,
            )
        except AppError as error:
            failed += 1
            ocr.update(
                status="error", error=error_payload(error.code, error.message)
            )
        finally:
            ocr["duration_seconds"] = perf_counter() - field_started

    return {
        "model": model.value,
        "engine": engine,
        "duration_seconds": perf_counter() - started,
        "processed_fields": processed,
        "failed_fields": failed,
    }
