import re
from typing import Any

from fastapi import UploadFile

from app.core.errors import AppError, ErrorCode, api_success_response
from app.modules.v1.crop.egrn import read_egrn_fields_upload
from app.modules.v1.crop.egrn_layout import FIELD_LABELS
from app.modules.v1.crop.egrn_schemas import EgrnOcrModel
from app.modules.v1.russia.egrn.schemas import EgrnData, EgrnModel


def field_value(name: str, text: str) -> str | None:
    value = " ".join(text.split())
    if name == "application_number":
        match = re.fullmatch(
            r"(?:Заявление\s*)?(?:№\s*[-°]?\s*)?([0-9]+)",
            value,
            flags=re.IGNORECASE,
        )
        return match[1] if match else None
    if name == "actuality_date":
        match = re.fullmatch(
            r"(?:Данные актуальны на\s+)?([0-9]{2}\.[0-9]{2}\.[0-9]{4})",
            value,
            flags=re.IGNORECASE,
        )
        return match[1] if match else None
    return value or None


async def recognize_egrn_upload(
    image: UploadFile, *, ocr_model: EgrnModel = EgrnModel.CYRILLIC
) -> dict[str, Any]:
    result = await read_egrn_fields_upload(
        image, ocr_model=EgrnOcrModel(ocr_model.value), include_page=False
    )
    values = dict.fromkeys(FIELD_LABELS)
    errors = []
    for field in result["fields"]:
        ocr = field["ocr"]
        if ocr["error"]:
            errors.append(ocr["error"])
        elif ocr["status"] == "success":
            values[field["name"]] = field_value(field["name"], ocr["text"])
    if not any(values.values()):
        if errors and len(errors) == len(result["fields"]):
            raise AppError(ErrorCode(errors[0]["code"]), errors[0]["message"])
        raise AppError(ErrorCode.OCR_FAILED, "Не удалось прочитать поля ЕГРН")
    data = EgrnData.model_validate(
        {
            **values,
            "meta": {
                "model": ocr_model,
                "duration_seconds": result["meta"]["duration_seconds"],
            },
        }
    )
    return api_success_response(data.model_dump(mode="json"))
