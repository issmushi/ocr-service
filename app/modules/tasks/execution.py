from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

from fastapi import UploadFile
from starlette.datastructures import Headers

from app.modules.tasks.enums import TaskOperation
from app.modules.v1.crop.service import (
    crop_document_image_bytes,
    crop_mrz_bytes,
)
from app.modules.v1.full.schemas import FullOcrModel, FullOcrResponseSchema
from app.modules.v1.full.service import recognize_full_upload
from app.modules.v1.idcard.schemas import (
    IdcardMrzResponseSchema,
    IdcardOcrModel,
)
from app.modules.v1.idcard.service import recognize_idcard_upload
from app.modules.v1.passport.schemas import (
    PassportMrzResponseSchema,
    PassportOcrModel,
    PassportPage,
    PassportPageOcrModel,
    PassportPageResponseSchema,
)
from app.modules.v1.passport.service import (
    recognize_passport_page_upload,
    recognize_passport_upload,
)


async def execute_task_operation(
    job: dict[str, Any],
    content: bytes,
) -> dict[str, Any]:
    """Execute the same V1 service and persist its public JSON or JPEG result.

    Each service retains its engine-specific timeout. The task worker keeps
    renewing its lease while an OCR or crop operation is running.
    """
    operation = TaskOperation(job["mode"])
    options = dict(job.get("options") or {})
    if operation in {TaskOperation.CROP_DOCUMENT, TaskOperation.CROP_MRZ}:
        crop = (
            crop_document_image_bytes
            if operation == TaskOperation.CROP_DOCUMENT
            else crop_mrz_bytes
        )
        jpeg = await crop(content, **options)
        return {
            "format": "jpeg",
            "content_base64": base64.b64encode(jpeg).decode("ascii"),
        }

    upload = UploadFile(
        file=BytesIO(content),
        filename=job["filename"],
        size=len(content),
        headers=Headers({"content-type": job["content_type"]}),
    )
    try:
        if operation == TaskOperation.PASSPORT_MRZ:
            if "model" in options:
                options["model"] = PassportOcrModel(options["model"])
            response = await recognize_passport_upload(upload, **options)
            schema = PassportMrzResponseSchema
        elif operation == TaskOperation.PASSPORT_PAGE:
            if "model" in options:
                options["model"] = PassportPageOcrModel(options["model"])
            if "page" in options:
                options["page"] = PassportPage(options["page"])
            response = await recognize_passport_page_upload(upload, **options)
            schema = PassportPageResponseSchema
        elif operation == TaskOperation.IDCARD_MRZ:
            if "model" in options:
                options["model"] = IdcardOcrModel(options["model"])
            response = await recognize_idcard_upload(upload, **options)
            schema = IdcardMrzResponseSchema
        elif operation == TaskOperation.FULL_V1:
            if "model" in options:
                options["model"] = FullOcrModel(options["model"])
            response = await recognize_full_upload(upload, **options)
            schema = FullOcrResponseSchema
        else:  # pragma: no cover - every TaskOperation has a handler above.
            raise ValueError(f"Unsupported task operation: {operation}")
        return {
            "format": "json",
            "response": schema.model_validate(response).model_dump(
                mode="json"
            ),
        }
    finally:
        await upload.close()
