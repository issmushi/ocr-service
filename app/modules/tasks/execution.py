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
from app.modules.v1.mrz_api.schemas import UnifiedMrzResponseSchema
from app.modules.v1.mrz_api.service import recognize_mrz_upload
from app.modules.v1.passport.schemas import (
    PassportInternalResponseSchema,
    PassportInternationalResponseSchema,
)
from app.modules.v1.passport.service import (
    recognize_passport_internal_upload,
    recognize_passport_international_upload,
)
from app.modules.v1.russia.driver_license.schemas import (
    DriverLicenseResponseSchema,
)
from app.modules.v1.russia.driver_license.service import (
    recognize_driver_license_upload,
)
from app.modules.v1.russia.snils.schemas import SnilsResponseSchema
from app.modules.v1.russia.snils.service import recognize_snils_upload


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
        if operation in {
            TaskOperation.MRZ,
            TaskOperation.PASSPORT_MRZ,
            TaskOperation.IDCARD_MRZ,
        }:
            # Previously queued tasks may still store a model choice.
            options.pop("model", None)
            response = await recognize_mrz_upload(upload, **options)
            schema = UnifiedMrzResponseSchema
        elif operation == TaskOperation.PASSPORT_PAGE:
            # Old queued jobs may contain page/model/crop options. The internal
            # passport endpoint now always runs RussianDocsOCR on the image.
            response = await recognize_passport_internal_upload(upload)
            schema = PassportInternalResponseSchema
        elif operation == TaskOperation.PASSPORT_INTERNATIONAL:
            response = await recognize_passport_international_upload(upload)
            schema = PassportInternationalResponseSchema
        elif operation == TaskOperation.SNILS:
            response = await recognize_snils_upload(upload)
            schema = SnilsResponseSchema
        elif operation == TaskOperation.DRIVER_LICENSE:
            response = await recognize_driver_license_upload(upload)
            schema = DriverLicenseResponseSchema
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
