from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

from fastapi import UploadFile
from starlette.datastructures import Headers

from app.modules.tasks.enums import TaskOperation
from app.modules.v1.crop.egrn import crop_egrn_bytes, crop_egrn_fields_upload
from app.modules.v1.crop.egrn_schemas import (
    EgrnFieldCropResponse,
    EgrnOcrModel,
)
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
from app.modules.v1.russia.egrn.schemas import EgrnModel, EgrnResponse
from app.modules.v1.russia.egrn.service import recognize_egrn_upload
from app.modules.v1.russia.snils.schemas import SnilsResponseSchema
from app.modules.v1.russia.snils.service import recognize_snils_upload


async def execute_task_operation(
    job: dict[str, Any],
    content: bytes,
) -> dict[str, Any]:
    """Execute the same V1 service and persist its public JSON or image result.

    Each service retains its engine-specific timeout. The task worker keeps
    renewing its lease while an OCR or crop operation is running.
    """
    operation = TaskOperation(job["mode"])
    options = dict(job.get("options") or {})
    image_operations = {
        TaskOperation.CROP_DOCUMENT: (crop_document_image_bytes, "jpeg"),
        TaskOperation.CROP_MRZ: (crop_mrz_bytes, "jpeg"),
        TaskOperation.CROP_EGRN: (crop_egrn_bytes, "png"),
    }
    if operation in image_operations:
        crop, image_format = image_operations[operation]
        image = await crop(content, **options)
        return {
            "format": image_format,
            "content_base64": base64.b64encode(image).decode("ascii"),
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
            # Old model/crop options must not alter direct scanner input.
            options.pop("model", None)
            options.pop("crop", None)
            response = await recognize_mrz_upload(upload, **options)
            schema = UnifiedMrzResponseSchema
        elif operation == TaskOperation.PASSPORT_PAGE:
            # Old queued jobs may contain page/model/crop options. The internal
            # passport endpoint now always runs RussianDocsOCR on the image.
            response = await recognize_passport_internal_upload(
                upload, fields=options.get("fields")
            )
            schema = PassportInternalResponseSchema
        elif operation == TaskOperation.PASSPORT_INTERNATIONAL:
            response = await recognize_passport_international_upload(
                upload, fields=options.get("fields")
            )
            schema = PassportInternationalResponseSchema
        elif operation == TaskOperation.EGRN:
            model = EgrnModel(options.get("ocr_model", EgrnModel.CYRILLIC))
            response = await recognize_egrn_upload(
                upload, ocr_model=model, fields=options.get("fields")
            )
            schema = EgrnResponse
        elif operation == TaskOperation.SNILS:
            response = await recognize_snils_upload(
                upload, fields=options.get("fields")
            )
            schema = SnilsResponseSchema
        elif operation == TaskOperation.DRIVER_LICENSE:
            response = await recognize_driver_license_upload(
                upload, fields=options.get("fields")
            )
            schema = DriverLicenseResponseSchema
        elif operation == TaskOperation.FULL_V1:
            if "model" in options:
                options["model"] = FullOcrModel(options["model"])
            response = await recognize_full_upload(upload, **options)
            schema = FullOcrResponseSchema
        elif operation == TaskOperation.CROP_EGRN_FIELDS:
            options["ocr_model"] = EgrnOcrModel(
                options.get("ocr_model", EgrnOcrModel.NONE)
            )
            response = await crop_egrn_fields_upload(upload, **options)
            schema = EgrnFieldCropResponse
        else:  # Retired operations cannot start new recognition.
            raise ValueError(f"Unsupported task operation: {operation}")
        return {
            "format": "json",
            "response": schema.model_validate(response).model_dump(
                mode="json",
                exclude_unset=operation
                not in {
                    TaskOperation.FULL_V1,
                    TaskOperation.CROP_EGRN_FIELDS,
                },
            ),
        }
    finally:
        await upload.close()
