from __future__ import annotations

from enum import StrEnum


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class TaskOperation(StrEnum):
    MRZ = "mrz"
    PASSPORT_MRZ = "passport_mrz"
    PASSPORT_PAGE = "passport_page"
    PASSPORT_INTERNATIONAL = "passport_international"
    EGRN = "egrn"
    SNILS = "snils"
    DRIVER_LICENSE = "driver_license"
    IDCARD_MRZ = "idcard_mrz"
    FULL_V1 = "full_v1"
    CROP_DOCUMENT = "crop_document"
    # Retained only to read statuses/results of historical tasks.
    CROP_DOCUMENT_FIELDS = "crop_document_fields"
    CROP_EGRN = "crop_egrn"
    CROP_EGRN_FIELDS = "crop_egrn_fields"
    CROP_MRZ = "crop_mrz"


ACTIVE_STATUSES = (
    JobStatus.QUEUED.value,
    JobStatus.PROCESSING.value,
)
TERMINAL_STATUSES = (
    JobStatus.DONE.value,
    JobStatus.FAILED.value,
    JobStatus.TIMEOUT.value,
    JobStatus.CANCELLED.value,
)
