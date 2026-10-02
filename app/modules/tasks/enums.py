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
    SNILS = "snils"
    DRIVER_LICENSE = "driver_license"
    IDCARD_MRZ = "idcard_mrz"
    FULL_V1 = "full_v1"
    CROP_DOCUMENT = "crop_document"
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
