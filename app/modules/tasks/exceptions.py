from __future__ import annotations

from datetime import datetime


class QueueCapacityExceeded(Exception):
    def __init__(self, active_jobs: int, limit: int) -> None:
        self.active_jobs = active_jobs
        self.limit = limit
        super().__init__(
            f"Queue capacity exceeded: {active_jobs}/{limit} active jobs"
        )


class JobLeaseBusy(Exception):
    def __init__(
        self,
        job_id: str,
        lease_expires_at: datetime | None,
        retry_after_seconds: float,
    ) -> None:
        self.job_id = job_id
        self.lease_expires_at = lease_expires_at
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"Job {job_id} is owned by another worker")


class JobLeaseLost(Exception):
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(f"Worker lease lost for job {job_id}")
