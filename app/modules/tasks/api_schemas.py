from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.core.api_schemas import ErrorSchema
from app.modules.tasks.enums import JobStatus, TaskOperation


class TaskCreatedDataSchema(BaseModel):
    task_id: str = Field(description="UUID созданной задачи.")
    status: Literal["queued"]


class TaskCreatedResponseSchema(BaseModel):
    success: bool
    data: TaskCreatedDataSchema | None = None
    error: ErrorSchema | None = None


class TaskStatusDataSchema(BaseModel):
    task_id: str = Field(description="UUID задачи.")
    status: JobStatus
    operation: TaskOperation
    created_at: str | None = Field(description="Время создания в ISO 8601.")
    started_at: str | None = Field(
        description="Время начала обработки в ISO 8601."
    )
    finished_at: str | None = Field(description="Время завершения в ISO 8601.")
    error: ErrorSchema | None = Field(
        description="Ошибка обработки задачи, если она возникла."
    )


class TaskStatusResponseSchema(BaseModel):
    success: bool
    data: TaskStatusDataSchema | None = None
    error: ErrorSchema | None = None


class TaskCancelledDataSchema(BaseModel):
    task_id: str
    status: JobStatus
    cancelled: bool = Field(
        description="Изменил ли этот запрос статус задачи на cancelled."
    )


class TaskCancelledResponseSchema(BaseModel):
    success: bool
    data: TaskCancelledDataSchema | None = None
    error: ErrorSchema | None = None
