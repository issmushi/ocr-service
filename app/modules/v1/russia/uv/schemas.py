"""Keep a preliminary UV result separate from physical authenticity."""

from typing import Literal

from pydantic import BaseModel, Field

from app.core.api_schemas import ErrorSchema


class UvCheckSchema(BaseModel):
    code: str
    status: Literal["passed", "failed", "not_checked"]
    message: str
    metrics: dict[str, float] = Field(default_factory=dict)
    region: tuple[float, float, float, float] | None = Field(
        default=None,
        description=(
            "Границы зоны [x1, y1, x2, y2] от 0 до 1 "
            "на изображении после rotation_degrees."
        ),
    )


class UvMetaSchema(BaseModel):
    engine: Literal["opencv"] = "opencv"
    algorithm: Literal["ru-passport-uv-heuristic-v1"] = (
        "ru-passport-uv-heuristic-v1"
    )
    experimental: Literal[True] = True
    duration_seconds: float = Field(ge=0)
    rotation_degrees: Literal[0, 90, 180, 270]
    analyzed_width: int
    analyzed_height: int


class UvDataSchema(BaseModel):
    is_authentic: None = Field(
        default=None,
        description=(
            "Всегда null: этот прототип не устанавливает подлинность."
        ),
    )
    uv_check_passed: bool | None = Field(
        description=(
            "true — тестовые UV-признаки найдены; false — есть отклонения; "
            "null — изображение нельзя оценить."
        ),
    )
    status: Literal["consistent", "suspicious", "inconclusive"]
    message: str
    checks: list[UvCheckSchema]
    limitations: list[str]
    meta: UvMetaSchema


class UvResponseSchema(BaseModel):
    success: bool
    data: UvDataSchema | None
    error: ErrorSchema | None
