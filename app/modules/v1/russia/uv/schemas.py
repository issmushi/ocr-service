"""Public response contract for passport authenticity screening."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.api_schemas import ErrorSchema


class UvCheckSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(
        description=(
            "Код проверки: resolution, page_geometry, exposure, sharpness, "
            "uv_color_signature, fluorescent_band, photo_zone_glow "
            "или background_glow."
        )
    )
    status: Literal["passed", "failed"] = Field(
        description="passed — условие выполнено; failed — не выполнено."
    )
    message: str = Field(description="Объяснение результата этой проверки.")


class UvMetaSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: Literal["opencv"] = Field(
        default="opencv", description="Движок обработки изображения."
    )
    model: Literal["ru-passport-uv-v1"] = Field(
        default="ru-passport-uv-v1",
        description="Профиль правил проверки UV-признаков паспорта РФ.",
    )
    duration_seconds: float = Field(
        ge=0,
        description=(
            "Время проверки файла, декодирования и анализа в секундах, "
            "без приёма HTTP-запроса и чтения загруженного файла."
        ),
    )


class UvDataSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_authentic: bool = Field(
        description=(
            "Результат автоматической проверки UV-признаков: true — "
            "все условия профиля выполнены; false — найдены отклонения. "
            "Это оценка по текущим правилам, не экспертное заключение. "
            "Если снимок нельзя оценить, API возвращает ошибку 422."
        ),
    )
    checks: list[UvCheckSchema] = Field(
        description="Проверки входного снимка и UV-признаков с причинами."
    )
    meta: UvMetaSchema = Field(
        description="Движок, профиль правил и время обработки."
    )


class UvResponseSchema(BaseModel):
    success: bool = Field(description="Запрос обработан успешно.")
    data: UvDataSchema | None = Field(
        description="Результат проверки; null при ошибке запроса."
    )
    error: ErrorSchema | None = Field(
        description="Код и описание ошибки; null при успешном запросе."
    )
