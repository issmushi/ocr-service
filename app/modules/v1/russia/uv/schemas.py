"""Keep a preliminary UV result separate from physical authenticity."""

from typing import Literal

from pydantic import BaseModel, Field

from app.core.api_schemas import ErrorSchema


class UvCheckSchema(BaseModel):
    code: str = Field(
        description=(
            "Код проверки: resolution, page_geometry, exposure, sharpness, "
            "uv_color_signature, fluorescent_band, photo_zone_glow "
            "или background_glow."
        )
    )
    status: Literal["passed", "failed", "not_checked"] = Field(
        description=(
            "passed — условие выполнено; failed — не выполнено; "
            "not_checked — проверка пропущена из-за неподходящего снимка."
        )
    )
    message: str = Field(description="Объяснение результата этой проверки.")
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Измеренные признаки. fraction и coverage — доли от 0 до 1; "
            "brightness — яркость от 0 до 255; short_side_pixels — пиксели; "
            "aspect_ratio — ширина/высота; laplacian_variance — показатель "
            "детализации; bright_components — число отдельных ярких "
            "элементов. Это не вероятность подлинности."
        ),
    )
    region: tuple[float, float, float, float] | None = Field(
        default=None,
        description=(
            "Границы зоны [x1, y1, x2, y2] от 0 до 1 "
            "на изображении после rotation_degrees; null для всего кадра."
        ),
    )


class UvMetaSchema(BaseModel):
    engine: Literal["opencv"] = Field(
        default="opencv", description="Библиотека обработки изображений."
    )
    algorithm: Literal["ru-passport-uv-heuristic-v1"] = Field(
        default="ru-passport-uv-heuristic-v1",
        description="Идентификатор и версия набора правил проверки.",
    )
    experimental: Literal[True] = Field(
        default=True,
        description=(
            "Точность алгоритма на подтверждённых подделках ещё не измерена. "
            "Флаг описывает зрелость алгоритма, а не название маршрута."
        ),
    )
    duration_seconds: float = Field(
        ge=0,
        description=(
            "Время проверки файла, декодирования и анализа в секундах, "
            "без приёма HTTP-запроса и чтения загруженного файла."
        ),
    )
    rotation_degrees: Literal[0, 90, 180, 270] = Field(
        description=(
            "Поворот по часовой стрелке, применённый алгоритмом после EXIF."
        )
    )
    analyzed_width: int = Field(
        description="Ширина изображения для анализа в пикселях."
    )
    analyzed_height: int = Field(
        description="Высота изображения для анализа в пикселях."
    )


class UvDataSchema(BaseModel):
    is_authentic: None = Field(
        default=None,
        description=(
            "Всегда null: текущий набор правил не устанавливает "
            "физическую подлинность документа."
        ),
    )
    uv_check_passed: bool | None = Field(
        description=(
            "true — UV-признаки найдены; false — есть отклонения; "
            "null — изображение нельзя оценить."
        ),
    )
    status: Literal["consistent", "suspicious", "inconclusive"] = Field(
        description=(
            "consistent — все три проверки UV-признаков пройдены; "
            "suspicious — есть отклонения; inconclusive — входной снимок "
            "не прошёл отбор и оценка невозможна."
        )
    )
    message: str = Field(description="Общее объяснение результата.")
    checks: list[UvCheckSchema] = Field(
        description="Пять проверок входного снимка и три проверки UV-зон."
    )
    limitations: list[str] = Field(
        description="Ограничения текущего алгоритма и условий съёмки."
    )
    meta: UvMetaSchema = Field(description="Алгоритм и параметры обработки.")


class UvResponseSchema(BaseModel):
    success: bool = Field(
        description="Запрос обработан успешно; не признак подлинности."
    )
    data: UvDataSchema | None = Field(
        description="Результат проверки; null при ошибке запроса."
    )
    error: ErrorSchema | None = Field(
        description="Код и описание ошибки; null при успешном запросе."
    )
