"""Small, deliberately uncalibrated screening profile for Regula UV scans.

The thresholds describe image appearance observed in local page-3 scans.
They are not security specifications or a classifier trained on verified
genuine/counterfeit documents. A rendered colour pattern can pass them.
No images, identities or descriptors of personal fields are retained here.
"""

from __future__ import annotations

from time import perf_counter

import cv2
import numpy as np

from app.core.errors import AppError, ErrorCode
from app.core.images.decoding import InvalidImageError, decode_image
from app.core.images.validation import validate_image_bytes
from app.modules.v1.russia.uv.schemas import (
    UvCheckSchema,
    UvDataSchema,
    UvMetaSchema,
)

ANALYSIS_WIDTH = 1000
MIN_SHORT_SIDE = 400
PHOTO_REGION = (0.03, 0.15, 0.36, 0.65)
BAND_REGION = (0.02, 0.55, 0.98, 0.80)
BACKGROUND_REGION = (0.40, 0.12, 0.90, 0.53)
LIMITATIONS = [
    "Тестовый профиль страницы с фотографией внутреннего паспорта РФ. "
    "Тип документа и версия бланка автоматически не подтверждаются.",
    "Нужен цветной UV-скан страницы целиком, почти без внешнего фона. "
    "Повороты на 90 градусов исправляются; перспектива не исправляется.",
    "Пороги подобраны по локальным сканам Regula, без подтверждённых "
    "образцов подделок. Точность обнаружения подделок не измерена.",
    "Рисунок, волокна, замена фотографии и данные документа не проверяются. "
    "Цветная имитация или повторное фото изображения могут пройти тест.",
]


def _region(image: np.ndarray, box: tuple[float, ...]) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    return image[
        int(y1 * height) : max(int(y1 * height) + 1, int(y2 * height)),
        int(x1 * width) : max(int(x1 * width) + 1, int(x2 * width)),
    ]


def _masks(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Float arithmetic avoids uint8 overflow in colour differences.
    red, green, blue = cv2.split(rgb.astype(np.float32))
    green_mask = (
        (green > 70)
        & (green > blue * 1.2)
        & (green > red * 0.9)
        & (green - blue > 15)
    )
    blue_mask = (blue > 80) & (blue > red * 1.25) & (blue > green * 1.12)
    return green_mask, blue_mask


def _band_metrics(green: np.ndarray) -> dict[str, float]:
    band = _region(green, BAND_REGION)
    _, _, stats, _ = cv2.connectedComponentsWithStats(
        band.astype(np.uint8), connectivity=8
    )
    areas = stats[1:, cv2.CC_STAT_AREA]
    return {
        "green_fraction": float(band.mean()),
        "horizontal_coverage": float((band.sum(axis=0) > 2).mean()),
        "bright_components": float(((areas >= 3) & (areas <= 800)).sum()),
    }


def _orient(rgb: np.ndarray) -> tuple[np.ndarray, int]:
    candidates = []
    for turns in range(4):
        rotated = np.rot90(rgb, k=-turns)
        height, width = rotated.shape[:2]
        if not 1.20 <= width / height <= 1.75:
            continue
        green, blue = _masks(rotated)
        band = _band_metrics(green)
        score = band["horizontal_coverage"] + float(
            _region(blue, PHOTO_REGION).mean()
        )
        candidates.append((score, turns, rotated))
    if not candidates:
        return rgb, 0
    _, turns, rotated = max(candidates, key=lambda item: item[0])
    return np.ascontiguousarray(rotated), turns * 90


def _check(
    code: str,
    passed: bool,
    message: str,
    metrics: dict[str, float],
    region: tuple[float, ...] | None = None,
) -> UvCheckSchema:
    return UvCheckSchema(
        code=code,
        status="passed" if passed else "failed",
        message=message,
        metrics={key: round(value, 4) for key, value in metrics.items()},
        region=region,
    )


def analyze_uv(content: bytes) -> UvDataSchema:
    started = perf_counter()
    validate_image_bytes(content)
    try:
        with decode_image(content) as decoded:
            original_width, original_height = decoded.size
            decoded.thumbnail((ANALYSIS_WIDTH, ANALYSIS_WIDTH))
            rgb = np.asarray(decoded).copy()
    except InvalidImageError as error:
        raise AppError(
            ErrorCode.IMAGE_READ_ERROR, "Изображение невозможно прочитать"
        ) from error

    rgb, rotation = _orient(rgb)
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    green, blue = _masks(rgb)
    checks = []

    resolution_ok = min(original_width, original_height) >= MIN_SHORT_SIDE
    checks.append(
        _check(
            "resolution",
            resolution_ok,
            "Разрешение достаточное."
            if resolution_ok
            else "Слишком маленькое изображение: загрузите исходный скан.",
            {"short_side_pixels": float(min(original_width, original_height))},
        )
    )
    geometry_ok = 1.20 <= width / height <= 1.75
    checks.append(
        _check(
            "page_geometry",
            geometry_ok,
            "Пропорции совместимы с тестовым профилем; тип не подтверждён."
            if geometry_ok
            else "Нужен скан одной страницы целиком без фона.",
            {"aspect_ratio": width / height},
        )
    )
    brightness = float(np.percentile(gray, 95))
    clipped = float((rgb.min(axis=2) >= 250).mean())
    exposure_ok = brightness >= 45 and clipped < 0.25
    checks.append(
        _check(
            "exposure",
            exposure_ok,
            "Есть различимые яркие участки без сильного белого пересвета."
            if exposure_ok
            else "Снимок слишком тёмный или пересвечен.",
            {"brightness_p95": brightness, "white_clipped_fraction": clipped},
        )
    )
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness_ok = sharpness >= 25
    checks.append(
        _check(
            "sharpness",
            sharpness_ok,
            "На снимке различимы мелкие детали."
            if sharpness_ok
            else "Недостаточно мелких деталей: возможен смаз или пустой кадр.",
            {"laplacian_variance": sharpness},
        )
    )
    color_fraction = float(blue.mean())
    color_ok = color_fraction >= 0.03
    checks.append(
        _check(
            "uv_color_signature",
            color_ok,
            "Цвета совместимы с UV-примерами; "
            "источник освещения не подтверждён."
            if color_ok
            else "Не найдены цвета из UV-профиля; возможно, фото "
            "обычное, чёрно-белое или снято в других условиях.",
            {"blue_fraction": color_fraction},
        )
    )

    can_evaluate = all(check.status == "passed" for check in checks)
    if can_evaluate:
        band = _band_metrics(green)
        band_ok = (
            0.035 <= band["green_fraction"] <= 0.55
            and band["horizontal_coverage"] >= 0.55
            and band["bright_components"] >= 20
        )
        checks.append(
            _check(
                "fluorescent_band",
                band_ok,
                "Найдена неоднородная зелёная полоса в ожидаемой зоне."
                if band_ok
                else "Зелёная полоса отсутствует или отличается "
                "от тестового профиля. Возможен другой бланк или освещение.",
                band,
                BAND_REGION,
            )
        )
        photo_fraction = float(_region(blue, PHOTO_REGION).mean())
        photo_ok = photo_fraction >= 0.15
        checks.append(
            _check(
                "photo_zone_glow",
                photo_ok,
                "Свечение в зоне фотографии совместимо с тестовым профилем."
                if photo_ok
                else "Свечение в зоне фотографии отличается "
                "от тестового профиля; причина не установлена.",
                {"blue_fraction": photo_fraction},
                PHOTO_REGION,
            )
        )
        background = float(np.median(_region(gray, BACKGROUND_REGION)))
        background_ok = background <= 100
        checks.append(
            _check(
                "background_glow",
                background_ok,
                "Фон достаточно тёмный относительно тестовых UV-примеров."
                if background_ok
                else "Фон светлее тестовых UV-примеров. "
                "Нужно исключить другое освещение или другой вариант бланка.",
                {"median_brightness": background},
                BACKGROUND_REGION,
            )
        )
        passed = band_ok and photo_ok and background_ok
        status = "consistent" if passed else "suspicious"
        message = (
            "Тестовые UV-признаки найдены. Подлинность не подтверждена."
            if passed
            else "Есть отличия от тестового UV-профиля. "
            "Нужна ручная проверка; это не заключение о подделке."
        )
    else:
        passed = None
        status = "inconclusive"
        message = (
            "Этот снимок нельзя оценить по тестовому UV-профилю. "
            "Причины указаны в checks."
        )
        for code in (
            "fluorescent_band",
            "photo_zone_glow",
            "background_glow",
        ):
            checks.append(
                UvCheckSchema(
                    code=code,
                    status="not_checked",
                    message=(
                        "Проверка пропущена: входной снимок не прошёл отбор."
                    ),
                )
            )

    return UvDataSchema(
        uv_check_passed=passed,
        status=status,
        message=message,
        checks=checks,
        limitations=LIMITATIONS,
        meta=UvMetaSchema(
            duration_seconds=perf_counter() - started,
            rotation_degrees=rotation,
            analyzed_width=width,
            analyzed_height=height,
        ),
    )
