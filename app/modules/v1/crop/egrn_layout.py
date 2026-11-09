"""Locate fields in the two-table first-page layout of a Gosuslugi extract.

The profile maps rows to fields. It does not read labels or infer that an
arbitrary two-column document is an EGRN extract.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

Box = tuple[int, int, int, int]

TOP_TABLE_FIELDS = {
    "object_type": "Вид объекта",
    "cadastral_number": "Кадастровый номер",
}
MAIN_TABLE_FIELDS = {
    "cadastral_quarter_number": "Номер кадастрового квартала",
    "cadastral_number_assignment_date": "Дата присвоения кадастрового номера",
    "previous_registration_number": "Ранее присвоенный учётный номер",
    "address": "Адрес (местоположение)",
    "floor": "Номер, тип этажа",
    "area": "Площадь, м²",
    "purpose": "Назначение",
    "object_name": "Наименование объекта",
    "residential_premises_type": "Вид жилого помещения",
    "cadastral_value": "Кадастровая стоимость, руб.",
    "parent_cadastral_numbers": "Кадастровые номера содержащих объектов",
    "cadastral_engineer": "Сведения о кадастровом инженере",
    "cultural_heritage": "Сведения о включении в реестр культурного наследия",
    "claims": "Сведения о правопритязаниях",
    "specialized_housing": "Отнесение к специализированному жилищному фонду",
}
FIELD_LABELS = {
    "application_number": "Номер заявления",
    **TOP_TABLE_FIELDS,
    **MAIN_TABLE_FIELDS,
    "actuality_date": "Дата актуальности",
}


@dataclass(frozen=True)
class _Grid:
    left: int
    divider: int
    right: int
    top_rows: tuple[int, ...]
    main_rows: tuple[int, ...]
    bottom: int


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    return [tuple(map(int, row)) for row in stats[1:]]


def _grid(binary: np.ndarray) -> _Grid | None:
    height, width = binary.shape
    horizontal = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, np.ones((3, 9), np.uint8)
    )
    horizontal = cv2.morphologyEx(
        horizontal, cv2.MORPH_OPEN, np.ones((1, width * 3 // 20), np.uint8)
    )
    vertical = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, np.ones((9, 3), np.uint8)
    )
    vertical = cv2.morphologyEx(
        vertical, cv2.MORPH_OPEN, np.ones((70, 1), np.uint8)
    )
    verticals = _components(vertical)
    columns = sorted(
        (x + w // 2, y, y + h)
        for x, y, w, h, _ in verticals
        if h > height * 0.3 and w < width * 0.015
    )
    if len(columns) != 3:
        return None
    left, divider, right = (column[0] for column in columns)
    span = right - left
    if span < width * 0.7 or not 0.3 < (divider - left) / span < 0.55:
        return None
    start = max(column[1] for column in columns)
    bottom = min(column[2] for column in columns)
    rules = sorted(
        y + h // 2
        for x, y, w, h, _ in _components(horizontal)
        if w >= span * 0.9 and abs(x - left) < 20 and abs(x + w - right) < 20
    )
    main_rows = tuple(y for y in rules if start - 12 <= y <= bottom + 5)
    # This profile has 15 characteristic rows. Its last row may have no
    # bottom rule in the printed extract. Reject incomplete/different grids.
    has_bottom_rule = bool(main_rows) and abs(main_rows[-1] - bottom) <= 8
    if len(main_rows) != (16 if has_bottom_rule else 15):
        return None
    short_columns = [
        (x + w // 2, y, y + h)
        for x, y, w, h, _ in verticals
        if 60 <= h < height * 0.2
        and y + h < start
        and any(
            abs(x + w // 2 - column) < 12 for column in (left, divider, right)
        )
    ]
    if len(short_columns) != 3:
        return None
    top_start = max(column[1] for column in short_columns)
    top_end = min(column[2] for column in short_columns)
    top_rows = tuple(y for y in rules if top_start - 12 <= y <= top_end + 5)
    if len(top_rows) != 3:
        return None
    if any(
        b - a < 15
        for rows in (top_rows, main_rows)
        for a, b in zip(rows, rows[1:])
    ):
        return None
    return _Grid(left, divider, right, top_rows, main_rows, bottom)


def _text_box(binary: np.ndarray, region: Box) -> Box | None:
    x1, y1, x2, y2 = region
    parts = [
        (x, y, w, h)
        for x, y, w, h, area in _components(binary[y1:y2, x1:x2])
        if area >= 6
        and h >= 5
        and w >= 1
        and h < (y2 - y1)
        and w < (x2 - x1) * 0.95
    ]
    if not parts:
        return None
    typical_height = float(np.median([h for _, _, _, h in parts]))
    parts = [part for part in parts if part[3] >= typical_height * 0.55]
    return (
        x1 + max(0, min(x for x, _, _, _ in parts) - 3),
        y1 + max(0, min(y for _, y, _, _ in parts) - 3),
        x1 + min(x2 - x1, max(x + w for x, _, w, _ in parts) + 3),
        y1 + min(y2 - y1, max(y + h for _, y, _, h in parts) + 3),
    )


def locate_egrn_fields(image: np.ndarray) -> dict[str, Box]:
    """Return tight field boxes in the supplied deskewed document's pixels."""
    height, width = image.shape[:2]
    analysis_width = 1600
    analysis_height = round(height * analysis_width / width)
    if not 500 <= analysis_height <= 3000:
        return {}
    small = cv2.resize(
        image,
        (analysis_width, analysis_height),
        interpolation=(
            cv2.INTER_AREA if width >= analysis_width else cv2.INTER_LINEAR
        ),
    )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        8,
    )
    grid = _grid(binary)
    if grid is None:
        # Downsampling can fade thin rules. Retry their detection at lower
        # contrast, while keeping the stricter text mask for tight crops.
        faint_rules = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            51,
            6,
        )
        grid = _grid(faint_rules)
    if grid is None:
        return {}
    boxes: dict[str, Box] = {}
    for names, rows in (
        (TOP_TABLE_FIELDS, grid.top_rows),
        (MAIN_TABLE_FIELDS, grid.main_rows),
    ):
        for index, name in enumerate(names):
            # Some extracts omit the final horizontal rule. The ends of
            # the vertical rules still bound the last value's cell.
            bottom = rows[index + 1] if index + 1 < len(rows) else grid.bottom
            region = (
                grid.divider + 5,
                rows[index] + 5,
                grid.right - 5,
                bottom - 5,
            )
            box = _text_box(binary, region)
            if box is not None:
                boxes[name] = box

    # Text blocks detached from the tables: right-aligned statement header
    # and the wide date line below the page number. Keep their prefixes.
    words = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, np.ones((3, 21), np.uint8)
    )
    span = grid.right - grid.left
    blocks = [
        (x, y, x + w, y + h)
        for x, y, w, h, area in _components(words)
        if 8 <= h <= 45
        and area > 80
        and x > grid.divider + (grid.right - grid.divider) * 0.35
        and abs(x + w - grid.right) < span * 0.08
    ]
    headers = [
        b
        for b in blocks
        if b[3] < grid.top_rows[0] - 12 and b[2] - b[0] > span * 0.08
    ]
    footers = [
        b
        for b in blocks
        if b[1] > grid.bottom + 20 and b[2] - b[0] > span * 0.15
    ]
    if len(headers) == 1:
        boxes["application_number"] = headers[0]
    if footers:
        boxes["actuality_date"] = max(footers, key=lambda b: b[1])

    result = {}
    for name, (x1, y1, x2, y2) in boxes.items():
        # The small analysis image is never returned as a field crop.
        padding = max(2, round(width / analysis_width * 2))
        result[name] = (
            max(0, int(x1 * width / analysis_width) - padding),
            max(0, int(y1 * height / analysis_height) - padding),
            min(width, int(np.ceil(x2 * width / analysis_width)) + padding),
            min(height, int(np.ceil(y2 * height / analysis_height)) + padding),
        )
    return result
