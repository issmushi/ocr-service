from __future__ import annotations

from typing import Any

OPENAPI_TAGS = [
    {
        "name": "MRZ Sync",
        "description": "Распознавание MRZ с ответом в том же запросе.",
    },
    {"name": "MRZ Async", "description": "Создание задачи распознавания MRZ."},
    {
        "name": "Russia Sync",
        "description": "Синхронное распознавание российских документов.",
    },
    {
        "name": "Russia Async",
        "description": "Создание задач распознавания российских документов.",
    },
    {
        "name": "Full Sync",
        "description": "Синхронное распознавание полного текста.",
    },
    {
        "name": "Full Async",
        "description": "Создание задачи распознавания полного текста.",
    },
    {
        "name": "Crop Sync",
        "description": "Синхронная обрезка документов и полей.",
    },
    {
        "name": "Crop Async",
        "description": "Создание задач обрезки документа или MRZ.",
    },
    {
        "name": "Authenticity",
        "description": "Проверка признаков подлинности документа.",
    },
    {
        "name": "Best frame",
        "description": "Выбор лучшего кадра из серии снимков.",
    },
    {
        "name": "Tasks",
        "description": "Статус, результат и отмена любых задач Async.",
    },
    {"name": "Health", "description": "Состояние сервиса и его компонентов."},
]


def normalize_binary_file_schemas(schema: dict[str, Any]) -> None:
    """Convert binary JSON Schema metadata to Swagger UI's format."""
    _normalize_node(schema)


def _normalize_node(node: Any) -> None:
    if isinstance(node, dict):
        if (
            node.get("type") == "string"
            and node.get("contentMediaType") == "application/octet-stream"
        ):
            node.pop("contentMediaType")
            node["format"] = "binary"

        for value in node.values():
            _normalize_node(value)
        return

    if isinstance(node, list):
        for value in node:
            _normalize_node(value)
