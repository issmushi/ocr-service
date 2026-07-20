from __future__ import annotations

from typing import Any


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
