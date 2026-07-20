from __future__ import annotations

import ast
from pathlib import Path

ENGINE_ROOT = Path(__file__).parents[1] / "app" / "ocr_engine"
CORE_ROOT = Path(__file__).parents[1] / "app" / "core"
FORBIDDEN_DEPENDENCIES = (
    "app.core",
    "app.modules",
    "app.workers",
)
CORE_FORBIDDEN_DEPENDENCIES = (
    "app.ocr_engine",
    "app.modules",
    "app.workers",
)


def find_dependency_violations(
    root: Path,
    forbidden_dependencies: tuple[str, ...],
) -> list[str]:
    violations: list[str] = []

    for source_path in root.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = ""
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(forbidden_dependencies):
                        violations.append(f"{source_path.name}: {alias.name}")
                continue

            if module.startswith(forbidden_dependencies):
                violations.append(f"{source_path.name}: {module}")

    return violations


def test_ocr_engine_has_no_application_layer_dependencies() -> None:
    violations = find_dependency_violations(
        ENGINE_ROOT,
        FORBIDDEN_DEPENDENCIES,
    )

    assert not violations, "ocr_engine dependency violations: " + ", ".join(
        violations
    )


def test_core_has_no_ocr_or_business_layer_dependencies() -> None:
    violations = find_dependency_violations(
        CORE_ROOT,
        CORE_FORBIDDEN_DEPENDENCIES,
    )

    assert not violations, "core dependency violations: " + ", ".join(
        violations
    )
