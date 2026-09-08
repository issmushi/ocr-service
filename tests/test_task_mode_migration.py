from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "alembic"
    / "versions"
    / "20260925_0007_current_task_modes.py"
)


def load_migration():
    spec = importlib.util.spec_from_file_location(
        "task_mode_migration_0007",
        MIGRATION_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover - fixed path.
        raise RuntimeError("Cannot load task mode migration")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CurrentTaskModeMigrationTests(unittest.TestCase):
    def test_upgrade_replaces_constraint_without_validating_legacy_rows(
        self,
    ) -> None:
        migration = load_migration()

        with (
            patch.object(migration.op, "drop_constraint") as drop_constraint,
            patch.object(
                migration.op, "create_check_constraint"
            ) as create_constraint,
        ):
            migration.upgrade()

        drop_constraint.assert_called_once_with(
            "ck_ocr_jobs_mode",
            "ocr_jobs",
            type_="check",
        )
        create_constraint.assert_called_once_with(
            "ck_ocr_jobs_mode",
            "ocr_jobs",
            migration.CURRENT_TASK_MODE_CHECK,
            postgresql_not_valid=True,
        )
        self.assertNotIn("'document'", migration.CURRENT_TASK_MODE_CHECK)
        self.assertNotIn("'full'", migration.CURRENT_TASK_MODE_CHECK)

    def test_downgrade_restores_the_previous_mode_set(self) -> None:
        migration = load_migration()

        with (
            patch.object(migration.op, "drop_constraint"),
            patch.object(
                migration.op, "create_check_constraint"
            ) as create_constraint,
        ):
            migration.downgrade()

        create_constraint.assert_called_once_with(
            "ck_ocr_jobs_mode",
            "ocr_jobs",
            migration.PREVIOUS_TASK_MODE_CHECK,
        )
        self.assertIn("'document'", migration.PREVIOUS_TASK_MODE_CHECK)
        self.assertIn("'full'", migration.PREVIOUS_TASK_MODE_CHECK)


if __name__ == "__main__":
    unittest.main()
