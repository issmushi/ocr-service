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
UNIFIED_MRZ_MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "alembic"
    / "versions"
    / "20261002_0008_unified_mrz_task_mode.py"
)
INTERNATIONAL_MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "alembic"
    / "versions"
    / "20261002_0009_russian_international_task_mode.py"
)
SNILS_MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "alembic"
    / "versions"
    / "20261002_0010_snils_task_mode.py"
)


def load_migration(path=MIGRATION_PATH):
    spec = importlib.util.spec_from_file_location(
        f"task_mode_migration_{path.stem}",
        path,
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


class UnifiedMrzTaskModeMigrationTests(unittest.TestCase):
    def test_upgrade_adds_unified_mrz_without_dropping_queued_modes(self):
        migration = load_migration(UNIFIED_MRZ_MIGRATION_PATH)
        with (
            patch.object(migration.op, "drop_constraint") as drop_constraint,
            patch.object(
                migration.op, "create_check_constraint"
            ) as create_constraint,
        ):
            migration.upgrade()

        drop_constraint.assert_called_once_with(
            "ck_ocr_jobs_mode", "ocr_jobs", type_="check"
        )
        create_constraint.assert_called_once_with(
            "ck_ocr_jobs_mode",
            "ocr_jobs",
            migration.UNIFIED_TASK_MODE_CHECK,
            postgresql_not_valid=True,
        )
        for mode in ("mrz", "passport_mrz", "idcard_mrz"):
            self.assertIn(f"'{mode}'", migration.UNIFIED_TASK_MODE_CHECK)

    def test_downgrade_restores_previous_constraint(self):
        migration = load_migration(UNIFIED_MRZ_MIGRATION_PATH)
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
            postgresql_not_valid=True,
        )
        self.assertNotIn("'mrz'", migration.PREVIOUS_TASK_MODE_CHECK)


class InternationalPassportTaskModeMigrationTests(unittest.TestCase):
    def test_upgrade_adds_international_mode_without_dropping_old_modes(self):
        migration = load_migration(INTERNATIONAL_MIGRATION_PATH)
        self.assertEqual(migration.down_revision, "20261002_0008")
        with (
            patch.object(migration.op, "drop_constraint") as drop_constraint,
            patch.object(
                migration.op, "create_check_constraint"
            ) as create_constraint,
        ):
            migration.upgrade()

        drop_constraint.assert_called_once_with(
            "ck_ocr_jobs_mode", "ocr_jobs", type_="check"
        )
        create_constraint.assert_called_once_with(
            "ck_ocr_jobs_mode",
            "ocr_jobs",
            migration.INTERNATIONAL_TASK_MODE_CHECK,
            postgresql_not_valid=True,
        )
        for mode in ("passport_international", "mrz", "passport_page"):
            self.assertIn(f"'{mode}'", migration.INTERNATIONAL_TASK_MODE_CHECK)

    def test_downgrade_restores_previous_constraint(self):
        migration = load_migration(INTERNATIONAL_MIGRATION_PATH)
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
            postgresql_not_valid=True,
        )
        self.assertNotIn(
            "'passport_international'", migration.PREVIOUS_TASK_MODE_CHECK
        )


class SnilsTaskModeMigrationTests(unittest.TestCase):
    def test_upgrade_adds_snils_and_preserves_existing_modes(self):
        migration = load_migration(SNILS_MIGRATION_PATH)
        self.assertEqual(migration.down_revision, "20261002_0009")
        with (
            patch.object(migration.op, "drop_constraint") as drop_constraint,
            patch.object(
                migration.op, "create_check_constraint"
            ) as create_constraint,
        ):
            migration.upgrade()

        drop_constraint.assert_called_once_with(
            "ck_ocr_jobs_mode", "ocr_jobs", type_="check"
        )
        create_constraint.assert_called_once_with(
            "ck_ocr_jobs_mode",
            "ocr_jobs",
            migration.SNILS_TASK_MODE_CHECK,
            postgresql_not_valid=True,
        )
        for mode in ("snils", "passport_international", "mrz"):
            self.assertIn(f"'{mode}'", migration.SNILS_TASK_MODE_CHECK)

    def test_downgrade_restores_previous_constraint(self):
        migration = load_migration(SNILS_MIGRATION_PATH)
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
            postgresql_not_valid=True,
        )
        self.assertNotIn("'snils'", migration.PREVIOUS_TASK_MODE_CHECK)


if __name__ == "__main__":
    unittest.main()
