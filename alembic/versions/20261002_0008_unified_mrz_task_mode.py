"""Allow unified MRZ tasks while retaining queued legacy MRZ modes."""

from __future__ import annotations

from alembic import op

revision = "20261002_0008"
down_revision = "20260925_0007"
branch_labels = None
depends_on = None

UNIFIED_TASK_MODE_CHECK = (
    "mode IN ('mrz', 'passport_mrz', 'passport_page', 'idcard_mrz', "
    "'full_v1', 'crop_document', 'crop_mrz')"
)
PREVIOUS_TASK_MODE_CHECK = (
    "mode IN ('passport_mrz', 'passport_page', 'idcard_mrz', "
    "'full_v1', 'crop_document', 'crop_mrz')"
)


def upgrade() -> None:
    op.drop_constraint("ck_ocr_jobs_mode", "ocr_jobs", type_="check")
    # The previous constraint was NOT VALID, so retain that behavior for rows
    # that predate the current task modes.
    op.create_check_constraint(
        "ck_ocr_jobs_mode",
        "ocr_jobs",
        UNIFIED_TASK_MODE_CHECK,
        postgresql_not_valid=True,
    )


def downgrade() -> None:
    op.drop_constraint("ck_ocr_jobs_mode", "ocr_jobs", type_="check")
    # Existing unified MRZ jobs may still be present after a rollback.
    op.create_check_constraint(
        "ck_ocr_jobs_mode",
        "ocr_jobs",
        PREVIOUS_TASK_MODE_CHECK,
        postgresql_not_valid=True,
    )
