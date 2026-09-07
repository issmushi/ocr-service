"""Reject new jobs that use the removed document and full task modes."""

from __future__ import annotations

from alembic import op

revision = "20260925_0007"
down_revision = "20260925_0006"
branch_labels = None
depends_on = None

CURRENT_TASK_MODE_CHECK = (
    "mode IN ('passport_mrz', 'passport_page', 'idcard_mrz', "
    "'full_v1', 'crop_document', 'crop_mrz')"
)
PREVIOUS_TASK_MODE_CHECK = (
    "mode IN ('document', 'full', 'passport_mrz', 'passport_page', "
    "'idcard_mrz', 'full_v1', 'crop_document', 'crop_mrz')"
)


def upgrade() -> None:
    op.drop_constraint("ck_ocr_jobs_mode", "ocr_jobs", type_="check")
    # Existing document/full rows remain readable. PostgreSQL still applies a
    # NOT VALID check constraint to every new or updated row, so no new legacy
    # task can enter the queue without making this migration scan old history.
    op.create_check_constraint(
        "ck_ocr_jobs_mode",
        "ocr_jobs",
        CURRENT_TASK_MODE_CHECK,
        postgresql_not_valid=True,
    )


def downgrade() -> None:
    op.drop_constraint("ck_ocr_jobs_mode", "ocr_jobs", type_="check")
    op.create_check_constraint(
        "ck_ocr_jobs_mode",
        "ocr_jobs",
        PREVIOUS_TASK_MODE_CHECK,
    )
