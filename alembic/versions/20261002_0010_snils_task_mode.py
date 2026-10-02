"""Allow asynchronous Russian SNILS OCR tasks."""

from __future__ import annotations

from alembic import op

revision = "20261002_0010"
down_revision = "20261002_0009"
branch_labels = None
depends_on = None

SNILS_TASK_MODE_CHECK = (
    "mode IN ('mrz', 'passport_mrz', 'passport_page', "
    "'passport_international', 'idcard_mrz', "
    "'snils', 'full_v1', 'crop_document', 'crop_mrz')"
)
PREVIOUS_TASK_MODE_CHECK = (
    "mode IN ('mrz', 'passport_mrz', 'passport_page', "
    "'passport_international', 'idcard_mrz', "
    "'full_v1', 'crop_document', 'crop_mrz')"
)


def upgrade() -> None:
    op.drop_constraint("ck_ocr_jobs_mode", "ocr_jobs", type_="check")
    op.create_check_constraint(
        "ck_ocr_jobs_mode",
        "ocr_jobs",
        SNILS_TASK_MODE_CHECK,
        postgresql_not_valid=True,
    )


def downgrade() -> None:
    op.drop_constraint("ck_ocr_jobs_mode", "ocr_jobs", type_="check")
    # Already queued SNILS jobs can remain after rollback.
    op.create_check_constraint(
        "ck_ocr_jobs_mode",
        "ocr_jobs",
        PREVIOUS_TASK_MODE_CHECK,
        postgresql_not_valid=True,
    )
