"""Allow Russian international passport OCR tasks."""

from __future__ import annotations

from alembic import op

revision = "20261002_0009"
down_revision = "20261002_0008"
branch_labels = None
depends_on = None

INTERNATIONAL_TASK_MODE_CHECK = (
    "mode IN ('mrz', 'passport_mrz', 'passport_page', "
    "'passport_international', 'idcard_mrz', "
    "'full_v1', 'crop_document', 'crop_mrz')"
)
PREVIOUS_TASK_MODE_CHECK = (
    "mode IN ('mrz', 'passport_mrz', 'passport_page', 'idcard_mrz', "
    "'full_v1', 'crop_document', 'crop_mrz')"
)


def upgrade() -> None:
    op.drop_constraint("ck_ocr_jobs_mode", "ocr_jobs", type_="check")
    op.create_check_constraint(
        "ck_ocr_jobs_mode",
        "ocr_jobs",
        INTERNATIONAL_TASK_MODE_CHECK,
        postgresql_not_valid=True,
    )


def downgrade() -> None:
    op.drop_constraint("ck_ocr_jobs_mode", "ocr_jobs", type_="check")
    # Existing jobs can remain in the table after rollback.
    op.create_check_constraint(
        "ck_ocr_jobs_mode",
        "ocr_jobs",
        PREVIOUS_TASK_MODE_CHECK,
        postgresql_not_valid=True,
    )
