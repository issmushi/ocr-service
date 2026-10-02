"""Allow asynchronous Russian driver license OCR tasks."""

from __future__ import annotations

from alembic import op

revision = "20261002_0011"
down_revision = "20261002_0010"
branch_labels = None
depends_on = None

DRIVER_LICENSE_TASK_MODE_CHECK = (
    "mode IN ('mrz', 'passport_mrz', 'passport_page', "
    "'passport_international', 'idcard_mrz', "
    "'snils', 'driver_license', 'full_v1', 'crop_document', 'crop_mrz')"
)
PREVIOUS_TASK_MODE_CHECK = (
    "mode IN ('mrz', 'passport_mrz', 'passport_page', "
    "'passport_international', 'idcard_mrz', "
    "'snils', 'full_v1', 'crop_document', 'crop_mrz')"
)


def upgrade() -> None:
    op.drop_constraint("ck_ocr_jobs_mode", "ocr_jobs", type_="check")
    op.create_check_constraint(
        "ck_ocr_jobs_mode",
        "ocr_jobs",
        DRIVER_LICENSE_TASK_MODE_CHECK,
        postgresql_not_valid=True,
    )


def downgrade() -> None:
    op.drop_constraint("ck_ocr_jobs_mode", "ocr_jobs", type_="check")
    # Existing driver license jobs may remain after rollback.
    op.create_check_constraint(
        "ck_ocr_jobs_mode",
        "ocr_jobs",
        PREVIOUS_TASK_MODE_CHECK,
        postgresql_not_valid=True,
    )
