"""Allow queued document field and EGRN crop operations."""

from __future__ import annotations

from alembic import op

revision = "20261117_0013"
down_revision = "20261106_0012"
branch_labels = None
depends_on = None

PREVIOUS_TASK_MODE_CHECK = (
    "mode IN ('mrz', 'passport_mrz', 'passport_page', "
    "'passport_international', 'idcard_mrz', "
    "'snils', 'driver_license', 'egrn', 'full_v1', "
    "'crop_document', 'crop_mrz')"
)
CROP_TASK_MODE_CHECK = (
    "mode IN ('mrz', 'passport_mrz', 'passport_page', "
    "'passport_international', 'idcard_mrz', "
    "'snils', 'driver_license', 'egrn', 'full_v1', "
    "'crop_document', 'crop_mrz', 'crop_document_fields', "
    "'crop_egrn', 'crop_egrn_fields')"
)


def upgrade() -> None:
    op.drop_constraint("ck_ocr_jobs_mode", "ocr_jobs", type_="check")
    op.create_check_constraint(
        "ck_ocr_jobs_mode",
        "ocr_jobs",
        CROP_TASK_MODE_CHECK,
        postgresql_not_valid=True,
    )


def downgrade() -> None:
    op.drop_constraint("ck_ocr_jobs_mode", "ocr_jobs", type_="check")
    # Keep already-created crop jobs when rolling back the allowed modes.
    op.create_check_constraint(
        "ck_ocr_jobs_mode",
        "ocr_jobs",
        PREVIOUS_TASK_MODE_CHECK,
        postgresql_not_valid=True,
    )
