"""Persist V1 task options and allow the six synchronous operation variants."""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "20260925_0006"
down_revision = "20260718_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "ocr_jobs",
        sa.Column(
            "options",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.drop_constraint("ck_ocr_jobs_mode", "ocr_jobs", type_="check")
    op.create_check_constraint(
        "ck_ocr_jobs_mode",
        "ocr_jobs",
        "mode IN ('document', 'full', 'passport_mrz', 'passport_page', "
        "'idcard_mrz', 'full_v1', 'crop_document', 'crop_mrz')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_ocr_jobs_mode", "ocr_jobs", type_="check")
    op.create_check_constraint(
        "ck_ocr_jobs_mode",
        "ocr_jobs",
        "mode IN ('document', 'full')",
    )
    op.drop_column("ocr_jobs", "options")
