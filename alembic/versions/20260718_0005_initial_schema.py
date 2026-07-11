from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# Retain the former head identifier so existing databases remain compatible
# after the migration history is consolidated into this baseline.
revision = "20260718_0005"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ocr_jobs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("mode", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("content_type", sa.String(), nullable=False),
        sa.Column("image_path", sa.Text(), nullable=False),
        sa.Column(
            "result",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "attempts",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
        sa.Column(
            "max_attempts",
            sa.Integer(),
            server_default="2",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "finished_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column("worker_id", sa.String(), nullable=True),
        sa.Column(
            "lease_expires_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "heartbeat_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.CheckConstraint(
            "mode IN ('document', 'full')",
            name="ck_ocr_jobs_mode",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_ocr_jobs_status",
        "ocr_jobs",
        ["status"],
    )
    op.create_index(
        "idx_ocr_jobs_created_at",
        "ocr_jobs",
        ["created_at"],
    )
    op.create_index(
        "idx_ocr_jobs_processing_lease",
        "ocr_jobs",
        ["status", "lease_expires_at"],
    )

    op.create_table(
        "ocr_job_outbox",
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column(
            "attempts",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
        sa.Column(
            "next_attempt_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "published_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["ocr_jobs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("job_id"),
    )
    op.create_index(
        "idx_ocr_job_outbox_pending",
        "ocr_job_outbox",
        ["published_at", "next_attempt_at", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_ocr_job_outbox_pending",
        table_name="ocr_job_outbox",
    )
    op.drop_table("ocr_job_outbox")

    op.drop_index(
        "idx_ocr_jobs_processing_lease",
        table_name="ocr_jobs",
    )
    op.drop_index(
        "idx_ocr_jobs_created_at",
        table_name="ocr_jobs",
    )
    op.drop_index(
        "idx_ocr_jobs_status",
        table_name="ocr_jobs",
    )
    op.drop_table("ocr_jobs")
