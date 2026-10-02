from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from app.modules.tasks.enums import TaskOperation
from app.modules.tasks.exceptions import QueueCapacityExceeded
from app.modules.tasks.models import OCRJob, OCRJobOutbox
from app.modules.tasks.repository import (
    claim_job,
    claim_stale_published_job,
    create_job,
    mark_done,
    recover_expired_job_leases,
    serialize_job,
)


class AsyncContext:
    def __init__(self, value: object = None) -> None:
        self.value = value

    async def __aenter__(self) -> object:
        return self.value

    async def __aexit__(self, *_args: object) -> None:
        return None


class TaskRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_job_and_outbox_are_added_in_one_transaction(self) -> None:
        session = MagicMock()
        session.begin.return_value = AsyncContext()
        session.execute = AsyncMock(
            side_effect=[
                None,
                SimpleNamespace(scalar_one=lambda: 49),
            ]
        )
        session_factory = MagicMock(
            return_value=AsyncContext(session),
        )

        await create_job(
            session_factory,
            job_id="job-1",
            mode=TaskOperation.PASSPORT_MRZ.value,
            filename="document.jpg",
            content_type="image/jpeg",
            image_path="/tmp/document.jpg",
        )

        objects = session.add_all.call_args.args[0]
        self.assertEqual(len(objects), 2)
        self.assertIsInstance(objects[0], OCRJob)
        self.assertIsInstance(objects[1], OCRJobOutbox)
        self.assertEqual(objects[0].options, {})
        self.assertEqual(objects[1].payload, {"job_id": "job-1"})
        self.assertEqual(session.execute.await_count, 2)

    async def test_v1_options_are_persisted_with_the_job_and_serialized(
        self,
    ) -> None:
        session = MagicMock()
        session.begin.return_value = AsyncContext()
        session.execute = AsyncMock(
            side_effect=[None, SimpleNamespace(scalar_one=lambda: 0)]
        )
        session_factory = MagicMock(return_value=AsyncContext(session))
        options = {
            "model": "qwen3-vl-8b",
            "crop": False,
            "normalize": False,
            "normalization": True,
        }

        await create_job(
            session_factory,
            job_id="v1-job",
            mode=TaskOperation.MRZ.value,
            filename="document.png",
            content_type="image/png",
            image_path="/tmp/document.png",
            options=options,
        )

        job, outbox = session.add_all.call_args.args[0]
        self.assertEqual(job.mode, "mrz")
        self.assertEqual(job.options, options)
        self.assertIsNot(job.options, options)
        self.assertEqual(serialize_job(job)["options"], options)
        self.assertEqual(outbox.payload, {"job_id": "v1-job"})

    def test_database_mode_constraint_accepts_v1_operations(self) -> None:
        constraint = next(
            item
            for item in OCRJob.__table__.constraints
            if item.name == "ck_ocr_jobs_mode"
        )
        constraint_sql = str(constraint.sqltext)
        for mode in TaskOperation:
            self.assertIn(f"'{mode.value}'", constraint_sql)
        self.assertNotIn("'document'", constraint_sql)
        self.assertNotIn("'full'", constraint_sql)
        column = OCRJob.__table__.c.options
        self.assertFalse(column.nullable)
        self.assertEqual(str(column.server_default.arg), "'{}'::jsonb")

    async def test_capacity_check_rejects_job_inside_transaction(self) -> None:
        session = MagicMock()
        session.begin.return_value = AsyncContext()
        session.execute = AsyncMock(
            side_effect=[
                None,
                SimpleNamespace(scalar_one=lambda: 50),
            ]
        )
        session_factory = MagicMock(
            return_value=AsyncContext(session),
        )

        with self.assertRaises(QueueCapacityExceeded) as error_context:
            await create_job(
                session_factory,
                job_id="job-2",
                mode=TaskOperation.PASSPORT_MRZ.value,
                filename="document.jpg",
                content_type="image/jpeg",
                image_path="/tmp/document.jpg",
            )

        self.assertEqual(error_context.exception.active_jobs, 50)
        self.assertEqual(error_context.exception.limit, 50)
        session.add_all.assert_not_called()

    async def test_claim_assigns_worker_and_returns_attempt(self) -> None:
        result = SimpleNamespace(scalar_one_or_none=lambda: 2)
        session = MagicMock()
        session.execute = AsyncMock(return_value=result)
        session.commit = AsyncMock()
        session_factory = MagicMock(
            return_value=AsyncContext(session),
        )

        attempts = await claim_job(
            session_factory,
            "job-3",
            "worker-1",
        )

        self.assertEqual(attempts, 2)
        statement = session.execute.await_args.args[0]
        statement_text = str(statement)
        self.assertIn("ocr_jobs.lease_expires_at", statement_text)
        self.assertIn("worker_id=:worker_id", statement_text)
        session.commit.assert_awaited_once()

    async def test_done_transition_is_guarded_by_worker_id(self) -> None:
        result = SimpleNamespace(scalar_one_or_none=lambda: None)
        session = MagicMock()
        session.execute = AsyncMock(return_value=result)
        session.commit = AsyncMock()
        session_factory = MagicMock(
            return_value=AsyncContext(session),
        )

        transitioned = await mark_done(
            session_factory,
            "job-4",
            {"result": "ok"},
            "old-worker",
        )

        self.assertFalse(transitioned)
        statement = session.execute.await_args.args[0]
        self.assertIn("ocr_jobs.worker_id", str(statement))

    async def test_expired_lease_is_requeued_without_republishing(
        self,
    ) -> None:
        now = datetime.now(UTC)
        job = SimpleNamespace(
            id="job-5",
            status="processing",
            attempts=1,
            max_attempts=2,
            worker_id="dead-worker",
            lease_expires_at=now - timedelta(seconds=1),
            heartbeat_at=now - timedelta(seconds=10),
            updated_at=now,
            error=None,
            finished_at=None,
        )
        selected_jobs = SimpleNamespace(
            scalars=lambda: [job],
        )
        session = MagicMock()
        session.begin.return_value = AsyncContext()
        session.execute = AsyncMock(return_value=selected_jobs)
        session_factory = MagicMock(
            return_value=AsyncContext(session),
        )

        result = await recover_expired_job_leases(session_factory)

        self.assertEqual(result.requeued, 1)
        self.assertEqual(result.failed, 0)
        self.assertEqual(job.status, "queued")
        self.assertIsNone(job.worker_id)
        self.assertIsNone(job.lease_expires_at)
        self.assertEqual(session.execute.await_count, 1)

    async def test_watchdog_atomically_claims_stale_published_job(
        self,
    ) -> None:
        now = datetime.now(UTC)
        job = SimpleNamespace(
            id="job-7",
            mode=TaskOperation.PASSPORT_MRZ.value,
            status="queued",
            filename="document.jpg",
            content_type="image/jpeg",
            image_path="/tmp/document.jpg",
            options={},
            result=None,
            error="delivery delayed",
            attempts=0,
            max_attempts=2,
            created_at=now - timedelta(minutes=2),
            updated_at=now - timedelta(minutes=2),
            started_at=None,
            finished_at=None,
            worker_id=None,
            lease_expires_at=None,
            heartbeat_at=None,
        )
        selected_job = SimpleNamespace(
            scalar_one_or_none=lambda: job,
        )
        session = MagicMock()
        session.begin.return_value = AsyncContext()
        session.execute = AsyncMock(return_value=selected_job)
        session_factory = MagicMock(
            return_value=AsyncContext(session),
        )

        claimed = await claim_stale_published_job(
            session_factory,
            "watchdog-worker",
        )

        self.assertIsNotNone(claimed)
        assert claimed is not None
        self.assertEqual(claimed["status"], "processing")
        self.assertEqual(claimed["attempts"], 1)
        self.assertEqual(claimed["worker_id"], "watchdog-worker")
        self.assertIsNotNone(claimed["lease_expires_at"])
        statement = session.execute.await_args.args[0]
        self.assertIn("ocr_job_outbox", str(statement))
        self.assertIn("published_at IS NOT NULL", str(statement))

    async def test_watchdog_fails_stale_job_after_last_attempt(self) -> None:
        now = datetime.now(UTC)
        job = SimpleNamespace(
            id="job-8",
            mode=TaskOperation.PASSPORT_MRZ.value,
            status="queued",
            filename="document.jpg",
            content_type="image/jpeg",
            image_path="/tmp/document.jpg",
            options={},
            result=None,
            error=None,
            attempts=2,
            max_attempts=2,
            created_at=now - timedelta(minutes=2),
            updated_at=now - timedelta(minutes=2),
            started_at=now - timedelta(minutes=1),
            finished_at=None,
            worker_id=None,
            lease_expires_at=None,
            heartbeat_at=None,
        )
        selected_job = SimpleNamespace(
            scalar_one_or_none=lambda: job,
        )
        session = MagicMock()
        session.begin.return_value = AsyncContext()
        session.execute = AsyncMock(return_value=selected_job)
        session_factory = MagicMock(
            return_value=AsyncContext(session),
        )

        claimed = await claim_stale_published_job(
            session_factory,
            "watchdog-worker",
        )

        self.assertIsNotNone(claimed)
        assert claimed is not None
        self.assertEqual(claimed["status"], "failed")
        self.assertIsNotNone(claimed["finished_at"])
        self.assertIsNone(claimed["worker_id"])

    async def test_expired_lease_fails_after_last_attempt(self) -> None:
        now = datetime.now(UTC)
        job = SimpleNamespace(
            id="job-6",
            status="processing",
            attempts=2,
            max_attempts=2,
            worker_id="dead-worker",
            lease_expires_at=now - timedelta(seconds=1),
            heartbeat_at=now - timedelta(seconds=10),
            updated_at=now,
            error=None,
            finished_at=None,
        )
        selected_jobs = SimpleNamespace(
            scalars=lambda: [job],
        )
        session = MagicMock()
        session.begin.return_value = AsyncContext()
        session.execute = AsyncMock(return_value=selected_jobs)
        session_factory = MagicMock(
            return_value=AsyncContext(session),
        )

        result = await recover_expired_job_leases(session_factory)

        self.assertEqual(result.requeued, 0)
        self.assertEqual(result.failed, 1)
        self.assertEqual(job.status, "failed")
        self.assertIsNotNone(job.finished_at)


if __name__ == "__main__":
    unittest.main()
