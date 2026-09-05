from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.modules.tasks.exceptions import JobLeaseBusy, JobLeaseLost
from app.workers.ocr_worker import (
    execute_ocr_job,
    handle_message,
    process_job,
    reconcile_stale_queued_job,
    run_with_job_lease,
)


def processing_job() -> dict[str, object]:
    return {
        "id": "job-1",
        "mode": "passport_mrz",
        "status": "processing",
        "filename": "document.jpg",
        "content_type": "image/jpeg",
        "image_path": "/tmp/document.jpg",
        "result": None,
        "error": None,
        "attempts": 1,
        "max_attempts": 2,
        "created_at": None,
        "updated_at": None,
        "started_at": datetime.now(UTC),
        "finished_at": None,
        "worker_id": "other-worker",
        "lease_expires_at": datetime.now(UTC) + timedelta(seconds=30),
        "heartbeat_at": datetime.now(UTC),
    }


class WorkerLeaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_execution_preserves_public_result_envelope(self) -> None:
        job = processing_job()
        job.update(mode="full_v1", options={"model": "qwen3-vl-8b"})
        path = MagicMock(spec=Path)
        path.read_bytes.return_value = b"original-image"
        payload = {
            "format": "json",
            "response": {
                "success": True,
                "data": {"text": "OCR"},
                "error": None,
            },
        }
        dispatcher = AsyncMock(return_value=payload)
        with patch(
            "app.workers.ocr_worker.execute_task_operation", dispatcher
        ):
            result = await execute_ocr_job("job-1", job, path)
        dispatcher.assert_awaited_once_with(job, b"original-image")
        self.assertIs(result, payload)

    async def test_v1_completion_losing_ownership_does_not_delete_input(
        self,
    ) -> None:
        job = processing_job()
        job.update(mode="crop_mrz", worker_id="worker-1", options={})
        payload = {"format": "jpeg", "content_base64": "anBlZw=="}

        async def keep_lease(*_args):
            await asyncio.Event().wait()

        with (
            patch(
                "app.workers.ocr_worker.execute_ocr_job",
                new=AsyncMock(return_value=payload),
            ),
            patch("app.workers.ocr_worker.maintain_job_lease", keep_lease),
            patch(
                "app.workers.ocr_worker.mark_done",
                new=AsyncMock(return_value=False),
            ) as done,
            patch("app.workers.ocr_worker.delete_upload") as delete,
        ):
            await process_job("job-1", worker_id="worker-1", claimed_job=job)
        self.assertEqual(
            done.await_args.args[1:], ("job-1", payload, "worker-1")
        )
        delete.assert_not_called()

    async def test_execution_outlasts_old_deadlines_while_lease_renews(
        self,
    ) -> None:
        heartbeat = asyncio.Event()

        async def renew(*_args):
            heartbeat.set()
            return True

        async def operation():
            await heartbeat.wait()
            await asyncio.sleep(0.01)
            return {"format": "json", "response": {"text": "result"}}

        with (
            patch(
                "app.workers.ocr_worker.renew_job_lease",
                new=AsyncMock(side_effect=renew),
            ) as renewal,
            patch(
                "app.workers.ocr_worker.settings",
                SimpleNamespace(
                    ocr_worker_heartbeat_seconds=0.001,
                ),
            ),
        ):
            result = await run_with_job_lease(
                "job-long", "worker-1", operation()
            )
        self.assertEqual(result["response"]["text"], "result")
        self.assertGreaterEqual(renewal.await_count, 1)

    async def test_malformed_message_is_confirmed_in_dlq_before_ack(
        self,
    ) -> None:
        message = AsyncMock()
        message.body = b'{"unexpected": true}'
        message.delivery_tag = 10
        message.redelivered = False
        message.message_id = "message-10"
        message.correlation_id = "request-10"
        message.content_type = "application/json"
        publisher = AsyncMock()

        with patch(
            "app.workers.ocr_worker.publish_dead_letter",
            new=publisher,
        ):
            await handle_message(message)

        publisher.assert_awaited_once()
        self.assertEqual(
            publisher.await_args.args[0],
            message.body,
        )
        message.ack.assert_awaited_once()
        message.nack.assert_not_awaited()
        message.reject.assert_not_awaited()

    async def test_malformed_message_is_requeued_when_dlq_fails(self) -> None:
        message = AsyncMock()
        message.body = b"not-json"
        message.delivery_tag = 11
        message.redelivered = False
        message.message_id = None
        message.correlation_id = None
        message.content_type = None

        with (
            patch(
                "app.workers.ocr_worker.publish_dead_letter",
                new=AsyncMock(
                    side_effect=RuntimeError("RabbitMQ unavailable")
                ),
            ),
            patch(
                "app.workers.ocr_worker.asyncio.sleep",
                new=AsyncMock(),
            ),
        ):
            await handle_message(message)

        message.nack.assert_awaited_once_with(requeue=True)
        message.ack.assert_not_awaited()
        message.reject.assert_not_awaited()

    async def test_watchdog_processes_preclaimed_job_without_publish(
        self,
    ) -> None:
        job = processing_job()
        claimed_job = AsyncMock(return_value=job)
        processor = AsyncMock()

        with (
            patch(
                "app.workers.ocr_worker.claim_stale_published_job",
                new=claimed_job,
            ),
            patch(
                "app.workers.ocr_worker.process_job",
                new=processor,
            ),
        ):
            recovered = await reconcile_stale_queued_job(asyncio.Lock())

        self.assertTrue(recovered)
        claimed_job.assert_awaited_once()
        processor.assert_awaited_once()
        self.assertEqual(processor.await_args.args[0], "job-1")
        self.assertIs(processor.await_args.kwargs["claimed_job"], job)

    async def test_active_processing_lease_is_not_acknowledged(self) -> None:
        job = processing_job()
        with (
            patch(
                "app.workers.ocr_worker.claim_job",
                new=AsyncMock(return_value=0),
            ),
            patch(
                "app.workers.ocr_worker.get_job",
                new=AsyncMock(return_value=job),
            ),
        ):
            with self.assertRaises(JobLeaseBusy):
                await process_job("job-1", worker_id="new-worker")

    async def test_busy_lease_requeues_rabbit_message(self) -> None:
        message = AsyncMock()
        message.body = b'{"job_id": "job-1"}'
        message.delivery_tag = 1
        message.redelivered = True
        busy = JobLeaseBusy(
            "job-1",
            datetime.now(UTC) + timedelta(seconds=30),
            0.1,
        )

        with (
            patch(
                "app.workers.ocr_worker.process_job",
                new=AsyncMock(side_effect=busy),
            ),
            patch(
                "app.workers.ocr_worker.asyncio.sleep",
                new=AsyncMock(),
            ),
        ):
            await handle_message(message)

        message.nack.assert_awaited_once_with(requeue=True)
        message.ack.assert_not_awaited()

    async def test_lost_lease_requeues_rabbit_message(self) -> None:
        message = AsyncMock()
        message.body = b'{"job_id": "job-2"}'
        message.delivery_tag = 2
        message.redelivered = True

        with (
            patch(
                "app.workers.ocr_worker.process_job",
                new=AsyncMock(side_effect=JobLeaseLost("job-2")),
            ),
            patch(
                "app.workers.ocr_worker.asyncio.sleep",
                new=AsyncMock(),
            ),
        ):
            await handle_message(message)

        message.nack.assert_awaited_once_with(requeue=True)
        message.ack.assert_not_awaited()

    async def test_lost_heartbeat_cancels_ocr_operation(self) -> None:
        operation_cancelled = False
        operation_started = asyncio.Event()

        async def operation() -> dict[str, object]:
            nonlocal operation_cancelled
            try:
                operation_started.set()
                await asyncio.Event().wait()
            finally:
                operation_cancelled = True

        async def lose_lease(_job_id: str, _worker_id: str) -> None:
            await operation_started.wait()
            raise JobLeaseLost("job-3")

        with patch(
            "app.workers.ocr_worker.maintain_job_lease",
            new=lose_lease,
        ):
            with self.assertRaises(JobLeaseLost):
                await run_with_job_lease(
                    "job-3",
                    "worker-1",
                    operation(),
                )

        self.assertTrue(operation_cancelled)


if __name__ == "__main__":
    unittest.main()
