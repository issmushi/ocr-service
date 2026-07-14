from __future__ import annotations

import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from app.modules.tasks.outbox import dispatch_pending_outbox_events


class AsyncContext:
    def __init__(self, value: object = None) -> None:
        self.value = value

    async def __aenter__(self) -> object:
        return self.value

    async def __aexit__(self, *_args: object) -> None:
        return None


def make_session_factory(event: object) -> MagicMock:
    session = MagicMock()
    session.begin.return_value = AsyncContext()
    session.execute = AsyncMock(
        return_value=SimpleNamespace(
            scalar_one_or_none=lambda: event,
        )
    )
    return MagicMock(return_value=AsyncContext(session))


class OutboxTests(unittest.IsolatedAsyncioTestCase):
    async def test_marks_event_published_after_confirmed_delivery(
        self,
    ) -> None:
        event = SimpleNamespace(
            job_id="job-1",
            payload={"job_id": "job-1"},
            attempts=0,
            next_attempt_at=datetime.now(UTC),
            last_error=None,
            published_at=None,
        )
        publisher = AsyncMock()

        result = await dispatch_pending_outbox_events(
            limit=1,
            session_factory=make_session_factory(event),
            publisher=publisher,
        )

        self.assertEqual(result.published, 1)
        self.assertEqual(result.failed, 0)
        self.assertEqual(event.attempts, 1)
        self.assertIsNotNone(event.published_at)
        self.assertIsNone(event.last_error)
        publisher.assert_awaited_once_with({"job_id": "job-1"})

    async def test_schedules_retry_when_rabbitmq_is_unavailable(self) -> None:
        current_time = datetime.now(UTC)
        event = SimpleNamespace(
            job_id="job-2",
            payload={"job_id": "job-2"},
            attempts=0,
            next_attempt_at=current_time,
            last_error=None,
            published_at=None,
        )
        publisher = AsyncMock(side_effect=ConnectionError("unavailable"))

        result = await dispatch_pending_outbox_events(
            limit=1,
            session_factory=make_session_factory(event),
            publisher=publisher,
        )

        self.assertEqual(result.published, 0)
        self.assertEqual(result.failed, 1)
        self.assertEqual(event.attempts, 1)
        self.assertIsNone(event.published_at)
        self.assertGreater(event.next_attempt_at, current_time)
        self.assertIn("ConnectionError", event.last_error)

    async def test_empty_outbox_does_not_publish(self) -> None:
        publisher = AsyncMock()

        result = await dispatch_pending_outbox_events(
            limit=10,
            session_factory=make_session_factory(None),
            publisher=publisher,
        )

        self.assertEqual(result.published, 0)
        self.assertEqual(result.failed, 0)
        publisher.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
