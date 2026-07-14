from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aio_pika import DeliveryMode

from app.core.infrastructure.queue import (
    get_queue_status,
    publish_dead_letter,
    publish_job,
)


class QueueStatusTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_queue_and_consumer_counts(self) -> None:
        connection = AsyncMock()
        channel = AsyncMock()
        channel.declare_queue.side_effect = [
            SimpleNamespace(
                declaration_result=SimpleNamespace(
                    message_count=7,
                    consumer_count=2,
                )
            ),
            SimpleNamespace(
                declaration_result=SimpleNamespace(
                    message_count=3,
                    consumer_count=0,
                )
            ),
        ]
        connection.channel.return_value = channel

        with (
            patch(
                "app.core.infrastructure.queue.settings",
                SimpleNamespace(
                    rabbitmq_url="amqp://rabbitmq/",
                    ocr_queue_name="ocr.jobs",
                    ocr_dead_letter_queue_name="ocr.jobs.dlq",
                ),
            ),
            patch(
                "app.core.infrastructure.queue.aio_pika.connect_robust",
                new=AsyncMock(return_value=connection),
            ),
        ):
            result = await get_queue_status()

        self.assertEqual(result["queue_name"], "ocr.jobs")
        self.assertEqual(result["message_count"], 7)
        self.assertEqual(result["consumer_count"], 2)
        self.assertEqual(
            result["dead_letter_queue_name"],
            "ocr.jobs.dlq",
        )
        self.assertEqual(result["dead_letter_message_count"], 3)
        connection.close.assert_awaited_once()

    async def test_publishes_persistent_message_with_confirmation(
        self,
    ) -> None:
        connection = AsyncMock()
        channel = AsyncMock()
        exchange = AsyncMock()
        channel.default_exchange = exchange
        connection.channel.return_value = channel

        with (
            patch(
                "app.core.infrastructure.queue.settings",
                SimpleNamespace(
                    rabbitmq_url="amqp://rabbitmq/",
                    ocr_queue_name="ocr.jobs",
                ),
            ),
            patch(
                "app.core.infrastructure.queue.aio_pika.connect_robust",
                new=AsyncMock(return_value=connection),
            ),
        ):
            await publish_job({"job_id": "job-1"})

        connection.channel.assert_awaited_once_with(publisher_confirms=True)
        channel.declare_queue.assert_awaited_once_with(
            "ocr.jobs",
            durable=True,
        )
        message = exchange.publish.await_args.args[0]
        self.assertEqual(json.loads(message.body), {"job_id": "job-1"})
        self.assertEqual(message.delivery_mode, DeliveryMode.PERSISTENT)
        self.assertEqual(message.message_id, "job-1")
        self.assertEqual(message.correlation_id, "job-1")
        self.assertEqual(
            exchange.publish.await_args.kwargs,
            {"routing_key": "ocr.jobs", "mandatory": True},
        )
        connection.close.assert_awaited_once()

    async def test_publishes_original_body_to_durable_dlq(self) -> None:
        connection = AsyncMock()
        channel = AsyncMock()
        exchange = AsyncMock()
        channel.default_exchange = exchange
        connection.channel.return_value = channel

        with (
            patch(
                "app.core.infrastructure.queue.settings",
                SimpleNamespace(
                    rabbitmq_url="amqp://rabbitmq/",
                    ocr_queue_name="ocr.jobs",
                    ocr_dead_letter_queue_name="ocr.jobs.dlq",
                ),
            ),
            patch(
                "app.core.infrastructure.queue.aio_pika.connect_robust",
                new=AsyncMock(return_value=connection),
            ),
        ):
            await publish_dead_letter(
                b"not-json",
                reason="JSONDecodeError: invalid payload",
                message_id="message-1",
                correlation_id="request-1",
                content_type="application/json",
            )

        connection.channel.assert_awaited_once_with(publisher_confirms=True)
        channel.declare_queue.assert_awaited_once_with(
            "ocr.jobs.dlq",
            durable=True,
        )
        message = exchange.publish.await_args.args[0]
        self.assertEqual(message.body, b"not-json")
        self.assertEqual(message.delivery_mode, DeliveryMode.PERSISTENT)
        self.assertEqual(message.message_id, "message-1")
        self.assertEqual(message.correlation_id, "request-1")
        self.assertEqual(message.headers["x-source-queue"], "ocr.jobs")
        self.assertIn("JSONDecodeError", message.headers["x-failure-reason"])
        self.assertEqual(
            exchange.publish.await_args.kwargs,
            {"routing_key": "ocr.jobs.dlq", "mandatory": True},
        )
        connection.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
