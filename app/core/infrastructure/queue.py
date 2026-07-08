from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import aio_pika

from app.core.config import settings


async def get_queue_status() -> dict[str, int | str]:
    connection = await aio_pika.connect_robust(settings.rabbitmq_url)
    try:
        channel = await connection.channel()
        queue = await channel.declare_queue(
            settings.ocr_queue_name,
            durable=True,
        )
        dead_letter_queue = await channel.declare_queue(
            settings.ocr_dead_letter_queue_name,
            durable=True,
        )
        declaration = queue.declaration_result
        dead_letter_declaration = dead_letter_queue.declaration_result
        return {
            "queue_name": settings.ocr_queue_name,
            "message_count": int(
                getattr(declaration, "message_count", 0) or 0
            ),
            "consumer_count": int(
                getattr(declaration, "consumer_count", 0) or 0
            ),
            "dead_letter_queue_name": settings.ocr_dead_letter_queue_name,
            "dead_letter_message_count": int(
                getattr(dead_letter_declaration, "message_count", 0) or 0
            ),
        }
    finally:
        await connection.close()


async def check_queue() -> bool:
    await get_queue_status()
    return True


async def publish_job(payload: dict[str, Any]) -> None:
    job_id = str(payload["job_id"])
    connection = await aio_pika.connect_robust(settings.rabbitmq_url)
    try:
        channel = await connection.channel(publisher_confirms=True)
        await channel.declare_queue(settings.ocr_queue_name, durable=True)
        message = aio_pika.Message(
            json.dumps(payload).encode("utf-8"),
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            message_id=job_id,
            correlation_id=job_id,
        )
        await channel.default_exchange.publish(
            message,
            routing_key=settings.ocr_queue_name,
            mandatory=True,
        )
    finally:
        await connection.close()


async def publish_dead_letter(
    body: bytes,
    *,
    reason: str,
    message_id: str | None = None,
    correlation_id: str | None = None,
    content_type: str | None = None,
) -> None:
    connection = await aio_pika.connect_robust(settings.rabbitmq_url)
    try:
        channel = await connection.channel(publisher_confirms=True)
        await channel.declare_queue(
            settings.ocr_dead_letter_queue_name,
            durable=True,
        )
        message = aio_pika.Message(
            body,
            content_type=content_type or "application/octet-stream",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            message_id=message_id,
            correlation_id=correlation_id,
            headers={
                "x-source-queue": settings.ocr_queue_name,
                "x-failure-reason": reason[:512],
                "x-failed-at": datetime.now(UTC).isoformat(),
            },
        )
        await channel.default_exchange.publish(
            message,
            routing_key=settings.ocr_dead_letter_queue_name,
            mandatory=True,
        )
    finally:
        await connection.close()
