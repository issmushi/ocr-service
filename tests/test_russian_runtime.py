from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.core.config import settings
from app.core.errors import AppError, ErrorCode
from app.core.infrastructure.inference import (
    InferenceInputError,
    InferenceTimeoutError,
    InferenceUnavailableError,
    RemoteInferenceError,
)
from app.modules.russian.runtime import RussianRuntime


@pytest.mark.parametrize(
    "error,code",
    [
        (InferenceTimeoutError(), ErrorCode.OCR_TIMEOUT),
        (InferenceUnavailableError(), ErrorCode.OCR_ENGINE_UNAVAILABLE),
        (InferenceInputError(), ErrorCode.IMAGE_READ_ERROR),
        (RemoteInferenceError("RuntimeError", "test"), ErrorCode.OCR_FAILED),
    ],
)
def test_worker_errors_have_standard_api_codes(error, code):
    runtime = RussianRuntime()
    executor = SimpleNamespace(run=AsyncMock(side_effect=error))
    with patch.object(
        runtime, "_ensure_ready", new=AsyncMock(return_value=executor)
    ):
        with pytest.raises(AppError) as caught:
            asyncio.run(
                runtime.run("passport_text", b"image", timeout_seconds=1)
            )
    assert caught.value.code == code


def test_requests_are_serialized_and_worker_is_reused():
    async def scenario():
        runtime = RussianRuntime()
        active = peak = 0

        async def recognize(*args, **kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return {"items": [], "passport": {}}

        executor = SimpleNamespace(run=AsyncMock(side_effect=recognize))
        with patch.object(
            runtime, "_ensure_ready", new=AsyncMock(return_value=executor)
        ):
            results = await asyncio.gather(
                *(
                    runtime.run("passport_text", b"image", timeout_seconds=1)
                    for _ in range(3)
                )
            )
        assert peak == 1
        assert executor.run.await_count == 3
        assert all(
            call.args == ("passport_text", b"image")
            for call in executor.run.await_args_list
        )
        assert len(results) == 3

    asyncio.run(scenario())


def test_start_uses_shared_supervisor_and_configured_device():
    with patch("app.modules.russian.runtime.InferenceExecutor") as factory:
        runtime = RussianRuntime()
    assert factory.call_args.args[0] == "russian-docs-ocr"
    assert factory.call_args.kwargs["worker_args"] == (
        settings.russian_ocr_device,
    )
    assert runtime._executor is factory.return_value


def test_shutdown_blocks_new_inference():
    async def scenario():
        runtime = RussianRuntime()
        runtime._executor = Mock()
        await runtime.stop()
        with pytest.raises(AppError) as caught:
            await runtime.run("passport_text", b"image", timeout_seconds=1)
        assert caught.value.code == ErrorCode.OCR_ENGINE_UNAVAILABLE
        runtime._executor.run.assert_not_called()

    asyncio.run(scenario())
