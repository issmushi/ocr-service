from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.core.images.decoding import InvalidImageError, decode_image

logger = logging.getLogger("uvicorn.error")
VlmModel = Literal["qwen3-vl-4b", "qwen3-vl-8b"]
VlmTask = Literal["ocr", "passport_fields"]
MODELS = ("qwen3-vl-4b", "qwen3-vl-8b")


class VlmUnavailableError(RuntimeError):
    pass


def create_engine(model: str) -> Any:
    if model in {"qwen3-vl-4b", "qwen3-vl-8b"}:
        from app.ocr_engine.vlm.qwen import QwenEngine

        return QwenEngine(model.rsplit("-", 1)[-1])
    raise ValueError("Unsupported VLM model")


async def _finish(future: asyncio.Future) -> Any:
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(future)
            break
        except asyncio.CancelledError:
            if future.cancelled():
                raise
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError
    return result


class VlmBackendRuntime:
    """Keep one model loaded; serialize its use on one thread."""

    def __init__(
        self, engine_factory: Callable[[str], Any] = create_engine
    ) -> None:
        self._engine_factory = engine_factory
        self._engine: Any = None
        self._model: str | None = None
        self._executor: ThreadPoolExecutor | None = None
        self.status = "stopped"

    @property
    def active_model(self) -> str | None:
        return self._model

    async def start(self) -> None:
        if self._executor is not None:
            raise RuntimeError("VLM backend has already started")
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="vlm-inference"
        )
        self.status = "ready"

    def _unload(self) -> None:
        if self._engine is not None:
            self._engine.stop()
        self._engine, self._model = None, None

    def _select(self, model: str) -> None:
        if self._model == model:
            return
        started = perf_counter()
        engine = None
        try:
            self._unload()
            engine = self._engine_factory(model)
            engine.start()
        except Exception as error:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    # Retain the owner so a later stop can finish releasing it.
                    self._engine, self._model = engine, None
            logger.error(
                "VLM load failed model=%s duration_seconds=%.3f error_type=%s",
                model,
                perf_counter() - started,
                type(error).__name__,
            )
            raise VlmUnavailableError(
                "Cannot load selected VLM model"
            ) from error
        self._engine, self._model = engine, model
        info = engine.runtime_info
        logger.info(
            "VLM loaded model=%s backend=%s device=%s duration_seconds=%.3f",
            model,
            info.get("backend", "unknown"),
            info.get("device", "unknown"),
            perf_counter() - started,
        )

    def _recognize(
        self,
        content: bytes,
        model: str,
        task: str = "ocr",
        cpu_only: bool = False,
        report_runtime: bool = False,
    ) -> dict[str, Any]:
        started = perf_counter()
        try:
            with decode_image(content) as image:
                self._select(model)
                if (
                    cpu_only
                    and self._engine.runtime_info.get("device") != "cpu"
                ):
                    raise VlmUnavailableError(
                        "CPU mode is required for this request"
                    )
                result = self._engine.recognize(image, task=task)
                if cpu_only or report_runtime:
                    result = {
                        **result,
                        "device": self._engine.runtime_info.get(
                            "device", "unknown"
                        ),
                    }
            logger.info(
                "VLM recognized model=%s blocks=%d duration_seconds=%.3f",
                model,
                len(result["items"]),
                perf_counter() - started,
            )
            return result
        except Exception as error:
            logger.warning(
                "VLM recognition failed model=%s duration_seconds=%.3f "
                "error_type=%s",
                model,
                perf_counter() - started,
                type(error).__name__,
            )
            raise

    async def recognize(
        self,
        content: bytes,
        model: str,
        *,
        task: str = "ocr",
        cpu_only: bool = False,
        report_runtime: bool = False,
    ) -> dict[str, Any]:
        if model not in MODELS:
            raise ValueError("Unsupported VLM model")
        if self.status != "ready" or self._executor is None:
            raise VlmUnavailableError("VLM backend is not ready")
        future = asyncio.get_running_loop().run_in_executor(
            self._executor,
            self._recognize,
            content,
            model,
            task,
            cpu_only,
            report_runtime,
        )
        future.add_done_callback(
            lambda done: None if done.cancelled() else done.exception()
        )
        return await asyncio.shield(future)

    async def stop(self) -> None:
        executor = self._executor
        if executor is None:
            return
        self.status = "stopping"
        try:
            await _finish(
                asyncio.get_running_loop().run_in_executor(
                    executor, self._unload
                )
            )
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
            self.status = "stopped"


def create_app(runtime: VlmBackendRuntime | None = None) -> FastAPI:
    runtime = runtime or VlmBackendRuntime()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await runtime.start()
        try:
            yield
        finally:
            await runtime.stop()

    app = FastAPI(title="Local VLM OCR", lifespan=lifespan)
    app.state.runtime = runtime

    @app.get("/health")
    async def health():
        return JSONResponse(
            {
                "status": runtime.status,
                "models": MODELS,
                "active_model": runtime.active_model,
            },
            status_code=200 if runtime.status == "ready" else 503,
        )

    @app.post("/recognize")
    async def recognize(
        image: UploadFile = File(...),
        model: VlmModel = Form(...),
        task: VlmTask = Form("ocr"),
        cpu_only: bool = Form(False),
        report_runtime: bool = Form(False),
    ):
        try:
            content = await image.read()
        finally:
            await image.close()
        try:
            return await runtime.recognize(
                content,
                model,
                task=task,
                cpu_only=cpu_only,
                report_runtime=report_runtime,
            )
        except InvalidImageError as exc:
            raise HTTPException(400, "Cannot read image") from exc
        except VlmUnavailableError as exc:
            raise HTTPException(
                503, "Selected VLM model is unavailable"
            ) from exc
        except TimeoutError as exc:
            raise HTTPException(504, "VLM inference timed out") from exc
        except Exception as exc:
            raise HTTPException(422, "VLM recognition failed") from exc

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8004, workers=1)
