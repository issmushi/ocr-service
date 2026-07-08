from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import signal
import threading
import traceback
from collections.abc import Callable
from multiprocessing.connection import Connection
from time import monotonic, perf_counter, sleep
from typing import Any, TypeVar, cast
from uuid import uuid4

ResultT = TypeVar("ResultT")
WorkerTarget = Callable[..., None]
Initializer = Callable[[], Any]
Dispatcher = Callable[[Any, str, tuple[Any, ...], dict[str, Any]], Any]


class InferenceBusyError(RuntimeError):
    """Raised when the single native inference slot is occupied."""


class InferenceTimeoutError(TimeoutError):
    """Raised after the native worker exceeded its hard deadline."""


class InferenceUnavailableError(RuntimeError):
    """Raised when the native worker is not ready or exited unexpectedly."""


class InferenceInputError(ValueError):
    """Represents an input validation error raised inside the worker."""


class RemoteInferenceError(RuntimeError):
    """Represents an inference error raised inside the worker process."""

    def __init__(
        self,
        error_type: str,
        message: str,
        remote_traceback: str | None = None,
    ) -> None:
        self.error_type = error_type
        self.remote_traceback = remote_traceback
        super().__init__(f"{error_type}: {message}" if message else error_type)


def _error_category(error: BaseException) -> str:
    if isinstance(error, ValueError):
        return "input"
    if isinstance(error, (ImportError, ModuleNotFoundError, OSError)):
        return "unavailable"
    return "internal"


def _send_worker_message(
    connection: Connection, message: dict[str, Any]
) -> bool:
    try:
        connection.send(message)
    except (BrokenPipeError, EOFError, OSError):
        return False
    return True


def serve_inference_worker(
    connection: Connection,
    initializer: Initializer,
    dispatcher: Dispatcher,
) -> None:
    """Initialize one engine and serve serializable inference commands."""

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        engine = initializer()
    except BaseException as error:
        _send_worker_message(
            connection,
            {
                "kind": "startup_error",
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        connection.close()
        return

    if not _send_worker_message(
        connection,
        {"kind": "ready", "worker_pid": os.getpid()},
    ):
        connection.close()
        return

    try:
        while True:
            try:
                command = connection.recv()
            except (EOFError, OSError):
                break

            if not isinstance(command, dict):
                continue
            if command.get("kind") == "shutdown":
                break
            if command.get("kind") != "request":
                continue

            request_id = str(command.get("request_id") or "")
            operation = str(command.get("operation") or "")
            args = tuple(command.get("args") or ())
            kwargs = dict(command.get("kwargs") or {})
            try:
                result = dispatcher(engine, operation, args, kwargs)
            except Exception as error:
                sent = _send_worker_message(
                    connection,
                    {
                        "kind": "error",
                        "request_id": request_id,
                        "category": _error_category(error),
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    },
                )
            else:
                sent = _send_worker_message(
                    connection,
                    {
                        "kind": "result",
                        "request_id": request_id,
                        "result": result,
                    },
                )
            if not sent:
                break
    finally:
        connection.close()


class InferenceExecutor:
    """Supervises one killable native inference process."""

    def __init__(
        self,
        engine: str,
        logger: logging.Logger,
        *,
        worker_target: WorkerTarget,
        worker_args: tuple[Any, ...] = (),
        startup_timeout_seconds: float = 300.0,
        shutdown_timeout_seconds: float = 1.0,
    ) -> None:
        self._engine = engine
        self._logger = logger
        self._worker_target = worker_target
        self._worker_args = worker_args
        self._startup_timeout_seconds = startup_timeout_seconds
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._context = multiprocessing.get_context("spawn")
        self._lock = threading.RLock()
        self._process: Any | None = None
        self._connection: Connection | None = None
        self._state = "stopped"
        self._last_error: str | None = None
        self._started_at: float | None = None
        self._request_active = False
        self._closing = False

    @property
    def busy(self) -> bool:
        with self._lock:
            self._refresh_state_locked()
            return self._request_active or self._state == "starting"

    @property
    def ready(self) -> bool:
        with self._lock:
            self._refresh_state_locked()
            return self._state == "ready"

    @property
    def status(self) -> str:
        with self._lock:
            self._refresh_state_locked()
            return self._state

    @property
    def worker_pid(self) -> int | None:
        with self._lock:
            process = self._process
            return (
                process.pid
                if process is not None and process.is_alive()
                else None
            )

    def start(self) -> None:
        self._closing = False
        with self._lock:
            self._refresh_state_locked()
            if self._state == "ready":
                return
            self._terminate_locked(graceful=False)
            self._spawn_locked()

        deadline = monotonic() + self._startup_timeout_seconds
        while True:
            with self._lock:
                self._refresh_state_locked()
                if self._state == "ready":
                    self._logger.info(
                        "Inference worker ready engine=%s worker_pid=%s",
                        self._engine,
                        self.worker_pid,
                    )
                    return
                if self._state == "unavailable":
                    raise InferenceUnavailableError(
                        self._last_error
                        or f"{self._engine} worker failed to start"
                    )

            if monotonic() >= deadline:
                with self._lock:
                    self._last_error = (
                        f"{self._engine} worker startup exceeded "
                        f"{self._startup_timeout_seconds:.1f} seconds"
                    )
                    self._terminate_locked(graceful=False)
                    self._state = "unavailable"
                raise InferenceUnavailableError(self._last_error)
            sleep(0.05)

    def stop(self) -> None:
        self._closing = True
        with self._lock:
            self._terminate_locked(graceful=not self._request_active)
            self._request_active = False
            self._state = "stopped"

    def restart(self) -> None:
        with self._lock:
            self._terminate_locked(graceful=False)
            if self._closing:
                self._state = "stopped"
                return
            self._spawn_locked()
        self._logger.info(
            "Inference worker restart started engine=%s", self._engine
        )

    async def run(
        self,
        operation: str,
        /,
        *args: Any,
        timeout_seconds: float,
        **kwargs: Any,
    ) -> ResultT:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        connection, process = self._begin_request()
        request_id = uuid4().hex
        command = {
            "kind": "request",
            "request_id": request_id,
            "operation": operation,
            "args": args,
            "kwargs": kwargs,
        }
        started_at = perf_counter()
        self._logger.debug(
            "Inference slot acquired engine=%s operation=%s worker_pid=%s",
            self._engine,
            operation,
            process.pid,
        )

        try:
            await asyncio.to_thread(connection.send, command)
            response = await self._receive_response(
                connection,
                process,
                request_id,
                timeout_seconds,
            )
            return cast(ResultT, self._decode_response(response))
        except InferenceTimeoutError:
            self._logger.error(
                "Inference hard timeout engine=%s operation=%s worker_pid=%s "
                "timeout_seconds=%.3f",
                self._engine,
                operation,
                process.pid,
                timeout_seconds,
            )
            await self._restart_after_failure()
            raise
        except asyncio.CancelledError:
            self._logger.warning(
                "Inference request cancelled; terminating worker engine=%s "
                "operation=%s worker_pid=%s",
                self._engine,
                operation,
                process.pid,
            )
            await self._restart_after_failure()
            raise
        except (
            BrokenPipeError,
            EOFError,
            OSError,
            InferenceUnavailableError,
        ) as error:
            self._logger.error(
                "Inference worker connection lost engine=%s operation=%s "
                "worker_pid=%s error=%s",
                self._engine,
                operation,
                process.pid,
                error,
            )
            await self._restart_after_failure()
            raise InferenceUnavailableError(
                f"{self._engine} worker exited during inference"
            ) from error
        finally:
            duration_seconds = perf_counter() - started_at
            with self._lock:
                self._request_active = False
            self._logger.debug(
                "Inference slot released engine=%s operation=%s "
                "duration_seconds=%.3f",
                self._engine,
                operation,
                duration_seconds,
            )

    async def _restart_after_failure(self) -> None:
        try:
            await asyncio.to_thread(self.restart)
        except Exception as error:
            with self._lock:
                self._last_error = f"worker restart failed: {error}"
                self._state = "unavailable"
            self._logger.exception(
                "Inference worker restart failed engine=%s",
                self._engine,
            )

    def _begin_request(self) -> tuple[Connection, Any]:
        with self._lock:
            self._refresh_state_locked()
            if self._request_active or self._state == "starting":
                raise InferenceBusyError(f"{self._engine} is busy")
            if self._state != "ready":
                raise InferenceUnavailableError(
                    self._last_error or f"{self._engine} worker is not ready"
                )
            connection = self._connection
            process = self._process
            if connection is None or process is None or not process.is_alive():
                raise InferenceUnavailableError(
                    f"{self._engine} worker is not available"
                )
            self._request_active = True
            return connection, process

    async def _receive_response(
        self,
        connection: Connection,
        process: Any,
        request_id: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        deadline = monotonic() + timeout_seconds
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise InferenceTimeoutError(
                    f"{self._engine} inference exceeded "
                    f"{timeout_seconds:.1f} seconds"
                )
            if not process.is_alive():
                raise InferenceUnavailableError(
                    f"{self._engine} worker exited with code "
                    f"{process.exitcode}"
                )

            readable = await asyncio.to_thread(
                connection.poll,
                min(0.1, remaining),
            )
            if not readable:
                continue

            response = await asyncio.to_thread(connection.recv)
            if not isinstance(response, dict):
                raise InferenceUnavailableError(
                    f"{self._engine} worker returned an invalid response"
                )
            if response.get("request_id") != request_id:
                raise InferenceUnavailableError(
                    f"{self._engine} worker returned a mismatched response"
                )
            return response

    def _decode_response(self, response: dict[str, Any]) -> Any:
        if response.get("kind") == "result":
            return response.get("result")
        if response.get("kind") != "error":
            raise InferenceUnavailableError(
                f"{self._engine} worker returned an invalid response kind"
            )

        category = str(response.get("category") or "internal")
        error_type = str(response.get("error_type") or "InferenceError")
        message = str(response.get("message") or "")
        remote_traceback = response.get("traceback")
        if category == "input":
            raise InferenceInputError(message)
        if category == "unavailable":
            raise InferenceUnavailableError(message)
        raise RemoteInferenceError(error_type, message, remote_traceback)

    def _spawn_locked(self) -> None:
        parent_connection, child_connection = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=self._worker_target,
            args=(child_connection, *self._worker_args),
            name=f"{self._engine}-inference",
            daemon=True,
        )
        try:
            process.start()
        except Exception:
            parent_connection.close()
            child_connection.close()
            self._state = "unavailable"
            raise

        child_connection.close()
        self._connection = parent_connection
        self._process = process
        self._state = "starting"
        self._last_error = None
        self._started_at = monotonic()
        self._logger.info(
            "Inference worker starting engine=%s worker_pid=%s",
            self._engine,
            process.pid,
        )

    def _refresh_state_locked(self) -> None:
        process = self._process
        connection = self._connection
        if process is None:
            return

        if self._state == "starting" and connection is not None:
            if connection.poll(0):
                try:
                    message = connection.recv()
                except (EOFError, OSError) as error:
                    self._mark_unavailable_locked(str(error))
                    return
                if (
                    isinstance(message, dict)
                    and message.get("kind") == "ready"
                ):
                    self._state = "ready"
                    self._started_at = None
                    return
                if isinstance(message, dict):
                    error_type = str(
                        message.get("error_type") or "StartupError"
                    )
                    error_message = str(message.get("message") or "")
                    self._mark_unavailable_locked(
                        f"{error_type}: {error_message}".rstrip(": ")
                    )
                    return
                self._mark_unavailable_locked(
                    "invalid worker startup response"
                )
                return

            if (
                self._started_at is not None
                and monotonic() - self._started_at
                > self._startup_timeout_seconds
            ):
                self._mark_unavailable_locked("worker startup timed out")
                return

        if self._state in {"starting", "ready"} and not process.is_alive():
            self._mark_unavailable_locked(
                f"worker exited with code {process.exitcode}"
            )

    def _mark_unavailable_locked(self, message: str) -> None:
        self._last_error = message
        self._terminate_locked(graceful=False)
        self._state = "unavailable"
        self._logger.error(
            "Inference worker unavailable engine=%s error=%s",
            self._engine,
            message,
        )

    def _terminate_locked(self, *, graceful: bool) -> None:
        process = self._process
        connection = self._connection

        if process is not None and process.is_alive():
            if graceful and connection is not None:
                try:
                    connection.send({"kind": "shutdown"})
                except (BrokenPipeError, EOFError, OSError):
                    pass
                process.join(timeout=self._shutdown_timeout_seconds)

            if process.is_alive():
                if graceful:
                    process.terminate()
                else:
                    process.kill()
                process.join(timeout=self._shutdown_timeout_seconds)
            if process.is_alive():
                process.kill()
                process.join(timeout=self._shutdown_timeout_seconds)

        if connection is not None:
            connection.close()
        if process is not None:
            try:
                process.close()
            except ValueError:
                self._logger.error(
                    "Inference worker could not be closed "
                    "engine=%s worker_pid=%s",
                    self._engine,
                    process.pid,
                )
        self._connection = None
        self._process = None
        self._started_at = None
