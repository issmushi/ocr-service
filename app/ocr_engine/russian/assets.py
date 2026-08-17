"""Install and verify the pinned RussianDocsOCR model assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, NoReturn
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

_MODELS_ENV = "RUSSIAN_OCR_MODELS_DIR"
_PACKAGE_VERSION = "1.0.1"
_DOWNLOAD_RETRIES = 3
_DOWNLOAD_WORKERS = 4
_DOWNLOAD_TIMEOUT_SECONDS = 120
_CHUNK_SIZE = 1024 * 1024
_MANIFEST_PATH = Path(__file__).with_name("models.lock.json")
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MODELS_DIR = (
    _PROJECT_ROOT
    / "storage"
    / "models"
    / "russian-docs-ocr"
    / _PACKAGE_VERSION
    / "models"
)


class RussianOcrModelsError(RuntimeError):
    """Raised when pinned RussianDocsOCR models cannot be used."""


@dataclass(frozen=True, slots=True)
class _ModelAsset:
    path: PurePosixPath
    size: int
    sha256: str


def _load_manifest() -> tuple[str, tuple[_ModelAsset, ...]]:
    try:
        payload = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
        base_url = payload["source"]["base_url"]
        raw_files = payload["files"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise RussianOcrModelsError(
            f"Cannot read RussianDocsOCR model manifest: {error}"
        ) from error

    if not isinstance(base_url, str) or not base_url.startswith("https://"):
        raise RussianOcrModelsError("Model manifest has an invalid base URL")
    if not isinstance(raw_files, list) or not raw_files:
        raise RussianOcrModelsError("Model manifest contains no files")

    assets: list[_ModelAsset] = []
    paths: set[PurePosixPath] = set()
    try:
        for item in raw_files:
            path = PurePosixPath(item["path"])
            size = item["size"]
            sha256 = item["sha256"].lower()
            if (
                path.is_absolute()
                or ".." in path.parts
                or str(path) in {"", "."}
            ):
                raise ValueError(f"unsafe model path: {path}")
            if path in paths:
                raise ValueError(f"duplicate model path: {path}")
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise ValueError(f"invalid size for {path}")
            if len(sha256) != 64 or any(
                character not in "0123456789abcdef" for character in sha256
            ):
                raise ValueError(f"invalid SHA-256 for {path}")
            paths.add(path)
            assets.append(_ModelAsset(path, size, sha256))
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise RussianOcrModelsError(
            f"Invalid RussianDocsOCR model manifest: {error}"
        ) from error
    return base_url.rstrip("/") + "/", tuple(assets)


_BASE_URL, _MODEL_FILES = _load_manifest()


def models_dir() -> Path:
    """Return the configured directory containing the model tree."""
    configured = os.getenv(_MODELS_ENV, "").strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        return path.resolve()
    return _DEFAULT_MODELS_DIR


def _root_path(root: Path | None) -> Path:
    return models_dir() if root is None else Path(root).expanduser().resolve()


def _local_path(root: Path, asset: _ModelAsset) -> Path:
    return root.joinpath(*asset.path.parts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_problem(path: Path, asset: _ModelAsset) -> str | None:
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return "missing"
    except OSError as error:
        return f"cannot inspect ({error})"
    if not path.is_file():
        return "not a regular file"
    if size != asset.size:
        return f"size is {size}, expected {asset.size}"
    try:
        actual_hash = _sha256(path)
    except OSError as error:
        return f"cannot hash ({error})"
    if actual_hash != asset.sha256:
        return f"SHA-256 is {actual_hash}, expected {asset.sha256}"
    return None


def verify_models(root: Path | None = None) -> None:
    """Verify every pinned model file, raising on any discrepancy."""
    target_root = _root_path(root)
    problems: list[str] = []
    for asset in _MODEL_FILES:
        problem = _asset_problem(_local_path(target_root, asset), asset)
        if problem is not None:
            problems.append(f"{asset.path}: {problem}")
    if problems:
        details = "\n".join(f"- {problem}" for problem in problems)
        raise RussianOcrModelsError(
            "RussianDocsOCR model verification failed in "
            f"{target_root}:\n{details}"
        )


def _copy_and_hash(source: BinaryIO, destination: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with destination.open("wb") as output:
        while chunk := source.read(_CHUNK_SIZE):
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        output.flush()
        os.fsync(output.fileno())
    return size, digest.hexdigest()


def _validate_download(
    asset: _ModelAsset,
    size: int,
    sha256: str,
) -> None:
    if size != asset.size:
        raise RussianOcrModelsError(
            f"Downloaded {asset.path} has size {size}, expected {asset.size}"
        )
    if sha256 != asset.sha256:
        raise RussianOcrModelsError(
            f"Downloaded {asset.path} has SHA-256 {sha256}, "
            f"expected {asset.sha256}"
        )


def _remote_source(asset: _ModelAsset) -> AbstractContextManager[BinaryIO]:
    url = _BASE_URL + quote(asset.path.as_posix(), safe="/")
    request = Request(
        url, headers={"User-Agent": "ocr-service-model-fetcher/1"}
    )
    return urlopen(request, timeout=_DOWNLOAD_TIMEOUT_SECONDS)


def _install_asset(
    asset: _ModelAsset,
    target_root: Path,
    source_dir: Path | None,
) -> None:
    destination = _local_path(target_root, asset)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    attempts = 1 if source_dir is not None else _DOWNLOAD_RETRIES

    try:
        for attempt in range(1, attempts + 1):
            try:
                if source_dir is None:
                    with _remote_source(asset) as source:
                        size, sha256 = _copy_and_hash(source, temporary_path)
                else:
                    source_path = _local_path(source_dir, asset)
                    with source_path.open("rb") as source:
                        size, sha256 = _copy_and_hash(source, temporary_path)
                _validate_download(asset, size, sha256)
                os.replace(temporary_path, destination)
                return
            except (HTTPError, URLError, TimeoutError, OSError) as error:
                if attempt == attempts:
                    raise RussianOcrModelsError(
                        f"Cannot install {asset.path}: {error}"
                    ) from error
                time.sleep(0.25 * (2 ** (attempt - 1)))
            except RussianOcrModelsError:
                if attempt == attempts:
                    raise
                time.sleep(0.25 * (2 ** (attempt - 1)))
        raise AssertionError("unreachable")
    finally:
        temporary_path.unlink(missing_ok=True)


def fetch_models(
    root: Path | None = None,
    source_dir: Path | None = None,
) -> None:
    """Fetch missing or corrupt assets, then verify the complete model set."""
    target_root = _root_path(root)
    offline_root = (
        None if source_dir is None else Path(source_dir).expanduser().resolve()
    )
    pending = [
        asset
        for asset in _MODEL_FILES
        if _asset_problem(_local_path(target_root, asset), asset) is not None
    ]
    if pending:
        target_root.mkdir(parents=True, exist_ok=True)
        errors: list[str] = []
        workers = min(_DOWNLOAD_WORKERS, len(pending))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _install_asset,
                    asset,
                    target_root,
                    offline_root,
                ): asset
                for asset in pending
            }
            for future in as_completed(futures):
                asset = futures[future]
                try:
                    future.result()
                except Exception as error:  # noqa: BLE001
                    errors.append(f"{asset.path}: {error}")
        if errors:
            details = "\n".join(f"- {error}" for error in sorted(errors))
            raise RussianOcrModelsError(
                f"Could not install RussianDocsOCR models:\n{details}"
            )
    verify_models(target_root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage pinned RussianDocsOCR model assets."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("fetch", "check"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument(
            "--root",
            type=Path,
            help=(
                f"model directory (default: ${_MODELS_ENV} or {models_dir()})"
            ),
        )
        if command == "fetch":
            command_parser.add_argument(
                "--source-dir",
                type=Path,
                help="copy from a local model tree instead of the network",
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "fetch":
            fetch_models(args.root, args.source_dir)
        else:
            verify_models(args.root)
    except RussianOcrModelsError as error:
        print(error, file=sys.stderr)
        return 1
    root = _root_path(args.root)
    action = "installed" if args.command == "fetch" else "verified"
    print(f"RussianDocsOCR models {action}: {root}")
    return 0


def _run() -> NoReturn:
    raise SystemExit(main())


if __name__ == "__main__":
    _run()
