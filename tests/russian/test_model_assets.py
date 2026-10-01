from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path, PurePosixPath
from unittest.mock import patch
from urllib.error import URLError

from app.ocr_engine.russian import assets


def model_asset(path: str, content: bytes) -> assets._ModelAsset:
    return assets._ModelAsset(
        PurePosixPath(path),
        len(content),
        hashlib.sha256(content).hexdigest(),
    )


class ModelManifestTests(unittest.TestCase):
    def test_application_package_does_not_embed_model_binaries(self) -> None:
        russian_engine = Path(assets.__file__).parent
        embedded = [
            path
            for path in russian_engine.rglob("*")
            if path.suffix.lower()
            in {".onnx", ".pkl", ".mlmodel", ".tflite", ".ir"}
        ]

        self.assertEqual(embedded, [])

    def test_manifest_pins_the_complete_onnx_model_set(self) -> None:
        manifest = json.loads(
            Path(assets.__file__)
            .with_name("models.lock.json")
            .read_text(encoding="utf-8")
        )
        expected_sets = {
            "Borders",
            "TextFields",
            "DocTypeAngles",
            "Words",
            "OCR/cyrillic_accurate",
            "OCR/latin_accurate",
        }
        expected_paths = {
            f"{model}/ONNX/{name}"
            for model in expected_sets
            for name in ("model.json", "model.onnx")
        } | {"DocTypeAngles/ONNX/resources/centers.npz"}
        self.assertEqual(manifest["version"], "4.6.1")
        self.assertEqual(manifest["models_version"], "v8")
        self.assertEqual(
            manifest["source"]["source_commit"],
            "bd0028aa8e9d3a30d2de4df8b672185bfcd984ba",
        )
        self.assertEqual(manifest["source"]["license"]["spdx"], "MIT")
        self.assertIn(
            "/releases/download/models-v8/", manifest["source"]["base_url"]
        )
        self.assertEqual(
            {item["path"] for item in manifest["files"]}, expected_paths
        )
        self.assertEqual(len(manifest["files"]), 13)
        self.assertEqual(
            sum(item["size"] for item in manifest["files"]), 141918502
        )
        for item in manifest["files"]:
            self.assertEqual(item["asset"], item["path"].replace("/", "__"))
        for item in manifest["files"]:
            self.assertGreater(item["size"], 0)
            self.assertRegex(item["sha256"], r"^[0-9a-f]{64}$")

    def test_models_dir_supports_default_absolute_and_relative_env_paths(
        self,
    ) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                assets.models_dir(),
                assets._PROJECT_ROOT
                / "storage/models/russian-docs-ocr/4.6.1/models",
            )
        with tempfile.TemporaryDirectory() as temporary_directory:
            absolute = Path(temporary_directory) / "models"
            with patch.dict(
                "os.environ",
                {"RUSSIAN_OCR_MODELS_DIR": str(absolute)},
                clear=True,
            ):
                self.assertEqual(assets.models_dir(), absolute.resolve())
        with patch.dict(
            "os.environ",
            {"RUSSIAN_OCR_MODELS_DIR": "custom/models"},
            clear=True,
        ):
            self.assertEqual(
                assets.models_dir(),
                (assets._PROJECT_ROOT / "custom/models").resolve(),
            )


class ModelAssetManagerTests(unittest.TestCase):
    def test_verify_reports_every_missing_or_corrupt_asset(self) -> None:
        first = model_asset("one/model.bin", b"first")
        second = model_asset("two/model.bin", b"second")
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_path = root / first.path
            first_path.parent.mkdir(parents=True)
            first_path.write_bytes(b"wrong")
            with (
                patch.object(assets, "_MODEL_FILES", (first, second)),
                self.assertRaises(assets.RussianOcrModelsError) as raised,
            ):
                assets.verify_models(root)

        message = str(raised.exception)
        self.assertIn("one/model.bin: SHA-256", message)
        self.assertIn("two/model.bin: missing", message)

    def test_offline_fetch_replaces_corrupt_and_skips_valid_files(
        self,
    ) -> None:
        first_data = b"first asset"
        second_data = b"second asset"
        first = model_asset("one/model.bin", first_data)
        second = model_asset("OCR/eng+nums/model.bin", second_data)
        with (
            tempfile.TemporaryDirectory() as target_directory,
            tempfile.TemporaryDirectory() as source_directory,
        ):
            target = Path(target_directory)
            source = Path(source_directory)
            for asset, content in (
                (first, first_data),
                (second, second_data),
            ):
                source_path = source / asset.path
                source_path.parent.mkdir(parents=True, exist_ok=True)
                source_path.write_bytes(content)
            valid_path = target / first.path
            valid_path.parent.mkdir(parents=True)
            valid_path.write_bytes(first_data)
            corrupt_path = target / second.path
            corrupt_path.parent.mkdir(parents=True)
            corrupt_path.write_bytes(b"corrupt")

            with (
                patch.object(assets, "_MODEL_FILES", (first, second)),
                patch.object(
                    assets,
                    "_install_asset",
                    wraps=assets._install_asset,
                ) as install,
            ):
                assets.fetch_models(target, source)
                assets.verify_models(target)
                assets.fetch_models(target, source)

            self.assertEqual(install.call_count, 1)
            self.assertEqual(corrupt_path.read_bytes(), second_data)
            self.assertFalse(list(target.rglob("*.tmp")))

    def test_failed_offline_fetch_keeps_existing_file_unchanged(self) -> None:
        asset = model_asset("model.bin", b"expected")
        with (
            tempfile.TemporaryDirectory() as target_directory,
            tempfile.TemporaryDirectory() as source_directory,
        ):
            target_path = Path(target_directory) / asset.path
            target_path.write_bytes(b"existing")
            source_path = Path(source_directory) / asset.path
            source_path.write_bytes(b"invalid!")

            with (
                patch.object(assets, "_MODEL_FILES", (asset,)),
                self.assertRaises(assets.RussianOcrModelsError),
            ):
                assets.fetch_models(
                    Path(target_directory), Path(source_directory)
                )

            self.assertEqual(target_path.read_bytes(), b"existing")
            self.assertFalse(list(Path(target_directory).rglob("*.tmp")))

    def test_network_fetch_retries_and_installs_atomically(self) -> None:
        content = b"downloaded model"
        asset = model_asset("model.bin", content)
        calls = 0

        def remote_source(
            _asset: assets._ModelAsset,
        ) -> nullcontext[io.BytesIO]:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise URLError("temporary failure")
            return nullcontext(io.BytesIO(content))

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with (
                patch.object(assets, "_MODEL_FILES", (asset,)),
                patch.object(assets, "_remote_source", remote_source),
                patch.object(assets.time, "sleep") as sleep,
            ):
                assets.fetch_models(root)

            self.assertEqual((root / asset.path).read_bytes(), content)
            self.assertEqual(calls, 3)
            self.assertEqual(sleep.call_count, 2)
            self.assertFalse(list(root.rglob("*.tmp")))

    def test_cli_has_stable_success_and_failure_exit_codes(self) -> None:
        output = io.StringIO()
        with (
            patch.object(assets, "verify_models") as verify,
            redirect_stdout(output),
        ):
            self.assertEqual(assets.main(["check", "--root", "/models"]), 0)
        verify.assert_called_once_with(Path("/models"))
        self.assertIn("models verified", output.getvalue())

        error = io.StringIO()
        with (
            patch.object(
                assets,
                "verify_models",
                side_effect=assets.RussianOcrModelsError("broken"),
            ),
            redirect_stderr(error),
        ):
            self.assertEqual(assets.main(["check", "--root", "/models"]), 1)
        self.assertEqual(error.getvalue().strip(), "broken")

        with patch.object(assets, "fetch_models") as fetch:
            self.assertEqual(
                assets.main(
                    [
                        "fetch",
                        "--root",
                        "/models",
                        "--source-dir",
                        "/offline",
                    ]
                ),
                0,
            )
        fetch.assert_called_once_with(Path("/models"), Path("/offline"))


if __name__ == "__main__":
    unittest.main()
