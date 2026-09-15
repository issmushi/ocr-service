"""Download pinned public OCR weights without handling document images."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "storage/vlm/huggingface"))

MODELS = {
    "qwen3-vl-4b": (
        "Qwen/Qwen3-VL-4B-Instruct-GGUF",
        "1cd86afb9a95c410a6038ab3b40d8b578c892266",
        [
            "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
            "mmproj-Qwen3VL-4B-Instruct-F16.gguf",
            "LICENSE",
        ],
    ),
    "qwen3-vl-8b": (
        "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "f982a07559d4a2f6c8744d840bf6fccab30eea96",
        [
            "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
            "mmproj-Qwen3VL-8B-Instruct-F16.gguf",
            "LICENSE",
        ],
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["all", *MODELS], default="all")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    selected = (
        MODELS if args.model == "all" else {args.model: MODELS[args.model]}
    )
    for name, (repo, revision, patterns) in selected.items():
        print(f"Downloading {name} ({repo}@{revision})", flush=True)
        path = snapshot_download(
            repo,
            revision=revision,
            allow_patterns=patterns,
        )
        print(f"Ready {name}: {path}", flush=True)


if __name__ == "__main__":
    main()
