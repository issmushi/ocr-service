from __future__ import annotations

import os

# The complete test suite exercises every public model. Local installation
# choices from .env must not silently disable Qwen scenarios during collection.
os.environ["OCR_MODELS"] = "paddle,mrz,russian-ocr,qwen3-vl-4b,qwen3-vl-8b"
