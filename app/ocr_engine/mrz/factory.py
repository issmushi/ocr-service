from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mrzscanner import MRZScanner


def create_mrz_scanner() -> MRZScanner:
    """Create the CPU detector and recognizer used for MRZ scanning."""
    from capybara import Backend
    from mrzscanner import ModelType, MRZScanner

    return MRZScanner(backend=Backend.cpu, model_type=ModelType.two_stage)
