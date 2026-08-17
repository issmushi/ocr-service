import os
from pathlib import Path

import yaml

from app.ocr_engine.russian.assets import models_dir

FILE = Path(__file__).resolve()
DEFAULT_CFG_PATH = FILE.parent.joinpath("models_path.yaml")
ROOT = models_dir().parent

with open(DEFAULT_CFG_PATH, "r") as f:
    DEFAULT_CFG = yaml.safe_load(f)
    for key, value in DEFAULT_CFG.items():
        if not Path(value).is_absolute():
            DEFAULT_CFG[key] = ROOT.joinpath(
                str(value).replace("\\", os.sep)
            ).as_posix()
