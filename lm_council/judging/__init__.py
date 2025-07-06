import os
import yaml
from pathlib import Path
from typing import Dict

from lm_council.judging.config import EvaluationConfig

PRESET_EVAL_CONFIGS: Dict[str, EvaluationConfig] = {}


def load_preset_configs():
    config_dir = Path(__file__).parent / "preset_configs"
    for file in config_dir.glob("*.yaml"):
        with open(file, "r") as f:
            data = yaml.safe_load(f)
            config = EvaluationConfig(**data)
            PRESET_EVAL_CONFIGS[file.stem] = config


load_preset_configs()
