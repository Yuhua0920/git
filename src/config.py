"""Load and validate the stage S0 base configuration."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML configuration mapping from *path*."""
    with Path(path).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return config


def validate_config(config: dict[str, Any]) -> int:
    """Return derived sample rate, or raise on an inconsistent configuration."""
    system = config["system"]
    derived_sample_rate = system["nfft"] * system["delta_f_hz"]
    if system["sample_rate_hz"] != derived_sample_rate:
        raise ValueError(
            "sample_rate_hz must equal nfft * delta_f_hz: "
            f"{system['sample_rate_hz']} != {derived_sample_rate}"
        )
    return derived_sample_rate
