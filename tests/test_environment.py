from copy import deepcopy
from pathlib import Path

import pytest

from src.config import load_config, validate_config


BASE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


def test_sample_rate_consistency() -> None:
    config = load_config(BASE_CONFIG)
    system = config["system"]
    assert system["nfft"] == 64
    assert system["delta_f_hz"] == 30000
    assert system["sample_rate_hz"] == 1920000
    assert system["nfft"] * system["delta_f_hz"] == system["sample_rate_hz"]
    assert validate_config(config) == 1920000


def test_incorrect_sample_rate_raises() -> None:
    config = deepcopy(load_config(BASE_CONFIG))
    config["system"]["sample_rate_hz"] = 1900000
    with pytest.raises(ValueError, match="sample_rate_hz must equal"):
        validate_config(config)
