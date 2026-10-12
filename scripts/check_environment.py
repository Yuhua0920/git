"""Print environment and configuration details, then write a JSON report."""

import importlib.metadata
import importlib.util
import json
import platform
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, validate_config  # noqa: E402


def package_version(distribution_name: str) -> str | None:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "base.yaml"
    config_text = config_path.read_text(encoding="utf-8")
    config = load_config(config_path)
    derived_sample_rate = validate_config(config)

    torch_version = package_version("torch")
    cuda_available = False
    cuda_version = None
    gpu_name = None
    if torch_version is not None and importlib.util.find_spec("torch") is not None:
        import torch

        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)

    report = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "numpy_version": package_version("numpy"),
        "scipy_version": package_version("scipy"),
        "matplotlib_version": package_version("matplotlib"),
        "pyyaml_version": package_version("PyYAML"),
        "pytest_version": package_version("pytest"),
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "config_path": str(config_path),
        "derived_sample_rate": derived_sample_rate,
        "config_validation_passed": True,
    }

    print("System information:")
    for key in (
        "platform", "cpu", "python_version", "numpy_version", "scipy_version",
        "matplotlib_version", "pyyaml_version", "pytest_version", "torch_version",
        "cuda_available", "cuda_version", "gpu_name",
    ):
        print(f"  {key}: {report[key]}")
    print(f"base.yaml ({config_path}):")
    print(config_text, end="" if config_text.endswith("\n") else "\n")
    print(f"Derived sample rate: {derived_sample_rate} Hz")
    print("Configuration validation: PASS")

    output_path = PROJECT_ROOT / "outputs" / "environment_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Report saved: {output_path}")


if __name__ == "__main__":
    main()
