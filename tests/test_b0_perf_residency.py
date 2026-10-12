"""Co-resident runtime/memory gate and frozen inference costs."""
import json
from pathlib import Path
import pytest
from src.lmmse_train import sha256_file

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def report():
    path = ROOT/"outputs/t12c_perf_results.json"
    if not path.exists():
        pytest.skip("benchmark not executed")
    result = json.loads(path.read_text(encoding="utf8"))
    if "combined_residency_verification" not in result:
        pytest.skip("co-resident measurement not executed")
    return result


def test_selected_cache_and_workers_together_pass_physical_memory_gate(report):
    row = report["combined_residency_verification"]
    assert row["full_train_samples"] == row["validation_inputs"] == 20000
    assert row["workers_remained_live_during_validation"]
    assert row["memory_safe"] and not row["memory"]["paging_detected"]
    assert row["memory"]["available_RAM_min_bytes"] >= 1.5*2**30
    assert row["full_update_state_exact"] and row["validation_metrics_exact"]
    assert row["train_H_long_hashes_matched"] == 2000


def test_speedup_scope_and_all_actual_output_hashes(report):
    for name, digest in report["output_SHA256"].items():
        assert sha256_file(ROOT/"outputs"/name) == digest
    for path, digest in report["provenance"]["source_sha256"].items():
        assert sha256_file(ROOT/path) == digest
    assert report["speedup"] == report["baseline_profile"]["seconds_per_epoch"]/report["selected_profile"]["seconds_per_epoch"]
    assert report["future_cost_estimate"]["T12D_30_additional_epochs_seconds"] == 30*report["selected_profile"]["seconds_per_epoch"]
    assert report["calibration_initial_state"]["weights_saved"] is False
