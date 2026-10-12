"""Regression evidence for the unexpectedly perfect five-frame classification."""
import json
from pathlib import Path

import pytest

from src.lmmse_train import sha256_file

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def audit():
    path=ROOT/"outputs/t13c_statistics_identifiability_independent_audit.json"
    if not path.exists():
        pytest.skip("independent actual-pilot audit pending")
    return json.loads(path.read_text(encoding="utf8"))


def test_actual_pilot_direct_solve_and_candidate_permutation(audit):
    records=audit["actual_replayed_pilot_direct_solve_checks"]
    assert len(records)==18
    assert max(r["direct_score_max_absolute_error"] for r in records)<1e-9
    assert all(r["permuted_bank_same_regime"] for r in records)
    assert audit["all_3600_decisions_minimize_stored_pilot_scores"]


def test_no_csv_changed_no_H_substitution_no_formal_test(audit):
    assert audit["CSV_before"]==audit["CSV_after"]
    assert all(sha256_file(ROOT/"outputs"/p)==h for p,h in audit["CSV_after"].items())
    assert audit["R7_not_equal_H_true"] and not audit["full_scan_rerun"]
    assert audit["formal_test_sealed"] and not audit["formal_test_used"]
    assert all(sha256_file(ROOT/p)==h for p,h in audit["source_sha256"].items())
