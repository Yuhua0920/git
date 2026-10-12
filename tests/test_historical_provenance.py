"""Versioned T11A artifacts, strict current sources, and metadata-only T11B fix."""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import finalize_nearest_validation_provenance as fix

ROOT = Path(__file__).resolve().parents[1]


def test_historical_t11a_commit_blobs_match_recorded_hashes():
    result = fix.verify_historical_t11a(ROOT)
    old = json.loads((ROOT/fix.T11A_JSON).read_text(encoding="utf-8"))
    assert result["historical_t11a_commit"]=="9fe2f165f9f511c9fe1c90585f2ec5023ef3c951"
    assert result["historical_registry_provenance_verified"] and result["historical_test_provenance_verified"]
    for path,digest in result["historical_source_sha256"].items():
        assert digest==old["provenance"]["source_sha256"][path]
    assert result["T11A_JSON_sha256"]==fix.T11A_JSON_SHA256


def test_historical_checks_do_not_bind_current_document_or_test_bytes():
    real_sha256 = fix.sha256_file
    def guarded(path):
        if Path(path).relative_to(ROOT).as_posix() in fix.HISTORICAL_PATHS:
            raise AssertionError("historical document/test must use commit bytes")
        return real_sha256(path)
    with patch.object(fix,"sha256_file",side_effect=guarded):
        assert fix.verify_historical_t11a(ROOT)["historical_test_provenance_verified"]


@pytest.mark.parametrize("path",fix.HISTORICAL_PATHS)
def test_tampered_historical_blob_is_rejected(path):
    original = fix.git_blob
    with patch.object(fix,"git_blob",side_effect=lambda root,p:original(root,p)+(b"tampered" if p==path else b"")):
        with pytest.raises(ValueError,match="commit-based historical provenance mismatch"):
            fix.verify_historical_t11a(ROOT)


def test_missing_historical_commit_never_falls_back_to_current_files():
    with patch.object(fix,"git_blob",side_effect=subprocess.CalledProcessError(128,["git","show"])):
        with pytest.raises(subprocess.CalledProcessError):
            fix.verify_historical_t11a(ROOT)


@pytest.mark.parametrize("path",("src/interpolation.py","scripts/validate_nearest_interpolation.py","configs/base.yaml"))
def test_tampered_current_core_script_or_config_is_rejected(path):
    if path == "scripts/validate_nearest_interpolation.py":
        with patch("src.provenance.plumbing_semantics_unchanged",return_value=False):
            with pytest.raises(ValueError,match="current T11A core/script/config source changed"):
                fix.verify_historical_t11a(ROOT)
        return
    original = fix.sha256_file
    with patch.object(fix,"sha256_file",side_effect=lambda p:"changed" if Path(p)==ROOT/path else original(p)):
        with pytest.raises(ValueError,match="current T11A core/script/config source changed"):
            fix.verify_historical_t11a(ROOT)


def test_tampered_t11a_json_is_rejected():
    original = fix.sha256_file
    with patch.object(fix,"sha256_file",side_effect=lambda p:"changed" if Path(p)==ROOT/fix.T11A_JSON else original(p)):
        with pytest.raises(ValueError,match="current T11A historical JSON changed"):
            fix.verify_historical_t11a(ROOT)


def test_changed_numerical_payload_stops_before_metadata_write(tmp_path):
    report = json.loads((ROOT/fix.T11B_JSON).read_text(encoding="utf-8"))
    report["r2_aggregate_results"][0]["nearest_nmse_db"] += 1
    path = tmp_path/fix.T11B_JSON
    path.parent.mkdir()
    path.write_text(json.dumps(report),encoding="utf-8")
    before = path.read_bytes()
    with patch.object(fix,"verify_protected_files",return_value=fix.PROTECTED_HASHES):
        with pytest.raises(ValueError,match="numerical/results payload changed"):
            fix.finalize(tmp_path)
    assert path.read_bytes()==before


def test_current_registry_scope_and_r1_metrics():
    assert fix.verify_current_registry(ROOT)["current_registry_updated"]


@pytest.mark.parametrize("old,new",[
    ("data BER=N/A","data BER=0"),
    ("held-out ID validation available","not evaluated"),
    ("formal test remains sealed.","formal test available."),
])
def test_incorrect_current_registry_is_rejected(tmp_path,old,new):
    path = tmp_path/"docs/baseline_registry.md"
    path.parent.mkdir()
    text = (ROOT/"docs/baseline_registry.md").read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old,new,1),encoding="utf-8")
    with pytest.raises(ValueError):
        fix.verify_current_registry(tmp_path)


def test_saved_metadata_gates_and_original_payload_are_verified():
    report = json.loads((ROOT/fix.T11B_JSON).read_text(encoding="utf-8"))
    assert report["historical_t11a_commit"]==fix.T11A_COMMIT
    assert all(report[k] for k in ("historical_registry_provenance_verified",
        "historical_test_provenance_verified","current_registry_updated","all_checks_passed"))
    assert fix.protected_payload_sha256(report)==fix.PRE_FIX_PAYLOAD_SHA256
    provenance = report["provenance_fix"]
    assert provenance["metadata_only_update"] and not provenance["numerical_scan_executed"]
    assert provenance["original_numerical_provenance_preserved"]
    assert provenance["protected_file_sha256_before_after"]["before"]==provenance["protected_file_sha256_before_after"]["after"]==fix.verify_protected_files(ROOT)
    assert provenance["current_registry_sha256"]==fix.sha256_file(ROOT/"docs/baseline_registry.md")
    for path,digest in provenance["fix_source_sha256"].items():
        from src.provenance import recorded_source_matches
        assert recorded_source_matches(ROOT,path,digest,"t11b",current_hash=fix.sha256_file)


def test_saved_finalization_is_idempotent_without_numerical_scan():
    before = (ROOT/fix.T11B_JSON).read_bytes()
    with patch("src.nearest_validation.run_replay",side_effect=AssertionError("no R2 scan")), \
         patch("src.trajectories.prepare_trajectory",side_effect=AssertionError("no channel generation")), \
         patch("src.lmmse_train_validation.run_validation",side_effect=AssertionError("no old scan")):
        result = fix.finalize(ROOT)
    assert result["all_checks_passed"]
    assert (ROOT/fix.T11B_JSON).read_bytes()==before
