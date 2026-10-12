"""Editorial regression guards protect frozen mathematics and science."""
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_paper_blueprint import ROOT, notation_audit, unexpected_changes
from check_paper import bibliography_metadata, script_firewall
from audit_paper_layout import scale_pt, table_rows, linear_rgb, srgb, CVD
from paper_support import json_read, sha256
from paper_history import historical_bytes_exact
import hashlib


def test_historical_t20d_equations_equal_committed_t20c():
    snapshot = json_read(ROOT / "outputs/t20d_before_manifest.json")["t20c_commit"]
    pattern = r"\\begin\{(equation|align)\}(.*?)\\end\{\1\}"
    for path in (ROOT / "paper/sections").glob("*.tex"):
        old = subprocess.check_output(["git", "show", snapshot + ":" + path.relative_to(ROOT).as_posix()], cwd=ROOT).decode()
        t20d = json_read(ROOT / "outputs/t20e_before_manifest.json")["t20d_commit"]
        current = subprocess.check_output(["git", "show", t20d + ":" + path.relative_to(ROOT).as_posix()], cwd=ROOT).decode()
        equations = lambda t: Counter((kind, re.sub(r"\s+", "", body)) for kind, body in re.findall(pattern, t, re.S))
        assert equations(old) == equations(current)


@pytest.mark.parametrize("protected", ["src/detectors.py", "outputs/t20b_results.json", "docs/t20a_final_method_definition.md"])
def test_governance_rejects_science_mutation(tmp_path, protected):
    path = tmp_path / protected
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"original")
    manifest = {"tracked_before_sha256": {protected: sha256(path)}}
    path.write_bytes(b"changed")
    assert unexpected_changes(tmp_path, manifest)[1] == [protected]


def test_current_bibliography_preserves_unresolved_access_and_version_identity():
    metadata = bibliography_metadata()
    assert metadata["technical_unknown_entries"] == ["ozdemir2006"]
    closure = json_read(ROOT / "paper/bibliography/technical_closure.json")["entries"]
    assert closure["ozdemir2006"]["verified_from_abstract_only"] is False
    assert closure["lee2026"]["publisher_final_pdf_obtained"] is False
    assert closure["luan2023channelformer"]["read_version"] == "2302.04368v1"


def test_historical_t20d_notation_snapshot_matches_frozen_fitting_dimensions(tmp_path):
    commit = json_read(ROOT / "outputs/t20e_before_manifest.json")["t20d_commit"]
    for name in ("paper/tables/notation.tex", "paper/sections/04_proposed_method.tex"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT))
    assert notation_audit(tmp_path)["all_checks_passed"]


@pytest.mark.parametrize("name", ["check_paper_blueprint.py", "audit_paper_layout.py"])
def test_new_manuscript_tools_do_not_call_science(name):
    assert not script_firewall((ROOT / "scripts" / name).read_text(encoding="utf8"))


def test_final_size_font_scaling_is_not_nominal_size():
    assert scale_pt(8, 180, 90) == 4
    assert scale_pt(8, 180, 174) == pytest.approx(7.733333333)


def test_color_qa_identity_roundtrip_and_neutral_preservation():
    values = np.array([0, .02, .2, .8, 1.])
    np.testing.assert_allclose(srgb(linear_rgb(values)), values, atol=1e-15)
    np.testing.assert_allclose(CVD @ np.ones(3), np.ones(3), atol=2e-6)
    assert CVD[0, 0] == .367322  # Official dataset's deuteranomaly1, not protanomaly.


def test_full_numeric_tables_are_retained():
    for filename in ("02_nmse.tex", "03_ber.tex"):
        assert len(table_rows((ROOT / "paper/tables" / filename).read_text(encoding="utf8"))) == 80


def test_historical_manuscript_outputs_and_scientific_files_unchanged():
    manifest = json_read(ROOT / "outputs/t20d_before_manifest.json")
    for name, digest in manifest["tracked_before_sha256"].items():
        if name.startswith(("outputs/", "src/", "paper/figures/")):
            if name.startswith("paper/figures/"):
                assert hashlib.sha256(historical_bytes_exact(manifest["t20c_commit"], name, digest)).hexdigest() == digest, name
            else:
                assert sha256(ROOT / name) == digest, name
