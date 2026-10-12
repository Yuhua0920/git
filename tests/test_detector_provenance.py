"""Commit provenance plus strict current legacy code and frozen-result regression."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np
import pytest

from src import detectors, link, provenance as pv
from scripts import validate_linear_mmse_detector as calibration

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path",sorted(pv.EXTENDED_MODULES))
def test_legacy_detector_link_exact_bytes_and_no_overridden_old_names(path):
    assert pv.legacy_prefix_unchanged(ROOT,path)
    old = ast.parse(pv.git_blob(str(ROOT),pv.T11B_COMMIT,path))
    current = ast.parse((ROOT/path).read_bytes())
    old_names = {n.name for n in old.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    for name in old_names:
        assert sum(isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name for n in current.body)==1
    assert len(current.body)>len(old.body)  # D1 really is an append-only extension.


@pytest.mark.parametrize("stage",["t09a","t09b","t10a","t10b","t10c","t10d","t11a","t11b"])
@pytest.mark.parametrize("path",sorted(pv.EXTENDED_MODULES))
def test_historical_detector_link_snapshot_digest(stage,path):
    old = pv.git_blob(str(ROOT),pv.STAGE_COMMITS[stage],path)
    assert pv.recorded_source_matches(ROOT,path,hashlib.sha256(old).hexdigest(),stage)
    assert not pv.recorded_source_matches(ROOT,path,"0"*64,stage)


@pytest.mark.parametrize("path",sorted(pv.EXTENDED_MODULES))
def test_changed_old_numerical_bytes_rejected_even_with_historical_hash(tmp_path,path):
    blob = pv.git_blob(str(ROOT),pv.T11B_COMMIT,path)
    target = tmp_path/path; target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(b"# tampered legacy\n"+blob)
    with patch.object(pv,"git_blob",return_value=blob):
        assert not pv.recorded_source_matches(tmp_path,path,hashlib.sha256(blob).hexdigest())


def test_missing_snapshot_never_falls_back_to_current_source():
    with patch.object(pv,"git_blob",side_effect=subprocess.CalledProcessError(128,["git","show"])):
        with pytest.raises(subprocess.CalledProcessError):
            pv.recorded_source_matches(ROOT,"src/detectors.py",calibration.digest(ROOT/"src/detectors.py"))


@pytest.mark.parametrize("path",sorted(p for p in pv.PROVENANCE_PLUMBING if not p.startswith("tests/")))
def test_non_guard_numerical_functions_and_constants_unchanged(path):
    assert pv.plumbing_semantics_unchanged(ROOT,path)


def test_plumbing_cannot_hide_changed_numerical_constants(tmp_path):
    path = "src/nearest_validation.py"
    old = pv.git_blob(str(ROOT),pv.T11B_COMMIT,path)
    current = (ROOT/path).read_text(encoding="utf8")
    assert 'def read_csv(path):' in current
    current = current.replace('def read_csv(path):','def read_csv(path):\n    raise RuntimeError("tampered")',1)
    target = tmp_path/path; target.parent.mkdir(parents=True); target.write_text(current,encoding="utf8")
    with patch.object(pv,"git_blob",return_value=old):
        assert not pv.plumbing_semantics_unchanged(tmp_path,path)
        # Existing numerical imports remain part of the strict AST comparison.
        imports_changed = (ROOT/path).read_text(encoding="utf8").replace(
            "import numpy as np", "import numpy as changed_alias", 1)
        target.write_text(imports_changed,encoding="utf8")
        assert not pv.plumbing_semantics_unchanged(tmp_path,path)


@pytest.mark.parametrize("path",["src/estimators.py","src/interpolation.py","configs/base.yaml"])
def test_other_core_sources_still_require_current_whole_file_hash(path):
    correct = calibration.digest(ROOT/path)
    assert pv.recorded_source_matches(ROOT,path,correct)
    assert not pv.recorded_source_matches(ROOT,path,correct,current_hash=lambda _:"changed")


def test_current_D0_grid_wrapper_and_slicer_exactly_equal_snapshot():
    old_detector = {}
    exec(compile(pv.git_blob(str(ROOT),pv.T11B_COMMIT,"src/detectors.py"),"D0_snapshot","exec"),old_detector)
    old_link = {}
    exec(compile(pv.git_blob(str(ROOT),pv.T11B_COMMIT,"src/link.py"),"link_snapshot","exec"),old_link)
    old_link["zf_detect"] = old_detector["zf_detect"]
    y = np.tile(calibration.Y_COMPLEX[:,None,None],(1,3,2))
    h = np.tile(calibration.H_COMPLEX[:,:,None,None],(1,1,3,2))
    mask = np.ones((2,3,2),dtype=bool); mask[:,0,0]=False
    old = old_link["detect_data_zf"](y,h,mask)
    now = link.detect_data_zf(y,h,mask)
    assert np.array_equal(old,now,equal_nan=True)
    for a,b in zip(old_link["demodulate_data_bits"](np.sqrt(2)*old,mask),link.demodulate_data_bits(np.sqrt(2)*now,mask)):
        assert np.array_equal(a,b)
    bits, master = link.generate_master_qpsk(bits_seeds=(20261001,20261002))
    layout = calibration.build_fdm_pilot_layout(2,64,16,16,4)
    assert np.array_equal(old_link["assemble_pilot_data_frame"](master,layout),link.assemble_pilot_data_frame(master,layout))


def test_all_112_accepted_output_files_are_byte_unchanged():
    frozen = json.loads(calibration.FROZEN_PATH.read_text(encoding="utf8"))
    assert frozen["accepted_t11b_commit"]==pv.T11B_COMMIT and len(frozen["sha256"])==112
    for path,digest in frozen["sha256"].items():
        assert calibration.digest(ROOT/path)==digest


def test_saved_migration_and_D0_fixed_sentinel():
    r = json.loads(calibration.RESULT_PATH.read_text(encoding="utf8"))["legacy_provenance_regression"]
    assert r["passed"] and r["fixed_D0_vs_Git_snapshot_exact_array_equal"]
    assert r["frozen_artifacts"]["unchanged"]
    assert r["frozen_artifacts"]["before"]==r["frozen_artifacts"]["after"]
    assert all(r["legacy_prefix_byte_checks"].values())
