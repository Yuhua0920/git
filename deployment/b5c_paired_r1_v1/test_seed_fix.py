"""Seed-record regressions using synthetic identities and synthetic arrays only.

Run with PYTHONPATH selecting deployment/b5c_paired_r1_v1.  No stored B2/B3
arrays, fitted models, or performance records are read by this test module.
"""
import ast
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import base as b
import science as s
import test_b5c as fixtures


def sample():
    proof, cfg, index = fixtures.proof_fixture()
    expected = {
        role: index["ID", 0, -1 if role in s.ROLES[:3] else 0, role]
        for role in s.ROLES
    }
    assert proof["channel_seeds"] == expected
    return proof, cfg, index


def original_checker():
    """Execute the unmodified, preserved original function, not a reconstruction."""
    path = b.ROOT / "deployment/b5c_paired_v1/science.py"
    tree = ast.parse(path.read_text(encoding="utf8"), filename=str(path))
    function = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "proof_check"
    )
    namespace = {"b": b, "ROLES": s.ROLES}
    isolated = ast.Module(body=[function], type_ignores=[])
    exec(compile(isolated, str(path), "exec"), namespace)
    return namespace["proof_check"]


def test_preserved_original_failure_reproduced():
    proof, cfg, index = sample()
    assert set(proof["channel_seeds"]) == set(s.ROLES)
    assert {
        role: proof["channel_seeds"][role] for role in s.ROLES[:3]
    } == {role: index["ID", 0, -1, role] for role in s.ROLES[:3]}
    with pytest.raises(RuntimeError, match="^CHANNEL_SEED_IDENTITY_DRIFT$"):
        original_checker()(proof, "ID", 0, cfg, index)
    assert len(s.proof_check(proof, "ID", 0, cfg, index)) == 70


@pytest.mark.parametrize("role", s.ROLES)
def test_each_channel_record_seed_value_must_match(role):
    proof, cfg, index = sample()
    proof["channel_seeds"][role] += 1
    with pytest.raises(RuntimeError, match="^CHANNEL_SEED_IDENTITY_DRIFT$"):
        s.proof_check(proof, "ID", 0, cfg, index)


@pytest.mark.parametrize("role", s.ROLES)
def test_missing_record_role_rejected(role):
    proof, cfg, index = sample()
    del proof["channel_seeds"][role]
    with pytest.raises(RuntimeError, match="^CHANNEL_SEED_IDENTITY_DRIFT$"):
        s.proof_check(proof, "ID", 0, cfg, index)


def test_extra_record_role_rejected():
    proof, cfg, index = sample()
    proof["channel_seeds"]["unregistered_seed"] = 123
    with pytest.raises(RuntimeError, match="^CHANNEL_SEED_IDENTITY_DRIFT$"):
        s.proof_check(proof, "ID", 0, cfg, index)


@pytest.mark.parametrize("role", s.ROLES)
def test_string_seed_not_coerced_to_integer(role):
    proof, cfg, index = sample()
    proof["channel_seeds"][role] = str(proof["channel_seeds"][role])
    with pytest.raises(RuntimeError, match="^CHANNEL_SEED_IDENTITY_DRIFT$"):
        s.proof_check(proof, "ID", 0, cfg, index)


@pytest.mark.parametrize("role", s.ROLES[3:])
def test_channel_record_requires_frame_zero_communication_provenance(role):
    proof, cfg, index = sample()
    proof["channel_seeds"][role] = index["ID", 0, 1, role]
    with pytest.raises(RuntimeError, match="^CHANNEL_SEED_IDENTITY_DRIFT$"):
        s.proof_check(proof, "ID", 0, cfg, index)


def test_json_object_order_does_not_change_identity():
    proof, cfg, index = sample()
    proof["channel_seeds"] = dict(reversed(list(proof["channel_seeds"].items())))
    proof["communication_seeds"] = [
        dict(reversed(list(frame.items()))) for frame in proof["communication_seeds"]
    ]
    restored = json.loads(json.dumps(proof))
    assert len(s.proof_check(restored, "ID", 0, cfg, index)) == 70


@pytest.mark.parametrize("condition", ("ID", "Profile", "Mobility", "Composed", "Boundary"))
def test_condition_mapping_accepts_same_registered_identity(condition):
    proof, cfg, index = sample()
    proof["condition"] = condition
    mapped = {(condition, trajectory, frame, role): seed
              for (_, trajectory, frame, role), seed in index.items()}
    assert len(s.proof_check(proof, condition, 0, cfg, mapped)) == 70


@pytest.mark.parametrize("condition,trajectory", [("Profile", 0), ("ID", 1)])
def test_requested_trajectory_identity_cannot_be_remapped(condition, trajectory):
    proof, cfg, index = sample()
    with pytest.raises(RuntimeError, match="^B2_RAW_IDENTITY_DRIFT$"):
        s.proof_check(proof, condition, trajectory, cfg, index)


@pytest.mark.parametrize("field,value", [("snr_db", 35), ("frame_id", 10)])
def test_out_of_grid_input_identity_rejected(field, value):
    proof, cfg, index = sample()
    proof["inputs"][-1][field] = value
    with pytest.raises(RuntimeError, match="^EXACT_INPUT_KEY_GRID_REQUIRED$"):
        s.proof_check(proof, "ID", 0, cfg, index)


def test_duplicate_input_identity_is_still_rejected():
    proof, cfg, index = sample()
    proof["inputs"][-1] = copy.deepcopy(proof["inputs"][0])
    with pytest.raises(RuntimeError, match="^EXACT_INPUT_KEY_GRID_REQUIRED$"):
        s.proof_check(proof, "ID", 0, cfg, index)


def test_communication_frame_seed_check_is_unchanged():
    proof, cfg, index = sample()
    proof["communication_seeds"][9]["noise_seed"] += 1
    with pytest.raises(RuntimeError, match="^COMM_SEED_IDENTITY_DRIFT$"):
        s.proof_check(proof, "ID", 0, cfg, index)


def test_grid_and_scoring_contract_is_unchanged():
    original = b.read(b.ROOT / "deployment/b5c_paired_v1/contract.json")
    current = b.contract()
    assert current == original
    assert tuple(current["conditions"]) == ("ID", "Profile", "Mobility", "Composed", "Boundary")
    assert current["conditions"] == {
        "ID": "CDL-C-30-100", "Profile": "CDL-B-30-100",
        "Mobility": "CDL-C-120-100", "Composed": "CDL-B-120-100",
        "Boundary": "CDL-C-30-200",
    }
    assert current["snrs"] == [0, 5, 10, 15, 20, 25, 30]
    assert current["N"] == 1000
    assert current["score_frames"] == [4, 5, 6, 7, 8, 9]
    assert len(current["conditions"]) * len(current["snrs"]) == 35
    assert len(current["conditions"]) * current["N"] * 70 == 350000
    assert len(current["conditions"]) * len(current["snrs"]) * current["N"] * len(current["score_frames"]) == 210000


def _synthetic_cache(tmp_path, monkeypatch):
    cfg, index, _ = fixtures.synthetic_cache(tmp_path, monkeypatch)
    cache = tmp_path / "outputs/sp_u12b2b_sealed/cache/ID"
    raw = tmp_path / "outputs/sp_u12b2b_sealed/raw_evaluation/ID"
    return cfg, index, cache, raw


def _update_archive_commit(path, info_path):
    record = b.read(info_path)
    assert len(record["artifacts"]) == 1
    record["artifacts"][0]["sha256"] = b.sha(path)
    record["artifacts"][0]["bytes"] = path.stat().st_size
    info_path.write_text(json.dumps(record), encoding="utf8")


def test_h_array_identity_check_is_unchanged(tmp_path, monkeypatch):
    cfg, index, cache, raw = _synthetic_cache(tmp_path, monkeypatch)
    path = cache / "H_0000.npz"
    with np.load(path, allow_pickle=False) as archive:
        changed = archive["H"].copy()
    changed.flat[0] += 0.01
    np.savez(path, H=changed)
    _update_archive_commit(path, cache / "H_0000.json")
    proof_path = raw / "0000.json"
    proof = b.read(proof_path)
    proof["H_artifact_sha256"] = b.sha(path)
    proof_path.write_text(json.dumps(proof), encoding="utf8")
    with pytest.raises(RuntimeError, match="^H_ARRAY_OR_STORAGE_IDENTITY_DRIFT$"):
        s.checked_input("ID", 0, cfg, index)


def test_noise_frame_hash_check_is_unchanged(tmp_path, monkeypatch):
    cfg, index, cache, raw = _synthetic_cache(tmp_path, monkeypatch)
    path = cache / "comm_0000.npz"
    with np.load(path, allow_pickle=False) as archive:
        changed = {name: archive[name].copy() for name in archive.files}
    changed["unit_noise"].flat[0] += 0.01
    np.savez(path, **changed)
    _update_archive_commit(path, cache / "comm_0000.json")
    proof_path = raw / "0000.json"
    proof = b.read(proof_path)
    proof["communication_sha256"] = b.sha(path)
    proof_path.write_text(json.dumps(proof), encoding="utf8")
    with pytest.raises(RuntimeError, match="^COMM_FRAME_HASH_DRIFT$"):
        s.checked_input("ID", 0, cfg, index)


def test_pilot_observation_identity_check_is_unchanged(tmp_path, monkeypatch):
    cfg, index, _, _ = _synthetic_cache(tmp_path, monkeypatch)
    h, comm, inputs, *_ = s.checked_input("ID", 0, cfg, index)
    inputs[10, 4]["pilot_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="^EXACT_PILOT_TRUTH_NOISE_PAIRING_FAILED:"):
        s.pilot_inputs(h, comm, inputs)


def test_npz_commit_check_is_unchanged(tmp_path, monkeypatch):
    cfg, index, cache, _ = _synthetic_cache(tmp_path, monkeypatch)
    path = cache / "comm_0000.npz"
    path.write_bytes(path.read_bytes() + b"synthetic-corruption")
    with pytest.raises(RuntimeError, match="^COMMITTED_INPUT_HASH_DRIFT$"):
        s.checked_input("ID", 0, cfg, index)


def test_existing_firewall_boundaries_remain_denied():
    code = """import control, base as b
control.scope(b.OUT/'synthetic', False)
cases=[
 (b.ROOT/'outputs/sp_u12b2b_sealed/raw_evaluation/ID/0000.npz','rb'),
 (b.ROOT/'outputs/sp_u12b3h_sealed/raw.npz','rb'),
 (b.ROOT/'outputs/sp_u12b5b/full_retraining.npy','rb'),
 (b.FINAL/'model/means.npy','wb')]
for path, mode in cases:
 try: open(path,mode)
 except RuntimeError: continue
 raise AssertionError('FIREWALL_EXPANDED')
print('FOUR_BOUNDARY_DENIALS_PASS')
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(b.HERE)
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], env=env, cwd=b.ROOT,
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "FOUR_BOUNDARY_DENIALS_PASS" in result.stdout


@pytest.mark.parametrize("marker", ["STOP.json", "FORMAL_START_CONSUMED.json", "STOP_REQUESTED.json"])
def test_original_stop_or_consumption_cannot_be_removed_by_preflight(marker, tmp_path, monkeypatch):
    import control
    monkeypatch.setattr(b, "OUT", tmp_path)
    path = tmp_path / marker
    original = b'{"historical_failure":true}\n'
    path.write_bytes(original)
    with pytest.raises(RuntimeError, match="^ONE_TIME_STOP_OR_START_PRESENT:"):
        control.absent()
    assert path.read_bytes() == original


def test_revision_cannot_start_formal_attempt_or_clear_failure(tmp_path, monkeypatch):
    import control
    monkeypatch.setattr(b, "OUT", tmp_path)
    records = {}
    for name in ("STOP.json", "FORMAL_START_CONSUMED.json", "stderr.log"):
        path = tmp_path / name
        path.write_bytes(b"preserved-original-failure\n")
        records[name] = path.read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError("FORMAL_ENTRY_REACHED_PREFLIGHT_OR_MONITOR")
    monkeypatch.setattr(control, "preflight", forbidden)
    monkeypatch.setattr(control, "monitor", forbidden)
    with pytest.raises(RuntimeError, match="^MENTOR_NEW_ATTEMPT_REQUIRED_R1_TECHNICAL_ONLY$"):
        control.start()
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == records


def test_revision_cli_run_formal_is_always_denied(monkeypatch):
    import control
    monkeypatch.setattr(sys, "argv", ["control.py", "Run", "--formal"])
    with pytest.raises(RuntimeError, match="^MENTOR_NEW_ATTEMPT_REQUIRED_R1_TECHNICAL_ONLY$"):
        control.main()


def test_direct_worker_evaluate_is_denied_before_any_execution_write(tmp_path):
    """A caller cannot bypass the disabled controller through the worker CLI."""
    output = tmp_path / "isolated_synthetic_output"
    output.mkdir()
    before = {}
    for name in ("STOP.json", "FORMAL_START_CONSUMED.json", "stderr.log"):
        path = output / name
        path.write_bytes(b"preserved-synthetic-failure\n")
        before[name] = path.read_bytes()
    code = """import sys
from pathlib import Path
import base as b
b.OUT=Path(sys.argv[1]).resolve()
import worker
sys.argv=['worker.py','--action','evaluate','--job',str(b.OUT/'formal'),'--blas','6','--openmp','1']
worker.main()
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(b.HERE)
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(output)], env=env,
        cwd=b.ROOT, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "MENTOR_NEW_ATTEMPT_REQUIRED_R1_TECHNICAL_ONLY" in result.stderr
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
    assert not (output / "formal").exists()


def test_metadata_only_hook_denies_existing_arrays_and_historical_access(tmp_path):
    """All readable fixtures are synthetic; real scientific paths are denied first."""
    code = """import json,sys
from pathlib import Path
import base as b
import r1_control
original_root=b.ROOT
root=Path(sys.argv[1]).resolve()/'Synthetic_Project'
b.ROOT=root
b.DATA=root/'lab_inputs/b5c_b2_v1'
b.OUT=root/'lab_runs/b5c_paired_r1'
b.HERE=root/'deployment/synthetic_r1'
b.HERE.mkdir(parents=True)
(b.HERE/'package_manifest.json').write_text(json.dumps({'files':{}}),encoding='utf8')
(b.HERE/'contract.json').write_text(json.dumps({'conditions':{'ID':'synthetic'}}),encoding='utf8')
allowed=b.DATA/'outputs/sp_u12b2b_sealed/raw_evaluation/ID/0000.json'
allowed.parent.mkdir(parents=True)
allowed.write_text(json.dumps({'synthetic_identity':True}),encoding='utf8')
blocked=[
 b.DATA/'outputs/sp_u12b2b_sealed/raw_evaluation/ID/0000.npz',
 root/'outputs/sp_u12b2b_sealed/raw_evaluation/ID/0000.json',
 root/'outputs/sp_u12b3h_sealed/raw_evaluation/ID/0000.json',
 root/'outputs/sp_u12b5b/full_retraining.npy',
 root/'lab_runs/r6_formal/final/model/means.npy',
 root/'lab_runs/b5c_paired/STOP.json']
for path in blocked:
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_bytes(b'SYNTHETIC_ONLY_EXISTING_FILE\\n')
before={str(path):path.read_bytes() for path in blocked}
guard=r1_control.metadata_scope()
denied=0
try:
 assert json.loads(allowed.read_text(encoding='utf8'))=={'synthetic_identity':True}
 for path in blocked[:4]:
  try:path.read_bytes()
  except RuntimeError as e:
   assert str(e).startswith(('R1_NUMERIC_ARRAY_READ_DENIED','R1_METADATA_READ_DENIED'))
   denied+=1
  else:raise AssertionError('EXISTING_UNAUTHORIZED_READ_ALLOWED')
 for path in [
  original_root/'outputs/sp_u12b2b_sealed/raw_evaluation/ID/0000.npz',
  original_root/'outputs/sp_u12b3h_sealed/raw.npz']:
  try:path.read_bytes()
  except RuntimeError as e:
   assert str(e)=='R1_NUMERIC_ARRAY_READ_DENIED'
   denied+=1
  else:raise AssertionError('REAL_SCIENTIFIC_PATH_READ_ALLOWED')
 try:blocked[4].write_bytes(b'OVERWRITE_ATTEMPT')
 except RuntimeError as e:
  assert str(e)=='R1_HISTORICAL_WRITE_DENIED'
  denied+=1
 else:raise AssertionError('EXISTING_MODEL_OVERWRITE_ALLOWED')
 try:blocked[5].unlink()
 except RuntimeError as e:
  assert str(e)=='R1_HISTORICAL_WRITE_DENIED'
  denied+=1
 else:raise AssertionError('EXISTING_OLD_STOP_DELETE_ALLOWED')
finally:guard['enabled']=False
assert {str(path):path.read_bytes() for path in blocked}==before
assert denied==8
print('METADATA_ONLY_ALLOWED_JSON_AND_EIGHT_DENIALS_PASS')
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(b.HERE)
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(tmp_path)], env=env,
        cwd=b.ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "METADATA_ONLY_ALLOWED_JSON_AND_EIGHT_DENIALS_PASS" in result.stdout
