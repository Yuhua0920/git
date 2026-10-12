"""Small complex mathematical sentinels and immutable T16A replay evidence."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pytest
from src.lowrank_filter import (WeightedCorrection, RANKS, SOURCE_MEAN_RANKS,
    T15C_COMMIT, closure, required_rank, design_gate, complexity, angles)

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    rng = np.random.default_rng(20261601)
    def z(shape): return rng.normal(size=shape)+1j*rng.normal(size=shape)
    source, target = z((9, 5)), z((9, 5))
    c = z((5, 5)); c = c@c.conj().T+.7*np.eye(5)
    return WeightedCorrection.build(source, target, c)


def test_T15C_accepted_committed_snapshot():
    subject = subprocess.check_output(["git", "show", "-s", "--format=%s", T15C_COMMIT], cwd=ROOT, text=True).strip()
    assert subject == "feat: add realistic CDL covariance-mismatch gate"
    subprocess.check_call(["git", "merge-base", "--is-ancestor", T15C_COMMIT, "HEAD"], cwd=ROOT)


def test_source_target_frozen_statistics_hashes():
    r = json.loads((ROOT/"outputs/t15c_cdl_mismatch_results.json").read_text(encoding="utf8"))
    for info in [r["source_model_provenance"], *r["matched_statistics_provenance"].values()]:
        assert hashlib.sha256((ROOT/info["path"]).read_bytes()).hexdigest() == info["sha256"]
        with np.load(ROOT/info["path"]) as f:
            for k, h in info["component_sha256"].items():
                assert hashlib.sha256(np.ascontiguousarray(f[k]).tobytes()).hexdigest() == h


@pytest.mark.parametrize("rank", range(6))
def test_complex_weighted_tail_and_right_solve(rank):
    c = fixture(); a, q = c.factors(rank); b = a@q
    residual = c.target-c.source-b
    weighted = np.linalg.norm(residual@c.lower)**2
    trace_loss = np.trace(residual@c.covariance@residual.conj().T).real
    np.testing.assert_allclose(weighted, trace_loss, rtol=1e-12, atol=1e-24)
    np.testing.assert_allclose(weighted, np.sum(c.singular[rank:]**2), rtol=1e-12, atol=1e-24)
    np.testing.assert_allclose(q@c.lower, c.right_h[:rank], rtol=1e-12, atol=1e-12)
    assert a.dtype == q.dtype == np.complex128


def test_non_diagonal_complex_covariance_and_adjoint_sentinel():
    c = fixture(); d = c.target-c.source
    assert np.linalg.norm(c.covariance.imag) > 1
    np.testing.assert_allclose(c.lower@c.lower.conj().T, c.covariance, atol=1e-12)
    # Wrong L^H weighting is measurably different, so this sentinel detects it.
    assert abs(np.linalg.norm(d@c.lower)**2-np.linalg.norm(d@c.lower.conj().T)**2) > 1
    a, q = c.factors(2)
    rng = np.random.default_rng(123)
    trial = rng.normal(size=(9, 2))+1j*rng.normal(size=(9, 2))
    assert np.linalg.norm((d-a@q)@c.lower)**2 < np.linalg.norm((d-trial@q)@c.lower)**2


def test_rank0_affine_source_and_fullrank_target():
    c = fixture(); z = np.arange(5)+1j*np.arange(5)[::-1]
    p = np.ones(5)*(.4+.3j); d = np.ones(9)*(.5-.7j)
    assert np.array_equal(c.estimate(z, p, d, c.factors(0)), d+(z-p)@c.source.T)
    np.testing.assert_allclose(c.estimate(z, p, d, c.factors(5)), d+(z-p)@c.target.T, rtol=1e-12, atol=1e-12)
    assert max(c.diagnostics()[k] for k in c.diagnostics() if "residual" in k) < 1e-12


@pytest.mark.parametrize("rank", [-1, 6, 1.5, True])
def test_invalid_rank_is_rejected(rank):
    with pytest.raises(ValueError): fixture().factors(rank)


def test_nonhermitian_and_nonPD_covariance_rejected():
    c = fixture(); wrong = c.covariance.copy(); wrong[0, 1] += .1j
    with pytest.raises(ValueError): WeightedCorrection.build(c.source, c.target, wrong)
    with pytest.raises(np.linalg.LinAlgError): WeightedCorrection.build(c.source, c.target, -np.eye(5))


def test_spectrum_monotone_and_effective_rank():
    c = fixture(); energy, er = c.spectrum()
    assert np.all(np.diff(c.singular) <= 0) and np.all(np.diff(energy) >= 0)
    assert energy[-1] == pytest.approx(1.) and 1 <= er <= 5
    same = WeightedCorrection.build(c.source, c.source, c.covariance)
    assert same.spectrum()[1] == 0 and not same.spectrum()[0].any()


def test_closure_unclipped_and_required_rank():
    assert closure(-10, -20, -18)["closure_r"] == .8
    assert closure(-10, -20, -21)["closure_r"] == 1.1
    assert closure(-10, -20, -9)["closure_r"] == -.1
    with pytest.raises(ValueError): closure(-10, -10, -10)
    rows = [dict(rank=4, closure_r=.79), dict(rank=8, closure_r=.85)]
    assert required_rank(rows, .8) == 8 and required_rank(rows, .9) is None


@pytest.mark.parametrize("level,expected", [(.81, "viable"), (.49, "weak"), (.6, "partial")])
def test_preregistered_gate(level, expected):
    rows = [dict(shift=n, snr_db=s, rank=r, closure_r=level) for n in ("B", "C") for s in (10, 20) for r in RANKS]
    flags = design_gate(rows)
    assert sum(flags.values()) == 1 and flags[f"low_rank_correction_representation_{expected}"]


@pytest.mark.parametrize("rank", [1, 2, 4, 8, 16])
def test_complexity_formula(rank):
    result = complexity(rank)
    assert result["extra_complex_MAC_per_frame"] == 4032*rank
    assert result["overhead_ratio"] == 4032*rank/253952


def test_principal_angles_identical_complex_subspaces():
    c = fixture()
    for space in ("left_data", "right_weighted_pilot"):
        assert np.max(angles(c, c, 4, space)) < 1e-10


def test_frozen_filter_Cholesky_residuals_and_actual_geometry():
    from scripts.run_lowrank_filter import statistics, prepare
    from src.cdl_benchmark import observation_geometry
    _, source, targets, variances = statistics()
    estimators, checks, _ = prepare(source, targets, variances)
    assert all(r["source_W_residual"] < 1e-11 and r["target_W_residual"] < 1e-11 for r in checks)
    assert len(estimators) == 12
    g = observation_geometry()
    assert g.selected.size == 64 and g.data.sum() == 3968 and g.layout.pilot_union.mean() == 1/32


def test_actual_small_preflight_if_present():
    path = ROOT/"outputs/t16a_lowrank_preflight.json"
    if not path.exists(): return
    r = json.loads(path.read_text(encoding="utf8"))
    assert r["all_checks_passed"] and len(r["evidence"]) == 2
    assert all(v["exact_T15C_and_unchanged_evaluator"] and v["fullrank_NMSE_and_BER"] for v in r["evidence"])


def test_CSV_only_plot_small_preflight_scoped_style(tmp_path):
    from scripts.plot_lowrank_filter import audit
    import matplotlib
    before = dict(matplotlib.rcParams)
    metric = []; spectrum = []
    for name in ("CDL-B-30", "CDL-C-120"):
        for s in (5, 10, 20):
            for rank in RANKS:
                metric.append(dict(shift=name, snr_db=s, means="target", rank=rank, closure_r=rank/64))
            for index in range(1, 65):
                spectrum.append(dict(shift=name, snr_db=s, singular_index=index, cumulative_weighted_energy=index/64))
    for name, rows in (("t16a_lowrank_filter_correction.csv", metric), ("t16a_lowrank_singular_spectrum.csv", spectrum)):
        with (tmp_path/name).open("w", encoding="utf8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    audit(tmp_path, tmp_path/"figures")
    assert dict(matplotlib.rcParams) == before


def test_completed_outputs_replay_coverage_and_sealed_if_present():
    path = ROOT/"outputs/t16a_lowrank_results.json"
    if not path.exists(): return
    r = json.loads(path.read_text(encoding="utf8"))
    assert r["representation_gate"] and not r["proposed_method"]
    assert r["formal_test_sealed"] and not r["formal_test_used"] and not r["new_model_trained"]
    assert r["replay_gate"]["all_exact"] and len(r["replay_gate"]["evidence"]) == 200
    old = json.loads((ROOT/"outputs/t15c_cdl_mismatch_results.json").read_text(encoding="utf8"))
    for proof in r["replay_gate"]["evidence"]:
        prior = next(x for x in old["paired_evaluation_evidence"] if x["shift"] == proof["shift"] and x["trajectory_id"] == proof["trajectory_id"])
        assert proof["H_sha256"] == prior["H_sha256"] and proof["paired_inputs_sha256"] == prior["paired_inputs_sha256"]
    assert len(r["rank_results"]) == 54 and len(r["singular_spectrum"]) == 384
    assert sum(r[k] for k in r if k.startswith("low_rank_correction_representation_")) == 1
    for p, h in r["frozen_history_sha256"].items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h
    for p, h in r["source_sha256"].items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h
    for d in r["matrix_diagnostics"]:
        assert d["source_W_residual"] < 1e-11 and d["target_W_residual"] < 1e-11
        assert d["covariance_min_eigenvalue"] > 0
        assert d["full_rank_filter_relative_residual"] < 1e-11
    rows = list(csv.DictReader((ROOT/"outputs/t16a_lowrank_trajectory_metrics.csv").open(encoding="utf8")))
    assert len(rows) == 10200
    for a in r["rank_results"]:
        v = [x for x in rows if x["shift"] == a["shift"] and int(x["snr_db"]) == a["snr_db"] and x["method"] == a["method"]]
        assert len(v) == 100 and {int(x["trajectory_id"]) for x in v} == set(range(500, 600))
        assert sum(int(x["bit_errors"]) for x in v) == a["bit_errors"]
        np.testing.assert_allclose(sum(float(x["error_energy"]) for x in v)/sum(float(x["truth_energy"]) for x in v), a["nmse_linear"], rtol=1e-14)
    for name in ("CDL-B-30", "CDL-C-120"):
        for s in (5, 10, 20):
            v = [a for a in r["rank_results"] if a["shift"] == name and a["snr_db"] == s]
            assert r["r80"][name][str(s)] == required_rank(v, .8)
            assert r["r90"][name][str(s)] == required_rank(v, .9)
            last = next(a for a in v if a["rank"] == 64)
            assert last["closure_r"] == pytest.approx(1., abs=1e-10)
