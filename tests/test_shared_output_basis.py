"""Complex covariance sentinels and T16B fixed representation/replay gates."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pytest
from src.shared_output_basis import (SharedOutputBasis, T16A_COMMIT, RANKS, VALIDATION_RANKS,
    covariance_mse, anchored_nmse, validation_qthreshold, decision_gate, complexity_memory)
from src.cdl_benchmark import FrozenStatistics
ROOT = Path(__file__).resolve().parents[1]


def matrices():
    rng = np.random.default_rng(20261611)
    return {(n,s):rng.normal(size=(40,12))+1j*rng.normal(size=(40,12))
            for n in ("B","C") for s in (5,10,20)}


def test_T16A_committed_snapshot():
    subject = subprocess.check_output(["git","show","-s","--format=%s",T16A_COMMIT], cwd=ROOT, text=True).strip()
    assert subject == "feat: add oracle low-rank LMMSE filter-correction representation gate"
    subprocess.check_call(["git","merge-base","--is-ancestor",T16A_COMMIT,"HEAD"], cwd=ROOT)


def test_normalized_concatenation_exact_and_equal_block_energy():
    m = matrices(); b = SharedOutputBasis.build(m)
    expected = np.concatenate([a/np.linalg.norm(a) for a in m.values()], axis=1)/np.sqrt(6)
    assert np.array_equal(b.normalized, expected) and b.normalized.shape == (40,72)
    for block in np.split(b.normalized,6,axis=1):
        assert np.linalg.norm(block)**2 == pytest.approx(1/6, abs=1e-14)
    assert b.normalized.dtype == np.complex128 and not b.normalized.flags.writeable


def test_SVD_order_orthonormal_and_capture_monotone():
    m = matrices(); b = SharedOutputBasis.build(m)
    assert np.all(np.diff(b.singular) <= 0)
    np.testing.assert_allclose(b.left.conj().T@b.left,np.eye(40),rtol=1e-12,atol=1e-12)
    energy, rank = b.spectrum()
    assert np.all(np.diff(energy) >= 0) and energy[-1] == pytest.approx(1.) and 1 <= rank <= 40
    for a in m.values():
        c = [b.energy(a,q) for q in RANKS]
        assert np.all(np.diff([v["capture"] for v in c]) >= -1e-12)
        assert np.all(np.diff([v["weighted_excess_MSE_per_frame"] for v in c]) <= 1e-10)
        assert b.energy(a,32)["weighted_excess_MSE_per_frame"] <= b.energy(a,16)["weighted_excess_MSE_per_frame"]


@pytest.mark.parametrize("q",[1,2,4,8,12,16,24,32])
def test_independent_full_complex_covariance_expected_MSE_identity(q):
    rng = np.random.default_rng(20261612)
    z = rng.normal(size=(52,52))+1j*rng.normal(size=(52,52))
    joint = z@z.conj().T+np.eye(52)
    pp, dp, dd = joint[:12,:12], joint[12:,:12], joint[12:,12:]
    sigma = .9
    target = FrozenStatistics(np.zeros(12),np.zeros(40),pp,dp,(),1).prepare(sigma).weight
    source = rng.normal(size=(40,12))+1j*rng.normal(size=(40,12))
    lower = np.linalg.cholesky(pp+sigma*np.eye(12)); delta = target-source
    m = matrices(); m["B",5] = delta@lower; b = SharedOutputBasis.build(m)
    u = b.output(q); g = b.coefficient_map(delta,q)
    approximation = source+u@g
    direct = covariance_mse(approximation,pp,dp,np.trace(dd).real,sigma)
    matched = covariance_mse(target,pp,dp,np.trace(dd).real,sigma)
    excess = np.linalg.norm((delta-u@g)@lower)**2
    np.testing.assert_allclose(direct,matched+excess,rtol=1e-11,atol=1e-11)
    np.testing.assert_allclose(excess,b.energy(delta@lower,q)["weighted_excess_MSE_per_frame"],rtol=1e-11,atol=1e-11)
    np.testing.assert_allclose(target@(pp+sigma*np.eye(12)),dp,rtol=1e-11,atol=1e-11)
    # Wrong ordinary transpose must be distinguishable in this complex sentinel.
    assert np.linalg.norm(u.T@delta-u.conj().T@delta) > 1


def test_complex_factored_inference_matches_projected_affine_filter():
    b = SharedOutputBasis.build(matrices()); rng = np.random.default_rng(55)
    delta = matrices()["B",5]; source = matrices()["C",5]
    pilot = rng.normal(size=12)+1j*rng.normal(size=12)
    mean_p = np.ones(12)*(.3+.2j); mean_d = np.ones(40)*(.2-.3j)
    g = b.coefficient_map(delta,8)
    actual = b.estimate(pilot,source,mean_p,mean_d,8,g)
    expected = mean_d+(pilot-mean_p)@(source+b.output(8)@g).T
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)


def test_cross_shift_construction_ignores_heldout_matrices():
    m = matrices(); original = SharedOutputBasis.for_shift(m,"B")
    poisoned = {c:a if c[0] == "B" else np.full(a.shape,np.nan+1j*np.nan) for c,a in m.items()}
    check = SharedOutputBasis.for_shift(poisoned,"B")
    assert check.conditions == (("B",5),("B",10),("B",20))
    assert np.array_equal(check.normalized,original.normalized) and np.array_equal(check.singular,original.singular)


def test_invalid_zero_energy_or_nonfinite_input_rejected():
    with pytest.raises(ValueError): SharedOutputBasis.build({("B",5):np.zeros((4,4))})
    with pytest.raises(ValueError): SharedOutputBasis.build({("B",5):np.full((4,4),np.nan)})


def test_anchor_per_frame_units_and_no_decision_use():
    # Total matched energy1000 and truth10000 over100frames =>10,100/frame.
    r = anchored_nmse(5.,1000.,10000.,100)
    assert r["anchored_predicted_nmse_linear"] == .15 and not r["anchor_used_for_decision"]
    with pytest.raises(ValueError): validation_qthreshold([dict(q=8,shared_closure=.9,metric_type="anchored_prediction")],.8)
    with pytest.raises(ValueError): decision_gate([dict(metric_type="anchored_prediction")],[])


def test_q80_q90_only_actual_measured_grid():
    rows = [dict(q=q,shared_closure=c,metric_type="actual_validation") for q,c in zip(VALIDATION_RANKS,(.7,.82,.95,.99))]
    assert validation_qthreshold(rows,.8) == 8 and validation_qthreshold(rows,.9) == 16
    with pytest.raises(ValueError): validation_qthreshold([dict(q=12,shared_closure=.99,metric_type="actual_validation")],.9)


@pytest.mark.parametrize("values,expected",[((.81,.85,.9,.95),"strong"),((.6,.7,.85,.95),"moderate"),((.1,.2,.3,.7),"weak"),((.6,.7,.79,.85),"none")])
def test_preregistered_design_gate(values,expected):
    rows = [dict(shift=n,snr_db=s,q=q,shared_closure=v,metric_type="actual_validation") for n in ("B","C") for s in (10,20) for q,v in zip(VALIDATION_RANKS,values)]
    r = decision_gate(rows,rows)
    assert r["cross_shift_basis_reusable"] == (max(values[:3]) >= .7)
    assert r[f"shared_output_basis_{expected}"] if expected != "none" else not any(r[k] for k in r if k.startswith("shared_output_basis_"))


@pytest.mark.parametrize("q",[4,8,16])
def test_complexity_and_memory_units(q):
    r = complexity_memory(q)
    assert r["correction_reconstruction_complex_MAC_per_frame"] == 3968*q
    assert r["reconstruction_overhead_ratio"] == q/64
    assert r["oracle_linear_coefficient_map_extra_MAC"] == 64*q
    assert r["basis_bytes_complex64"] == 3968*q*8 and r["basis_bytes_complex128"] == 3968*q*16


def test_actual_results_if_present():
    p = ROOT/"outputs/t16b_shared_basis_results.json"
    if not p.exists(): return
    r = json.loads(p.read_text(encoding="utf8"))
    assert r["shared_representation_gate"] and not r["proposed_method"]
    assert r["formal_test_sealed"] and not r["formal_test_used"] and not r["new_model_trained"]
    assert not r["absolute_theoretical_expected_nmse_available"] and r["exact_excess_mse_available"]
    assert not r["anchored_nmse_used_for_decision_gate"] and r["BER_evaluations"] == r["detector_calls"] == 0
    assert len(r["capture_by_condition"]) == 48 and len(r["validation_nmse"]) == 24 and len(r["cross_shift_results"]) == 24
    assert len(r["replay_gate"]["evidence"]) == 200 and r["replay_gate"]["all_exact"]
    for p,h in {**r["source_sha256"],**r["frozen_history_sha256"]}.items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h
    for scope,info in r["shared_basis_definition"].items():
        assert info["orthonormality_relative_residual"] < 1e-11
        if scope != "all_conditions": assert all(x[0] == scope for x in info["basis_conditions"])
    with (ROOT/"outputs/t16b_shared_basis_trajectory_metrics.csv").open(encoding="utf8") as f: rows = list(csv.DictReader(f))
    assert len(rows) == 7200 and "ber" not in rows[0] and "bit_errors" not in rows[0]
    for a in r["validation_nmse"]:
        v = [x for x in rows if x["shift"] == a["shift"] and int(x["snr_db"]) == a["snr_db"] and x["method"] == a["method"]]
        assert len(v) == 100 and {int(x["trajectory_id"]) for x in v} == set(range(500,600))
        np.testing.assert_allclose(a["nmse_linear"],sum(float(x["error_energy"]) for x in v)/sum(float(x["truth_energy"]) for x in v),rtol=1e-14)
    for n in ("CDL-B-30","CDL-C-120"):
        for s in (5,10,20):
            group = [x for x in r["validation_nmse"] if x["shift"] == n and x["snr_db"] == s]
            assert r["q80"][n][str(s)] == validation_qthreshold(group,.8)
            assert r["q90"][n][str(s)] == validation_qthreshold(group,.9)
            energy = [x for x in r["capture_by_condition"] if x["shift"] == n and x["snr_db"] == s]
            assert np.all(np.diff([x["capture"] for x in energy]) >= -1e-12)
            assert energy[-1]["weighted_excess_MSE_per_frame"] <= energy[5]["weighted_excess_MSE_per_frame"]+1e-10


def test_frozen_statistics_and_T16A_implementation_hashes():
    from scripts.run_shared_output_basis import accepted_snapshot
    old = accepted_snapshot()
    for info in [old["source_filter_provenance"],*old["target_filter_provenance"].values()]:
        assert hashlib.sha256((ROOT/info["path"]).read_bytes()).hexdigest() == info["sha256"]
    assert all(c["historical_M_hash_available"] is False for c in
        json.loads((ROOT/"outputs/t16b_shared_basis_results.json").read_text(encoding="utf8"))["matrix_regression"]) if (ROOT/"outputs/t16b_shared_basis_results.json").exists() else True


def test_T16B_evaluator_cannot_call_historical_detector(monkeypatch):
    from scripts import run_shared_output_basis as run
    from src import link
    def forbidden(*args,**kwargs): raise AssertionError("detector must not be called in T16B")
    monkeypatch.setattr(link,"detect_data_zf",forbidden)
    # Tiny independent evaluator sentinel; no channel generator or historical BER evaluator.
    class Geometry:
        data = np.array([True,True,False,False]); selected = np.array([0,1])
        class Layout:
            pilot_union = np.array([True,False,False,False])
        layout = Layout()
    class Estimate:
        def estimate(self,pilot): return pilot
    z = np.zeros(4,dtype=complex); truth = np.array([1+2j,2-1j,0,0]); bits = np.zeros(2,dtype=int)
    clean = truth.copy(); symbols = np.ones(4,dtype=complex)
    frame = dict(truth=truth,bits=bits,symbols=symbols,z=z,clean=clean,energy=10.,
        input_hash=[run.array_hash(a) for a in (truth,bits,symbols,z)])
    monkeypatch.setattr(run,"estimate_ls_linear_channel",lambda y,layout:(y,None))
    est = {("B",m,s):Estimate() for m in run.BASELINES for s in run.SNRS}
    rows,_,digest = run.evaluate("B",500,[frame],Geometry(),{s:1. for s in run.SNRS},est,None,None,None,None)
    assert len(rows) == 6 and all(r["error_energy"] == 0 for r in rows)
    assert all("bit_errors" not in r and "ber" not in r for r in rows) and len(digest) == 64


def test_closure_uses_correct_source_matched_approximation_order():
    from src.lowrank_filter import closure
    assert closure(-10.,-20.,-18.)["closure_r"] == .8


def test_plot_scoped_style_and_actual_rank_contract(tmp_path):
    import matplotlib
    from scripts.plot_shared_output_basis import plot,FILES,STEMS
    from scripts.run_cdl_mismatch import csvsave
    before = dict(matplotlib.rcParams)
    rows = [dict(shift=n,snr_db=s,q=q,shared_closure=.5+q/100) for n in ("CDL-B-30","CDL-C-120")
        for s in (5,10,20) for q in VALIDATION_RANKS]
    for name in FILES: csvsave(tmp_path/name,rows)
    plot(tmp_path,tmp_path/"images")
    assert dict(matplotlib.rcParams) == before
    assert all((tmp_path/"images"/f"{stem}.{ext}").exists() for stem in STEMS for ext in ("png","pdf","svg"))
