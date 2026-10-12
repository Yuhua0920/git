"""T16D small mathematical sentinels before communication replay integration."""
import json
from pathlib import Path
import numpy as np
import pytest
from src.modular_correction import (orthonormal_union, frozen_module_unions, weighted_projection,
    ScalarComposition, interaction_spectrum, scientific_gate, BUDGETS, RANK_TOLERANCE)

ROOT = Path(__file__).resolve().parents[1]


def test_T16C_historical_snapshot():
    from scripts.run_modular_correction import accepted_snapshot
    r=accepted_snapshot()
    assert r["all_checks_passed"] and r["formal_test_sealed"]


def test_frozen_validation_manifest_H_and_pooled_baselines_without_generator(monkeypatch):
    from src.heldout_basis import HeldoutCDL
    from scripts.run_modular_correction import frozen_validation_tables
    def forbidden(*a,**kw):raise AssertionError("read-only validation audit must not generate H")
    monkeypatch.setattr(HeldoutCDL,"generate",forbidden)
    old,tables=frozen_validation_tables()
    assert len(tables["manifest"])==12000 and len(tables["trajectory_metrics"])==5600
    assert len(old["paired_input_evidence"])==200


def test_replay_authorization_gate_stops_before_generator(monkeypatch):
    from src.heldout_basis import HeldoutCDL
    from scripts import run_modular_correction as script
    def forbidden(*a,**kw):raise AssertionError("generator called before authorization")
    monkeypatch.setattr(HeldoutCDL,"generate",forbidden)
    monkeypatch.setattr(script,"VALIDATION_REPLAY_AUTHORIZED",False)
    with pytest.raises(ValueError,match="authorization pending"):script.replay_gate(None)


def test_historical_capture_preserves_original_calls_and_observed_Y(monkeypatch):
    from scripts import run_heldout_basis as old
    from scripts.run_modular_correction import capture_historical_trajectory
    counts=dict(communication=0,ls=0)
    def communication(seed):
        counts["communication"]+=1
        return np.array([seed],np.uint8),np.array([1j*seed]),np.array([seed+2j])
    def ls(y,layout):counts["ls"]+=1;return y,None
    def evaluate(name,i,h,g,var,source,target,estimates,controls,maps,bases):
        for j in range(10):
            bits,symbols,z=old.communication(j)
            for s in (5,10,20):old.estimate_ls_linear_channel(np.array([j+s*1j]),g.layout)
        return "original-result"
    monkeypatch.setattr(old,"communication",communication);monkeypatch.setattr(old,"estimate_ls_linear_channel",ls)
    monkeypatch.setattr(old,"evaluate",evaluate)
    result,inputs=capture_historical_trajectory("CDL-C-75",500,None,dict(var={},source=None,targets={"CDL-C-75":None},bases={}),None,None,None)
    assert result=="original-result" and counts==dict(communication=10,ls=30)
    assert old.communication is communication and old.estimate_ls_linear_channel is ls
    np.testing.assert_array_equal(inputs["y"][:,:,0],[[j+s*1j for s in (5,10,20)] for j in range(10)])
    np.testing.assert_array_equal(inputs["bits"][:,0],np.arange(10))


def test_formal_inference_rejects_regenerated_communication_stream():
    from scripts.run_modular_correction import evaluate
    with pytest.raises(ValueError,match="same historical replay input cache"):
        evaluate("CDL-C-75",500,None,None,baseline_only=False)


def test_replay_is_only_original_validation_generator_scope():
    import inspect
    from scripts.run_modular_correction import replay_gate,run
    source=inspect.getsource(replay_gate)
    assert 'range(500,600)' in source and 'split="validation_dev"' in source
    assert 'split="train_dev"' not in source and 'formal_test' not in source.split('proof=dict')[0]
    formal=inspect.getsource(run)
    assert '.generate(' not in formal and 'cached_inputs(info,i)' in formal


def test_replay_cache_preserves_original_noncontiguous_H_layout(tmp_path,monkeypatch):
    from scripts import run_modular_correction as script
    from src.cdl_benchmark import array_hash
    original=np.arange(2*2*160*64).reshape(2,2,160,64).transpose(0,1,3,2).astype(complex)
    np.save(tmp_path/"replay.npy",original.ravel(order="K")[None,:])
    monkeypatch.setattr(script,"ROOT",tmp_path)
    loaded=script.cached_H(dict(path="replay.npy",shape=original.shape,strides=original.strides),0)
    assert array_hash(loaded)==array_hash(original) and loaded.strides==original.strides
    assert not loaded.flags.writeable


def test_historical_baseline_comparison_rejects_tiny_nonexact_change():
    from scripts.run_modular_correction import check_baselines
    old=dict(shift="CDL-C-75",trajectory_id="500",snr_db="10",method="source",error_energy="1",truth_energy="2",nmse_linear="0.5")
    current={**old,"trajectory_id":500,"snr_db":10,"error_energy":1.,"truth_energy":2.,"nmse_linear":.5}
    tables=dict(trajectory_metrics=[old],baselines=[],shared_basis=[])
    check_baselines([current],[],tables)
    current["error_energy"]+=1e-12
    with pytest.raises(ValueError,match="not exact"):check_baselines([current],[],tables)


def test_no_channel_detector_training_or_formal_calls_in_math_preflight():
    import ast
    import inspect
    from scripts.run_modular_correction import construct,math_preflight
    for function in (construct,math_preflight):
        tree=ast.parse(inspect.getsource(function))
        calls=[n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id
               for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Attribute,ast.Name))]
        assert not set(calls)&{"generate","detect_data_zf","demodulate_data_bits","train","fit_network"}
    protocol=(ROOT/"docs/modular_correction_protocol.md").read_text(encoding="utf8")
    assert "1e-12" in protocol and "No reproduction" in protocol


def test_development_operators_and_unions_without_any_channel_call(monkeypatch):
    from src.cdl_benchmark import OfficialCDL
    from src.heldout_basis import HeldoutCDL
    from scripts.run_modular_correction import construct
    def forbidden(*a,**kw):raise AssertionError("channel forbidden during module construction")
    monkeypatch.setattr(OfficialCDL,"generate",forbidden)
    monkeypatch.setattr(HeldoutCDL,"generate",forbidden)
    c=construct()
    assert len(c["provenance"]["development_operator_regression"])==6
    assert all(r["historical_exact"] for r in c["provenance"]["development_operator_regression"])
    assert not c["provenance"]["heldout_used_for_module_construction"]
    assert not c["provenance"]["module_statistics_refitted"]
    assert len(c["capture"])==18 and len(c["scalar"])==6 and len(c["interaction"])==3


@pytest.mark.parametrize("k",(4,8,12))
def test_complex_union_span_dimension_and_orthonormality(k):
    rng=np.random.default_rng(20261640)
    p,_=np.linalg.qr(rng.normal(size=(40,16))+1j*rng.normal(size=(40,16)))
    m,_=np.linalg.qr(rng.normal(size=(40,16))+1j*rng.normal(size=(40,16)))
    q,info=orthonormal_union(p,m,k)
    assert info["dimension_budget"]==info["numerical_rank"]==2*k
    np.testing.assert_allclose(q.conj().T@q,np.eye(2*k),atol=2e-14)
    d=np.column_stack((p[:,:k],m[:,:k]))
    np.testing.assert_allclose(q@(q.conj().T@d),d,atol=2e-14)
    assert not q.flags.writeable


def test_union_overlap_rank_deficiency_uses_fixed_tolerance():
    p=np.eye(24,dtype=complex)[:,:16]
    q,info=orthonormal_union(p,p,8)
    assert info["numerical_rank"]==8 and info["dimension_budget"]==16
    assert info["relative_rank_tolerance"]==RANK_TOLERANCE==1e-12
    np.testing.assert_allclose(q@q.conj().T,np.diag([1.]*8+[0.]*16),atol=1e-14)


def test_frozen_modules_exclude_heldout_and_preserve_all_arrays():
    from src.heldout_basis import load_bases
    from src.cdl_benchmark import array_hash
    bases,proof=load_bases(ROOT)
    before={k:array_hash(b.u32) for k,b in bases.items()}
    unions=frozen_module_unions(bases)
    assert set(unions)==set(BUDGETS)
    assert before=={k:array_hash(b.u32) for k,b in bases.items()}
    assert all(n not in ("CDL-C-75","CDL-B-120") for scope in ("CDL-B-30","CDL-C-120") for n,_ in proof["basis_artifacts"][scope]["condition_ids"])
    with pytest.raises(ValueError):
        frozen_module_unions({"CDL-B-30":bases["all_conditions"],"CDL-C-120":bases["CDL-C-120"]})


def test_weighted_projection_matches_independent_energy_and_normal_residual():
    rng=np.random.default_rng(20261641)
    delta=rng.normal(size=(20,6))+1j*rng.normal(size=(20,6))
    a=rng.normal(size=(6,6))+1j*rng.normal(size=(6,6));c=a@a.conj().T+np.eye(6)
    lower=np.linalg.cholesky(c)
    q,_=np.linalg.qr(rng.normal(size=(20,8))+1j*rng.normal(size=(20,8)))
    g,info=weighted_projection(delta,lower,q)
    m=delta@lower;residual=(np.eye(20)-q@q.conj().T)@m
    assert info["capture"]==pytest.approx(1-np.linalg.norm(residual)**2/np.linalg.norm(m)**2,rel=1e-13)
    assert info["weighted_residual"]==pytest.approx(np.linalg.norm(residual)**2,rel=1e-13)
    np.testing.assert_allclose(q.conj().T@residual,0,atol=3e-14)
    np.testing.assert_allclose(g,q.conj().T@delta,atol=1e-14)


def test_one_real_scalar_analytic_solution_beats_brute_force_reference():
    target=np.array([[1+2j,3],[2-1j,-1+1j]],complex)
    mode=np.array([[2,1j],[1+1j,2-1j]],complex)
    lower=np.array([[2,0],[1j,1]],complex)
    fit=ScalarComposition.fit(target,[mode],lower)
    design=(mode@lower).ravel();response=(target@lower).ravel()
    reference=np.linalg.lstsq(np.concatenate([design.real,design.imag])[:,None],np.concatenate([response.real,response.imag]),rcond=None)[0]
    np.testing.assert_allclose(fit.coefficients,reference,atol=1e-14)
    grid=np.linspace(fit.coefficients[0]-1,fit.coefficients[0]+1,1001)
    brute=np.array([np.linalg.norm((target-beta*mode)@lower)**2 for beta in grid])
    assert fit.diagnostics["weighted_residual"]==pytest.approx(brute.min(),rel=1e-13)
    assert grid[brute.argmin()]==pytest.approx(fit.coefficients[0])
    assert fit.coefficients.dtype==np.float64


def test_two_real_modes_2x2_normal_equation_and_stacked_real_reference():
    rng=np.random.default_rng(20261642)
    modes=[rng.normal(size=(8,5))+1j*rng.normal(size=(8,5)) for _ in range(2)]
    target=rng.normal(size=(8,5))+1j*rng.normal(size=(8,5))
    a=rng.normal(size=(5,5))+1j*rng.normal(size=(5,5));lower=np.linalg.cholesky(a@a.conj().T+np.eye(5))
    fit=ScalarComposition.fit(target,modes,lower)
    design=np.column_stack([(m@lower).ravel() for m in modes]);response=(target@lower).ravel()
    reference=np.linalg.lstsq(np.vstack([design.real,design.imag]),np.r_[response.real,response.imag],rcond=None)[0]
    np.testing.assert_allclose(fit.coefficients,reference,atol=2e-15)
    gram=np.array(fit.diagnostics["normal_equation_gram"]);rhs=np.array(fit.diagnostics["normal_equation_rhs"])
    np.testing.assert_allclose(gram@fit.coefficients,rhs,atol=1e-12)
    np.testing.assert_allclose(fit.correction,reference[0]*modes[0]+reference[1]*modes[1],atol=1e-14)


def test_scalar_degenerate_modes_fail_without_regularization():
    a=np.ones((3,2),complex)
    with pytest.raises(ValueError,match="rank deficient"):
        ScalarComposition.fit(a,[a,a],np.eye(2))


def test_interaction_effective_rank_and_zero_residual():
    info=interaction_spectrum(np.diag([3.,2.,1.]).astype(complex),np.eye(3))
    p=np.array([9,4,1])/14
    assert info["effective_rank"]==pytest.approx(np.exp(-sum(p*np.log(p))))
    assert info["numerical_rank"]==3 and info["singular_values"]==[3.,2.,1.]
    assert interaction_spectrum(np.zeros((3,2)),np.eye(2))["effective_rank"]==0


def test_figure_preflight_all_QA_and_CSV_only_rebuild(tmp_path):
    import matplotlib
    from scripts.plot_modular_correction import audit,FILES
    from scripts.run_cdl_mismatch import csvsave
    before=dict(matplotlib.rcParams)
    union=[dict(shift=n,snr_db=s,dimension_budget=q,closure=.4+q/45,generic_closure=.3+q/50)
           for n in ("CDL-C-75","CDL-B-120") for s in (5,10,20) for q in BUDGETS]
    scalar=[dict(shift=n,snr_db=s,source_nmse_db=-s/2,nmse_db=-s/2-3,matched_nmse_db=-s/2-6)
            for n in ("CDL-C-75","CDL-B-120") for s in (5,10,20)]
    for name,rows in zip(FILES,(union,scalar)):csvsave(tmp_path/name,rows)
    audit(tmp_path,tmp_path/"figures")
    assert dict(matplotlib.rcParams)==before


def test_actual_results_if_present():
    path=ROOT/"outputs/t16d_modular_results.json"
    if not path.exists():pytest.skip("formal T16D validation results not yet generated")
    r=json.loads(path.read_text(encoding="utf8"))
    assert r["new_channel_realizations"]==r["train_channel_generator_calls"]==r["formal_inference_generator_calls"]==0
    assert r["BER_evaluations"]==r["detector_calls"]==0
    assert not r["new_model_trained"] and not r["formal_test_used"] and r["formal_test_sealed"]
    assert r["shared_T16C_baselines_exact"] and r["formal_validation_pairing_exact"]
    assert r["deterministic_validation_replay_used"] and not r["new_validation_realizations_generated"]
    assert r["t16c_historical_replay_gate_passed"] and r["t16d_methods_evaluated_on_same_replayed_realizations"]
    assert not r["historical_full_tensor_hashes_available"]
    assert r["formal_communication_regeneration_calls"]==r["formal_noise_rescaling_calls"]==0
    assert len(r["validation_nmse"])==18 and len(r["scalar_mode_results"])==6 and len(r["interaction_residual"])==3
    assert r["validation_replay_gate"]["all_200_H_hashes_exact"] and r["validation_replay_gate"]["all_200_paired_input_digests_exact"]
    assert r["validation_replay_gate"]["all_5600_historical_trajectory_rows_exact"]
    assert r["validation_replay_gate"]["all_56_historical_pooled_rows_exact"]
    info=r["union_basis_artifact"]
    from scripts.run_cdl_mismatch import sha
    from src.cdl_benchmark import array_hash
    assert sha(ROOT/info["path"])==info["file_sha256"]
    with np.load(ROOT/info["path"],allow_pickle=False) as f:
        for q in BUDGETS:
            assert array_hash(f[f"Q{q}"])==info["Q_sha256"][str(q)]
            assert f[f"Q{q}"].shape==(3968,r["union_numerical_rank"][str(q)])
    import csv
    with (ROOT/"outputs/t16d_trajectory_metrics.csv").open(encoding="utf8",newline="") as f:rows=list(csv.DictReader(f))
    assert len(rows)==5400 and "ber" not in rows[0]
    for aggregate in [*r["validation_nmse"],*r["scalar_mode_results"]]:
        selected=[x for x in rows if x["shift"]==aggregate["shift"] and int(x["snr_db"])==aggregate["snr_db"] and x["method"]==aggregate["method"]]
        assert len(selected)==100
        assert sum(float(x["error_energy"]) for x in selected)/sum(float(x["truth_energy"]) for x in selected)==aggregate["nmse_linear"]
    gate=scientific_gate(r["validation_nmse"],r["scalar_mode_results"])
    assert gate=={k:r[k] for k in gate}
    for p,h in {**r["source_sha256"],**r["frozen_history_sha256"],**r["numeric_artifact_sha256"]}.items():assert sha(ROOT/p)==h


@pytest.mark.parametrize("closures,label",[((.81,.9,.95),"q8"),((.79,.8,.9),"q16"),((.3,.4,.79),"weak"),((.3,.79,.8),"none")])
def test_preregistered_modular_gate_boundaries(closures,label):
    rows=[dict(shift=n,snr_db=s,dimension_budget=q,closure=v,metric_type="actual_validation") for n in ("CDL-C-75","CDL-B-120") for s in (10,20) for q,v in zip(BUDGETS,closures)]
    scalar=[dict(shift=n,snr_db=s,closure=.8,metric_type="actual_validation") for n in ("CDL-C-75","CDL-B-120") for s in (10,20)]
    gate=scientific_gate(rows,scalar)
    assert gate["scalar_mode_composition_promising"]
    assert gate["modular_union_q8_strong"]==(label=="q8")
    assert gate["modular_union_q16_strong"]==(label=="q16")
    assert gate["modular_union_weak"]==(label=="weak")
    scalar[0]["closure"] = .5
    assert scientific_gate(rows,scalar)["scalar_mode_composition_partial"]
    scalar[0]["closure"] = .499
    assert scientific_gate(rows,scalar)["scalar_mode_composition_insufficient"]
    scalar[0]["metric_type"]="anchor"
    with pytest.raises(ValueError):scientific_gate(rows,scalar)
