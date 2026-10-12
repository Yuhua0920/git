"""T16B frozen-statistics shared output basis and NMSE-only validation replay."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.linalg import cholesky
from src.shared_output_basis import (SharedOutputBasis, T16A_COMMIT, RANKS, VALIDATION_RANKS,
    SOURCE_MEAN_RANKS, EXPERIMENT_TYPE, anchored_nmse, validation_qthreshold, decision_gate,
    complexity_memory, closure)
from src.lowrank_filter import WeightedCorrection
from src.cdl_mismatch import ShiftCDL, SHIFTS, SNRS
from src.cdl_benchmark import observation_geometry, array_hash
from src.link import estimate_ls_linear_channel
from scripts.run_lowrank_filter import statistics, prepare, cache_trajectory, readcsv, BASELINES
from scripts.run_cdl_mismatch import load, save, sha, csvsave, history, environments
OUT = ROOT/"outputs"
SOURCES = ("src/shared_output_basis.py", "scripts/run_shared_output_basis.py", "scripts/plot_shared_output_basis.py",
    "scripts/audit_shared_output_basis.py", "tests/test_shared_output_basis.py", "docs/shared_output_basis_protocol.md")
RESULT = OUT/"t16b_shared_basis_results.json"


def accepted_snapshot():
    subprocess.check_call(["git", "merge-base", "--is-ancestor", T16A_COMMIT, "HEAD"], cwd=ROOT)
    subject = subprocess.check_output(["git", "show", "-s", "--format=%s", T16A_COMMIT], cwd=ROOT, text=True).strip()
    if subject != "feat: add oracle low-rank LMMSE filter-correction representation gate":
        raise ValueError("accepted T16A snapshot required")
    old = load(OUT/"t16a_lowrank_results.json")
    if not old["all_checks_passed"]: raise ValueError("T16A not accepted")
    for p,h in {**old["source_sha256"], **old["frozen_history_sha256"], **old["numeric_artifact_sha256"]}.items():
        if sha(ROOT/p) != h: raise ValueError(f"frozen T16A history changed: {p}")
    for p in old["source_sha256"]:
        blob = subprocess.check_output(["git", "show", f"{T16A_COMMIT}:{p}"], cwd=ROOT)
        if blob.replace(b"\r\n",b"\n") != (ROOT/p).read_bytes().replace(b"\r\n",b"\n"):
            raise ValueError(f"accepted implementation snapshot drift: {p}")
    return old


def construct():
    old = accepted_snapshot(); prior, source, targets, variances = statistics()
    estimates, matrix, seconds = prepare(source, targets, variances)
    matrices = {}; deltas = {}; checks = []; start = time.perf_counter()
    for name in SHIFTS:
        for s in SNRS:
            ws = estimates[name,BASELINES[0],s].weight; wt = estimates[name,BASELINES[1],s].weight
            covariance = targets[name].rpp+variances[s]*np.eye(64)
            c = WeightedCorrection.build(ws,wt,covariance)
            saved = [r["singular_value"] for r in old["singular_spectrum"] if r["shift"] == name and r["snr_db"] == s]
            if not np.array_equal(c.singular,np.array(saved)): raise ValueError("STOP: T16A singular values not exact")
            lower = cholesky(covariance,lower=True)
            # Match T16A's copy/layout and original Cholesky multiplication order.
            delta = c.target-c.source; m = delta@lower
            reconstructed = (c.left*c.singular)@c.right_h
            relative = float(np.linalg.norm(m-reconstructed)/np.linalg.norm(m))
            if relative > 1e-11: raise ValueError("M SVD reconstruction failed")
            old_diag = next(r for r in old["matrix_diagnostics"] if r["shift"] == name and r["snr_db"] == s)
            for key in ("source_W_residual","target_W_residual"):
                current = next(r for r in matrix if r["shift"] == name and r["snr_db"] == s)[key]
                if current != old_diag[key]: raise ValueError("T16A W residual drift")
            for a in (m,delta): a.flags.writeable = False
            matrices[name,s] = m; deltas[name,s] = delta
            checks.append(dict(shift=name,snr_db=s,all_64_historical_singular_values_exact=True,
                historical_M_hash_available=False,current_M_sha256=array_hash(m),
                reconstruction_relative_residual=relative,source_W_sha256=array_hash(ws),target_W_sha256=array_hash(wt)))
    bases = {"all_conditions":SharedOutputBasis.build(matrices)}
    bases.update({name:SharedOutputBasis.for_shift(matrices,name) for name in SHIFTS})
    maps = {(scope,name,s,q):b.coefficient_map(deltas[name,s],q) for scope,b in bases.items()
            for name in SHIFTS for s in SNRS for q in VALIDATION_RANKS}
    return old,prior,source,targets,variances,estimates,matrices,bases,maps,checks,dict(
        filter_construction_seconds=seconds,M_and_basis_SVD_seconds=time.perf_counter()-start)


def methods(name, representation):
    if not representation: return [(m,None,"source" if m == BASELINES[0] else "target","baseline") for m in BASELINES]
    other = next(n for n in SHIFTS if n != name)
    return ([(f"Shared-q{q}",q,"target","all_conditions") for q in VALIDATION_RANKS]
        +[(f"Shared-q{q}-SourceMeans",q,"source","all_conditions") for q in SOURCE_MEAN_RANKS]
        +[(f"Cross-{other}-q{q}",q,"target",other) for q in VALIDATION_RANKS])


def evaluate(name,i,cached,g,variances,estimates,source,targets,bases,maps,*,representation=False):
    """NMSE only. Truth enters metrics, never estimate or coefficient map."""
    declared = methods(name,representation)
    accum = {(s,m):dict(error_energy=0.,truth_energy=0.) for s in SNRS for m,_,_,_ in declared}
    digest = hashlib.sha256(); seconds = 0.
    for frame in cached:
        truth,z,clean = (frame[k] for k in ("truth","z","clean"))
        for s in SNRS:
            y = clean+np.sqrt(variances[s])*z; y.flags.writeable = False; yh = array_hash(y)
            sparse,_ = estimate_ls_linear_channel(y,g.layout)
            pilot = sparse.ravel()[g.selected]; pilot.flags.writeable = False
            for method,q,mean_kind,scope in declared:
                start = time.perf_counter()
                if not representation: coeff = estimates[name,method,s].estimate(pilot)
                else:
                    mean = targets[name] if mean_kind == "target" else source
                    coeff = bases[scope].estimate(pilot,estimates[name,BASELINES[0],s].weight,
                        mean.mean_p,mean.mean_d,q,maps[scope,name,s,q])
                seconds += time.perf_counter()-start
                accum[s,method]["truth_energy"] += frame["energy"]
                accum[s,method]["error_energy"] += float(np.sum(abs(coeff-truth.ravel()[g.data])**2))
                if array_hash(y) != yh: raise ValueError("shared Y changed")
            if frame["input_hash"] != [array_hash(frame[k]) for k in ("truth","bits","symbols","z")]:
                raise ValueError("shared historical input changed")
            for h in (*frame["input_hash"],yh): digest.update(h.encode("ascii"))
    rows = []
    for s in SNRS:
        for m,q,mean_kind,scope in declared:
            e = accum[s,m]
            rows.append(dict(shift=name,trajectory_id=i,split="validation_dev",frames=10,dF=16,snr_db=s,
                method=m,q=q,means=mean_kind,basis_scope=scope,pilot_ratio=float(g.layout.pilot_union.mean()),
                **e,nmse_linear=e["error_energy"]/e["truth_energy"]))
    return rows,seconds,digest.hexdigest()


def aggregate(rows):
    result = []
    for name in SHIFTS:
        for s in SNRS:
            for method in dict.fromkeys(r["method"] for r in rows if r["shift"] == name and r["snr_db"] == s):
                v = [r for r in rows if r["shift"] == name and r["snr_db"] == s and r["method"] == method]
                e = sum(r["error_energy"] for r in v); d = sum(r["truth_energy"] for r in v)
                result.append(dict(shift=name,snr_db=s,method=method,q=v[0]["q"],means=v[0]["means"],basis_scope=v[0]["basis_scope"],
                    trajectories=len(v),frames_per_trajectory=10,error_energy=e,truth_energy=d,nmse_linear=e/d,
                    nmse_db=float(10*np.log10(e/d)),trajectory_nmse_mean=float(np.mean([r["nmse_linear"] for r in v])),
                    trajectory_nmse_std=float(np.std([r["nmse_linear"] for r in v],ddof=1)),metric_type="actual_validation"))
    return result


def compare(actual,frozen,trajectory=True):
    fields = ("error_energy","truth_energy","nmse_linear") if trajectory else (
        "error_energy","truth_energy","nmse_linear","nmse_db","trajectory_nmse_mean","trajectory_nmse_std")
    count = 0
    index = {(r["shift"],int(r["snr_db"]),r["method"],int(r["trajectory_id"]) if trajectory else None):r for r in frozen}
    for r in actual:
        old = index[r["shift"],r["snr_db"],r["method"],r["trajectory_id"] if trajectory else None]
        for k in fields:
            if str(r[k]) != str(old[k]): raise ValueError(f"STOP: historical NMSE mismatch {r['shift']} {r['snr_db']} {r['method']} {k}")
            count += 1
    return count


def evidence(prior,old,name,i,h,digest):
    a = next(r for r in prior["paired_evaluation_evidence"] if r["shift"] == name and r["trajectory_id"] == i)
    b = next(r for r in old["replay_gate"]["evidence"] if r["shift"] == name and r["trajectory_id"] == i)
    if array_hash(h) != a["H_sha256"] or digest != a["paired_inputs_sha256"]:
        raise ValueError("STOP: historical H/paired input digest mismatch")
    if b["H_sha256"] != a["H_sha256"] or b["paired_inputs_sha256"] != digest:
        raise ValueError("STOP: T16A paired digest differs")
    return dict(shift=name,trajectory_id=i,H_sha256=array_hash(h),paired_inputs_sha256=digest,exact=True)


def preflight():
    require_environment()
    old,prior,source,targets,var,est,m,bases,maps,checks,timing = construct()
    g = observation_geometry(); proofs = []; frozen = readcsv("t15c_cdl_mismatch_trajectory_metrics.csv")
    for name in SHIFTS:
        h = ShiftCDL(name).generate(500,split="validation_dev"); cached = cache_trajectory(name,500,h,g)
        rows,_,digest = evaluate(name,500,cached,g,var,est,source,targets,bases,maps)
        compare(rows,frozen); compare(rows,readcsv("t16a_lowrank_trajectory_metrics.csv"))
        proofs.append(evidence(prior,old,name,500,h,digest))
        actual,_,second = evaluate(name,500,cached,g,var,est,source,targets,bases,maps,representation=True)
        if digest != second or len(actual) != 30: raise ValueError("small shared replay input/count drift")
    save(OUT/"t16b_shared_basis_preflight.json",dict(all_checks_passed=True,evidence=proofs,
        matrix_regression=checks,runtime=timing,BER_evaluations=0,detector_calls=0,
        source_sha256={p:sha(ROOT/p) for p in SOURCES}))
    print("T16B small frozen-statistics / exact historical NMSE / factored shared and cross replay PASS",flush=True)


def require_environment():
    if Path(sys.prefix).resolve() != (ROOT/".venv-cdl").resolve(): raise ValueError("frozen CDL environment required")


def run():
    require_environment()
    if RESULT.exists() or (OUT/"t16b_shared_correction_basis.csv").exists(): raise ValueError("existing numerical outputs protected")
    gate = load(OUT/"t16b_shared_basis_preflight.json")
    if not gate["all_checks_passed"] or any(sha(ROOT/p) != h for p,h in gate["source_sha256"].items()):
        raise ValueError("preflight missing or implementation drift")
    start = time.perf_counter(); before = history(); env = environments()
    old,prior,source,targets,var,est,m,bases,maps,checks,runtime = construct()
    g = observation_geometry()
    if g.selected.size != 64 or int(g.data.sum()) != 3968 or float(g.layout.pilot_union.mean()) != 1/32:
        raise ValueError("actual observation geometry drift")
    cache = {}; rows = []; proofs = []; comparisons = 0; replay_start = time.perf_counter()
    frozen_rows = readcsv("t15c_cdl_mismatch_trajectory_metrics.csv")
    t16a_rows = readcsv("t16a_lowrank_trajectory_metrics.csv")
    for name in SHIFTS:
        adapter = ShiftCDL(name)
        if adapter.provenance() != prior["sionna_provenance"][name]: raise ValueError("CDL provenance changed")
        for i in range(500,600):
            h = adapter.generate(i,split="validation_dev"); cached = cache_trajectory(name,i,h,g); cache[name,i] = cached
            actual,_,digest = evaluate(name,i,cached,g,var,est,source,targets,bases,maps)
            comparisons += compare(actual,frozen_rows)+compare(actual,t16a_rows)
            proofs.append(evidence(prior,old,name,i,h,digest)); rows.extend(actual)
            if (i-499)%25 == 0: print(f"Historical exact NMSE replay {name}: {i-499}/100",flush=True)
    baselines = aggregate(rows)
    comparisons += compare(baselines,readcsv("t15c_cdl_mismatch.csv"),False)
    comparisons += compare(baselines,readcsv("t16a_lowrank_filter_correction.csv"),False)
    runtime["historical_validation_replay_seconds"] = time.perf_counter()-replay_start
    replay = dict(all_exact=True,evidence=proofs,exact_NMSE_field_comparisons=comparisons,
        BER_evaluations=0,detector_calls=0,passed_before_shared_validation=True)
    save(OUT/"t16b_historical_replay_gate.json",replay)
    print("All 200 historical H/input and source/matched NMSE replay gates PASS; starting shared validation",flush=True)
    second_start = time.perf_counter(); estimator_seconds = 0.
    for name in SHIFTS:
        for i in range(500,600):
            actual,seconds,digest = evaluate(name,i,cache[name,i],g,var,est,source,targets,bases,maps,representation=True)
            expected = next(p for p in proofs if p["shift"] == name and p["trajectory_id"] == i)
            if digest != expected["paired_inputs_sha256"]: raise ValueError("shared validation input differs")
            rows.extend(actual); estimator_seconds += seconds
            if (i-499)%25 == 0: print(f"Shared/cross NMSE validation {name}: {i-499}/100",flush=True)
    runtime["representation_evaluation_seconds"] = time.perf_counter()-second_start
    runtime["factored_estimator_seconds"] = estimator_seconds
    pooled = aggregate(rows); primary = []; sensitivity = []; cross = []
    for r in pooled:
        if r["q"] is None: continue
        a = next(x for x in baselines if x["shift"] == r["shift"] and x["snr_db"] == r["snr_db"] and x["method"] == BASELINES[0])
        b = next(x for x in baselines if x["shift"] == r["shift"] and x["snr_db"] == r["snr_db"] and x["method"] == BASELINES[1])
        recovery = closure(a["nmse_db"],b["nmse_db"],r["nmse_db"])
        r["shared_closure"] = recovery["closure_r"]
        r["full_mismatch_gap_db"] = recovery["full_mismatch_gap_db"]
        r["recovery_db"] = recovery["rank_recovery_db"]
        (cross if r["basis_scope"] != "all_conditions" else primary if r["means"] == "target" else sensitivity).append(r)
    # Anchors enter only this descriptive energy report, AFTER validation and construction.
    captured = []; cross_energy = []; anchors = []; residuals = []
    frozen_aggregate = readcsv("t15c_cdl_mismatch.csv")
    for name in SHIFTS:
        for s in SNRS:
            matched = next(x for x in frozen_aggregate if x["shift"] == name and int(x["snr_db"]) == s and x["method"] == BASELINES[1])
            total_error,total_truth = float(matched["error_energy"]),float(matched["truth_energy"])
            count = int(matched["trajectories"])*int(matched["frames_per_trajectory"])
            anchors.append(dict(shift=name,snr_db=s,frame_count=count,matched_error_total=total_error,truth_energy_total=total_truth,
                source_csv="outputs/t15c_cdl_mismatch.csv",source_sha256=sha(OUT/"t15c_cdl_mismatch.csv"),q_independent=True))
            for scope in ("all_conditions",next(n for n in SHIFTS if n != name)):
                basis = bases[scope]
                for q in RANKS if scope == "all_conditions" else VALIDATION_RANKS:
                    energy = basis.energy(m[name,s],q)
                    delta = est[name,BASELINES[1],s].weight-est[name,BASELINES[0],s].weight
                    lower = cholesky(targets[name].rpp+var[s]*np.eye(64),lower=True)
                    direct = float(np.linalg.norm((delta-basis.output(q)@maps[scope,name,s,q])@lower)**2) if q in VALIDATION_RANKS else float(
                        np.linalg.norm((delta-basis.output(q)@basis.coefficient_map(delta,q))@lower)**2)
                    np.testing.assert_allclose(direct,energy["weighted_excess_MSE_per_frame"],rtol=1e-11,atol=1e-11)
                    prediction = anchored_nmse(energy["weighted_excess_MSE_per_frame"],total_error,total_truth,count)
                    entry = dict(shift=name,snr_db=s,q=q,basis_scope=scope,**energy,**prediction,
                        independent_weighted_residual_MSE_per_frame=direct)
                    (captured if scope == "all_conditions" else cross_energy).append(entry)
                    if scope == "all_conditions" and q in VALIDATION_RANKS:
                        observed = next(x for x in primary if x["shift"] == name and x["snr_db"] == s and x["q"] == q)
                        residuals.append(dict(shift=name,snr_db=s,q=q,
                            anchored_minus_validation_nmse_linear=prediction["anchored_predicted_nmse_linear"]-observed["nmse_linear"],
                            anchored_minus_validation_nmse_db=prediction["anchored_predicted_nmse_db"]-observed["nmse_db"]))
    definitions = {}; spectra = []; effective = {}
    for scope,basis in bases.items():
        cumulative,rank = basis.spectrum(); effective[scope] = rank
        u = basis.output(32)
        definitions[scope] = dict(basis_conditions=[list(c) for c in basis.conditions],Z_shape=list(basis.normalized.shape),
            Z_sha256=array_hash(basis.normalized),U32_sha256=array_hash(u),
            block_squared_energy=[float(np.linalg.norm(v)**2) for v in np.split(basis.normalized,len(basis.conditions),axis=1)],
            orthonormality_relative_residual=float(np.linalg.norm(u.conj().T@u-np.eye(32))/np.linalg.norm(np.eye(32))),
            heldout_statistics_used_for_basis=False,effective_rank=rank)
        for j,(value,energy) in enumerate(zip(basis.singular,cumulative),1):
            spectra.append(dict(basis_scope=scope,singular_index=j,singular_value=float(value),cumulative_squared_energy=float(energy)))
    q80 = {}; q90 = {}; comparison = []
    for name in SHIFTS:
        q80[name] = {}; q90[name] = {}
        for s in SNRS:
            group = [r for r in primary if r["shift"] == name and r["snr_db"] == s]
            q80[name][str(s)] = validation_qthreshold(group,.8); q90[name][str(s)] = validation_qthreshold(group,.9)
            previous = next(r for r in old["rank_results"] if r["shift"] == name and r["snr_db"] == s and r["rank"] == 4)
            comparison.append(dict(shift=name,snr_db=s,T16A_condition_r4_closure=previous["closure_r"],
                **{f"shared_q{q}_closure":next(r["shared_closure"] for r in group if r["q"] == q) for q in (4,8,16)}))
    # CSV schemas are uniform; unmeasured ranks have empty observed metrics.
    primary_csv = []
    for e in captured:
        actual = next((r for r in primary if r["shift"] == e["shift"] and r["snr_db"] == e["snr_db"] and r["q"] == e["q"]),None)
        primary_csv.append(dict(**e,validation_nmse_linear=None if actual is None else actual["nmse_linear"],
            validation_nmse_db=None if actual is None else actual["nmse_db"],shared_closure=None if actual is None else actual["shared_closure"],
            metric_type="exact_energy_and_anchor_only" if actual is None else "actual_validation_with_descriptive_anchor"))
    cross_csv = []
    for e in cross_energy:
        actual = next(r for r in cross if r["shift"] == e["shift"] and r["snr_db"] == e["snr_db"] and r["q"] == e["q"])
        cross_csv.append(dict(**e,validation_nmse_linear=actual["nmse_linear"],validation_nmse_db=actual["nmse_db"],
            shared_closure=actual["shared_closure"],metric_type="actual_validation_with_descriptive_anchor"))
    for name,data in (("t16b_shared_correction_basis.csv",primary_csv),("t16b_shared_basis_spectrum.csv",spectra),
        ("t16b_cross_shift_basis.csv",cross_csv),("t16b_shared_basis_trajectory_metrics.csv",rows),
        ("t16b_condition_vs_shared.csv",comparison),("t16b_source_mean_sensitivity.csv",sensitivity),
        ("t16b_baseline_nmse.csv",baselines)):
        csvsave(OUT/name,data)
    if any(sha(ROOT/p) != h for p,h in before.items()): raise ValueError("frozen historical files changed")
    runtime["total_numeric_wall_seconds"] = time.perf_counter()-start
    result = dict(task="T16B",experiment_type=EXPERIMENT_TYPE,literature_gate_passed=True,proposed_method=False,
        shared_representation_gate=True,accepted_T16A_commit=T16A_COMMIT,shared_basis_definition=definitions,
        condition_normalization="M/||M||F, equal 1/sqrt(6) main and 1/sqrt(3) cross weights",
        shared_spectrum=spectra,effective_rank=effective,capture_by_condition=captured,
        exact_excess_MSE=captured,expected_nmse=None,expected_nmse_unavailable_reason="Frozen NPZ lacks Rdd/data-domain prior trace",
        absolute_theoretical_expected_nmse_available=False,exact_excess_mse_available=True,
        anchored_predicted_nmse_available=True,anchor_source="frozen_matched_validation",anchored_nmse_used_for_decision_gate=False,
        anchor_used_for_q80_q90=False,anchor_used_for_basis=False,anchors=anchors,anchored_predicted_nmse=captured,
        anchored_prediction_residual=residuals,validation_nmse=primary,baseline_nmse=baselines,source_mean_sensitivity=sensitivity,
        shared_closure=primary,q80=q80,q90=q90,q80_validation=q80,q90_validation=q90,
        q_threshold_scope="minimum only among actually replayed q=4/8/16/24; no anchored thresholds",
        cross_shift_results=cross,cross_shift_exact_energy=cross_energy,
        cross_shift_scope="development representation transfer with oracle heldout coefficient maps, not formal OOD",
        condition_vs_shared=comparison,complexity=[complexity_memory(q) for q in (4,8,16)],memory=[complexity_memory(q) for q in (4,8,16)],
        **decision_gate(primary,cross),matrix_regression=checks,replay_gate=replay,snr_policy=prior["snr_policy"],
        source_filter_provenance=old["source_filter_provenance"],target_filter_provenance=old["target_filter_provenance"],
        statistics_refitted=False,new_channel_realizations=False,deterministic_existing_validation_replay=True,
        BER_evaluations=0,detector_calls=0,new_model_trained=False,formal_test_used=False,formal_test_sealed=True,
        runtime=runtime,source_sha256={p:sha(ROOT/p) for p in SOURCES},frozen_history_sha256=before,
        frozen_history_unchanged=True,environment_before=env,all_checks_passed=False,completion_state="numeric complete; final audit pending",
        warnings=["Missing Rdd: no absolute theoretical expected NMSE; fixed frozen-validation anchor is descriptive only.",
            "q80/q90 use measured q=4/8/16/24 only; unmeasured ranks are not validation results.",
            "Oracle target means/statistics/coefficient maps; no deployable predictor or proposed method.",
            "Development validation only; formal test sealed; ten continuous frames are correlated.",
            "Closure is not clipped; actual heldout NMSE need not be monotone.",
            "Existing frequency-domain per-symbol abstraction retained; no within-symbol ICI.",
            "Optional Triton flop counter unavailable in frozen CDL environment."])
    save(RESULT,result)
    print({"gate":decision_gate(primary,cross),"q80":q80,"q90":q90,"runtime":runtime},flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--preflight",action="store_true")
    if p.parse_args().preflight: preflight()
    else: run()
