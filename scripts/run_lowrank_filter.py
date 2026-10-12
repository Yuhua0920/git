"""T16A exact historical replay followed by an oracle weighted-rank gate."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.lowrank_filter import (WeightedCorrection, RANKS, SOURCE_MEAN_RANKS,
    T15C_COMMIT, EXPERIMENT_TYPE, closure, required_rank, design_gate, complexity, angles)
from src.cdl_mismatch import ShiftCDL, SHIFTS, SNRS, seeds, manifest_rows, frozen_variances
from src.cdl_benchmark import FrozenStatistics, frames, array_hash, observation_geometry
from src.link import (assemble_pilot_data_frame, apply_frequency_mimo_channel,
    estimate_ls_linear_channel, extract_reference_data_bits, detect_data_zf, demodulate_data_bits)
from scripts.run_cdl_mismatch import (source_statistics, communication, environments, history,
    load, save, sha, csvsave, evaluate_trajectory)
OUT = ROOT/"outputs"
SOURCES = ("src/lowrank_filter.py", "scripts/run_lowrank_filter.py", "scripts/plot_lowrank_filter.py",
           "scripts/audit_lowrank_filter.py", "tests/test_lowrank_filter.py", "docs/lowrank_filter_protocol.md")
BASELINES = ("R10-C-frozen", "R10-matched")
MEAN_AUDIT = "R10-targetW-sourceMeans"


def readcsv(name):
    with (OUT/name).open(encoding="utf8", newline="") as f:
        return list(csv.DictReader(f))


def require_snapshot():
    subprocess.check_call(["git", "merge-base", "--is-ancestor", T15C_COMMIT, "HEAD"], cwd=ROOT)
    subject = subprocess.check_output(["git", "show", "-s", "--format=%s", T15C_COMMIT], cwd=ROOT, text=True).strip()
    if subject != "feat: add realistic CDL covariance-mismatch gate":
        raise ValueError("accepted T15C snapshot required")
    r = load(OUT/"t15c_cdl_mismatch_results.json")
    if not r["all_checks_passed"]: raise ValueError("T15C not accepted")
    # Historical implementation and result files remain byte-identical today.
    for p, h in {**r["frozen_history_sha256"], **r["source_sha256"], **r["numeric_artifact_sha256"]}.items():
        if sha(ROOT/p) != h: raise ValueError(f"T15C historical file changed: {p}")
    # Git stores LF for text; compare declared snapshot content without inventing
    # an old checkout hash. Binary statistics are exact Git-blob hashes.
    for p in r["source_sha256"]:
        blob = subprocess.check_output(["git", "show", f"{T15C_COMMIT}:{p}"], cwd=ROOT)
        current = (ROOT/p).read_bytes()
        if blob.replace(b"\r\n", b"\n") != current.replace(b"\r\n", b"\n"):
            raise ValueError(f"accepted snapshot implementation differs: {p}")
    actual = readcsv("t15c_cdl_manifest.csv")
    expected = [{k: str(v) for k, v in row.items()} for row in manifest_rows()]
    if actual != expected: raise ValueError("manifest or seed policy drift")
    return r


def statistics():
    old = require_snapshot(); source, source_prov = source_statistics(); targets = {}
    if source_prov["component_sha256"] != old["source_model_provenance"]["component_sha256"]:
        raise ValueError("source component drift")
    for name, info in old["matched_statistics_provenance"].items():
        path = ROOT/info["path"]
        if sha(path) != info["sha256"]: raise ValueError("target file hash drift")
        blob = subprocess.check_output(["git", "show", f"{T15C_COMMIT}:{info['path']}"], cwd=ROOT)
        if hashlib.sha256(blob).hexdigest() != info["sha256"]: raise ValueError("target historical snapshot drift")
        with np.load(path) as f:
            values = [f[k].copy() for k in ("mean_p", "mean_d", "rpp", "rdp")]
            ids = tuple(map(int, f["train_trajectory_ids"])); count = int(f["samples"])
        if ids != tuple(range(500)) or count != 5000: raise ValueError("target train protocol drift")
        for k, a in zip(("mean_p", "mean_d", "rpp", "rdp"), values):
            if array_hash(a) != info["component_sha256"][k]: raise ValueError("target component drift")
            a.flags.writeable = False
        targets[name] = FrozenStatistics(*values, ids, count)
    _, variances = frozen_variances(load(OUT/"t15a_cdl_results.json"))
    if {str(k): v for k, v in variances.items()} != old["snr_policy"]["production_variances"]:
        raise ValueError("T15C production noise drift")
    return old, source, targets, variances


def prepare(source, targets, variances):
    start = time.perf_counter(); estimators = {}; checks = []
    for name, target in targets.items():
        for s in SNRS:
            for m, stats in ((BASELINES[0], source), (BASELINES[1], target)):
                e = stats.prepare(variances[s])
                if e.weight.shape != (3968, 64) or e.weight.dtype != np.complex128 or e.solve_relative_residual > 1e-11:
                    raise ValueError("W shape/type/residual gate failed")
                e.weight.flags.writeable = False; estimators[name, m, s] = e
            checks.append(dict(shift=name, snr_db=s, source_W_residual=estimators[name, BASELINES[0], s].solve_relative_residual,
                               target_W_residual=estimators[name, BASELINES[1], s].solve_relative_residual))
    return estimators, checks, time.perf_counter()-start


def cache_trajectory(name, i, h, g):
    h.flags.writeable = False; cached = []
    for j, truth in enumerate(frames(h)):
        bits, symbols, z = communication(seeds(name, "validation_dev", i, j))
        x = assemble_pilot_data_frame(symbols, g.layout)
        clean = apply_frequency_mimo_channel(truth, x); clean.flags.writeable = False
        reference = extract_reference_data_bits(bits, g.layout.data_mask)
        for a in reference: a.flags.writeable = False
        cached.append(dict(truth=truth, bits=bits, symbols=symbols, z=z, clean=clean, reference=reference,
            energy=float(np.sum(abs(truth.ravel()[g.data])**2)),
            input_hash=[array_hash(a) for a in (truth, bits, symbols, z)]))
    return cached


def bit_errors(y, coefficients, g, reference, timing):
    estimate = np.full(4096, np.nan+1j*np.nan, np.complex128)
    estimate[g.data] = coefficients; estimate.flags.writeable = False
    start = time.perf_counter()
    detected = detect_data_zf(y, estimate.reshape(2, 2, 64, 16), g.layout.data_mask)*np.sqrt(2)
    timing["ZF_seconds"] += time.perf_counter()-start
    predicted = demodulate_data_bits(detected, g.layout.data_mask)
    return sum(int(np.count_nonzero(a != b)) for a, b in zip(predicted, reference))


def evaluate_cached(name, i, cached, g, variances, estimators, source, *, corrections=None, targets=None):
    """Same scalar baseline arithmetic and digest order as frozen T15C.

    True H is used only for error metrics. Factors/means are constructed before
    this function, independent of validation. No mutable inputs are passed on.
    """
    methods = [(m, None, "source" if m == BASELINES[0] else "target", True) for m in BASELINES]
    if corrections is None:
        methods += [(MEAN_AUDIT, None, "source", False)]
    else:
        methods = [(f"R11-{r}-OracleCorrection", r, "target", True) for r in RANKS]
        methods += [(f"R11-{r}-SourceMeans", r, "source", False) for r in SOURCE_MEAN_RANKS]
    accum = {(s, m): dict(error_energy=0., truth_energy=0., bit_errors=0 if ber else None, total_bits=0 if ber else None)
             for s in SNRS for m, _, _, ber in methods}
    timing = dict(ZF_seconds=0., estimator_seconds=0.); digest = hashlib.sha256()
    for frame in cached:
        truth, z, clean, reference = (frame[k] for k in ("truth", "z", "clean", "reference"))
        for s in SNRS:
            y = clean+np.sqrt(variances[s])*z; y.flags.writeable = False; yh = array_hash(y)
            sparse, _ = estimate_ls_linear_channel(y, g.layout)
            pilot = sparse.ravel()[g.selected]; pilot.flags.writeable = False
            for method, rank, mean_kind, ber in methods:
                start = time.perf_counter()
                if corrections is None:
                    if method == MEAN_AUDIT:
                        e = estimators[name, BASELINES[1], s]
                        coeff = source.mean_d+(pilot-source.mean_p)@e.weight.T
                    else:
                        coeff = estimators[name, method, s].estimate(pilot)
                else:
                    weighted, factors = corrections[name, s]
                    mean = targets[name] if mean_kind == "target" else source
                    coeff = weighted.estimate(pilot, mean.mean_p, mean.mean_d, factors[rank])
                    if rank == 64:
                        target_e = estimators[name, BASELINES[1], s]
                        exact = (target_e.estimate(pilot) if mean_kind == "target" else
                                 source.mean_d+(pilot-source.mean_p)@target_e.weight.T)
                        np.testing.assert_allclose(coeff, exact, rtol=1e-10, atol=1e-10)
                timing["estimator_seconds"] += time.perf_counter()-start
                entry = accum[s, method]
                entry["truth_energy"] += frame["energy"]
                entry["error_energy"] += float(np.sum(abs(coeff-truth.ravel()[g.data])**2))
                if ber:
                    entry["bit_errors"] += bit_errors(y, coeff, g, reference, timing)
                    entry["total_bits"] += sum(a.size for a in reference)
                if array_hash(y) != yh: raise ValueError("shared Y mutated")
            if frame["input_hash"] != [array_hash(frame[k]) for k in ("truth", "bits", "symbols", "z")]:
                raise ValueError("shared H/bits/symbols/unit noise mutated")
            for h in (*frame["input_hash"], yh): digest.update(h.encode("ascii"))
    rows = []
    for s in SNRS:
        for m, rank, mean_kind, ber in methods:
            e = accum[s, m]
            rows.append(dict(shift=name, trajectory_id=i, split="validation_dev", frames=10, dF=16, snr_db=s,
                method=m, rank=rank, means=mean_kind, pilot_ratio=float(g.layout.pilot_union.mean()), **e, nmse_linear=e["error_energy"]/e["truth_energy"],
                ber=e["bit_errors"]/e["total_bits"] if ber else None))
    return rows, timing, digest.hexdigest()


def compare_rows(actual, frozen, *, trajectory=True):
    compared = 0
    fields = ("error_energy", "truth_energy", "bit_errors", "total_bits", "nmse_linear", "ber")
    if not trajectory: fields += ("nmse_db", "trajectory_nmse_mean", "trajectory_nmse_std", "trajectory_ber_std")
    for row in actual:
        if row["method"] not in BASELINES: continue
        old = next(r for r in frozen if r["shift"] == row["shift"] and r["method"] == row["method"] and int(r["snr_db"]) == row["snr_db"]
                   and (not trajectory or int(r["trajectory_id"]) == row["trajectory_id"]))
        for k in fields:
            if str(row[k]) != str(old[k]):
                raise ValueError(f"STOP: exact T15C replay mismatch {row['shift']} {row.get('trajectory_id')} {row['snr_db']} {row['method']} {k}: {row[k]} != {old[k]}")
            compared += 1
    return compared


def aggregate(rows):
    result = []
    for name in SHIFTS:
        for s in SNRS:
            methods = list(dict.fromkeys(r["method"] for r in rows if r["shift"] == name and r["snr_db"] == s))
            for m in methods:
                v = [r for r in rows if r["shift"] == name and r["snr_db"] == s and r["method"] == m]
                e = sum(r["error_energy"] for r in v); d = sum(r["truth_energy"] for r in v)
                ber_available = v[0]["bit_errors"] is not None
                errors = sum(r["bit_errors"] for r in v) if ber_available else None
                bits = sum(r["total_bits"] for r in v) if ber_available else None
                result.append(dict(shift=name, snr_db=s, dF=16, pilot_ratio=v[0]["pilot_ratio"], method=m, rank=v[0]["rank"], means=v[0]["means"],
                    trajectories=len(v), frames_per_trajectory=10, error_energy=e, truth_energy=d,
                    nmse_linear=e/d, nmse_db=float(10*np.log10(e/d)), bit_errors=errors, total_bits=bits,
                    ber=errors/bits if ber_available else None,
                    trajectory_nmse_mean=float(np.mean([r["nmse_linear"] for r in v])),
                    trajectory_nmse_std=float(np.std([r["nmse_linear"] for r in v], ddof=1)),
                    trajectory_ber_std=float(np.std([r["ber"] for r in v], ddof=1)) if ber_available else None))
    return result


def preflight():
    if Path(sys.prefix).resolve() != (ROOT/".venv-cdl").resolve(): raise ValueError("frozen CDL environment required")
    old, source, targets, variances = statistics(); g = observation_geometry()
    estimates, matrix, seconds = prepare(source, targets, variances)
    proofs = []; frozen = readcsv("t15c_cdl_mismatch_trajectory_metrics.csv")
    for name in SHIFTS:
        h = ShiftCDL(name).generate(500, split="validation_dev")
        historical = next(e for e in old["paired_evaluation_evidence"] if e["shift"] == name and e["trajectory_id"] == 500)
        if array_hash(h) != historical["H_sha256"]: raise ValueError("small H replay failed")
        cached = cache_trajectory(name, 500, h, g)
        actual, _, digest = evaluate_cached(name, 500, cached, g, variances, estimates, source)
        compare_rows(actual, frozen)
        # Independent unchanged T15C evaluator at the same existing sample.
        legacy = {(m, s): estimates[name, m, s] for m in BASELINES for s in SNRS}
        legacy_rows = evaluate_trajectory(name, 500, h, g, legacy, variances)[0]
        compare_rows(actual, legacy_rows)
        if digest != historical["paired_inputs_sha256"]: raise ValueError("small paired digest replay failed")
        corrections = {}
        for s in SNRS:
            c = WeightedCorrection.build(estimates[name, BASELINES[0], s].weight, estimates[name, BASELINES[1], s].weight,
                                         targets[name].rpp+variances[s]*np.eye(64))
            corrections[name, s] = (c, {r: c.factors(r) for r in RANKS})
            if c.diagnostics()["full_rank_filter_relative_residual"] > 1e-11: raise ValueError("small fullrank gate failed")
        rank_rows, _, rank_digest = evaluate_cached(name, 500, cached, g, variances, estimates, source, corrections=corrections, targets=targets)
        if digest != rank_digest: raise ValueError("small rank input drift")
        for s in SNRS:
            matched = next(r for r in actual if r["method"] == BASELINES[1] and r["snr_db"] == s)
            full = next(r for r in rank_rows if r["rank"] == 64 and r["means"] == "target" and r["snr_db"] == s)
            np.testing.assert_allclose(full["nmse_linear"], matched["nmse_linear"], rtol=1e-10)
            if full["bit_errors"] != matched["bit_errors"]: raise ValueError("small fullrank integer BER drift")
        proofs.append(dict(shift=name, trajectory_id=500, H_sha256=array_hash(h), paired_inputs_sha256=digest,
                           exact_T15C_and_unchanged_evaluator=True, fullrank_NMSE_and_BER=True))
    result = dict(all_checks_passed=True, accepted_T15C_commit=T15C_COMMIT, evidence=proofs,
                  matrix_checks=matrix, filter_construction_seconds=seconds, source_sha256={p: sha(ROOT/p) for p in SOURCES})
    save(OUT/"t16a_lowrank_preflight.json", result)
    print("T16A small exact historical H/input/NMSE/BER replay and weighted r64 gate PASS", flush=True)


def subspaces(corrections):
    from itertools import combinations
    rows = []
    pairs = [("within_shift_SNR", name, a, name, b) for name in SHIFTS for a, b in combinations(SNRS, 2)]
    names = tuple(SHIFTS)
    pairs += [("cross_shift_same_SNR", names[0], s, names[1], s) for s in SNRS]
    for kind, n1, s1, n2, s2 in pairs:
        first, second = corrections[n1, s1][0], corrections[n2, s2][0]
        for r in (4, 8):
            for space in ("left_data", "right_weighted_pilot"):
                gap1 = float((first.singular[r-1]-first.singular[r])/first.singular[0])
                gap2 = float((second.singular[r-1]-second.singular[r])/second.singular[0])
                for index, angle in enumerate(angles(first, second, r, space), 1):
                    rows.append(dict(comparison=kind, first_shift=n1, first_snr_db=s1, second_shift=n2, second_snr_db=s2,
                        rank=r, space=space, angle_index_descending=index, angle_degrees=float(angle),
                        first_boundary_gap_relative_to_sigma1=gap1, second_boundary_gap_relative_to_sigma1=gap2))
    return rows


def run():
    if Path(sys.prefix).resolve() != (ROOT/".venv-cdl").resolve(): raise ValueError("frozen CDL environment required")
    if (OUT/"t16a_lowrank_filter_correction.csv").exists(): raise ValueError("existing numerical results protected")
    start = time.perf_counter(); old, source, targets, variances = statistics()
    gate = load(OUT/"t16a_lowrank_preflight.json")
    if not gate["all_checks_passed"] or any(sha(ROOT/p) != h for p, h in gate["source_sha256"].items()):
        raise ValueError("small preflight or source gate missing/drifted")
    env = environments(); before = history(); g = observation_geometry()
    if g.selected.size != 64 or int(g.data.sum()) != 3968 or float(g.layout.pilot_union.mean()) != 1/32:
        raise ValueError("actual pilot/data geometry drift")
    estimates, matrix, filter_seconds = prepare(source, targets, variances)
    frozen_rows = readcsv("t15c_cdl_mismatch_trajectory_metrics.csv")
    frozen_agg = readcsv("t15c_cdl_mismatch.csv")
    cache = {}; rows = []; proofs = []; comparisons = 0
    runtime = dict(filter_construction_seconds=filter_seconds, SVD_seconds=0., validation_replay_seconds=0.,
                   representation_evaluation_seconds=0., ZF_seconds=0., deterministic_channel_replay_seconds=0.)
    replay_start = time.perf_counter()
    for name in SHIFTS:
        adapter = ShiftCDL(name)
        if adapter.provenance() != old["sionna_provenance"][name]: raise ValueError("CDL environment/geometry implementation provenance drift")
        for i in range(500, 600):
            tick = time.perf_counter(); h = adapter.generate(i, split="validation_dev")
            runtime["deterministic_channel_replay_seconds"] += time.perf_counter()-tick
            expected = next(e for e in old["paired_evaluation_evidence"] if e["shift"] == name and e["trajectory_id"] == i)
            if array_hash(h) != expected["H_sha256"]: raise ValueError("STOP: historical H hash replay failed")
            cached = cache_trajectory(name, i, h, g); cache[name, i] = cached
            values, timing, digest = evaluate_cached(name, i, cached, g, variances, estimates, source)
            comparisons += compare_rows(values, frozen_rows)
            if digest != expected["paired_inputs_sha256"]: raise ValueError("STOP: paired input replay failed")
            rows.extend(values); runtime["ZF_seconds"] += timing["ZF_seconds"]
            proofs.append(dict(shift=name, trajectory_id=i, H_sha256=array_hash(h), paired_inputs_sha256=digest,
                               exact_T15C_H_and_inputs=True, exact_source_target_NMSE_BER=True))
            if (i-499) % 20 == 0: print(f"{name} exact historical replay {i-499}/100", flush=True)
    comparisons += compare_rows(aggregate(rows), frozen_agg, trajectory=False)
    runtime["validation_replay_seconds"] = time.perf_counter()-replay_start
    replay_gate = dict(all_exact=True, tolerance_used=False, trajectory_field_comparisons=comparisons,
                       evidence=proofs, gate_passed_before_rank_performance=True)
    save(OUT/"t16a_lowrank_replay_gate.json", replay_gate)
    print("Global T15C replay gate PASS: all200 H/input digests and source/target trajectory/pooled NMSE/BER exact", flush=True)

    # Mean-only audit is calculated before weighted representation results.
    baseline_agg = aggregate(rows); mean_audit = []
    for name in SHIFTS:
        for s in SNRS:
            a = next(r for r in baseline_agg if r["shift"] == name and r["snr_db"] == s and r["method"] == MEAN_AUDIT)
            matched = next(r for r in baseline_agg if r["shift"] == name and r["snr_db"] == s and r["method"] == BASELINES[1])
            mean_audit.append(dict(shift=name, snr_db=s, targetW_sourceMeans_nmse_db=a["nmse_db"], targetW_targetMeans_nmse_db=matched["nmse_db"],
                                   mean_mismatch_penalty_db=a["nmse_db"]-matched["nmse_db"], descriptive_only=True))
    corrections = {}; spectra = []; effective = []
    for diag in matrix:
        name, s = diag["shift"], diag["snr_db"]; tick = time.perf_counter()
        c = WeightedCorrection.build(estimates[name, BASELINES[0], s].weight, estimates[name, BASELINES[1], s].weight,
                                     targets[name].rpp+variances[s]*np.eye(64))
        runtime["SVD_seconds"] += time.perf_counter()-tick
        diag.update(c.diagnostics())
        if diag["covariance_min_eigenvalue"] <= 0 or any(v > 1e-11 for k, v in diag.items() if "residual" in k):
            raise ValueError("weighted matrix residual/PD gate failed")
        tick = time.perf_counter(); corrections[name, s] = (c, {r: c.factors(r) for r in RANKS})
        runtime["factor_triangular_solve_seconds"] = runtime.get("factor_triangular_solve_seconds", 0.)+time.perf_counter()-tick
        energy, er = c.spectrum(); effective.append(dict(shift=name, snr_db=s, effective_rank=er))
        spectra.extend(dict(shift=name, snr_db=s, singular_index=j+1, singular_value=float(c.singular[j]),
                            cumulative_weighted_energy=float(energy[j])) for j in range(64))
    subspace = subspaces(corrections)
    rankstart = time.perf_counter()
    for name in SHIFTS:
        for i in range(500, 600):
            values, timing, digest = evaluate_cached(name, i, cache[name, i], g, variances, estimates, source,
                                                     corrections=corrections, targets=targets)
            expected = next(e for e in proofs if e["shift"] == name and e["trajectory_id"] == i)
            if digest != expected["paired_inputs_sha256"]: raise ValueError("rank pass differs from exact replay inputs")
            for s in SNRS:
                full = next(r for r in values if r["rank"] == 64 and r["means"] == "target" and r["snr_db"] == s)
                target = next(r for r in rows if r["shift"] == name and r["trajectory_id"] == i and r["snr_db"] == s and r["method"] == BASELINES[1])
                np.testing.assert_allclose(full["nmse_linear"], target["nmse_linear"], rtol=1e-10)
                if full["bit_errors"] != target["bit_errors"]: raise ValueError("r64 integer BER mismatch")
            rows.extend(values); runtime["ZF_seconds"] += timing["ZF_seconds"]
            if (i-499) % 20 == 0: print(f"{name} oracle ranks {i-499}/100", flush=True)
    runtime["representation_evaluation_seconds"] = time.perf_counter()-rankstart
    aggregates = aggregate(rows); ranks = []; sensitivity = []; r80 = {}; r90 = []
    for name in SHIFTS:
        r80[name] = {}; ranks90 = {}
        for s in SNRS:
            group = [r for r in aggregates if r["shift"] == name and r["snr_db"] == s]
            source_row = next(r for r in group if r["method"] == BASELINES[0]); target_row = next(r for r in group if r["method"] == BASELINES[1])
            current = []
            for r in group:
                r.update(full_mismatch_gap_db=None, rank_recovery_db=None, closure_r=None)
                if r["rank"] is not None:
                    r.update(closure(source_row["nmse_db"], target_row["nmse_db"], r["nmse_db"]))
                    if r["means"] == "target": current.append(r); ranks.append(r)
                    else: sensitivity.append(r)
            r80[name][str(s)] = required_rank(current, .8); ranks90[str(s)] = required_rank(current, .9)
        r90.append((name, ranks90))
    source_mean_penalties = []
    for r in sensitivity:
        target_mean = next(x for x in ranks if x["shift"] == r["shift"] and x["snr_db"] == r["snr_db"] and x["rank"] == r["rank"])
        source_mean_penalties.append(dict(shift=r["shift"], snr_db=r["snr_db"], rank=r["rank"],
                                          source_minus_target_means_nmse_db=r["nmse_db"]-target_mean["nmse_db"]))
    csvsave(OUT/"t16a_lowrank_filter_correction.csv", aggregates)
    csvsave(OUT/"t16a_lowrank_singular_spectrum.csv", spectra)
    csvsave(OUT/"t16a_lowrank_subspace_angles.csv", subspace)
    csvsave(OUT/"t16a_lowrank_trajectory_metrics.csv", rows)
    if any(sha(ROOT/p) != h for p, h in before.items()): raise ValueError("historical files modified")
    runtime["total_numeric_wall_seconds"] = time.perf_counter()-start
    result = dict(task="T16A", experiment_type=EXPERIMENT_TYPE, literature_gate_passed=True, proposed_method=False, representation_gate=True,
        accepted_T15C_commit=T15C_COMMIT, source_filter_provenance=old["source_model_provenance"],
        target_filter_provenance=old["matched_statistics_provenance"], statistics_refitted=False, new_channel_realizations=False,
        deterministic_existing_validation_replay=True, mean_mismatch=mean_audit, source_mean_sensitivity=sensitivity,
        source_mean_sensitivity_penalties=source_mean_penalties,
        weighted_svd_definition=dict(Cz="Rpp_target+sigma2 I = L L^H", M="(Wt-Ws)L", rank_correction="A=U_r Sigma_r; Q L=V_r^H; B=A Q",
            objective="E||(DeltaW-B)x||^2 = ||(DeltaW-B)L||_F^2", primary_means="target", source_mean_ranks=list(SOURCE_MEAN_RANKS),
            inverse_used=False, dense_correction_inference=False, oracle_statistics_used=True),
        matrix_diagnostics=matrix, rank_results=ranks, r80=r80, r90=dict(r90), singular_spectrum=spectra,
        effective_rank=effective, subspace_similarity=subspace, complexity=[complexity(r) for r in (1, 2, 4, 8, 16)],
        design_gate_definition=dict(primary_snrs=[10,20], strong="common declared rank<=8 with all four closures>=0.80",
            weak="otherwise any primary r16 closure<0.50", partial="otherwise", thresholds_preregistered=True),
        **design_gate(ranks), replay_gate=replay_gate, snr_policy=old["snr_policy"], runtime=runtime,
        new_model_trained=False, formal_test_used=False, formal_test_sealed=True, all_checks_passed=False,
        completion_state="numeric complete; final audit pending", source_sha256={p: sha(ROOT/p) for p in SOURCES},
        frozen_history_sha256=before, frozen_history_unchanged=True, environment_before=env,
        storage=dict(readonly_cached_validation_trajectories=200, continuous_frames=2000, raw_H_saved=False,
            cache_description="same existing H/bits/symbols/unit noise/clean Y in RAM; noisy Y exactly reconstructed and digest checked"),
        warnings=["Oracle target means/statistics/filters: representation opportunity only, no deployable adapter.",
            "Development validation only; final formal test sealed.", "Ten continuous frames are correlated; independent unit is trajectory.",
            "Source-mean sensitivity is NMSE-only; primary BER is secondary ZF context.",
            "Closure is not clipped and primary r0 uses target means, unlike historical source baseline.",
            "Right subspace comparisons use condition-specific Cholesky-weighted coordinates; angles are descriptive, not basis-reuse proof.",
            "Small singular boundary gaps can make principal subspaces sensitive.",
            "Existing per-symbol frequency-domain abstraction retained; no within-symbol ICI.",
            "Optional Triton flop counter may be unavailable in the frozen CDL environment."])
    save(OUT/"t16a_lowrank_results.json", result)
    print(json.dumps(dict(scientific_flags=design_gate(ranks), r80=r80, r90=dict(r90), runtime=runtime), indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--preflight", action="store_true")
    if p.parse_args().preflight: preflight()
    else: run()
