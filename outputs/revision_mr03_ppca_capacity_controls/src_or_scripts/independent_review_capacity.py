"""Independent read-only review of persisted MR03 small energy statistics.

No original inference, random generator, analyzer, model fit, or formal outcomes
are imported. Bootstrap multiplicity matrices independently reproduce the exact
persisted draws. This script creates no new draws and never alters input files.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import zipfile

import numpy as np

ATOL = 1e-10
RTOL = 1e-10
EXPECTED_CONDITIONS = ("ID", "Profile", "Mobility", "Composed", "Boundary")
EXPECTED_SNRS = (10, 20, 30)
EXPECTED_METHODS = (
    "SOURCE", "POOLED_NATURAL", "DENSE_TW5", "LR4_TW5",
    "WHOLE_POOL_M2_R23", "WHOLE_POOL_M2_R24", "DENSE_TW1", "LR4_TW1",
    "PPCA_DIRECT_R16", "PPCA_DIRECT_R32", "PPCA_DIRECT_R64",
)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def assert_close(actual, expected, label, report=None):
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("NONFINITE_OR_SHAPE:" + label)
    error = float(np.max(np.abs(a-b), initial=0.))
    scale = max(float(np.max(np.abs(a), initial=0.)),
                float(np.max(np.abs(b), initial=0.)))
    if error > ATOL + RTOL * scale:
        raise ValueError(f"INDEPENDENT_MISMATCH:{label}:{error}:{scale}")
    if report is not None:
        report["comparison_count"] += 1
        report["max_scaled_error"] = max(report["max_scaled_error"], error/(1.+scale))
    return error


def cell_statistics(frame_error, frame_truth, draws):
    """Use counts of each trajectory in a draw, not original advanced indexing."""
    fe = np.asarray(frame_error, dtype=np.float64)
    fq = np.asarray(frame_truth, dtype=np.float64)
    draws = np.asarray(draws)
    if fe.ndim != 4 or fq.shape != (fe.shape[0], fe.shape[-1]):
        raise ValueError("ENERGY_AXES")
    n, snrs, methods, scored_frames = fe.shape
    if draws.ndim != 2 or draws.shape[1] != n or draws.dtype != np.int64:
        raise ValueError("DRAW_AXES")
    if not len(draws) or draws.min() < 0 or draws.max() >= n:
        raise ValueError("DRAW_RANGE")
    if not np.isfinite(fe).all() or not np.isfinite(fq).all() or np.any(fe < 0) or np.any(fq <= 0):
        raise ValueError("ENERGY_FINITE_POSITIVE")
    trajectory_error = fe.sum(axis=-1)
    trajectory_truth = fq.sum(axis=-1)
    numerator = fe.sum(axis=(0, 3))
    denominator = float(fq.sum())
    linear = numerator / denominator
    counts = np.stack([np.bincount(row, minlength=n) for row in draws])
    if not np.all(counts.sum(axis=1) == n):
        raise ValueError("DRAW_MULTIPLICITY")
    boot_num = counts @ trajectory_error.reshape(n, snrs*methods)
    boot_den = counts @ trajectory_truth
    boot_linear = (boot_num / boot_den[:, None]).reshape(len(draws), snrs, methods)
    if not np.isfinite(linear).all() or not np.isfinite(boot_linear).all() or np.any(linear <= 0) or np.any(boot_linear <= 0):
        raise ValueError("NMSE_FINITE_POSITIVE")
    return dict(error=trajectory_error, truth=trajectory_truth,
                numerator=numerator, denominator=denominator,
                nmse_linear=linear, nmse_db=10*np.log10(linear),
                bootstrap_linear=boot_linear, bootstrap_db=10*np.log10(boot_linear))


def linear_percentiles(values):
    """Independent explicit linear-order-statistic interpolation for 95% CI."""
    a = np.sort(np.asarray(values, dtype=np.float64), axis=0)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError("PERCENTILE_INPUT")
    answer = []
    for p in (.025, .975):
        q = (len(a)-1)*p
        lo, hi = math.floor(q), math.ceil(q)
        answer.append(a[lo] + (q-lo)*(a[hi]-a[lo]))
    return np.stack(answer)


def expected_contrasts(methods):
    result = []
    if "MFA_K1_R16_EM_FROZEN" in methods:
        result.append(("EM16_minus_Direct16", "MFA_K1_R16_EM_FROZEN", "PPCA_DIRECT_R16"))
    result.extend([
        ("Direct16_minus_Direct32", "PPCA_DIRECT_R16", "PPCA_DIRECT_R32"),
        ("Direct32_minus_Direct64", "PPCA_DIRECT_R32", "PPCA_DIRECT_R64"),
    ])
    for rank in (16, 32, 64):
        second = f"PPCA_DIRECT_R{rank}"
        for first in ("POOLED_NATURAL", "LR4_TW1", "LR4_TW5", "DENSE_TW5", "WHOLE_POOL_M2_R23", "WHOLE_POOL_M2_R24"):
            result.append((first+"_minus_"+second, first, second))
    result.extend([
        ("LR4_TW5_minus_LR4_TW1", "LR4_TW5", "LR4_TW1"),
        ("DENSE_TW5_minus_DENSE_TW1", "DENSE_TW5", "DENSE_TW1"),
    ])
    return result


def csv_by(path, fields):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keyed = {tuple(row[k] for k in fields): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError("DUPLICATE_CSV_IDENTITY:"+str(path))
    return keyed


def inside(root, relative):
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


def check_interval(row, report):
    if int(row["bootstrap"]) != 2000 or int(row["trajectory_count"]) != 100:
        raise ValueError("STATISTICAL_UNIT_OR_RESAMPLES")
    text = row["interval"].lower()
    if not all(t in text for t in ("unadjusted", "exploratory", "95%", "trajectory", "not confirmatory")):
        raise ValueError("INTERVAL_SCOPE")


def dataset_review(out, cfg, dataset, mr01_bootstrap=None):
    methods, conditions, snrs = cfg["methods"], cfg["conditions"], cfg["snrs"]
    report = dict(dataset=dataset, PASS=False, comparison_count=0, max_scaled_error=0.,
                  raw_NPZ_checked=0, raw_JSON_checked=0, raw_CRC_checked=0,
                  bootstrap_resamples=2000, new_random_draws=0)
    receipt = read_json(out/f"reports/{dataset}_receipt.json")
    if receipt.get("PASS") is not True or receipt["physical_trajectories"] != 500 or not receipt["raw_saved_before_statistics"]:
        raise ValueError("INCOMPLETE_RAW_RECEIPT")
    receipts = receipt["receipts"]
    mapping = {(r["condition"], int(r["trajectory_id"])): r for r in receipts}
    if len(receipts) != 500 or len(mapping) != 500 or set(mapping) != {(c,i) for c in conditions for i in range(100)}:
        raise ValueError("RAW_IDENTITY_GRID")
    frame_error = np.empty((5,100,3,len(methods),6), dtype=np.float64)
    frame_truth = np.empty((5,100,6), dtype=np.float64)
    for ci,c in enumerate(conditions):
        for i in range(100):
            row = mapping[c,i]
            p = inside(out,row["path"])
            jp = p.with_suffix(".json")
            if digest(p) != row["sha256"] or digest(jp) != row["json_sha256"]:
                raise ValueError("RAW_HASH")
            meta = read_json(jp)
            if (meta["dataset"],meta["condition"],int(meta["trajectory_id"]),meta["complete"]) != (dataset,c,i,True):
                raise ValueError("RAW_PAIRING")
            if meta["methods"] != methods:
                raise ValueError("JSON_METHOD_AXES")
            with zipfile.ZipFile(p) as handle:
                if handle.testzip() is not None:
                    raise ValueError("RAW_CRC")
            with np.load(p,allow_pickle=False) as raw:
                if list(raw["method_names"]) != methods or list(raw["snr_db"]) != snrs or list(raw["frame_ids"]) != list(range(4,10)):
                    raise ValueError("RAW_AXES")
                if raw["frame_error"].shape != (3,len(methods),6) or raw["frame_truth"].shape != (6,):
                    raise ValueError("RAW_ENERGY_SHAPE")
                frame_error[ci,i] = raw["frame_error"]
                frame_truth[ci,i] = raw["frame_truth"]
                if np.any(raw["fit_failed"]):
                    raise ValueError("FAILED_FIT")
                for name in ("coordinates_tw1","coordinates_tw5","gram_tw1","gram_tw5","rhs_tw1","rhs_tw5","normal_residual_tw1","normal_residual_tw5"):
                    if not np.isfinite(raw[name]).all():
                        raise ValueError("NONFINITE_SOLVE:"+name)
            for key in ("raw_NPZ_checked","raw_JSON_checked","raw_CRC_checked"):
                report[key] += 1
    with np.load(out/f"data/bootstrap_{dataset}.npz",allow_pickle=False) as handle:
        draws = {c:handle[c] for c in conditions}
        if set(handle.files) != set(conditions) or any(v.shape != (2000,100) for v in draws.values()):
            raise ValueError("DRAW_GRID")
    if dataset == "D_eval":
        original_key="outputs/revision_mr01_pooled_lmmse/data/exploratory_bootstrap_indices.npz"
        # Portable review accepts only an exact authenticated original copy.
        # It never treats absent old files as permission to skip this check.
        expected=read_json(out/"reports/preservation_before.json")["hashes"][original_key]
        old=Path(mr01_bootstrap) if mr01_bootstrap is not None else out.parents[1]/original_key
        if digest(old) != expected:
            raise ValueError("D_ORIGINAL_BOOTSTRAP_HASH_DRIFT")
        with np.load(old,allow_pickle=False) as original:
            if set(original.files) != set(conditions) or any(not np.array_equal(original[c],draws[c]) for c in conditions):
                raise ValueError("D_ORIGINAL_DRAWS_DRIFT")
        report["original_MR01_bootstrap_SHA256_verified"]=True
        report["original_MR01_bootstrap_copy_override"]=mr01_bootstrap is not None
        report["original_MR01_bootstrap_sha256"]=expected
    cells = [cell_statistics(frame_error[j],frame_truth[j],draws[c]) for j,c in enumerate(conditions)]
    with np.load(out/f"data/analysis_{dataset}.npz",allow_pickle=False) as analysis:
        for key, expected in (("frame_error",frame_error),("frame_truth",frame_truth),("error",np.stack([z["error"] for z in cells])),("truth",np.stack([z["truth"] for z in cells]))):
            assert_close(analysis[key],expected,dataset+":"+key,report)
        if list(analysis["methods"]) != methods or list(analysis["conditions"]) != conditions or list(analysis["snrs"]) != snrs:
            raise ValueError("ANALYSIS_AXES")
    definitions = expected_contrasts(methods)
    bs = np.stack([z["bootstrap_linear"] for z in cells])
    bd = np.stack([z["bootstrap_db"] for z in cells])
    cd = np.stack([bd[:,:,:,methods.index(a)]-bd[:,:,:,methods.index(b)] for _,a,b in definitions],axis=-1)
    with np.load(out/f"data/bootstrap_statistics_{dataset}.npz",allow_pickle=False) as saved:
        for key,expected in (("nmse_linear",bs),("nmse_db",bd),("contrast_db",cd)):
            assert_close(saved[key],expected,dataset+":"+key,report)
        for key,expected in (("contrast_names",[d[0] for d in definitions]),("contrast_first",[d[1] for d in definitions]),("contrast_second",[d[2] for d in definitions]),("methods",methods),("conditions",conditions),("snrs",snrs)):
            if list(saved[key]) != expected:
                raise ValueError("BOOTSTRAP_STATISTIC_AXES:"+key)
    nmse = csv_by(out/f"tables/nmse_{dataset}.csv",("condition","snr_db","method"))
    contrasts = csv_by(out/f"tables/contrasts_{dataset}.csv",("condition","snr_db","contrast"))
    if len(nmse) != 15*len(methods) or len(contrasts) != 15*len(definitions):
        raise ValueError("CSV_GRID")
    for ci,c in enumerate(conditions):
        z = cells[ci]
        lim = linear_percentiles(z["bootstrap_db"])
        for si,s in enumerate(snrs):
            for mi,method in enumerate(methods):
                row=nmse[c,str(s),method];check_interval(row,report)
                for field,value in (("error_energy",z["numerator"][si,mi]),("truth_energy",z["denominator"]),("nmse_linear",z["nmse_linear"][si,mi]),("nmse_db",z["nmse_db"][si,mi]),("CI_lower",lim[0,si,mi]),("CI_upper",lim[1,si,mi])):
                    assert_close(float(row[field]),value,dataset+":"+field,report)
                if int(row["scored_frames"]) != 600:
                    raise ValueError("CELL_SCORE_COUNT")
            for di,(name,first,second) in enumerate(definitions):
                row=contrasts[c,str(s),name];check_interval(row,report)
                if row["first"] != first or row["second"] != second or "positive favors second" not in row["sign"]:
                    raise ValueError("CONTRAST_DIRECTION")
                ai,bi=methods.index(first),methods.index(second)
                bounds=linear_percentiles(cd[ci,:,si,di])
                for field,value in (("estimate_db",z["nmse_db"][si,ai]-z["nmse_db"][si,bi]),("CI_lower",bounds[0]),("CI_upper",bounds[1]),("first_worse_trajectory_fraction",np.mean(z["error"][:,si,ai]>z["error"][:,si,bi]))):
                    assert_close(float(row[field]),value,dataset+":"+field,report)
    if dataset == "V_new":
        macro_rows=csv_by(out/"tables/validation_capacity_macros.csv",("scope","rank"))
        recommendation=read_json(out/"reports/validation_recommendations.json")
        if len(macro_rows) != 6 or recommendation.get("PASS") is not True or not recommendation["fixed_scopes_before_V"] or recommendation["final_deployment_selected"]:
            raise ValueError("VALIDATION_SCOPE")
        for scope,subset in (("origin-only",cfg["origin_conditions"]),("expanded",conditions)):
            cis=[conditions.index(c) for c in subset]
            for rank in (16,32,64):
                mi=methods.index(f"PPCA_DIRECT_R{rank}")
                point=float(sum(float(cells[j]["nmse_linear"][s,mi]) for j in cis for s in range(3))/(3*len(cis)))
                # Do not combine physical trajectories across different conditions.
                macro_boot=sum(bs[j,:,s,mi] for j in cis for s in range(3))/(3*len(cis))
                bounds=linear_percentiles(macro_boot)
                row=macro_rows[scope,str(rank)]
                for field,value in (("macro_linear_pooled_nmse",point),("db_of_macro_linear",10*np.log10(point)),("CI_lower_linear",bounds[0]),("CI_upper_linear",bounds[1])):
                    assert_close(float(row[field]),value,"V:"+scope+":"+field,report)
                if int(row["cells"]) != 3*len(cis) or int(row["physical_trajectories"]) != 100*len(cis) or int(row["bootstrap"]) != 2000:
                    raise ValueError("MACRO_UNITS")
            # Use exact persisted macro floats for the exact-equality rule; independent
            # resummation above checks them numerically without changing tie semantics.
            candidates=[macro_rows[scope,str(r)] for r in (16,32,64)]
            chosen=min(candidates,key=lambda row:(float(row["macro_linear_pooled_nmse"]),int(row["rank"])))
            rec=recommendation["recommendations"][scope]
            if int(rec["rank"]) != int(chosen["rank"]):
                raise ValueError("RECOMMENDATION_RULE")
            assert_close(float(rec["macro_linear_pooled_nmse"]),float(chosen["macro_linear_pooled_nmse"]),scope+":recommendation",report)
        report["validation_macro_rows_checked"]=6
        report["recommendation_scopes_checked"]=2
    report.update(PASS=True,methods=len(methods),condition_SNR_cells=15,
                  physical_trajectories=500,scored_positions_per_method=9000,
                  nmse_rows_checked=len(nmse),contrast_rows_checked=len(contrasts),
                  interval="Unadjusted exploratory 95% trajectory-paired percentile; not confirmatory")
    return report


def review(output_root,datasets=("V_new","D_eval"),mr01_bootstrap=None):
    out=Path(output_root).resolve()
    if not set(datasets).issubset({"V_new","D_eval"}) or not datasets:
        raise ValueError("DIAGNOSTIC_DATASETS_ONLY")
    freeze=read_json(out/"reports/execution_freeze.json")
    for name,expected in freeze["hashes"].items():
        if digest(inside(out,name)) != expected:
            raise ValueError("FROZEN_SOURCE_OR_INPUT_CHANGED:"+name)
    cfg=read_json(out/"config.json")
    if tuple(cfg["conditions"]) != EXPECTED_CONDITIONS or tuple(cfg["snrs"]) != EXPECTED_SNRS or tuple(cfg["methods"]) != EXPECTED_METHODS:
        raise ValueError("UNEXPECTED_FROZEN_GRID")
    if cfg["origin_conditions"] != ["ID","Profile","Mobility"] or cfg["per_condition"] != 100:
        raise ValueError("UNEXPECTED_SCOPE_OR_N")
    answer=dict(PASS=True,independent_code_sha256=digest(__file__),new_random_draws=0,
                inference_executed=False,formal_results_read=False,input_files_modified=0,
                tolerance=dict(absolute=ATOL,relative=RTOL),datasets=[])
    for dataset in datasets:
        answer["datasets"].append(dataset_review(out,cfg,dataset,mr01_bootstrap=mr01_bootstrap))
    if "D_eval" in datasets:
        binding=read_json(out/"reports/D_eval_validation_binding.json")
        if binding["validation_recommendations_sha256"] != digest(out/"reports/validation_recommendations.json") or not binding["all_predeclared_ranks_evaluated"] or not binding["no_D_eval_selection"]:
            raise ValueError("VALIDATION_BEFORE_D_BINDING")
        answer["validation_before_D_binding_checked"]=True
    return answer


def tiny_toy_tests():
    # Deterministic literal arrays; no new scientific channels or random draws.
    e=np.arange(1.,49.).reshape(4,2,3,2)
    q=np.array([[1.,2.],[3.,4.],[5.,6.],[7.,8.]])
    draws=np.array([[0,1,2,3],[3,3,0,1],[2,2,2,2],[1,0,1,0]],dtype=np.int64)
    actual=cell_statistics(e,q,draws)
    for s in range(2):
        for m in range(3):
            literal=sum(float(e[t,s,m,f]) for t in range(4) for f in range(2))/sum(float(x) for x in q.ravel())
            assert_close(actual["nmse_linear"][s,m],literal,"toy_point")
            for b,row in enumerate(draws):
                reference=sum(float(e[t,s,m,f]) for t in row for f in range(2))/sum(float(q[t,f]) for t in row for f in range(2))
                assert_close(actual["bootstrap_linear"][b,s,m],reference,"toy_count_vs_literal")
    for values in (np.array([9.,1.,7.,4.]),np.arange(2000.,dtype=float)):
        assert_close(linear_percentiles(values),np.quantile(values,[.025,.975],method="linear"),"toy_percentile")
    # Exact ties preserve lower rank, unlike an implicit preference for larger rank.
    assert min([(1.,64),(1.,16),(1.,32)],key=lambda row:(row[0],row[1])) == (1.,16)
    bad=q.copy();bad[0,0]=np.nan
    try:cell_statistics(e,bad,draws)
    except ValueError:pass
    else:raise AssertionError("nonfinite truth was accepted")
    bad_draw=draws.copy();bad_draw[0,0]=4
    try:cell_statistics(e,q,bad_draw)
    except ValueError:pass
    else:raise AssertionError("out-of-range identity was accepted")
    return dict(PASS=True,test_groups=6,new_random_draws=0,real_data_read=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path)
    parser.add_argument("--dataset",choices=("all","V_new","D_eval"),default="all")
    parser.add_argument("--report",type=Path)
    parser.add_argument("--mr01-bootstrap",type=Path,
                        help="Exact original MR01 draw NPZ copy; SHA must match preservation manifest")
    parser.add_argument("--toy",action="store_true")
    args=parser.parse_args()
    if args.toy:
        answer=tiny_toy_tests()
    else:
        if args.out is None:parser.error("--out is required for persisted-data review")
        datasets=("V_new","D_eval") if args.dataset=="all" else (args.dataset,)
        answer=review(args.out,datasets,mr01_bootstrap=args.mr01_bootstrap)
    text=json.dumps(answer,indent=2,allow_nan=False)
    if args.report:
        # Never overwrite a report or write into raw/data locations.
        if args.out and args.report.resolve().is_relative_to((args.out/"data").resolve()):
            raise ValueError("REPORT_MUST_NOT_MODIFY_DATA")
        with args.report.open("x",encoding="utf8") as handle:handle.write(text+"\n")
    print(text)


if __name__=="__main__":
    main()
