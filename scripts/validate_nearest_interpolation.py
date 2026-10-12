"""T11A: R2 nearest mathematical calibration and ten calibration trajectories."""

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_tdl_noiseless_scan import check_config
from src.estimators import ls_linear_interpolate, pilot_ls_estimate
from src.interpolation import (
    ls_nearest_interpolate, nearest_interpolate_frequency, nearest_interpolate_time,
)
from src.link import (
    PILOT_SYMBOLS, apply_frequency_mimo_channel, assemble_pilot_data_frame,
    demodulate_data_bits, extract_reference_data_bits, generate_master_qpsk,
)
from src.metrics import channel_error_energies, count_bit_errors
from src.pilots import build_fdm_pilot_layout, count_layout_resources, map_pilot_symbols
from src.tdl_noiseless_scan import (
    BITS_SEED_BASE, NoiselessLinkFailure, complex_record, safe_detect,
)
from src.trajectories import CHANNEL_SEED_BASE, prepare_trajectory

DF_OPTIONS = (2,4,8,16)
T10D_COMMIT = "6129743ca9940ad41a5d9de2f70005c9160822c2"
PILOT_LS_TOLERANCE = 1e-12
MACHINE_TOLERANCE = 1e-14
OUTPUT = ROOT / "outputs/t11a_nearest_interpolation_results.json"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sparse_observations(truth, mask):
    """Synthetic calibration inputs only; not another pilot extractor."""
    observed = np.full(truth.shape,np.nan+1j*np.nan,dtype=np.complex128)
    selected = np.broadcast_to(mask[None],truth.shape)
    observed[selected] = truth[selected]
    return observed


def mathematical_sentinels():
    # Explicit hand arrays are independent of the production anchor selector.
    frequency_expected_indices = np.array([0,0,0,4,4,4,4,8,8,8,8,8])
    time_expected_indices = np.array([0,0,0,4,4,4,4,8,8,8,8,12,12,12,12,12])
    mask = np.zeros((1,12,1),dtype=bool)
    mask[0,[0,4,8],0] = True
    observed = np.full((1,1,12,1),np.nan+1j*np.nan,dtype=np.complex128)
    observed[0,0,[0,4,8],0] = [10,20,30]
    frequency,_ = nearest_interpolate_frequency(observed,mask)
    frequency_expected = np.array([10,10,10,20,20,20,20,30,30,30,30,30])
    frequency_manual = {"anchors":[0,4,8],"anchor_values":[10,20,30],
        "expected_anchor_indices":frequency_expected_indices.tolist(),
        "expected_values":frequency_expected.tolist(),"actual_values":frequency[0,0,:,0].real.tolist(),
        "passed":bool(np.array_equal(frequency[0,0,:,0],frequency_expected))}
    mask_left = np.zeros_like(mask); mask_left[0,[2,6,10],0] = True
    left_input = np.full_like(observed,np.nan+1j*np.nan)
    left_input[0,0,[2,6,10],0] = [10,20,30]
    left,_ = nearest_interpolate_frequency(left_input,mask_left)
    frequency_manual["left_edge_shifted_anchors"] = [2,6,10]
    frequency_manual["left_edge_passed"] = bool(np.array_equal(left[0,0,:,0],[10,10,10,10,10,20,20,20,20,30,30,30]))

    time_input = np.full((1,1,1,16),np.nan+1j*np.nan,dtype=np.complex128)
    time_input[..., [0,4,8,12]] = [1,2,3,4]
    time = nearest_interpolate_time(time_input,[0,4,8,12])
    time_expected = np.array([1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,4])
    time_manual = {"anchors":[0,4,8,12],"anchor_values":[1,2,3,4],
        "expected_anchor_indices":time_expected_indices.tolist(),
        "expected_values":time_expected.tolist(),"actual_values":time[0,0,0].real.tolist(),
        "tie_anchor_at_t2_t6_t10":[0,4,8],"trailing_anchor_at_t13_t14_t15":[12,12,12],
        "passed":bool(np.array_equal(time[0,0,0],time_expected))}

    complex_input = observed.copy()
    complex_input[0,0,[0,4,8],0] = [1+2j,-3+0.5j,7-4j]
    complex_frequency,_ = nearest_interpolate_frequency(complex_input,mask)
    complex_expected = np.array([1+2j,1+2j,1+2j,-3+0.5j,-3+0.5j,-3+0.5j,-3+0.5j,7-4j,7-4j,7-4j,7-4j,7-4j])
    time_input[..., [0,4,8,12]] = [1+2j,-3+0.5j,7-4j,-0.25+6j]
    complex_time = nearest_interpolate_time(time_input,[0,4,8,12])
    complex_time_expected = np.array([1+2j,1+2j,1+2j,-3+0.5j,-3+0.5j,-3+0.5j,-3+0.5j,7-4j,7-4j,7-4j,7-4j,-0.25+6j,-0.25+6j,-0.25+6j,-0.25+6j,-0.25+6j])
    complex_sentinel = {"frequency_anchor_values":complex_record(complex_input[0,0,[0,4,8],0]),
        "frequency_expected":complex_record(complex_expected),"frequency_actual":complex_record(complex_frequency[0,0,:,0]),
        "frequency_exact":bool(np.array_equal(complex_frequency[0,0,:,0],complex_expected)),
        "time_exact":bool(np.array_equal(complex_time[0,0,0],complex_time_expected)),
        "real_imag_from_same_anchor":bool(np.array_equal(complex_frequency[0,0,:,0],complex_expected))}

    constants = np.array([[0.8+0.2j,-0.3+0.5j],[0.4-0.1j,0.6+0.7j]])
    truth = np.broadcast_to(constants[:,:,None,None],(2,2,64,16))
    constant_tests = []
    preserved, finite = True, True
    for df in DF_OPTIONS:
        layout = build_fdm_pilot_layout(2,64,16,df,4)
        x = map_pilot_symbols(layout,PILOT_SYMBOLS)
        ls,ls_mask = pilot_ls_estimate(apply_frequency_mimo_channel(truth,x),x,layout.pilot_mask)
        result = ls_nearest_interpolate(ls,ls_mask)
        selected = np.broadcast_to(ls_mask[None],ls.shape)
        error = float(np.max(np.abs(result-truth)))
        exact = bool(np.array_equal(result[selected],ls[selected]))
        full = bool(result.shape==(2,2,64,16) and result.dtype==np.complex128 and np.isfinite(result).all())
        constant_tests.append({"dF":df,"max_absolute_error":error,"tolerance":MACHINE_TOLERANCE,
            "pilot_preservation_exact":exact,"full_grid_finite":full,"passed":bool(error<=MACHINE_TOLERANCE and exact and full)})
        preserved &= exact; finite &= full

    layout = build_fdm_pilot_layout(2,64,16,4,4)
    axis_truth = np.broadcast_to(np.arange(64)[None,None,:,None],(2,2,64,16)).astype(np.complex128)
    edge_result = ls_nearest_interpolate(sparse_observations(axis_truth,layout.pilot_mask),layout.pilot_mask)
    edge = {"TX0_at_k63":float(edge_result[0,0,63,0].real),"expected_TX0_at_k63":60,
        "TX1_at_k0":float(edge_result[0,1,0,0].real),"expected_TX1_at_k0":2,
        "passed":bool(edge_result[0,0,63,0]==60 and edge_result[0,1,0,0]==2)}
    separation_truth = np.broadcast_to(np.array([[1+2j,-30+50j],[4-3j,80-90j]])[:,:,None,None],truth.shape)
    ls = sparse_observations(separation_truth,layout.pilot_mask)
    reference = ls_nearest_interpolate(ls,layout.pilot_mask)
    tx_checks = {}
    for q in range(2):
        changed = ls.copy(); changed[:,q,layout.pilot_mask[q]] += 1000+2000j
        result = ls_nearest_interpolate(changed,layout.pilot_mask)
        tx_checks[f"mutating_TX{q}_does_not_change_TX{1-q}"] = bool(np.array_equal(result[:,1-q],reference[:,1-q]))
        tx_checks[f"mutating_TX{q}_changes_own_output"] = not np.array_equal(result[:,q],reference[:,q])
    tx_checks["deliberate_cross_TX_value_rejected"] = not np.array_equal(reference[:,0,2,0],ls[:,1,2,0])

    a = np.array([[1+2j,-1+0.25j],[0.5-1j,3-2j]])
    b = np.array([[0.125-0.0625j,-0.25+0.125j],[0.375+0.25j,0.0625-0.125j]])
    c = np.array([[0.25+0.125j,0.125-0.25j],[-0.0625+0.375j,-0.125+0.0625j]])
    affine_truth = a[:,:,None,None]+b[:,:,None,None]*np.arange(12)[None,None,:,None]+c[:,:,None,None]*np.arange(16)[None,None,None,:]
    affine_mask = np.zeros((2,12,16),dtype=bool)
    for t in (0,4,8,12):
        affine_mask[0,[0,4,8],t] = True; affine_mask[1,[2,6,10],t] = True
    ls = sparse_observations(affine_truth,affine_mask)
    r2 = ls_nearest_interpolate(ls,affine_mask); r3 = ls_linear_interpolate(ls,affine_mask)
    r3_error, r2_oracle_error = 0.,0.
    for q,anchors in enumerate((frequency_expected_indices,np.array([2,2,2,2,2,6,6,6,6,10,10,10]))):
        expected = a[:,q,None,None]+b[:,q,None,None]*anchors[None,:,None]+c[:,q,None,None]*time_expected_indices[None,None,:]
        r2_oracle_error = max(r2_oracle_error,float(np.max(np.abs(r2[:,q]-expected))))
        interior = np.arange(0,9) if q==0 else np.arange(2,11)
        r3_error = max(r3_error,float(np.max(np.abs(r3[:,q,interior,:13]-affine_truth[:,q,interior,:13]))))
    difference = float(np.max(np.abs(r2-r3)))
    affine = {"definition":"H=a+b*k+c*t; independent explicit nearest anchor arrays",
        "interior_only_for_R3":True,"R3_interior_max_absolute_error":r3_error,
        "R2_hand_oracle_max_absolute_error":r2_oracle_error,"R2_piecewise_constant":bool(r2_oracle_error==0),
        "R2_R3_max_absolute_difference":difference,
        "passed":bool(r3_error<=MACHINE_TOLERANCE and r2_oracle_error==0 and difference>0.1)}
    checks = {"constant_all_dF":all(r["passed"] for r in constant_tests),
        "frequency_manual":frequency_manual["passed"] and frequency_manual["left_edge_passed"],
        "time_manual":time_manual["passed"],"complex":complex_sentinel["frequency_exact"] and complex_sentinel["time_exact"],
        "non_circular_edges":edge["passed"],"tx_separation":all(tx_checks.values()),
        "pilot_preservation":preserved,"full_grid_finite":finite,"affine_distinction":affine["passed"]}
    return {"constant_channel_tests":constant_tests,"frequency_manual_sentinel":frequency_manual,
        "time_manual_sentinel":time_manual,"complex_sentinel":complex_sentinel,
        "frequency_edge_sentinel":edge,"tx_separation":tx_checks,"pilot_preservation":preserved,
        "affine_R2_vs_R3_sentinel":affine,"mathematical_checks":checks}


def estimate_pair(h_ls, pilot_mask):
    """Both estimators receive the identical actual Pilot-LS/mask objects."""
    return ls_nearest_interpolate(h_ls,pilot_mask), ls_linear_interpolate(h_ls,pilot_mask)


def small_tdl_integration(trajectory_ids=tuple(range(10))):
    ids = tuple(trajectory_ids)
    if (not ids or len(ids)>10 or len(set(ids))!=len(ids)
            or any(isinstance(tid,bool) or not isinstance(tid,int) or not 0<=tid<10 for tid in ids)):
        raise ValueError("only calibration trajectory IDs 0..9, at most ten unique IDs, are allowed")
    started = perf_counter()
    model = prepare_trajectory()
    layouts = {df:build_fdm_pilot_layout(2,64,16,df,4) for df in DF_OPTIONS}
    totals = {df:{"dF":df,"pilot_ratio":count_layout_resources(layout)["union_ratio"],
        "frames_processed":0,"total_data_bits":0,"nmse_truth_energy":0.,"r2_error_energy":0.,"r3_error_energy":0.,
        "perfect_bit_errors":0,"r2_bit_errors":0,"r3_bit_errors":0,"pilot_LS_max_absolute_error":0.,
        "pilot_preservation_exact":True,"full_grid_finite":True,
        "perfect_ZF_successes":0,"r2_ZF_successes":0,"r3_ZF_successes":0} for df,layout in layouts.items()}
    visited, seeds, frame_records = [],[],[]
    for tid in ids:
        trajectory = model.generate(tid)
        if (trajectory.channel_seed!=CHANNEL_SEED_BASE+tid or trajectory.h_long.shape!=(2,2,64,160)
                or trajectory.h_frames.shape!=(10,2,2,64,16) or not np.shares_memory(trajectory.h_long,trajectory.h_frames)):
            raise ValueError("calibration trajectory must have ten continuous frame views and its calibration seed")
        visited.append(tid); seeds.append(trajectory.channel_seed)
        for frame,h_true in enumerate(trajectory.h_frames):
            bits_seeds = tuple(BITS_SEED_BASE+100*tid+2*frame+q for q in range(2))
            bits,symbols = generate_master_qpsk(bits_seeds=bits_seeds)
            frame_records.append({"trajectory_id":tid,"frame_index":frame,"bits_seeds":list(bits_seeds)})
            for df,layout in layouts.items():
                x = assemble_pilot_data_frame(symbols,layout)
                y = apply_frequency_mimo_channel(h_true,x)  # Noise OFF; no noise draw or scaling.
                known_pilots = map_pilot_symbols(layout,PILOT_SYMBOLS)
                h_ls,mask = pilot_ls_estimate(y,known_pilots,layout.pilot_mask)
                r2,r3 = estimate_pair(h_ls,mask)
                selected = np.broadcast_to(mask[None],h_ls.shape)
                metric = totals[df]
                metric["pilot_LS_max_absolute_error"] = max(metric["pilot_LS_max_absolute_error"],float(np.max(np.abs(h_ls[selected]-h_true[selected]))))
                metric["pilot_preservation_exact"] &= bool(np.array_equal(r2[selected],h_ls[selected]))
                metric["full_grid_finite"] &= bool(r2.dtype==np.complex128 and r2.shape==h_true.shape and np.isfinite(r2).all())
                reference = extract_reference_data_bits(bits,layout.data_mask)
                expected_bits = sum(stream.size for stream in reference)
                for name,h_used in (("perfect",h_true),("r2",r2),("r3",r3)):
                    try:
                        detected = safe_detect(y,h_used,layout,h_true,x,tid,df,name)
                    except NoiselessLinkFailure as exc:
                        exc.record.update({"frame_index":frame,"bits_seeds":list(bits_seeds)})
                        raise
                    errors,nbits = count_bit_errors(demodulate_data_bits(np.sqrt(2)*detected,layout.data_mask),reference)
                    if nbits!=expected_bits or nbits!=2*np.count_nonzero(layout.data_mask):
                        raise ValueError("receiver bit denominators differ from the actual data mask")
                    metric[f"{name}_bit_errors"] += errors; metric[f"{name}_ZF_successes"] += 1
                    if name=="perfect": metric["total_data_bits"] += nbits
                e2,truth_energy = channel_error_energies(r2,h_true,layout.data_mask)
                e3,truth_energy3 = channel_error_energies(r3,h_true,layout.data_mask)
                if truth_energy!=truth_energy3:
                    raise ValueError("R2/R3 data energy denominators differ")
                metric["r2_error_energy"] += e2; metric["r3_error_energy"] += e3
                metric["nmse_truth_energy"] += truth_energy; metric["frames_processed"] += 1
    for metric in totals.values():
        for name in ("r2","r3"):
            nmse = metric[f"{name}_error_energy"]/metric["nmse_truth_energy"]
            metric[f"{name}_data_nmse_linear"] = nmse
            metric[f"{name}_data_nmse_db"] = float(10*np.log10(nmse)) if nmse>0 else None
    expected_frames = 10*len(ids)
    checks = {"all_requested_trajectories_retained":visited==list(ids),"ten_continuous_frames_each":len(frame_records)==expected_frames,
        "Pilot_LS_exact_to_machine_precision":all(m["pilot_LS_max_absolute_error"]<=PILOT_LS_TOLERANCE for m in totals.values()),
        "R2_full_grid_finite":all(m["full_grid_finite"] for m in totals.values()),
        "R2_pilots_preserved_exactly":all(m["pilot_preservation_exact"] for m in totals.values()),
        "Perfect_CSI_zero_BER":all(m["perfect_bit_errors"]==0 for m in totals.values()),
        "all_ZF_paths_succeeded":all(m[f"{name}_ZF_successes"]==expected_frames for m in totals.values() for name in ("perfect","r2","r3"))}
    return {"purpose":"diagnostic / small-sample integration values; no performance ordering conclusion",
        "num_independent_calibration_trajectories":len(ids),"frames_per_trajectory":10,
        "num_correlated_calibration_frames":expected_frames,"trajectory_ids":visited,"channel_seeds":seeds,
        "channel_seed_rule":"40261001 + trajectory_id","bits_seed_rule":"20261001 + 100*trajectory_id + 2*frame_index + tx",
        "noise":"OFF","snr_scan_performed":False,"formal_manifest_used":False,
        "physical_model":{"channel":"TDL-A","RMS_DS_ns":100,"speed_kmh":30,"carrier_frequency_hz":3.5e9,
            "symbol_wise_quasi_static":True,"ICI":False},
        "frame_seed_records":frame_records,"per_dF":list(totals.values()),
        "num_filtered_trajectories":0,"numerical_failures":[],"checks":checks,"all_checks_passed":all(checks.values()),
        "runtime_seconds":perf_counter()-started}


def accepted_provenance():
    from src.provenance import recorded_source_matches
    old = json.loads((ROOT/"outputs/t10d_lmmse_train_validation_results.json").read_text(encoding="utf-8"))
    sources = old["provenance"]["source_sha256"]
    checks = {path:recorded_source_matches(ROOT,path,digest,"t10d",current_hash=sha256_file)
              for path,digest in sources.items()}
    checks.update({"outputs/"+name:sha256_file(ROOT/"outputs"/name)==digest for name,digest in old["output_csv_sha256"].items()})
    if not old["all_checks_passed"] or not all(checks.values()):
        raise ValueError("accepted T10D runtime/regression sources or data changed")
    message = subprocess.check_output(["git","show","-s","--format=%s",T10D_COMMIT],cwd=ROOT,text=True).strip()
    if message!="feat: add validated held-out LMMSE-Train full-link integration":
        raise ValueError("accepted standalone T10D commit is missing")
    frozen = {"manifest_sha256":sha256_file(ROOT/"outputs/t10c_train_validation_manifest.csv"),
        "covariance_file_sha256":sha256_file(ROOT/"outputs/t10c_lmmse_train_covariance.npz")}
    if frozen!={key:old["covariance_hash_before_after"]["after"][key] for key in frozen}:
        raise ValueError("frozen T10C artifacts changed")
    return checks,frozen


def main():
    started = perf_counter()
    config = check_config()
    accepted,before = accepted_provenance()
    try:
        report = mathematical_sentinels()
        if not all(report["mathematical_checks"].values()):
            raise ValueError("mathematical calibration failed before realistic integration")
        integration = small_tdl_integration()
    except (ValueError,np.linalg.LinAlgError) as exc:
        OUTPUT.parent.mkdir(exist_ok=True)
        failure = {"task":"T11A","all_checks_passed":False,"num_filtered_trajectories":0,
            "failure":getattr(exc,"record",{"reason":str(exc)}),"formal_test_used":False,"formal_test_sealed":True}
        OUTPUT.write_text(json.dumps(failure,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        raise
    accepted_after,after = accepted_provenance()
    report.update({"task":"T11A","experiment_type":"R2 LS-nearest interpolation mathematical and integration calibration",
        "nearest_definition":"separable 2D: actual Pilot-LS -> frequency nearest -> time nearest",
        "frequency_tie_rule":"equal index distances choose lower physical frequency index",
        "time_tie_rule":"equal index distances choose earlier time anchor",
        "edge_rule":"hold first/last anchor on the linear physical grid; no circular wrapping",
        "frequency_wraparound":False,"estimator_shape":[2,2,64,16],"estimator_dtype":"complex128",
        "frame_block_estimator":True,"causal_online_tracking_claimed":False,
        "small_tdl_integration":integration,"config":config,
        "r1_semantics":{"definition":"raw Pilot-LS only: H_LS(kp,tp)=Yp/Xp","domain":"actual pilot observations only",
            "pilot_MSE_NMSE_supported":True,"data_RE_channel_estimate":False,"data_BER":"N/A",
            "implicit_filling_allowed":False},
        "baseline_registry":{"path":"docs/baseline_registry.md","sha256":sha256_file(ROOT/"docs/baseline_registry.md")},
        "frozen_artifact_hash_before_after":{"before":before,"after":after},
        "formal_validation_scan_performed":False,"formal_test_used":False,"formal_test_sealed":True,
        "OOD_generated":False,"performance_figures_generated":False,"performance_ordering_required":False,
        "num_filtered_trajectories":0,"numerical_failures":[],
        "checks":{**report["mathematical_checks"],**integration["checks"],"accepted_R3_and_other_sources_unchanged":all(accepted_after.values()),
            "frozen_covariance_unchanged":before==after,"ten_calibration_trajectories":integration["trajectory_ids"]==list(range(10))}})
    new_sources = ("src/interpolation.py","scripts/validate_nearest_interpolation.py","tests/test_nearest_interpolation.py",
        "docs/baseline_registry.md","configs/base.yaml")
    report["provenance"] = {"T10D_commit":T10D_COMMIT,"T10D_committed_cleanly_at_task_start":True,
        "git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "python_version":platform.python_version(),"numpy_version":np.__version__,
        "accepted_source_and_data_checks":accepted,
        "source_sha256":{path:sha256_file(ROOT/path) for path in new_sources}}
    report["all_checks_passed"] = all(report["checks"].values())
    report["runtime_seconds"] = perf_counter()-started
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("T11A: R2 nearest mathematical and small-sample integration calibration")
    for key in ("constant_channel_tests","frequency_manual_sentinel","time_manual_sentinel","complex_sentinel",
                "frequency_edge_sentinel","tx_separation","affine_R2_vs_R3_sentinel"):
        print(key+":",json.dumps(report[key]))
    print("Small calibration: 10 independent trajectories / 100 correlated frames; noise OFF")
    for row in integration["per_dF"]:
        print("diagnostic / small-sample integration values:",json.dumps(row))
    print("All checks:",report["all_checks_passed"])
    print("Runtime seconds:",report["runtime_seconds"])
    print("JSON saved:",OUTPUT)
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
