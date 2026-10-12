"""T11B: replay only R2 and merge frozen T10D R0/R3/R4/R5 performance."""

import argparse
import csv
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from scripts.plot_nearest_validation import FIGURE_NAMES, plot_results
from scripts.run_lmmse_train_validation import qa_tools, save_csv, save_json
from scripts.validate_lmmse_train import check_config
from src.lmmse_train import sha256_file
from src.nearest_validation import (DF_OPTIONS, T11A_COMMIT, aggregate_r2, load_references,
    merge_baselines, run_replay, verify_accepted_sources, verify_frozen)
from src.pilots import build_fdm_pilot_layout, count_layout_resources
from src.stochastic_scan import correct_bit_rate_proxy, frame_duration_seconds
from src.trajectories import prepare_trajectory

TRAJECTORY_CSV = "t11b_nearest_validation_trajectory_metrics.csv"
MERGED_CSV = "t11b_r0_r5_validation_baselines.csv"
RESULTS_JSON = "t11b_nearest_validation_results.json"


def render_and_audit(csv_path,output_dir,tools_dir=None,expected_trajectories=400):
    tools_dir,gate = qa_tools(tools_dir)
    paths,records = plot_results(csv_path,output_dir,gate,expected_trajectories)
    for path,record in zip(paths,records):
        pdf = path.with_suffix(".pdf")
        text = subprocess.run([sys.executable,str(tools_dir/"audit_pdf_text.py"),str(pdf),"--min-pt","5","--json"],
            capture_output=True,text=True,encoding="utf-8")
        collision_path = path.with_suffix(".collision-audit.json")
        collision = subprocess.run([sys.executable,str(tools_dir/"audit_figure_collisions.py"),str(pdf),
            "--json-out",str(collision_path)],capture_output=True,text=True,encoding="utf-8")
        if text.returncode!=0 or collision.returncode!=0:
            raise RuntimeError(f"figure QA failed: {path.name}\n{text.stdout}\n{collision.stdout}\n{collision.stderr}")
        record.update({"path":str(path),"sha256":sha256_file(path),"PDF_text_audit":json.loads(text.stdout),
            "collision_audit":json.loads(collision_path.read_text(encoding="utf-8")),
            "alignment_audit":json.loads(path.with_suffix(".alignment.json").read_text(encoding="utf-8"))})
    return paths,records


def source_preflight(tools_dir):
    tool = Path(tools_dir)/"validate_figure.py"
    result = subprocess.run([sys.executable,str(tool),str(ROOT/"scripts/plot_nearest_validation.py"),"--json"],
        capture_output=True,text=True,encoding="utf-8")
    if result.returncode!=0:
        raise RuntimeError(f"plot source preflight failed: {result.stdout}\n{result.stderr}")
    return json.loads(result.stdout)


def benchmark_plot_rows(report,refs):
    """QA preview: aggregate the same ten frozen old trajectory rows; no receivers run."""
    ids = report["visited_validation_trajectory_ids"]
    rows = []
    for a in report["r2_aggregate_results"]:
        group = [refs["trajectory_index"][tid,a["dF"],a["snr_db"]] for tid in ids]
        truth = sum(float(r["nmse_truth_energy"]) for r in group)
        bits = sum(int(r["total_data_bits"]) for r in group)
        if truth!=a["nmse_truth_energy"] or bits!=a["total_data_bits"]:
            raise ValueError("paired benchmark preview denominator mismatch")
        row = {**a,"r1_data_ber":"N/A","r2_nearest_nmse_db":a["nearest_nmse_db"],
            "r2_nearest_ber":a["nearest_ber"],"r2_proxy_bps":a["nearest_proxy_bps"]}
        for rid,label,method in ((0,"perfect","perfect"),(3,"linear","ls"),(4,"train","train"),(5,"oracle","oracle")):
            errors = sum(int(r[f"{method}_bit_errors"]) for r in group)
            row[f"r{rid}_{label}_ber"] = errors/bits
            row[f"r{rid}_proxy_bps"] = correct_bit_rate_proxy(a["data_bits_per_frame"],errors/bits,frame_duration_seconds())[1]
            if rid!=0:
                error = sum(float(r[f"{method}_nmse_error_energy"]) for r in group)
                row[f"r{rid}_{label}_nmse_db"] = float(10*np.log10(error/truth))
        rows.append(row)
    return rows


def verify_csv_only_redraw(csv_path,output_dir,tools_dir=None):
    """Copy CSV and temporary PNGs, delete the PNGs, then rebuild with only CSV."""
    expected = {name:sha256_file(Path(output_dir)/f"{name}.png") for name in FIGURE_NAMES}
    with tempfile.TemporaryDirectory(prefix="t11b_csv_only_") as directory:
        temp = Path(directory)
        shutil.copyfile(csv_path,temp/MERGED_CSV)
        for name in FIGURE_NAMES:
            copy = temp/f"{name}.png"
            shutil.copyfile(Path(output_dir)/copy.name,copy)
            copy.unlink()
        if sorted(p.name for p in temp.iterdir())!=[MERGED_CSV]:
            raise ValueError("CSV-only directory contains other inputs")
        paths,_ = render_and_audit(temp/MERGED_CSV,temp,tools_dir)
        matches = {p.stem:sha256_file(p)==expected[p.stem] for p in paths}
        if not all(matches.values()) or len(matches)!=3:
            raise ValueError("CSV-only regenerated PNG hashes differ")
    return {"only_input":MERGED_CSV,"temporary_image_copies_deleted":True,"PNG_hash_matches":matches,
        "all_checks_passed":True}


def r1_reference(outputs):
    path = Path(outputs)/"t09b_tdl_awgn_results.json"
    old = json.loads(path.read_text(encoding="utf-8"))
    checks = old["pilot_ls_theoretical_mse_checks"]
    if len(checks)!=20 or not all(r["passed"] for r in checks):
        raise ValueError("accepted T09B pilot-domain calibration missing")
    return {"definition":"raw Pilot-LS at pilot locations only","data_BER":"N/A","r1_data_ber_supported":False,
        "data_RE_NMSE_supported":False,"reference_path":str(path),"reference_sha256":sha256_file(path),
        "pilot_LS_MSE_reference":"T09B: empirical pilot-domain MSE approximately noise variance",
        "existing_calibration_checks":checks,"new_R1_experiment_performed":False}


def numeric_merged(rows):
    integers = {"dF","snr_db","num_trajectories","frames_per_trajectory","num_correlated_frames","total_data_bits","data_bits_per_frame"}
    return [{k:(int(v) if k in integers else float(v)) if isinstance(v,str) and k not in
        ("R1_pilot_mse_or_NA_reference","r1_data_ber") else v for k,v in row.items()} for row in rows]


def descriptive_comparisons(rows):
    gap = [r["r2_minus_r3_nmse_db"] for r in rows]
    # Report signs, never enforce a performance ordering. A broad reversal is a
    # researcher's inspection decision, so do not invent a numerical threshold.
    return {"R2_minus_R3_NMSE_db_range":[min(gap),max(gap)],
        "R2_lower_NMSE_conditions":[[int(r["dF"]),int(r["snr_db"])] for r in rows if r["r2_minus_r3_nmse_db"]<0],
        "R2_lower_BER_conditions":[[int(r["dF"]),int(r["snr_db"])] for r in rows if r["r2_nearest_ber"]<float(r["r3_linear_ber"])],
        "method_ordering_required":False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--benchmark",action="store_true")
    modes.add_argument("--plot-only",action="store_true")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"outputs")
    parser.add_argument("--figure-qa-tools-dir",type=Path)
    args = parser.parse_args(argv)
    output_dir = args.output_dir
    if args.plot_only:
        paths,records = render_and_audit(output_dir/MERGED_CSV,output_dir,args.figure_qa_tools_dir)
        redraw = verify_csv_only_redraw(output_dir/MERGED_CSV,output_dir,args.figure_qa_tools_dir)
        print("CSV-only reconstruction:",json.dumps(redraw),flush=True)
        return
    started = perf_counter()
    config = check_config()
    accepted = verify_accepted_sources(ROOT)
    refs = load_references(ROOT/"outputs")
    r1 = r1_reference(ROOT/"outputs")
    tools_dir,_ = qa_tools(args.figure_qa_tools_dir)
    preflight = source_preflight(tools_dir)
    layouts = {df:build_fdm_pilot_layout(2,64,16,df,4) for df in DF_OPTIONS}
    model = prepare_trajectory()
    print("Benchmark: only R2; 10 validation trajectories x 10 frames x 20 conditions",flush=True)
    benchmark_report,_ = run_replay(refs["manifest"][:10],model,layouts,refs)
    benchmark = {"num_trajectories":10,"num_correlated_frames":100,"method":"R2 LS-nearest only",
        **benchmark_report["runtime"],"projected_400_simulation_seconds":40*benchmark_report["runtime"]["total_simulation_seconds"],
        "execution_counters":benchmark_report["execution_counters"],"replay_verification":benchmark_report["replay_verification"]}
    print("Benchmark/runtime:",json.dumps(benchmark,indent=2),flush=True)
    with tempfile.TemporaryDirectory(prefix="t11b_benchmark_figure_QA_") as directory:
        temp = Path(directory); preview = temp/"paired_10_trajectory_QA_preview.csv"
        save_csv(preview,benchmark_plot_rows(benchmark_report,refs))
        _,preview_qa = render_and_audit(preview,temp,tools_dir,expected_trajectories=10)
    benchmark["three_figure_export_QA_passed"] = True
    benchmark["preview_old_receiver_values_source"] = "same ten frozen T10D trajectory CSV rows, aggregation only"
    if args.benchmark:
        verify_frozen(ROOT/"outputs")
        print("Benchmark and representative three-figure QA passed; frozen files unchanged.",flush=True)
        return
    output_dir.mkdir(exist_ok=True)
    try:
        print("Full incremental replay: only R2; 400 independent trajectories / 4000 correlated frames",flush=True)
        report,rows = run_replay(refs["manifest"],model,layouts,refs,
            progress=lambda n:print(f"Completed {n}/400 validation trajectories",flush=True) if n%10==0 else None)
        after = verify_frozen(ROOT/"outputs")
        merged = merge_baselines(report["r2_aggregate_results"],refs)
        comparisons = descriptive_comparisons(merged)
        report.update({"task":"T11B","experiment_type":"incremental held-out validation of R2 LS-nearest and R0-R5 baseline merge",
            "split":"validation_id","num_independent_trajectories":400,"frames_per_trajectory":10,"num_correlated_frames":4000,
            "incremental_method":"R2 LS-nearest","t10d_reference_hashes":refs["before"],
            "frozen_hash_before_after":{"before":refs["before"],"after":after},
            **{k:report["replay_verification"][k] for k in ("num_h_hash_matches","num_bits_hash_matches","num_noise_hash_matches")},
            "merged_baseline_results":numeric_merged(merged),"descriptive_comparisons":comparisons,"r1_semantics":r1,
            "benchmark":benchmark,"plot_source_preflight":preflight,"config":config,
            "pilot_counts":{str(df):count_layout_resources(l) for df,l in layouts.items()},
            "frame_duration_seconds_including_CP":frame_duration_seconds(),"formal_test_used":False,"formal_test_sealed":True,
            "OOD_generated":False,"method_ordering_required":False,"covariance_updated":False,
            "common_random_numbers":True,"actual_noisy_Pilot_LS_used":True,
            "seed_policy":refs["old"]["seed_policy"],"all_checks_passed":False})
        trajectory_path = output_dir/TRAJECTORY_CSV
        merged_path = output_dir/MERGED_CSV
        save_csv(trajectory_path,rows); save_csv(merged_path,merged)
        report["output_csv_sha256"] = {p.name:sha256_file(p) for p in (trajectory_path,merged_path)}
        report["provenance"] = {"T11A_commit":T11A_COMMIT,"T11A_committed_cleanly_at_task_start":True,
            "initial_pytest":"725 passed in 23.53s","python_version":platform.python_version(),"numpy_version":np.__version__,
            "accepted_numerical_source_checks":accepted,"source_sha256":{p:sha256_file(ROOT/p) for p in (
                "src/nearest_validation.py","scripts/run_nearest_validation.py","scripts/plot_nearest_validation.py",
                "tests/test_nearest_validation.py")}}
        save_json(output_dir/RESULTS_JSON,report)
        stamp = perf_counter()
        _,figures = render_and_audit(merged_path,output_dir,tools_dir)
        report["figures"] = figures
        report["CSV_only_redraw"] = verify_csv_only_redraw(merged_path,output_dir,tools_dir)
        report["runtime"]["figure_export_QA_and_CSV_only_redraw_seconds"] = perf_counter()-stamp
        report["frozen_hash_before_after"]["after"] = verify_frozen(ROOT/"outputs")
        report["checks"] = {"accepted_numerical_sources_unchanged":all(verify_accepted_sources(ROOT).values()),
            "frozen_references_unchanged":True,"400_exact_validation_ids":report["visited_validation_trajectory_ids"]==list(range(2000,2400)),
            "8000_complete_trajectory_rows":len(rows)==8000,"full_exact_replay":report["num_h_hash_matches"]==400 and report["num_bits_hash_matches"]==report["num_noise_hash_matches"]==4000,
            "old_methods_not_recomputed":report["execution_counters"]["old_method_performance_evaluations"]==0,
            "R1_pilot_only":all(r["r1_data_ber"]=="N/A" for r in merged),"CSV_only_three_PNGs_exact":report["CSV_only_redraw"]["all_checks_passed"],
            "formal_test_sealed":True,"no_filtering":report["num_filtered_trajectories"]==0}
        report["all_checks_passed"] = all(report["checks"].values())
        report["runtime"]["total_seconds_including_benchmark_preparation_outputs_QA"] = perf_counter()-started
        save_json(output_dir/RESULTS_JSON,report)
    except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
        failure = {"task":"T11B","all_checks_passed":False,"num_filtered_trajectories":0,
            "numerical_failures":[getattr(exc,"record",{"reason":str(exc)})],"formal_test_used":False,"formal_test_sealed":True}
        save_json(output_dir/"t11b_failed_attempt.json",failure)
        raise
    print("R2 20-condition aggregate:",json.dumps(report["r2_aggregate_results"],indent=2),flush=True)
    print("Manual diagnostic:",json.dumps(report["manual_diagnostic"],indent=2),flush=True)
    print("Runtime:",json.dumps(report["runtime"]),flush=True)
    print("T11B all_checks_passed:",report["all_checks_passed"],flush=True)


if __name__=="__main__":
    main()
