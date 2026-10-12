"""T10D: 400 held-out trajectories, ten continuous frames, four paired receivers."""

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from scripts.plot_lmmse_train_validation import FIGURE_NAMES,plot_results
from scripts.validate_lmmse_train import check_config
from src.lmmse_train import sha256_file
from src.lmmse_train_validation import (T10C_COMMIT,DF_OPTIONS,THEORY_TOLERANCE_DB,
    frozen_hashes,load_frozen_inputs,prepare_pools,run_validation)
from src.pilots import build_fdm_pilot_layout,count_layout_resources
from src.trajectories import prepare_trajectory
from src.provenance import recorded_source_matches

PREFIX = "t10d_lmmse_train_validation"


def save_csv(path,rows):
    with Path(path).open("w",encoding="utf-8",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def save_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def qa_tools(tools_dir):
    if tools_dir is None:
        candidate = Path.home()/".codex/skills/nature-figure/scripts"
        tools_dir = Path(os.environ.get("NATURE_FIGURE_QA_TOOLS_DIR",candidate))
    tools_dir = Path(tools_dir)
    needed = ("audit_panel_alignment.py","audit_pdf_text.py","audit_figure_collisions.py")
    if not all((tools_dir/name).is_file() for name in needed):
        raise ValueError("figure QA tools missing; supply --figure-qa-tools-dir or NATURE_FIGURE_QA_TOOLS_DIR")
    sys.path.insert(0,str(tools_dir))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    return tools_dir,require_matplotlib_panel_alignment


def render_and_audit(csv_path,output_dir,tools_dir=None):
    tools_dir,gate = qa_tools(tools_dir)
    paths,records = plot_results(csv_path,output_dir,gate)
    for path,record in zip(paths,records):
        pdf = path.with_suffix(".pdf")
        text = subprocess.run([sys.executable,str(tools_dir/"audit_pdf_text.py"),str(pdf),"--min-pt","5","--json"],capture_output=True,text=True,encoding="utf-8")
        collision_path = path.with_suffix(".collision-audit.json")
        collision = subprocess.run([sys.executable,str(tools_dir/"audit_figure_collisions.py"),str(pdf),
            "--json-out",str(collision_path)],capture_output=True,text=True,encoding="utf-8")
        if text.returncode!=0 or collision.returncode!=0:
            raise RuntimeError(f"rendered figure QA failed: {path.name}\n{text.stdout}\n{collision.stdout}\n{collision.stderr}")
        record.update({"path":str(path),"sha256":sha256_file(path),"PDF_text_audit":json.loads(text.stdout),
            "collision_audit":json.loads(collision_path.read_text(encoding="utf-8")),
            "alignment_audit":json.loads(path.with_suffix(".alignment.json").read_text(encoding="utf-8"))})
    return paths,records


def source_checks():
    old = json.loads((ROOT/"outputs/t10c_lmmse_train_results.json").read_text(encoding="utf-8"))
    checks = {p:recorded_source_matches(ROOT,p,h,"t10c",current_hash=sha256_file)
              for p,h in old["provenance"]["source_sha256"].items()}
    if not all(checks.values()):
        raise ValueError("accepted T10C source/config/protocol changed")
    subprocess.run(["git","cat-file","-e",f"{T10C_COMMIT}^{{commit}}"],cwd=ROOT,check=True,capture_output=True)
    message = subprocess.check_output(["git","show","-s","--format=%s",T10C_COMMIT],cwd=ROOT,text=True).strip()
    if message!="feat: add frozen training-only LMMSE covariance":
        raise ValueError("accepted T10C standalone commit is missing")
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--benchmark",action="store_true",help="10 validation trajectories, all ten frames/20 conditions; no output files")
    modes.add_argument("--plot-only",action="store_true",help="rebuild all four figures from aggregate CSV alone")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"outputs")
    parser.add_argument("--figure-qa-tools-dir",type=Path)
    args = parser.parse_args(argv)
    output_dir = args.output_dir
    if args.plot_only:
        before = {n:sha256_file(output_dir/f"{n}.png") for n in FIGURE_NAMES if (output_dir/f"{n}.png").exists()}
        paths,figure_records = render_and_audit(output_dir/f"{PREFIX}_scan.csv",output_dir,args.figure_qa_tools_dir)
        checks = {p.name:before[p.stem]==sha256_file(p) for p in paths if p.stem in before}
        print("CSV-only reconstruction:",json.dumps({"existing_PNG_hash_matches":checks,"created_without_previous_images":[p.name for p in paths if p.stem not in before]}))
        if not all(checks.values()):
            raise RuntimeError("CSV-only redraw differs from previous PNG")
        result_path = output_dir/f"{PREFIX}_results.json"
        if result_path.exists():
            previous = json.loads(result_path.read_text(encoding="utf-8"))
            if sha256_file(output_dir/f"{PREFIX}_scan.csv")!=previous["output_csv_sha256"][f"{PREFIX}_scan.csv"]:
                raise ValueError("aggregate source CSV changed; cannot refresh the existing figure report")
            previous["figures"] = figure_records
            previous["CSV_only_redraw"] = {"source_only_entry_point":True,"PNG_hash_matches":checks,
                "all_checks_passed":all(checks.values())}
            previous["provenance"]["source_sha256"]["scripts/plot_lmmse_train_validation.py"] = sha256_file(ROOT/"scripts/plot_lmmse_train_validation.py")
            save_json(result_path,previous)
        return
    started = perf_counter()
    config = check_config()
    accepted_sources = source_checks()
    frozen,validation,theory,before = load_frozen_inputs(ROOT/"outputs")
    layouts = {df:build_fdm_pilot_layout(2,64,16,df,4) for df in DF_OPTIONS}
    model = prepare_trajectory()
    train,oracle,preparation = prepare_pools(frozen,layouts,theory)
    print("Actual preparation counters:",json.dumps({k:preparation[k] for k in (
        "prepared_train_estimators","prepared_oracle_estimators","train_Cholesky_calls","oracle_Cholesky_calls")}),flush=True)
    print("Benchmark: 10 validation trajectories x 10 frames x 20 conditions x 4 methods",flush=True)
    benchmark_report,_ = run_validation(validation[:10],model,layouts,train,oracle,theory)
    benchmark = {"num_trajectories":10,"num_correlated_frames":100,**benchmark_report["runtime"],
        "projected_400_simulation_seconds":40*benchmark_report["runtime"]["total_simulation_seconds"],
        "preparation_seconds":preparation["seconds"],"ZF_backend":"batched square np.linalg.solve; regression checked against existing per-RE ZF"}
    print("Benchmark/runtime:",json.dumps(benchmark,indent=2),flush=True)
    if args.benchmark:
        if frozen_hashes(ROOT/"outputs",frozen)!=before:
            raise RuntimeError("benchmark altered frozen covariance")
        return
    print("Full held-out integration: 400 independent trajectory units / 4000 correlated frames",flush=True)
    try:
        report,rows = run_validation(validation,model,layouts,train,oracle,theory,
            progress=lambda n:print(f"Completed {n}/400 validation trajectories",flush=True) if n%10==0 else None)
    except (ValueError,np.linalg.LinAlgError) as exc:
        output_dir.mkdir(exist_ok=True)
        save_json(output_dir/f"{PREFIX}_results.json",{"task":"T10D","all_checks_passed":False,
            "num_filtered_trajectories":0,"numerical_failures":[getattr(exc,"record",{"reason":str(exc)})],
            "covariance_hash_before_after":{"before":before,"after":frozen_hashes(ROOT/"outputs",frozen)},
            "formal_test_used":False,"formal_test_sealed":True})
        raise
    after = frozen_hashes(ROOT/"outputs",frozen)
    report["checks"].update({"T10C_commit_exists":True,"accepted_sources_unchanged":all(accepted_sources.values()),
        "frozen_hashes_unchanged":before==after,"400_independent_trajectories":len(validation)==400,
        "8000_trajectory_condition_rows":len(rows)==8000,"80_prepared_estimators":
            preparation["train_Cholesky_calls"]==preparation["oracle_Cholesky_calls"]==40})
    report.update({"task":"T10D","experiment_type":"held-out ID full-link LMMSE-Train integration validation",
        "split":"validation_id","num_independent_trajectories":400,"frames_per_trajectory":10,"num_correlated_frames":4000,
        "manifest_sha256":before["manifest_sha256"],"covariance_sha256":before["covariance_file_sha256"],
        "R_train_raw_sha256":before["R_train_raw_sha256"],"covariance_hash_before_after":{"before":before,"after":after},
        "seed_policy":{"channel":"manifest: 60261001 + trajectory_id","validation_bits_seed_base":80261001,
            "bits_rule":"80261001 + 100*trajectory_id + 2*frame_index + tx",
            "validation_full_link_noise_seed_base":90261001,"noise_rule":"90261001 + 100*trajectory_id + frame_index",
            "noise_draw":"Z~CN(0,1) [2,64,16] once per frame; N=sqrt(10^(-SNR/10))*Z"},
        "common_random_numbers":True,"same_y_for_all_methods":report["checks"]["same_y_for_all_methods"],
        "prepared_train_estimators":40,"prepared_oracle_estimators":40,"preparation":preparation,
        "benchmark":benchmark,"theory_tolerance_db":THEORY_TOLERANCE_DB,"config":config,
        "physical_model":{"channel":"TDL-A","RMS_delay_spread_ns":100,"speed_kmh":30,"carrier_frequency_hz":3.5e9,
            "symbol_wise_quasi_static":True,"ICI":False,"dtype":"complex128","fft_frequency_order":"fftshift physical ascending"},
        "pilot_counts":{str(df):count_layout_resources(layout) for df,layout in layouts.items()},
        "explicit_inverse_used":False,"pseudo_inverse_used":False,"jitter_used":False,"covariance_reestimated":False,
        "covariance_updated":False,"diagonal_renormalization_used":False,"shrinkage_used":False,
        "formal_test_used":False,"formal_test_sealed":True,"OOD_generated":False,
        "bootstrap_CI_computed":False,"p_value_computed":False,"method_ordering_required":False,
        "all_checks_passed":all(report["checks"].values())})
    report["diagnostic_anomalies"] = {"train_ber_above_ls_conditions":[[r["dF"],r["snr_db"]] for r in report["aggregate_results"] if r["train_ber"]>r["ls_ber"]],
        "oracle_ber_above_ls_conditions":[[r["dF"],r["snr_db"]] for r in report["aggregate_results"] if r["oracle_ber"]>r["ls_ber"]],
        "train_minus_oracle_nmse_db_range":[min(r["train_minus_oracle_nmse_db"] for r in report["aggregate_results"]),max(r["train_minus_oracle_nmse_db"] for r in report["aggregate_results"])]}
    output_dir.mkdir(exist_ok=True)
    trajectory_path = output_dir/f"{PREFIX}_trajectory_metrics.csv"
    aggregate_path = output_dir/f"{PREFIX}_scan.csv"
    save_csv(trajectory_path,rows); save_csv(aggregate_path,report["aggregate_results"])
    source_paths = ["src/lmmse_train_validation.py","scripts/run_lmmse_train_validation.py",
        "scripts/plot_lmmse_train_validation.py","tests/test_lmmse_train_validation.py",*accepted_sources]
    report["provenance"] = {"T10C_commit":T10C_COMMIT,"T10C_committed_cleanly_at_task_start":True,
        "git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "python_version":platform.python_version(),"numpy_version":np.__version__,"scipy_version":scipy.__version__,
        "accepted_source_checks":accepted_sources,"source_sha256":{p:sha256_file(ROOT/p) for p in dict.fromkeys(source_paths)}}
    report["output_csv_sha256"] = {p.name:sha256_file(p) for p in (trajectory_path,aggregate_path)}
    # Preserve the numerical results even if a later figure QA gate fails.
    save_json(output_dir/f"{PREFIX}_results.json",report)
    if report["all_checks_passed"]:
        stamp = perf_counter()
        _,figures = render_and_audit(aggregate_path,output_dir,args.figure_qa_tools_dir)
        report["figures"] = figures
        report["runtime"]["figure_export_and_QA_seconds"] = perf_counter()-stamp
    report["runtime"]["total_seconds_including_benchmark_preparation_and_outputs"] = perf_counter()-started
    save_json(output_dir/f"{PREFIX}_results.json",report)
    print("20-condition aggregate:",json.dumps(report["aggregate_results"],indent=2),flush=True)
    print("Manual diagnostic:",json.dumps(report["manual_diagnostic"],indent=2),flush=True)
    print("Runtime:",json.dumps(report["runtime"]),flush=True)
    print("T10D all_checks_passed:",report["all_checks_passed"],flush=True)
    if not report["all_checks_passed"]:
        raise RuntimeError("held-out calibration failed; original CSV/JSON preserved")


if __name__=="__main__":
    main()
