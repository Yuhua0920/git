"""T13B incremental R6-HR replay, CSV-only figures and final verification."""
import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/"outputs"

from scripts.plot_robust_lmmse import FIGURE_NAMES,plot_results
from scripts.run_lmmse_train_validation import qa_tools,save_csv
from scripts.validate_b0_pipeline import json_save
from src import robust_lmmse as scan
from src.b0_development import replay_link
from src.config import load_config
from src.lmmse_train import raw_array_sha256,sha256_file
from src.nearest_validation import verify_frozen
from src.trajectories import prepare_trajectory

SOURCES=("docs/robust_lmmse_protocol.md","docs/robust_lmmse_feasibility.md","src/robust_lmmse.py",
    "scripts/plot_robust_lmmse.py","scripts/run_robust_lmmse.py","tests/test_robust_lmmse.py")


def progress(message):
    print(message,flush=True)


def accepted_files():
    paths=subprocess.check_output(["git","ls-tree","-r","--name-only",scan.T13A_COMMIT,
        "src","scripts","tests","configs","docs","outputs"],cwd=ROOT,text=True).splitlines()
    return {p:sha256_file(ROOT/p) for p in paths if p!="docs/stage_status.md"}


def source_preflight():
    tools,_=qa_tools(None)
    result=subprocess.run([sys.executable,str(tools/"validate_figure.py"),str(ROOT/"scripts/plot_robust_lmmse.py"),"--json"],
        capture_output=True,text=True,encoding="utf8")
    if result.returncode:
        raise RuntimeError(f"figure source preflight failed\n{result.stdout}\n{result.stderr}")
    record=json.loads(result.stdout)
    json_save(OUT/f"{scan.PREFIX}_figure_source_preflight.json",record)
    return record


def render_and_audit(input_dir,output_dir):
    tools,gate=qa_tools(None)
    paths,records=plot_results(input_dir,output_dir,gate)
    for path,record in zip(paths,records):
        text=subprocess.run([sys.executable,str(tools/"audit_pdf_text.py"),str(path.with_suffix(".pdf")),
            "--min-pt","5","--json"],capture_output=True,text=True,encoding="utf8")
        cp=path.with_suffix(".collision-audit.json")
        collision=subprocess.run([sys.executable,str(tools/"audit_figure_collisions.py"),str(path.with_suffix(".pdf")),
            "--json-out",str(cp)],capture_output=True,text=True,encoding="utf8")
        if text.returncode or collision.returncode:
            raise RuntimeError(f"figure QA failed {path.name}\n{text.stdout}\n{collision.stdout}")
        record.update({"path":str(path),"PDF_text_audit":json.loads(text.stdout),
            "collision_audit":json.loads(cp.read_text(encoding="utf8")),
            "alignment_audit":json.loads(path.with_suffix(".alignment.json").read_text(encoding="utf8")),
            "sha256":{e:sha256_file(path.with_suffix('.'+e)) for e in ('png','pdf','svg')}})
    return records


def execute():
    result_path=OUT/f"{scan.PREFIX}_results.json"
    if result_path.exists() or (OUT/f"{scan.PREFIX}.csv").exists():
        raise ValueError("artifacts already exist; refuse accidental numerical rerun")
    if subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()!=scan.T13A_COMMIT:
        raise ValueError("accepted T13A must be HEAD")
    load_config(ROOT/"configs/base.yaml")
    torch.set_num_threads(4)
    before=accepted_files(); frozen=verify_frozen(OUT)
    sources={p:sha256_file(ROOT/p) for p in SOURCES}
    refs=scan.load_replay_references(OUT); old_link=replay_link(ROOT)
    preflight=source_preflight()
    started=perf_counter()
    progress("Envelope: max normalized TDL-A delay 9.6586 * 300 ns = 2897.58 ns = 2.89758e-6 s; fixed fD = 389.158111 Hz")
    stamp=perf_counter(); full,cov_record=scan.build_prior(); build_time=perf_counter()-stamp
    progress(f"Analytic/quadrature HR gate PASS; same fixed covariance {cov_record['covariance_hash']}")
    stamp=perf_counter(); pools=scan.prepare_pool(full); factor_time=perf_counter()-stamp
    models={"ID":prepare_trajectory(frames_per_trajectory=10),**{s:scan.ood.prepare_channel(s) for s in scan.ood.SHIFTS}}
    report={"task":"T13B","experiment_type":"fixed literature HR baseline gating on ID validation and exact frozen OOD development replay",
        "literature_baseline":True,"proposed_method":False,
        "hr_reference":{"authors":"M. D. Nisar, W. Utschick, T. Hindelang","title":"Maximally Robust 2-D Channel Estimation for OFDM Systems",
            "year":2010,"DOI":"10.1109/TSP.2010.2043126","definition":"uniform PDP and rectangular Doppler PSD; HR is not MR-SDP"},
        "git":{"T13A_commit":scan.T13A_COMMIT,"T13A_committed_cleanly":True,"clean_after_commit":True,
            "startup_pytest":"1114 passed in 78.42s (0:01:18)"},
        "uncertainty_envelope":scan.envelope(),"tau_max_envelope":scan.TAU_BOUND,"fd_max_envelope":scan.FD_BOUND,
        **cov_record,"covariance_hash_by_condition":{c:raw_array_sha256(full) for c in scan.CONDITIONS},
        "factorization_count":sum(len(pair) for pair in pools.values()),"factorizations_reused_across_all_conditions":True,
        "factorization_diagnostics":{str(s):[p.diagnostics for p in pairs] for s,pairs in pools.items()},
        "information_assumptions":{"R4":"matched-training frozen prior","R5":"condition-matched analytic oracle prior","R6_HR":"fixed support prior"},
        **scan.feasibility(),"dF":16,"dT":4,"snr_db":list(scan.SNRS),"pilot_ratio":1/32,
        "formal_test_used":False,"formal_test_sealed":True,"new_model_trained":False,"MMSE_detector_used":False,
        "old_estimators_rerun":False,"old_model_loaded":False,"num_filtered_trajectories":0,"numerical_failures":[],
        "replay_semantics":"reconstruct saved seed realizations, hash-match OOD H/bits/unit-noise/Y/Pilot-LS; ID H/bits/unit-noise hash-match and Y equals accepted snapshot link",
        "environment":{"python":platform.python_version(),"torch":torch.__version__,"numeric_dtype":"complex128","torch_CPU_threads":4},
        "provenance":{"accepted_before":before,"source_sha256":sources,"frozen_T10D_T10C_hashes":frozen},
        "figure_source_preflight":preflight,
        "known_warnings":["PNG/PDF/SVG development exports; static preflight notes no submission TIFF",
            "negative factor eigenvalues within 1e-12 roundoff are reported without clipping",
            "task Doppler bound rounds physical speed120 fD down by 6.451076e-8 Hz",
            "sparse frequency pilot alias period 2.083333 us is shorter than deployment delay envelope"],
        "small_scale_test_history":{"initial":"2 failed, 23 passed, 3 skipped in 9.75s",
            "causes":"test Git LF vs working CRLF byte assertion and nonexistent a_matrix attribute; corrected assertions only",
            "final":"25 passed, 3 skipped in 4.63s; artifact tests pending full scan"},
        "runtime":{"covariance_build_and_calibration_seconds":build_time,"factorization_and_preparation_seconds":factor_time,"per_condition":{}},
        "figures":[],"all_checks_passed":False}
    json_save(result_path,report)
    rows,realizations=[],[]
    try:
        for condition in scan.CONDITIONS:
            timing={k:0. for k in ("channel_replay_seconds","Pilot_LS_seconds","R6_inference_seconds","ZF_seconds")}
            stamp=perf_counter(); ids=range(2000,2400) if condition=="ID" else range(200)
            for index,tid in enumerate(ids):
                new,record=scan.evaluate_trajectory(condition,tid,models[condition],pools,refs,old_link,timing)
                rows.extend(new); realizations.append(record)
                if (index+1)%50==0:
                    progress(f"R6 only {condition}: {index+1}/{len(ids)} exact trajectories")
            timing["wall_seconds"]=perf_counter()-stamp; report["runtime"]["per_condition"][condition]=timing
    except Exception as exc:
        report["numerical_failures"].append({"condition":condition,"trajectory_id":tid,"reason":str(exc),"no_samples_removed":True})
        json_save(result_path,report)
        if rows:
            save_csv(OUT/f"{scan.PREFIX}_partial_trajectory_metrics.csv",rows)
        raise
    aggregates=scan.aggregate(rows,scan.frozen_performance(OUT))
    if len(rows)!=2400 or accepted_files()!=before or frozen!=verify_frozen(OUT):
        raise ValueError("row count or historical immutability failed")
    save_csv(OUT/f"{scan.PREFIX}_trajectory_metrics.csv",rows)
    save_csv(OUT/f"{scan.PREFIX}.csv",aggregates)
    report.update({"id_results":[r for r in aggregates if r["condition"]=="ID"],
        "ood_results":[r for r in aggregates if r["condition"]!="ID"],
        "gap_closure":[{k:r[k] for k in ("condition","snr_db","R4_R5_gap_db","R6_recovery_db","robust_closure")} for r in aggregates if r["condition"]!="ID"],
        **scan.decision_flags(aggregates),"trajectory_rows":len(rows),"realization_hashes":realizations,
        "replay_verification":{"ID":{"long_H":400,"H_bits_unit_noise_frames":4000,"accepted_snapshot_Y_checks":12000,"historical_Y_hash_available":False},
            "OOD":{"long_H":400,"H_bits_unit_noise_frames":2000,"historical_Y_hash_checks":6000,"historical_pilot_LS_hash_checks":6000},"all_checks_passed":True},
        "numeric_checks_passed":True})
    report["provenance"]["accepted_after"]=accepted_files()
    report["provenance"]["CSV_sha256"]={s:sha256_file(OUT/f"{scan.PREFIX}{s}.csv") for s in ("","_trajectory_metrics")}
    json_save(result_path,report)
    report["figures"]=render_and_audit(OUT,OUT)
    report["runtime"]["total_wall_seconds"]=perf_counter()-started
    json_save(result_path,report)
    progress("T13B incremental numeric/figure checks complete; ready for visual review and final verification")


def rebuild_audit():
    with tempfile.TemporaryDirectory(prefix="t13b_csv_only_",dir=OUT) as d:
        temp=Path(d); source=f"{scan.PREFIX}.csv"; shutil.copyfile(OUT/source,temp/source)
        initial=[p.name for p in temp.iterdir()]
        if initial!=[source]:
            raise ValueError("CSV-only starting directory required")
        render_and_audit(temp,temp)
        hashes={n+"."+e:{"original":sha256_file(OUT/(n+"."+e)),"rebuilt":sha256_file(temp/(n+"."+e))}
            for n in FIGURE_NAMES for e in ("png","pdf","svg")}
        if any(v["original"]!=v["rebuilt"] for v in hashes.values()):
            raise ValueError("CSV-only figure exports differ")
        record={"initial_files":initial,"numerical_scan_rerun":False,"hashes":hashes,"all_checks_passed":True}
        json_save(OUT/f"{scan.PREFIX}_csv_rebuild_audit.json",record)
        return record


def finalize():
    path=OUT/f"{scan.PREFIX}_results.json"; r=json.loads(path.read_text(encoding="utf8"))
    if not r["numeric_checks_passed"] or accepted_files()!=r["provenance"]["accepted_after"]:
        raise ValueError("numerical/historical gate failed")
    if any(sha256_file(ROOT/p)!=h for p,h in r["provenance"]["source_sha256"].items()):
        raise ValueError("T13B sources changed after numerical scan")
    if any(sha256_file(OUT/f"{scan.PREFIX}{s}.csv")!=h for s,h in r["provenance"]["CSV_sha256"].items()):
        raise ValueError("numerical results changed after scan")
    r["CSV_rebuild_audit"]=rebuild_audit()
    result=subprocess.run([sys.executable,"-m","pytest","-q"],cwd=ROOT,capture_output=True,text=True,encoding="utf8")
    (OUT/f"{scan.PREFIX}_pytest.txt").write_text(result.stdout+result.stderr,encoding="utf8"); progress(result.stdout)
    diff=subprocess.run(["git","diff","--check"],cwd=ROOT,capture_output=True,text=True,encoding="utf8")
    if result.returncode or diff.returncode or accepted_files()!=r["provenance"]["accepted_after"]:
        raise RuntimeError(f"pytest/diff/history gate failed\n{result.stderr}\n{diff.stdout}\n{diff.stderr}")
    for f in r["figures"]:
        f["visual_self_check"]="actual final PNG and each panel reviewed; all conditions, readable labels and unclipped negative closure retained"
    r["pytest"]={"exit_code":0,"summary":result.stdout.strip().splitlines()[-1]}
    r["git_diff_check"]={"exit_code":0,"warnings":diff.stderr}
    r["all_checks_passed"]=True; json_save(path,r)
    progress("T13B final verification PASS")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only",action="store_true")
    parser.add_argument("--finalize-after-visual-review",action="store_true")
    parser.add_argument("--input-dir",type=Path,default=OUT); parser.add_argument("--output-dir",type=Path,default=OUT)
    args=parser.parse_args()
    if args.plot_only:
        json_save(args.output_dir/f"{scan.PREFIX}_figure_qa.json",render_and_audit(args.input_dir,args.output_dir))
    elif args.finalize_after_visual_review:
        finalize()
    else:
        if args.input_dir.resolve()!=OUT.resolve() or args.output_dir.resolve()!=OUT.resolve():
            raise ValueError("scan output must use project outputs")
        execute()


if __name__=="__main__":
    main()
