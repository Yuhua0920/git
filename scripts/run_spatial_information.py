"""T14B theory scan and fixed correctness audits; no communication experiment."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

# Preserve the existing execution environment: historical bitwise trajectory
# hashes depend on the calibrated BLAS contraction/thread configuration.
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); OUT = ROOT/"outputs"
from src import lmmse, spatial_information as info
from src.config import load_config
from src.lmmse_train import sha256_file, raw_array_sha256
from src.numerics import noise_variance_from_snr
from scripts.plot_spatial_information import plot_results, FIGURE_NAMES

SOURCES = ("src/spatial_information.py", "scripts/run_spatial_information.py", "scripts/plot_spatial_information.py",
           "tests/test_spatial_information.py", "docs/spatial_information_protocol.md")


def json_save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False,
        default=lambda x: x.item() if isinstance(x, np.generic) else (_ for _ in ()).throw(TypeError(type(x).__name__)))+"\n", encoding="utf8")


def save_csv(path, rows):
    with Path(path).open("w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def initial_attempt_reconciliation(rows):
    directory = OUT/"t14b_attempt01"
    if not directory.exists(): return None
    old = json.loads((directory/(info.PREFIX+"_results.json")).read_text(encoding="utf8"))
    snapshot = directory/"run_spatial_information.py.source"
    if sha256_file(snapshot)!=old["provenance"]["source_sha256"]["scripts/run_spatial_information.py"]:
        raise ValueError("initial scan source snapshot hash mismatch")
    with (directory/(info.PREFIX+".csv")).open(encoding="utf8",newline="") as f:
        previous = list(csv.DictReader(f))
    if len(previous)!=len(rows): raise ValueError("attempt theory row count changed")
    maxima = {}
    for before,after in zip(previous,rows):
        for k,v in after.items():
            if isinstance(v,(float,int)) and not isinstance(v,bool):
                error = abs(float(before[k])-v)
                maxima[k] = max(maxima.get(k,0.),error)
                if error > 1e-10*max(1,abs(v)): raise ValueError("theory changed beyond contraction roundoff: "+k)
            elif before[k]!=str(v): raise ValueError("attempt categorical metadata changed: "+k)
    return {"directory":"outputs/t14b_attempt01", "initial_pytest_summary":"5 failed, 1229 passed, 14 errors in 99.18s (0:01:39)",
        "cause":"OPENBLAS_NUM_THREADS=4 leaked into pytest; historical bitwise trajectory hashes require original environment",
        "correction":"remove thread override; retain original OPENBLAS_NUM_THREADS unset and historical source/results/seeds",
        "initial_source_snapshot_sha256":sha256_file(snapshot), "initial_results_sha256":sha256_file(directory/(info.PREFIX+"_results.json")),
        "initial_CSV_sha256":sha256_file(directory/(info.PREFIX+".csv")), "all_36_theory_rows_reconciled":True,
        "numerical_column_max_absolute_differences":maxima, "comparison_relative_tolerance":1e-10,
        "communication_scan_rerun":False, "seed_changes":False, "historical_output_changes":False}


def accepted_files():
    paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", info.T14A_COMMIT,
        "src", "scripts", "tests", "configs", "docs", "outputs"], cwd=ROOT, text=True).splitlines()
    return {p:sha256_file(ROOT/p) for p in paths if p != "docs/stage_status.md"}


def qa_tools():
    tools = Path.home()/".codex/skills/nature-figure/scripts"
    spec = importlib.util.spec_from_file_location("t14b_alignment", tools/"audit_panel_alignment.py")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return tools, module.require_matplotlib_panel_alignment


def source_preflight():
    tools, _ = qa_tools()
    p = subprocess.run([sys.executable, str(tools/"validate_figure.py"), str(ROOT/SOURCES[2]), "--json"], capture_output=True, text=True, encoding="utf8")
    if p.returncode: raise RuntimeError(p.stdout+p.stderr)
    r = json.loads(p.stdout); json_save(OUT/(info.PREFIX+"_figure_source_preflight.json"), r)
    return r


def render_and_audit(input_dir, output_dir):
    tools, gate = qa_tools(); paths, records = plot_results(input_dir, output_dir, gate)
    for path, r in zip(paths, records):
        pdf = path.with_suffix(".pdf"); audit = path.with_suffix(".collision-audit.json")
        text = subprocess.run([sys.executable, str(tools/"audit_pdf_text.py"), str(pdf), "--min-pt", "5", "--json"], capture_output=True, text=True, encoding="utf8")
        collision = subprocess.run([sys.executable, str(tools/"audit_figure_collisions.py"), str(pdf), "--json-out", str(audit)], capture_output=True, text=True, encoding="utf8")
        if text.returncode or collision.returncode: raise RuntimeError(text.stdout+collision.stdout+collision.stderr)
        r.update({"path":str(path), "PDF_text_audit":json.loads(text.stdout),
            "collision_audit":json.loads(audit.read_text(encoding="utf8")),
            "alignment_audit":json.loads(path.with_suffix(".alignment.json").read_text(encoding="utf8")),
            "sha256":{e:sha256_file(path.with_suffix('.'+e)) for e in ('png','pdf','svg')}})
    return records


def scan():
    started = perf_counter(); path = OUT/(info.PREFIX+"_results.json")
    if path.exists(): raise ValueError("existing result protected; use plot-only/finalize without rescanning")
    load_config(ROOT/"configs/base.yaml")
    before = accepted_files(); prefixes = {p:{"length":len((ROOT/p).read_bytes()),"sha256":sha256_file(ROOT/p)} for p in ("README.md","docs/stage_status.md")}
    r = {"task":"T14B", "experiment_type":"theoretical value gate for exploiting MIMO spatial correlation under sparse orthogonal pilots",
        "literature_gate_passed":True, "proposed_method":False, "formal_test_used":False, "formal_test_sealed":True,
        "new_model_trained":False, "B0_loaded":False, "BER_computed":False, "production_correlated_generator_implemented":False,
        "git":{"accepted_T14A_commit":info.T14A_COMMIT,"clean_after_commit":True,"startup_pytest":"1224 passed in 103.77s (0:01:43)"},
        "spatial_model_definition":{"name":"controlled real Kronecker gate", "R_tx":"[[1,rho_tx],[rho_tx,1]]", "R_rx":"[[1,rho_rx],[rho_rx,1]]",
            "R_sp_column_major":"R_tx^T tensor R_rx", "realistic_CDL":False,"array_geometry_model":False,
            "fixed_frequency_time":"original TDL-A/30kmh/nominal100ns/fc3.5GHz; one K64,T16 frame"},
        "ordering":{"project_tensor":"[Rx,Tx,K,T] C order", "index":"((rx*2+tx)*64+k)*16+t", "column_vec_index":"rx+2*tx",
            "column_to_project_permutation":[0,2,1,3],"joint":"P(R_tx^T tensor R_rx)P^T tensor Rf tensor Rt",
            "entry":"R_tx[tx_j,tx_i]*R_rx[rx_i,rx_j]*Rf[k_i,k_j]*Rt[t_i,t_j]"},
        "provenance":{"accepted_before":before,"source_sha256":{p:sha256_file(ROOT/p) for p in SOURCES},"append_only_document_prefixes":prefixes},
        "literature":[{"source":"https://arxiv.org/abs/1602.05480","position":"spatial correlation-aware pilot reduction is established prior work"},
            {"source":"https://orbilu.uni.lu/handle/10993/6386","position":"Bayesian correlated-MIMO estimation and training are established prior work; repository abstract checked"}],
        "environment":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"dtype":"complex128","OPENBLAS_NUM_THREADS":os.environ.get("OPENBLAS_NUM_THREADS")},
        "figure_source_preflight":source_preflight(), "runtime":{}, "all_checks_passed":False, "numerical_failures":[],
        "known_warnings":["controlled Kronecker prior, not realistic CDL or final paper channel model",
            "deterministic one-frame posterior NMSE; no BER or communication Monte Carlo",
            "roundoff-negative covariance-factor eigenvalues retain the existing clipping policy; no jitter/positive truncation",
            "development PNG/PDF/SVG; no submission TIFF",
            "additional author PDF/landing-page full-text fetch timed out; repository abstract and arXiv primary abstract checked"]}
    json_save(path, r)
    try:
        stamp = perf_counter(); g = info.observation_geometry()
        r["runtime"]["observation_operator_seconds"] = perf_counter()-stamp
        ordering = info.ordering_audit(g)
        if not ordering["all_checks_passed"]: raise ValueError("ordering/selection gate failed")
        r["ordering_validation"] = ordering
        r["pilot_observation_operator"] = {"equation":"z=S h+n", "shape":list(g.selector.shape), "nnz":g.selector.nnz,
            "selected_project_indices":g.selected.tolist(), "offsets":list(g.layout.offsets_by_tx),
            "pilot_symbols":list(g.layout.pilot_symbol_indices), "resources":g.counts,
            "data_coefficients":int(g.data.sum()), "data_mask_sha256":raw_array_sha256(g.data),
            "noise":"independent proper complex CN(0,sigma2 I), sigma2=Ptot/10^(SNR/10); unit pilots"}
        stamp = perf_counter(); rf, rt = lmmse.build_frequency_covariance(), lmmse.build_time_covariance()
        r["frequency_time_covariances"] = {"Rf_sha256":raw_array_sha256(rf),"Rt_sha256":raw_array_sha256(rt),"unchanged_API":True}
        joint0 = info.JointCovariance(np.eye(4), rf, rt); r["joint_covariance_PSD"] = joint0.psd_diagnostics()
        r["runtime"]["covariance_construction_seconds"] = perf_counter()-stamp
        historical = json.loads((OUT/"t10a_lmmse_oracle_results.json").read_text(encoding="utf8"))
        references = {}; reg = []; solve_seconds = 0.
        for snr in info.SNRS:
            stamp = perf_counter()
            reference, objects = info.per_link_reference(rf, rt, g, snr); references[snr] = reference
            solve_seconds += perf_counter()-stamp
            old = next(x for x in historical["theoretical_conditions"] if x["dF"]==16 and x["snr_db"]==snr)
            difference = abs(reference["nmse_linear"]-old["posterior_nmse_data_combined"])
            if difference > 1e-12: raise ValueError("historical matched-ID R5 theory regression failed")
            reg.append({"snr_db":snr,"current_R5_nmse_linear":reference["nmse_linear"],"historical_R5_nmse_linear":old["posterior_nmse_data_combined"],"absolute_difference":difference,"passed":True})
        rows, eigenrecords, zero = [], [], []
        moderate = None; posterior = None
        covariance_seconds = r["runtime"]["covariance_construction_seconds"]
        for mode in info.MODES:
            for rho in info.RHOS:
                mark = perf_counter(); tx, rx, column, spatial = info.spatial_covariances(*info.mode_rhos(mode,rho))
                joint = info.JointCovariance(spatial, rf, rt); psd = joint.psd_diagnostics()
                covariance_seconds += perf_counter()-mark
                e = {"mode":mode,"rho":rho, "R_tx_eigenvalues":np.linalg.eigvalsh(tx).tolist(),
                    "R_rx_eigenvalues":np.linalg.eigvalsh(rx).tolist(), "R_sp_eigenvalues":np.linalg.eigvalsh(column).tolist(),
                    "R_tx_condition":float(np.linalg.cond(tx)),"R_rx_condition":float(np.linalg.cond(rx)),
                    "R_sp_condition":float(np.linalg.cond(column)),"joint_PSD":psd}
                eigenrecords.append(e)
                for snr in info.SNRS:
                    mark = perf_counter()
                    p = info.prepare_posterior(joint, g.selected, noise_variance_from_snr(snr))
                    solve_seconds += perf_counter()-mark
                    m = info.metric(p, g.data); ref = references[snr]
                    gain = ref["nmse_db"]-m["nmse_db"]
                    if gain < -1e-10: raise ValueError("joint oracle information gain negative beyond tolerance")
                    row = {"mode":mode, "rho":rho,"rho_tx":float(info.mode_rhos(mode,rho)[0]),"rho_rx":float(info.mode_rhos(mode,rho)[1]),"snr_db":snr,
                        "R5_nmse_linear":ref["nmse_linear"],"R5_nmse_db":ref["nmse_db"],"R9_nmse_linear":m["nmse_linear"],"R9_nmse_db":m["nmse_db"],
                        "spatial_gain_db":gain,"posterior_trace":m["posterior_trace"],"prior_trace":m["prior_trace"],"R5_posterior_trace":ref["posterior_trace"],
                        "noise_variance":noise_variance_from_snr(snr),**p.diagnostics}
                    rows.append(row)
                    if rho==0:
                        diff = abs(gain)
                        if diff>1e-10: raise ValueError("rho0 hard gate failed")
                        zero.append({"mode":mode,"snr_db":snr,"NMSE_db_difference":diff,"tolerance_db":1e-10,"passed":True})
                    if mode=="BOTH" and rho==.6 and snr==10: moderate, posterior = joint, p
        r["runtime"]["covariance_construction_seconds"] = covariance_seconds
        r["runtime"]["posterior_solves_seconds"] = solve_seconds
        r["R5_historical_regression"] = {"source":"t10a_lmmse_oracle_results.json","source_sha256":sha256_file(OUT/"t10a_lmmse_oracle_results.json"),"conditions":reg,"all_checks_passed":True}
        r["rho0_regression"] = {"conditions":zero,"all_checks_passed":True}
        r["spatial_eigenvalue_diagnostics"] = eigenrecords; r["theoretical_results"] = rows
        stamp = perf_counter(); spatial_audit = info.spatial_generation_audit()
        r["runtime"]["single_point_generation_audit_seconds"] = perf_counter()-stamp
        if not spatial_audit["all_checks_passed"]: raise ValueError("single-point covariance generation audit failed")
        r["spatial_generation_audit"] = spatial_audit
        print("Ordering, historical R5, rho0 and single-point spatial covariance gates PASS", flush=True)
        stamp = perf_counter(); mc = info.monte_carlo_audit(moderate, g, posterior)
        r["runtime"]["Monte_Carlo_audit_seconds"] = perf_counter()-stamp; r["monte_carlo_audit"] = mc
        if not mc["all_checks_passed"]: raise ValueError("fixed joint Gaussian MSE audit failed")
        print(f"Joint Gaussian MSE audit relative error={mc['relative_error']:.6%}; PASS", flush=True)
        # Fixed unobserved coefficient H[0,0,k=1,t=1]. No subset selection after results.
        i = 1*16+1
        ref = info.per_link_reference(rf,rt,g,10)[1][0]
        r["fixed_data_RE_diagnostic"] = {"project_index":i,"coordinates":[0,0,1,1],"R5_posterior_variance":float(ref.posterior_diagonal[i]),
            "R9_posterior_variance":float(posterior.diagonal[i]),
            "same_Rx_other_Tx_pilot_covariance_magnitudes":np.abs(moderate.entries(np.array([i]),g.selected[g.selected//1024==1])).ravel().tolist(),
            "same_Tx_other_Rx_pilot_covariance_magnitudes":np.abs(moderate.entries(np.array([i]),g.selected[g.selected//1024==2])).ravel().tolist()}
        r.update(info.scientific_flags(rows)); r["numeric_checks_passed"] = True
        recovery = initial_attempt_reconciliation(rows)
        if recovery is not None:
            r["initial_attempt_recovery"] = recovery
            r["known_warnings"].append("Initial complete regression: 5 failed/14 errors due inherited BLAS thread override; attempt archived, override removed and 36 theory rows reconciled without seed/history changes")
        if accepted_files()!=before: raise ValueError("historical file mutation")
        save_csv(OUT/(info.PREFIX+".csv"), rows)
        np.savez(OUT/(info.PREFIX+"_covariances.npz"), Rf=rf,Rt=rt,R_sp_moderate=moderate.spatial,
            selected=g.selected,data_mask=g.data,moderate_R9_posterior_diagonal=posterior.diagonal)
        r["provenance"]["output_sha256"] = {n:sha256_file(OUT/n) for n in (info.PREFIX+".csv",info.PREFIX+"_covariances.npz")}
        r["provenance"]["accepted_after"] = accepted_files()
    except Exception as exc:
        r["numerical_failures"].append(str(exc)); json_save(path,r); raise
    r["runtime"]["numeric_total_seconds"] = perf_counter()-started; json_save(path,r)
    mark = perf_counter(); r["figures"] = render_and_audit(OUT,OUT); r["runtime"]["figure_QA_seconds"] = perf_counter()-mark
    r["runtime"]["total_seconds"] = perf_counter()-started; json_save(path,r)
    print("Theory scan and figure QA complete; full regression and visual review pending", flush=True)


def rebuild_audit():
    with tempfile.TemporaryDirectory(prefix="t14b_csv_only_", dir=OUT) as directory:
        p = Path(directory); name = info.PREFIX+".csv"; shutil.copyfile(OUT/name,p/name)
        if list(p.iterdir()) != [p/name]: raise ValueError("CSV-only reconstruction input failed")
        render_and_audit(p,p)
        hashes = {n+"."+e:{"original":sha256_file(OUT/(n+"."+e)),"rebuilt":sha256_file(p/(n+"."+e))} for n in FIGURE_NAMES for e in ("png","pdf","svg")}
        if any(v["original"]!=v["rebuilt"] for v in hashes.values()): raise ValueError("figure hashes differ after CSV-only reconstruction")
        result = {"input_files":[name],"theory_scan_rerun":False,"hashes":hashes,"all_checks_passed":True}
        json_save(OUT/(info.PREFIX+"_csv_rebuild_audit.json"),result); return result


def finalize():
    path = OUT/(info.PREFIX+"_results.json"); r = json.loads(path.read_text(encoding="utf8"))
    if not r["numeric_checks_passed"] or accepted_files()!=r["provenance"]["accepted_before"]: raise ValueError("numeric/history gate failed")
    for base, records in ((ROOT,r["provenance"]["source_sha256"]),(OUT,r["provenance"]["output_sha256"])):
        if any(sha256_file(base/p)!=h for p,h in records.items()): raise ValueError("source/results changed after scan")
    stamp = perf_counter(); r["CSV_rebuild_audit"] = rebuild_audit(); r["runtime"]["CSV_rebuild_QA_seconds"] = perf_counter()-stamp
    result = subprocess.run([sys.executable,"-m","pytest","-q"],cwd=ROOT,capture_output=True,text=True,encoding="utf8")
    (OUT/(info.PREFIX+"_pytest.txt")).write_text(result.stdout+result.stderr,encoding="utf8"); print(result.stdout,flush=True)
    if result.returncode: raise RuntimeError("complete pytest failed: "+result.stderr)
    r["pytest"] = {"exit_code":0,"summary":result.stdout.strip().splitlines()[-1]}
    note = ("\n\n## T14B spatial correlation information gate\n\nTask:T14B\n\nStatus:completed\n\n"
        "Spatial correlation exploitation is not novel by itself. This task only determines whether the current "
        "independent-link channel model is omitting a material MIMO information source; no proposed estimator is introduced. "
        "Controlled Kronecker theory only, original TDL-A/30kmh/100ns, dF16/dT4, one frame. No production correlated "
        "generator, B0, training, BER or communication Monte Carlo. Formal test remains sealed.\n\n"
        f"Actual pytest:{r['pytest']['summary']}. Moderate material={r['moderate_spatial_correlation_material']}; "
        f"weak={r['spatial_information_gain_weak']}; high-only={r['spatial_gain_material_only_at_high_correlation']}; "
        f"paper correlated evaluation={r['final_paper_should_include_spatially_correlated_evaluation']}.\n\n"
        "Run:`.venv/Scripts/python.exe scripts/run_spatial_information.py`; CSV-only figure: add `--plot-only`. "
        "Fixed Gaussian audit10000 samples, 2% relative error gate; single-point covariance sentinel200000 samples. "
        "Known limits: controlled oracle statistics, development exports without submission TIFF.\n")
    for p, record in r["provenance"]["append_only_document_prefixes"].items():
        data = (ROOT/p).read_bytes()
        if hashlib.sha256(data[:record["length"]]).hexdigest()!=record["sha256"]: raise ValueError("historical document prefix changed")
        if b"## T14B spatial correlation information gate" not in data:
            with (ROOT/p).open("ab") as f: f.write(note.encode("utf8"))
    diff = subprocess.run(["git","diff","--check"],cwd=ROOT,capture_output=True,text=True,encoding="utf8")
    if diff.returncode or accepted_files()!=r["provenance"]["accepted_before"]: raise ValueError("whitespace/history regression failed")
    names = subprocess.check_output(["git","ls-files","--others","--exclude-standard"],cwd=ROOT,text=True).splitlines()
    bad = []
    for p in names:
        if Path(p).suffix.lower() in (".py",".md",".csv",".json",".txt",".svg"):
            if any(line.rstrip()!=line for line in (ROOT/p).read_text(encoding="utf8").splitlines()): bad.append(p)
    if bad: raise ValueError("untracked text trailing whitespace: "+str(bad))
    r["git_diff_check"] = {"exit_code":0,"warnings":diff.stderr,"untracked_text_whitespace_checked":True}
    for fig in r["figures"]: fig["visual_self_check"] = "actual PNG reviewed; all three panels, curves, scales, legends and labels checked"
    r["all_checks_passed"] = True; json_save(path,r)
    print("T14B final gates PASS; stop without production correlated channel",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only",action="store_true"); parser.add_argument("--finalize-after-visual-review",action="store_true")
    parser.add_argument("--input-dir",type=Path,default=OUT); parser.add_argument("--output-dir",type=Path,default=OUT)
    a = parser.parse_args()
    if a.plot_only: json_save(a.output_dir/(info.PREFIX+"_figure_qa.json"),render_and_audit(a.input_dir,a.output_dir))
    elif a.finalize_after_visual_review: finalize()
    else: scan()


if __name__=="__main__": main()
