"""Audit T16B saved results; no numerical rescan or channel generation."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_shared_output_basis import OUT, RESULT, load, save, sha, accepted_snapshot, environments, require_environment


def visual_review():
    r = load(OUT/"t16b_shared_basis_figure_audit.json")
    if r["source_returncode"] or not all(r["CSV_only_rebuild"].values()) or any(v["text_returncode"] or v["collision_returncode"] for v in r["rendered_checks"].values()):
        raise ValueError("automated figure gates incomplete")
    r["visual_review"] = "Actual final PNGs inspected: both primary and cross-shift panels; legends, direction labels, curves, measured rank ticks, thresholds and margins clear."
    r["reviewed_source_warnings"] = ["PNG300dpi development preview; editable PDF/SVG supplied","No CI claimed; trajectory dispersion saved"]
    r["all_checks_passed"] = True; save(OUT/"t16b_shared_basis_figure_audit.json",r)
    print("Actual T16B visual review recorded")


def finalize():
    require_environment(); start = time.perf_counter(); r = load(RESULT)
    for p,h in {**r["source_sha256"],**r["frozen_history_sha256"]}.items():
        if sha(ROOT/p) != h: raise ValueError(f"source/history changed: {p}")
    accepted_snapshot()
    figure = load(OUT/"t16b_shared_basis_figure_audit.json")
    if not figure["all_checks_passed"]: raise ValueError("figure gates incomplete")
    numeric = {str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in OUT.glob("t16b_*.csv")}
    for name,args in (("provenance",["tests/test_shared_output_basis.py","-k","snapshot or frozen_statistics or actual_results"]),("full",[])):
        cmd = [str(ROOT/".venv/Scripts/python.exe"),"-m","pytest","-q",*args]
        t = subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding="utf8",errors="replace")
        output = t.stdout+t.stderr
        (OUT/f"t16b_shared_basis_pytest_{name}.txt").write_text(output,encoding="utf8")
        r[f"pytest_{name}"] = dict(command=" ".join(cmd),returncode=t.returncode,output=output)
        save(RESULT,r)
        if t.returncode: raise ValueError("pytest failed; numerical outputs retained")
        print(output.strip().splitlines()[-1],flush=True)
    if any(sha(ROOT/p) != h for p,h in numeric.items()): raise ValueError("numerical CSV changed in audit")
    r["environment_after"] = environments()
    if r["environment_after"] != r["environment_before"]: raise ValueError("environment payload changed")
    diff = subprocess.run(["git","diff","--check"],cwd=ROOT,capture_output=True,text=True)
    if diff.returncode: raise ValueError("git diff --check failed")
    if any(sha(ROOT/p) != h for p,h in r["frozen_history_sha256"].items()): raise ValueError("historical files changed")
    r["git_diff_check"] = dict(returncode=diff.returncode,output=diff.stdout+diff.stderr)
    r["numeric_artifact_sha256"] = numeric; r["figure_audit"] = figure
    r["startup_pytest"] = (ROOT/".pytest_cache/t16b_startup_pytest.txt").read_text(encoding="utf8")
    r["small_math_pytest"] = (ROOT/".pytest_cache/t16b_math_pytest.txt").read_text(encoding="utf8")
    r["runtime"]["final_audit_seconds"] = time.perf_counter()-start
    r["runtime"]["total_numeric_plus_final_audit_seconds"] = r["runtime"]["total_numeric_wall_seconds"]+r["runtime"]["final_audit_seconds"]
    r["all_checks_passed"] = True; r["completion_state"] = "completed"
    save(RESULT,r); print("T16B engineering gates PASS; formal test remains sealed",flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--record-visual-review",action="store_true")
    if p.parse_args().record_visual_review: visual_review()
    else: finalize()
