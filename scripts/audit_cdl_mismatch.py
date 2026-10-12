"""Finalize T15C by auditing saved data only; no generation or refitting."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_cdl_mismatch import (OUT,load,save,sha,environments,source_statistics)


def visual_review():
    r=load(OUT/"t15c_cdl_figure_audit.json")
    if r["source_returncode"] or not all(r["CSV_only_rebuild"].values()) or any(v["text_returncode"] or v["collision_returncode"] for v in r["rendered_checks"].values()):raise ValueError("automated figure gates incomplete")
    r["visual_review"]="Actual final PNGs inspected: both NMSE panels and penalty panel; labels, legends, curves and margins clear."
    r["reviewed_source_warnings"]=["300dpi PNG preview/no TIFF: development figures with PDF/SVG line art", "No CI: pooled ratios; independent-trajectory dispersion and paired diagnostics in results"]
    r["all_checks_passed"]=True;save(OUT/"t15c_cdl_figure_audit.json",r)
    print("Actual T15C figure visual review recorded")


def finalize():
    if Path(sys.prefix).resolve()!=(ROOT/".venv-cdl").resolve():raise ValueError("original CDL environment required")
    r=load(OUT/"t15c_cdl_mismatch_results.json")
    if any(sha(ROOT/p)!=h for p,h in r["source_sha256"].items()):raise ValueError("numerical run source drift")
    if any(sha(ROOT/p)!=h for p,h in r["frozen_history_sha256"].items()):raise ValueError("frozen history drift")
    source_statistics()
    for info in r["matched_statistics_provenance"].values():
        if sha(ROOT/info["path"])!=info["sha256"]:raise ValueError("target frozen statistics drift")
    figure=load(OUT/"t15c_cdl_figure_audit.json")
    if not figure["all_checks_passed"]:raise ValueError("figure audit incomplete")
    numeric={str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in OUT.glob("t15c_*") if p.suffix in (".csv",".npz")}
    test=subprocess.run([str(ROOT/".venv/Scripts/python.exe"),"-m","pytest","-q"],cwd=ROOT,capture_output=True,text=True,encoding="utf8",errors="replace")
    (OUT/"t15c_cdl_pytest_full.txt").write_text(test.stdout+test.stderr,encoding="utf8")
    r["pytest"]=dict(command=".venv/Scripts/python.exe -m pytest -q",returncode=test.returncode,output=test.stdout+test.stderr)
    save(OUT/"t15c_cdl_mismatch_results.json",r)
    if test.returncode:raise ValueError("full regression failed; numerical artifacts preserved")
    if any(sha(ROOT/p)!=h for p,h in numeric.items()):raise ValueError("numeric artifacts changed during audit")
    r["environment_after"]=environments()
    diff=subprocess.run(["git","diff","--check"],cwd=ROOT,capture_output=True,text=True)
    if diff.returncode:raise ValueError("git diff --check failed")
    r["git_diff_check"]=dict(returncode=diff.returncode,output=diff.stdout+diff.stderr)
    r["numeric_artifact_sha256"]=numeric;r["figure_audit"]=figure
    r["initial_pytest"]=(ROOT/".pytest_cache/t15c_startup_pytest.txt").read_text(encoding="utf8")
    r["provenance_pytest"]=(ROOT/".pytest_cache/t15c_provenance_pytest.txt").read_text(encoding="utf8")
    r["all_checks_passed"]=True;r["completion_state"]="completed"
    save(OUT/"t15c_cdl_mismatch_results.json",r)
    print(test.stdout.strip().splitlines()[-1]);print("T15C final engineering gates PASS")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--record-visual-review",action="store_true")
    if p.parse_args().record_visual_review:visual_review()
    else:finalize()
