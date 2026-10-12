"""T15A manifest-only correction, preserved numerical evidence and final figure review."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.cdl_benchmark import manifest_rows,OfficialCDL,array_hash
OUT=ROOT/"outputs"
ARCHIVE=OUT/"t15a_metadata_and_figure_pre_audit"


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def repair_metadata():
    path=OUT/"t15a_cdl_results.json"; result=json.loads(path.read_text(encoding="utf8"))
    original=json.loads((ARCHIVE/"outputs/t15a_cdl_results.json").read_text(encoding="utf8"))
    if "metadata_only_amendment" in result: raise ValueError("metadata amendment already applied")
    def functions(p):
        tree=ast.parse(Path(p).read_text(encoding="utf8"))
        return {node.name:ast.dump(node,include_attributes=False) for node in tree.body
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name!="manifest_rows"}
    unchanged_ast=functions(ROOT/"src/cdl_benchmark.py")==functions(ARCHIVE/"src/cdl_benchmark.py")
    if not unchanged_ast: raise ValueError("channel/statistics implementations changed")
    if sha(ROOT/"scripts/run_cdl_benchmark.py")!=original["provenance"]["source_sha256"]["scripts/run_cdl_benchmark.py"]:
        raise ValueError("numeric run script changed")
    numeric={p:sha(OUT/p) for p in ("t15a_cdl_lmmse.csv","t15a_cdl_trajectory_metrics.csv")}
    if any(sha(ARCHIVE/"outputs"/p)!=h for p,h in numeric.items()): raise ValueError("numeric CSV changed")
    rows=manifest_rows()
    with (OUT/"t15a_cdl_manifest.csv").open(encoding="utf8",newline="") as f: old=list(csv.DictReader(f))
    channel_roles={"sionna_seed","torch_seed","numpy_seed"}
    for a,b in zip(old,rows,strict=True):
        if any(a[k]!=str(b[k]) for k in b if k not in channel_roles): raise ValueError("nonchannel manifest data changed")
    with (OUT/"t15a_cdl_manifest.csv").open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    adapter=OfficialCDL()
    repeat={i:array_hash(adapter.generate(int(i)))==h for i,h in original["calibration"]["selected_trajectory_hashes"].items()}
    if not all(repeat.values()): raise ValueError("actual channel seeds drifted")
    result["metadata_only_amendment"]=dict(reason="manifest channel seed fields for frames 1..9 must repeat actual trajectory frame-0 seed; generation always used frame-0 seed",
        original_archive="outputs/t15a_metadata_and_figure_pre_audit",algorithm_AST_except_manifest_unchanged=unchanged_ast,
        numeric_script_unchanged=True,numerical_CSV_sha256=numeric,numerical_CSV_unchanged=True,
        selected_channel_hash_regressions=repeat,nonchannel_manifest_fields_unchanged=True,
        original_source_sha256=original["provenance"]["source_sha256"],
        current_source_sha256={p:sha(ROOT/p) for p in original["provenance"]["source_sha256"]},
        figure_only_revision="legend moved away from grid strokes; audit_pdf_text uses its actual --json option",
        tests_only_revision="add actual-seed manifest, frozen snapshot and aggregate regression checks",
        numerical_results_recomputed=False)
    result["provenance"]["source_sha256"]=result["metadata_only_amendment"]["current_source_sha256"]
    result["provenance"]["metadata_audit_source_sha256"]=sha(Path(__file__))
    result["warnings"].append("First figure QA failed due to legend/grid overlap and incorrect text-auditor option; preserved and corrected, no numerical CSV changes.")
    result["warnings"].append("First manifest export had unused frame-varying channel seed labels beyond frame 0; corrected to actual trajectory seed, unchanged generation/seeds/numeric CSVs and six hash regressions verified.")
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print("Manifest metadata correction PASS; numeric CSV hashes and six channel hashes unchanged.")


def visual_review():
    # Invoked only after the agent actually viewed the final exported image.
    path=OUT/"t15a_cdl_figure_audit.json"; result=json.loads(path.read_text(encoding="utf8"))
    if not all(result["csv_only_rebuild_hash_checks"].values()) or any(r["returncode"] for r in result["automated_checks"]):
        raise ValueError("automated figure checks failed")
    result["visual_self_check"]="passed after viewing final PNG: both complete three-point series, legible labels, separate legend, no cropping/collision; no significance annotation"
    result["reviewed_warnings"]=["Development preview width 137.2 mm and no submission TIFF are intentional; PNG600dpi/PDF/SVG supplied."]
    result["all_checks_passed"]=True
    path.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print("T15A final figure visual review recorded.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--repair-metadata",action="store_true");p.add_argument("--record-visual-review",action="store_true")
    args=p.parse_args()
    if args.repair_metadata: repair_metadata()
    elif args.record_visual_review: visual_review()
    else: p.error("choose one explicit audit action")
