"""CSV-only T15A CDL-C development NMSE benchmark; deterministic export and QA."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
SKILL=Path("C:/Users/yuhua/.codex/skills/nature-figure")
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".pytest_cache/matplotlib-cdl"))


def plot(input_dir,output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    with (Path(input_dir)/"t15a_cdl_lmmse.csv").open(encoding="utf8",newline="") as f: rows=list(csv.DictReader(f))
    groups={method:sorted([r for r in rows if r["method"]==method],key=lambda r:int(r["snr_db"])) for method in ("R3-LS-LIN","R10-500")}
    if any([int(r["snr_db"]) for r in g]!=[5,10,20] for g in groups.values()): raise ValueError("three complete SNR values required")
    with matplotlib.rc_context({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"font.size":9,
        "axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False,
        "pdf.fonttype":42,"svg.fonttype":"none","svg.hashsalt":"t15a-cdl"}):
        fig,ax=plt.subplots(figsize=(5.4,3.9)); fig.subplots_adjust(left=.15,right=.97,bottom=.24,top=.76)
        for method,color,marker,label in (("R3-LS-LIN","#0072B2","o","R3: LS + linear"),("R10-500","#D55E00","s","R10: empirical LMMSE")):
            group=groups[method]
            ax.plot([int(r["snr_db"]) for r in group],[float(r["nmse_db"]) for r in group],
                color=color,marker=marker,markerfacecolor="none",linewidth=1.4,markersize=5,label=label)
        ax.set_xlabel("Ensemble receive SNR reference (dB)"); ax.set_ylabel("Data-RE NMSE (dB)")
        ax.set_xticks([5,10,20]); ax.set_xlim(4,21); ax.grid(alpha=.2)
        fig.legend(*ax.get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.53,.90),ncol=2,fontsize=8)
        fig.suptitle("CDL-C sparse-pilot validation benchmark",fontsize=10,y=.98)
        fig.text(.5,.10,"2 x 2; dF16/dT4; 100 validation trajectories x 10 frames",ha="center",fontsize=8)
        fig.text(.5,.05,"R10 fitted on 500 train trajectories; development results",ha="center",fontsize=8)
        fig.canvas.draw()
        name="t15a_cdl_nmse"
        fig.savefig(output_dir/(name+".png"),dpi=600)
        fig.savefig(output_dir/(name+".pdf"),metadata={"CreationDate":None,"ModDate":None})
        fig.savefig(output_dir/(name+".svg"),metadata={"Date":None})
        plt.close(fig)
        svg=output_dir/(name+".svg")
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf8").splitlines())+"\n",encoding="utf8")


def audit():
    out=ROOT/"outputs"; plot(out,out)
    def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="t15a_figure_rebuild_",dir=ROOT/".pytest_cache") as temporary:
        plot(out,temporary)
        checks={ext:h(out/f"t15a_cdl_nmse.{ext}")==h(Path(temporary)/f"t15a_cdl_nmse.{ext}") for ext in ("png","pdf","svg")}
    commands=[([sys.executable,str(SKILL/"scripts/validate_figure.py"),str(Path(__file__).resolve()),"--json"],"source"),
        ([sys.executable,str(SKILL/"scripts/audit_pdf_text.py"),str(out/"t15a_cdl_nmse.pdf"),"--json"],"text"),
        ([sys.executable,str(SKILL/"scripts/audit_figure_collisions.py"),str(out/"t15a_cdl_nmse.pdf"),"--json-out",str(out/"t15a_cdl_nmse.collision-audit.json"),"--overlay-pdf",str(out/"t15a_cdl_nmse.collision-audit.pdf")],"collisions")]
    results=[]
    for command,label in commands:
        p=subprocess.run(command,capture_output=True,text=True,encoding="utf8",errors="replace")
        results.append(dict(check=label,command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
        if label=="text" and p.returncode==0:
            (out/"t15a_cdl_nmse.text-audit.json").write_text(p.stdout,encoding="utf8")
    # Visual review is recorded only after actually viewing the exported image.
    result=dict(figure="outputs/t15a_cdl_nmse.png",source_csv="outputs/t15a_cdl_lmmse.csv",panels=1,
        panel_alignment="N/A single panel",csv_only_rebuild_hash_checks=checks,automated_checks=results,
        visual_self_check="pending",publication_stage="development benchmark; no journal submission",
        uncertainty="aggregate ratios, no significance claim; trajectory dispersion in source CSV",
        all_checks_passed=False)
    (out/"t15a_cdl_figure_audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print(json.dumps(dict(rebuild=checks,checks=[(r["check"],r["returncode"]) for r in results]),indent=2))
    if not all(checks.values()) or any(r["returncode"] for r in results): raise ValueError("figure rebuild/QA failed")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--audit",action="store_true")
    p.add_argument("--input-dir",type=Path,default=ROOT/"outputs");p.add_argument("--output-dir",type=Path,default=ROOT/"outputs")
    args=p.parse_args()
    if args.audit: audit()
    else: plot(args.input_dir,args.output_dir)
