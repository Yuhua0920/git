"""Two CSV-only T15B development figures; never imports channel or GMM runtime."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
SKILL=Path("C:/Users/yuhua/.codex/skills/nature-figure/scripts")
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".pytest_cache/matplotlib-t15b"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0,str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment

STYLE={"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
                     "font.size":10,"axes.labelsize":10,"xtick.labelsize":10,"ytick.labelsize":10,
                     "legend.fontsize":10,"svg.fonttype":"none","svg.hashsalt":"T15B-fixed-probe","pdf.fonttype":42,
                     "axes.spines.right":False,"axes.spines.top":False,"legend.frameon":False}
STYLES={5:("#0072B2","o","-"),10:("#E69F00","s","--"),20:("#009E73","^","-.")}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot(input_dir,output_dir):
    with matplotlib.rc_context(STYLE):
        return _plot(input_dir,output_dir)


def _plot(input_dir,output_dir):
    input_dir,output_dir=Path(input_dir),Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    with (input_dir/"t15b_gmm_probe_validation.csv").open(encoding="utf8") as f:rows=list(csv.DictReader(f))
    names=[]
    for stem,column,label in (("t15b_gmm_probe_nmse","nmse_db","Pooled data-probe NMSE (dB)"),
                              ("t15b_gmm_nonlinear_gain","nonlinear_gain_db","Nonlinear gain over K1 (dB)")):
        fig,ax=plt.subplots(figsize=(7.2,4.8))
        fig.subplots_adjust(left=.15,right=.97,bottom=.29,top=.75)
        for snr,(color,marker,style) in STYLES.items():
            chosen=sorted((r for r in rows if int(r["snr_db"])==snr),key=lambda r:int(r["K"]))
            if not chosen:continue
            x=[int(r["K"]) for r in chosen];y=[float(r[column]) for r in chosen]
            ax.plot(x,y,color=color,marker=marker,linestyle=style,linewidth=1.4,
                    markersize=5,markerfacecolor="white",label=f"{snr} dB")
        ax.set_xticks([1,2,4,8]);ax.set_xlabel("Mixture components K");ax.set_ylabel(label)
        ax.grid(axis="y",color="0.9",linewidth=.6);ax.set_axisbelow(True)
        ax.axvline(1,color="0.65",linestyle=":",linewidth=.8)
        if column=="nonlinear_gain_db":ax.axhline(0,color="0.65",linewidth=.8,zorder=0)
        fig.suptitle("CDL-C full-covariance complex-GMM probe",fontsize=13,y=.98)
        handles,labels=ax.get_legend_handles_labels()
        fig.legend(handles,labels,ncol=3,loc="upper center",bbox_to_anchor=(.56,.91))
        fig.text(.15,.11,"K1 Gaussian reference; 64 fixed data coefficients/frame",fontsize=9)
        fig.text(.15,.045,"100 validation trajectories x 10 frames; pooled metric, no CI",fontsize=9)
        require_matplotlib_panel_alignment(fig,json_out=str(output_dir/f"{stem}.alignment.json"),strict=True)
        fig.savefig(output_dir/f"{stem}.png",dpi=300)
        fig.savefig(output_dir/f"{stem}.pdf",metadata={"CreationDate":None,"ModDate":None})
        fig.savefig(output_dir/f"{stem}.svg",metadata={"Date":None})
        plt.close(fig);names.append(stem)
    return names


def audit(input_dir,output_dir):
    input_dir,output_dir=Path(input_dir),Path(output_dir)
    names=plot(input_dir,output_dir);checks={}
    source=subprocess.run([sys.executable,str(SKILL/"validate_figure.py"),str(Path(__file__)),"--json"],capture_output=True,text=True)
    for stem in names:
        text=subprocess.run([sys.executable,str(SKILL/"audit_pdf_text.py"),str(output_dir/f"{stem}.pdf"),"--min-pt","5","--json"],capture_output=True,text=True)
        collision=subprocess.run([sys.executable,str(SKILL/"audit_figure_collisions.py"),str(output_dir/f"{stem}.pdf"),
                                  "--json-out",str(output_dir/f"{stem}.collision-audit.json"),"--overlay-pdf",str(output_dir/f"{stem}.collision-audit.pdf")],capture_output=True,text=True)
        checks[stem]=dict(text_returncode=text.returncode,collision_returncode=collision.returncode,
                          text_output=text.stdout+text.stderr,collision_output=collision.stdout+collision.stderr)
    with tempfile.TemporaryDirectory(dir=ROOT/".pytest_cache") as directory:
        temp=Path(directory);csvdir=temp/"csv";csvdir.mkdir()
        shutil.copy2(input_dir/"t15b_gmm_probe_validation.csv",csvdir)
        plot(csvdir,temp/"rebuilt")
        rebuild={f"{stem}.{ext}":sha(output_dir/f"{stem}.{ext}")==sha(temp/"rebuilt"/f"{stem}.{ext}")
                 for stem in names for ext in ("png","pdf","svg")}
    result=dict(source_returncode=source.returncode,source_output=source.stdout+source.stderr,rendered_checks=checks,
                CSV_only_rebuild=rebuild,visual_review="pending",all_checks_passed=False,
                figure_contract=dict(archetype="quantitative single-panel curves",backend="python",width_mm=182.88,
                    conclusion="Probe conditional-estimation NMSE and gain as mixture capacity increases",
                    statistics="pooled error/truth energy across 100 trajectories; paired diagnostics saved separately; no CI",exports=["PNG300dpi","editable PDF","editable SVG"],
                    uncertainty_exemption="pooled ratio, not a mean across EM seeds; seeds selected by training likelihood only"))
    (output_dir/"t15b_gmm_figure_audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    if source.returncode or any(v["text_returncode"] or v["collision_returncode"] for v in checks.values()) or not all(rebuild.values()):
        raise ValueError("figure automated gate failed; numeric CSVs preserved")
    print("T15B two figures: source/text/collision and CSV-only byte rebuild PASS; visual review pending")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input-dir",default=str(ROOT/"outputs"));p.add_argument("--output-dir",default=str(ROOT/"outputs"));p.add_argument("--audit",action="store_true")
    args=p.parse_args()
    if args.audit:audit(args.input_dir,args.output_dir)
    else:plot(args.input_dir,args.output_dir)
