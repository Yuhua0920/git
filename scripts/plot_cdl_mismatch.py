"""T15C two CSV-only quantitative figures; scoped style and deterministic export."""
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
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".pytest_cache/matplotlib-t15c"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0,str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment
STYLE={"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
       "font.size":10,"legend.fontsize":10,"svg.fonttype":"none","svg.hashsalt":"T15C-fixed",
       "pdf.fonttype":42,"axes.spines.right":False,"axes.spines.top":False,"legend.frameon":False}
CURVES={"R3":("#777777","o","--","R3 LS-Linear"),
        "R10-C-frozen":("#D55E00","s","-","R10 C-frozen"),
        "R10-matched":("#0072B2","^","-.","R10 Matched")}
SHIFTS=("CDL-B-30","CDL-C-120")
STEMS=("t15c_cdl_mismatch_nmse","t15c_cdl_mismatch_penalty")


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(fig,directory,stem):
    require_matplotlib_panel_alignment(fig,json_out=str(directory/f"{stem}.alignment.json"),strict=True)
    fig.savefig(directory/f"{stem}.png",dpi=300)
    fig.savefig(directory/f"{stem}.pdf",metadata={"CreationDate":None,"ModDate":None})
    fig.savefig(directory/f"{stem}.svg",metadata={"Date":None})
    plt.close(fig)


def plot(input_dir,output_dir):
    directory=Path(output_dir);directory.mkdir(parents=True,exist_ok=True)
    with (Path(input_dir)/"t15c_cdl_mismatch.csv").open(encoding="utf8") as f:rows=list(csv.DictReader(f))
    with matplotlib.rc_context(STYLE):
        fig,axes=plt.subplots(1,2,figsize=(7.2,4.8),sharey=True)
        fig.subplots_adjust(left=.12,right=.97,bottom=.28,top=.72,wspace=.22)
        for ax,name,letter,title in zip(axes,SHIFTS,("a","b"),("CDL-B / 30 km/h","CDL-C / 120 km/h")):
            for method,(color,marker,linestyle,label) in CURVES.items():
                chosen=sorted((r for r in rows if r["shift"]==name and r["method"]==method),key=lambda r:float(r["snr_db"]))
                ax.plot([float(r["snr_db"]) for r in chosen],[float(r["nmse_db"]) for r in chosen],
                    color=color,marker=marker,linestyle=linestyle,markerfacecolor="white",linewidth=1.5,label=label)
            ax.set_title(title,fontsize=11,pad=12);ax.set_xticks([5,10,20]);ax.set_xlabel("Source-reference SNR (dB)")
            ax.grid(axis="y",color="0.9",linewidth=.6);ax.set_axisbelow(True)
            ax.annotate(letter,xy=(0,1),xycoords="axes fraction",xytext=(-20,12),textcoords="offset points",fontweight="bold",fontsize=11)
        axes[0].set_ylabel("Pooled data-RE NMSE (dB)")
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,ncol=3,loc="upper center",bbox_to_anchor=(.55,.90),columnspacing=1.)
        fig.suptitle("Frozen CDL-C statistics under two environment shifts",fontsize=12,y=.98)
        fig.text(.12,.105,"100 validation trajectories per shift; 10 continuous frames",fontsize=9)
        fig.text(.12,.045,"dF=16; pooled energy ratios, no CI; source noise reference fixed",fontsize=9)
        export(fig,directory,STEMS[0])
        fig,ax=plt.subplots(figsize=(7.2,4.8));fig.subplots_adjust(left=.15,right=.97,bottom=.28,top=.76)
        for name,color,marker,label in zip(SHIFTS,("#0072B2","#D55E00"),("o","s"),("CDL-B / 30 km/h","CDL-C / 120 km/h")):
            selected={}
            for r in rows:
                if r["shift"]==name:selected[float(r["snr_db"]),r["method"]]=r
            x=[5,10,20];y=[float(selected[s,"R10-C-frozen"]["nmse_db"])-float(selected[s,"R10-matched"]["nmse_db"]) for s in x]
            ax.plot(x,y,color=color,marker=marker,markerfacecolor="white",linewidth=1.5,label=label)
        ax.axhline(2.,color="0.65",linewidth=.8,linestyle=":")
        ax.axhline(0.,color="0.65",linewidth=.8)
        ax.set_xticks([5,10,20]);ax.set_xlabel("Source-reference SNR (dB)")
        ax.set_ylabel("Frozen - matched NMSE penalty (dB)")
        ax.grid(axis="y",color="0.9",linewidth=.6);ax.set_axisbelow(True)
        handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,ncol=2,loc="upper center",bbox_to_anchor=(.56,.91))
        fig.suptitle("Empirical-LMMSE covariance mismatch penalty",fontsize=12,y=.98)
        fig.text(.15,.105,"Positive values favour matched statistics; dotted line = 2 dB",fontsize=9)
        fig.text(.15,.045,"100 paired trajectories/shift; pooled ratio, no CI",fontsize=9)
        export(fig,directory,STEMS[1])
    return STEMS


def audit(input_dir,output_dir):
    directory=Path(output_dir);names=plot(input_dir,directory);checks={}
    source=subprocess.run([sys.executable,str(SKILL/"validate_figure.py"),str(Path(__file__)),"--json"],capture_output=True,text=True)
    for stem in names:
        text=subprocess.run([sys.executable,str(SKILL/"audit_pdf_text.py"),str(directory/f"{stem}.pdf"),"--min-pt","5","--json"],capture_output=True,text=True)
        collision=subprocess.run([sys.executable,str(SKILL/"audit_figure_collisions.py"),str(directory/f"{stem}.pdf"),"--json-out",str(directory/f"{stem}.collision-audit.json")],capture_output=True,text=True)
        checks[stem]=dict(text_returncode=text.returncode,collision_returncode=collision.returncode,
                          text_output=text.stdout+text.stderr,collision_output=collision.stdout+collision.stderr)
    with tempfile.TemporaryDirectory(dir=ROOT/".pytest_cache") as d:
        temp=Path(d);shutil.copy2(Path(input_dir)/"t15c_cdl_mismatch.csv",temp)
        plot(temp,temp/"rebuilt")
        rebuilt={f"{stem}.{ext}":sha(directory/f"{stem}.{ext}")==sha(temp/"rebuilt"/f"{stem}.{ext}") for stem in names for ext in ("png","pdf","svg")}
    result=dict(source_returncode=source.returncode,source_output=source.stdout+source.stderr,
        rendered_checks=checks,CSV_only_rebuild=rebuilt,visual_review="pending",all_checks_passed=False,
        figure_contract=dict(backend="python",archetype="quantitative grid",width_mm=182.88,
            conclusion="Compare profile and mobility covariance mismatch against independent target-training statistics",
            uncertainty="pooled energy ratios; trajectory dispersion saved separately, no confidence interval",
            source_data="t15c_cdl_mismatch.csv",excluded_rows=0,exports=["PNG300dpi","editable PDF","editable SVG"]))
    (directory/"t15c_cdl_figure_audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    if source.returncode or any(v["text_returncode"] or v["collision_returncode"] for v in checks.values()) or not all(rebuilt.values()):
        raise ValueError("figure gate failed; numeric CSV preserved")
    print("T15C automated figure source/alignment/text/collision/CSV-only byte rebuild PASS; visual review pending")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input-dir",default=str(ROOT/"outputs"));p.add_argument("--output-dir",default=str(ROOT/"outputs"));p.add_argument("--audit",action="store_true")
    a=p.parse_args()
    if a.audit:audit(a.input_dir,a.output_dir)
    else:plot(a.input_dir,a.output_dir)
