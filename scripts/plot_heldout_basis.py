"""T16C two CSV-only measured validation figures; scoped deterministic exports."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
SKILL=Path("C:/Users/yuhua/.codex/skills/nature-figure/scripts")
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".pytest_cache/matplotlib-t16c"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment
from scripts.run_cdl_mismatch import sha
STYLE={"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "font.size":10,"legend.fontsize":10,"svg.fonttype":"none","svg.hashsalt":"T16C-fixed",
    "pdf.fonttype":42,"axes.spines.right":False,"axes.spines.top":False,"legend.frameon":False}
STEMS=("t16c_heldout_shared_closure","t16c_heldout_oracle_vs_shared")
FILES=("t16c_heldout_shared_basis.csv","t16c_heldout_oracle_lowrank.csv")
CURVES=((5,"#0072B2","o","-"),(10,"#D55E00","s","--"),(20,"#009E73","^","-."))


def plot(input_dir,output_dir):
    directory=Path(output_dir);directory.mkdir(parents=True,exist_ok=True)
    tables=[]
    for filename in FILES:
        with (Path(input_dir)/filename).open(encoding="utf8",newline="") as f: tables.append(list(csv.DictReader(f)))
    with matplotlib.rc_context(STYLE):
        for index,stem in enumerate(STEMS):
            fig,axes=plt.subplots(1,2,figsize=(7.2,4.8),sharey=True)
            fig.subplots_adjust(left=.12,right=.97,bottom=.28,top=.72,wspace=.22)
            for ax,name,letter,title in zip(axes,("CDL-C-75","CDL-B-120"),("a","b"),("CDL-C / 75 km/h","CDL-B / 120 km/h")):
                for s,color,marker,linestyle in CURVES if index==0 else CURVES[1:]:
                    v=sorted((r for r in tables[0] if r["shift"]==name and int(r["snr_db"])==s),key=lambda r:int(r["rank"]))
                    if [int(r["rank"]) for r in v]!=[4,8,16,24] or any(r["closure"]=="" for r in v):
                        raise ValueError("measured ranks/defined closures required for plot")
                    if index==0:
                        ax.plot([int(r["rank"]) for r in v],[float(r["closure"]) for r in v],color=color,
                            marker=marker,linestyle=linestyle,markerfacecolor="white",linewidth=1.5,markersize=4,label=f"{s} dB")
                    else:
                        original=next(r for r in tables[1] if r["shift"]==name and int(r["snr_db"])==s and int(r["rank"])==4)
                        values=[float(original["closure"])]+[float(r["closure"]) for r in v if int(r["rank"]) in (4,8,16)]
                        ax.bar(np.arange(4)+(-.18 if s==10 else .18),values,width=.34,color=color,label=f"{s} dB",edgecolor="white",linewidth=.4)
                ax.axhline(.8,color="0.55",linewidth=.8,linestyle="--")
                ax.set_title(title,fontsize=11,pad=12)
                if index==0: ax.set_xticks([4,8,16,24]);ax.set_xlim(3,25);ax.set_xlabel("Frozen shared output rank q")
                else:
                    ax.set_xticks(range(4),["r4\noracle","q4\nshared","q8\nshared","q16\nshared"])
                    ax.set_xlabel("Correction representation")
                ax.grid(axis="y",color="0.9",linewidth=.6);ax.set_axisbelow(True)
                ax.annotate(letter,xy=(0,1),xycoords="axes fraction",xytext=(-20,12),textcoords="offset points",fontweight="bold",fontsize=11)
            axes[0].set_ylabel("Frozen-to-matched gap closure")
            handles,labels=axes[0].get_legend_handles_labels()
            fig.legend(handles,labels,ncol=3 if index==0 else 2,loc="upper center",bbox_to_anchor=(.55,.90),columnspacing=2.)
            fig.suptitle("Frozen shared basis: held-out validation recovery" if index==0 else "Intrinsic low rank versus frozen basis coverage",fontsize=12,y=.98)
            fig.text(.12,.105,"100 dedicated validation trajectories/held-out; target means; dF=16",fontsize=9)
            fig.text(.12,.045,"Unclipped actual closure; dashed 0.8; pooled ratios, no CI",fontsize=9)
            require_matplotlib_panel_alignment(fig,json_out=str(directory/f"{stem}.alignment.json"),strict=True)
            fig.savefig(directory/f"{stem}.png",dpi=300)
            fig.savefig(directory/f"{stem}.pdf",metadata={"CreationDate":None,"ModDate":None})
            fig.savefig(directory/f"{stem}.svg",metadata={"Date":None});plt.close(fig)
    return STEMS


def audit(input_dir,output_dir):
    directory=Path(output_dir);plot(input_dir,directory);checks={}
    source=subprocess.run([sys.executable,str(SKILL/"validate_figure.py"),str(Path(__file__)),"--json"],capture_output=True,text=True)
    for stem in STEMS:
        text=subprocess.run([sys.executable,str(SKILL/"audit_pdf_text.py"),str(directory/f"{stem}.pdf"),"--min-pt","5","--json"],capture_output=True,text=True)
        collision=subprocess.run([sys.executable,str(SKILL/"audit_figure_collisions.py"),str(directory/f"{stem}.pdf"),"--json-out",str(directory/f"{stem}.collision-audit.json")],capture_output=True,text=True)
        checks[stem]=dict(text_returncode=text.returncode,collision_returncode=collision.returncode,
            text_output=text.stdout+text.stderr,collision_output=collision.stdout+collision.stderr)
    with tempfile.TemporaryDirectory(dir=ROOT/".pytest_cache") as d:
        temp=Path(d)
        for name in FILES:shutil.copy2(Path(input_dir)/name,temp)
        plot(temp,temp/"rebuilt")
        rebuilt={f"{stem}.{ext}":sha(directory/f"{stem}.{ext}")==sha(temp/"rebuilt"/f"{stem}.{ext}") for stem in STEMS for ext in ("png","pdf","svg")}
    r=dict(source_returncode=source.returncode,source_output=source.stdout+source.stderr,rendered_checks=checks,
        CSV_only_rebuild=rebuilt,visual_review="pending",all_checks_passed=False,
        figure_contract=dict(backend="python",archetype="paired quantitative grid",width_mm=182.88,
            conclusion="Separate intrinsic correction rank from frozen shared coverage on unseen environments",
            source_data=list(FILES),uncertainty="pooled ratios; trajectory dispersion saved; no CI claimed",
            exclusions="no anchors, unmeasured ranks or source means in primary panels",exports=["PNG300dpi","editable PDF","editable SVG"]))
    (directory/"t16c_heldout_figure_audit.json").write_text(json.dumps(r,indent=2)+"\n",encoding="utf8")
    if source.returncode or not all(rebuilt.values()) or any(v["text_returncode"] or v["collision_returncode"] for v in checks.values()):
        raise ValueError("figure gates failed; numeric CSV protected")
    print("T16C source/alignment/text/collision/CSV-only byte rebuild PASS; actual visual review pending")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input-dir",default=str(ROOT/"outputs"))
    p.add_argument("--output-dir",default=str(ROOT/"outputs"));p.add_argument("--audit",action="store_true");a=p.parse_args()
    if a.audit:audit(a.input_dir,a.output_dir)
    else:plot(a.input_dir,a.output_dir)
