"""T16D two CSV-only figures: equal-budget union and real scalar capacity."""
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
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".pytest_cache/matplotlib-t16d"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment
from scripts.run_cdl_mismatch import sha
STYLE={"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
       "font.size":10,"legend.fontsize":10,"svg.fonttype":"none","svg.hashsalt":"T16D-fixed",
       "pdf.fonttype":42,"axes.spines.right":False,"axes.spines.top":False,"legend.frameon":False}
STEMS=("t16d_modular_union_closure","t16d_scalar_composition_nmse")
FILES=("t16d_modular_union.csv","t16d_scalar_composition.csv")
CURVES=((5,"#0072B2","o"),(10,"#D55E00","s"),(20,"#009E73","^"))


def plot(input_dir,output_dir):
    directory=Path(output_dir);directory.mkdir(parents=True,exist_ok=True)
    tables=[]
    for filename in FILES:
        with (Path(input_dir)/filename).open(encoding="utf8",newline="") as f:tables.append(list(csv.DictReader(f)))
    with matplotlib.rc_context(STYLE):
        for index,stem in enumerate(STEMS):
            if index==0:
                fig,ax=plt.subplots(figsize=(7.2,4.8));axes=[ax]
                fig.subplots_adjust(left=.12,right=.97,bottom=.28,top=.68)
                for s,color,marker in CURVES:
                    values=sorted((r for r in tables[0] if r["shift"]=="CDL-B-120" and int(r["snr_db"])==s),key=lambda r:int(r["dimension_budget"]))
                    if [int(r["dimension_budget"]) for r in values]!=[8,16,24]:raise ValueError("complete equal-budget comparisons required")
                    for key,label,style in (("generic_closure","Shared U","--"),("closure","Modular Q","-")):
                        if any(r[key]=="" for r in values):raise ValueError("defined actual closure required")
                        ax.plot([8,16,24],[float(r[key]) for r in values],color=color,marker=marker,
                            linestyle=style,markerfacecolor="white" if key=="generic_closure" else color,
                            linewidth=1.4,markersize=4,label=f"{s} dB: {label}")
                ax.axhline(.8,color="0.55",linewidth=.8,linestyle=":")
                ax.set_xticks([8,16,24]);ax.set_xlim(7,25);ax.set_xlabel("Output dimension budget")
                ax.set_ylabel("Actual frozen-to-matched gap closure")
                ax.set_title("CDL-B / 120 km/h: equal-dimension representation",fontsize=11,pad=12)
                fig.suptitle("Generic shared basis versus development-module union",fontsize=12,y=.98)
                footer="Unclipped pooled closure; dotted 0.8; solid Q / dashed U; no CI"
            else:
                fig,axes=plt.subplots(1,2,figsize=(7.2,4.8),sharey=True)
                fig.subplots_adjust(left=.12,right=.97,bottom=.28,top=.72,wspace=.22)
                for ax,name,letter,title in zip(axes,("CDL-C-75","CDL-B-120"),("a","b"),("CDL-C / 75 km/h","CDL-B / 120 km/h")):
                    values=sorted((r for r in tables[1] if r["shift"]==name and int(r["snr_db"]) in (10,20)),key=lambda r:int(r["snr_db"]))
                    if [int(r["snr_db"]) for r in values]!=[10,20]:raise ValueError("both primary SNRs required")
                    for key,label,color,marker,style in (("source_nmse_db","Source","0.45","s","--"),
                        ("nmse_db","Real scalar mode","#0072B2","o","-"),("matched_nmse_db","Matched","#D55E00","^","-.")):
                        ax.plot([10,20],[float(r[key]) for r in values],color=color,marker=marker,linestyle=style,
                            linewidth=1.5,markersize=4,label=label)
                    ax.set_xticks([10,20]);ax.set_xlim(9,21);ax.set_xlabel("Source deployment SNR label (dB)")
                    ax.set_title(title,fontsize=11,pad=12)
                    ax.annotate(letter,xy=(0,1),xycoords="axes fraction",xytext=(-20,12),textcoords="offset points",fontweight="bold",fontsize=11)
                axes[0].set_ylabel("Actual validation NMSE (dB)")
                fig.suptitle("Oracle real scalar composition capacity",fontsize=12,y=.98)
                footer="Target-weighted real coefficients; target means; pooled NMSE; no CI"
            for ax in axes:ax.grid(axis="y",color="0.9",linewidth=.6);ax.set_axisbelow(True)
            handles,labels=axes[0].get_legend_handles_labels()
            fig.legend(handles,labels,ncol=3,loc="upper center",bbox_to_anchor=(.55,.91),columnspacing=1.4)
            fig.text(.12,.105,"100 frozen T16C validation trajectories/held-out; 10 frames; dF=16",fontsize=9)
            fig.text(.12,.045,footer,fontsize=9)
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
        figure_contract=dict(backend="python",archetype="quantitative comparison",
            conclusion="Test module unions at equal budgets, then the stronger real scalar composition hypothesis",
            source_data=list(FILES),uncertainty="pooled ratios; trajectory dispersion saved; no CI claimed",
            exclusions="figure1 B120 only; figure2 10/20 only, as task specifies",
            export_width_mm=182.88,exports=["PNG300dpi","editable PDF","editable SVG"],
            adaptation="style-only inheritance; new data schema and scientific comparisons"))
    (directory/"t16d_modular_figure_audit.json").write_text(json.dumps(r,indent=2)+"\n",encoding="utf8")
    if source.returncode or not all(rebuilt.values()) or any(v["text_returncode"] or v["collision_returncode"] for v in checks.values()):
        raise ValueError("figure gates failed; numerical CSVs protected")
    print("T16D source/alignment/text/collision/CSV-only byte rebuild PASS; actual visual review pending")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input-dir",default=str(ROOT/"outputs"))
    p.add_argument("--output-dir",default=str(ROOT/"outputs"));p.add_argument("--audit",action="store_true");a=p.parse_args()
    if a.audit:audit(a.input_dir,a.output_dir)
    else:plot(a.input_dir,a.output_dir)
