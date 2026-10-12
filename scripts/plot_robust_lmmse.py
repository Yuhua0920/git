"""CSV-only fixed HR comparison: ID cost, OOD benefit and unclipped closure."""
import csv
import os
import tempfile
from pathlib import Path

import numpy as np

PREFIX="t13b_robust_lmmse"
FIGURE_NAMES=(PREFIX+"_nmse",PREFIX+"_closure")


def plot_results(input_dir,output_dir,require_matplotlib_panel_alignment):
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    with (Path(input_dir)/f"{PREFIX}.csv").open(encoding="utf8",newline="") as f:
        rows=list(csv.DictReader(f))
    expected={(c,s) for c in ("ID","speed120","DS300") for s in (5,10,20)}
    if len(rows)!=9 or {(r["condition"],int(r["snr_db"])) for r in rows}!=expected:
        raise ValueError("all nine condition/SNR observations required")
    with tempfile.TemporaryDirectory(prefix="t13b_mpl_",dir=output) as cache:
        old=os.environ.get("MPLCONFIGDIR"); os.environ["MPLCONFIGDIR"]=cache
        try:
            return render(rows,output,require_matplotlib_panel_alignment)
        finally:
            if old is None:
                os.environ.pop("MPLCONFIGDIR",None)
            else:
                os.environ["MPLCONFIGDIR"]=old


def render(rows,output,require_matplotlib_panel_alignment):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths,records=[],[]
    methods=(("R4","R4 frozen","#CC79A7","-.","s"),
        ("R6_HR","R6-HR fixed robust","#0072B2","-","o"),
        ("R5","R5 matched Oracle","#D55E00",":","^"))
    def group(condition):
        return sorted((r for r in rows if r["condition"]==condition),key=lambda r:int(r["snr_db"]))
    def values(part,column):
        data=np.array([float(r[column]) for r in part])
        if not np.all(np.isfinite(data)):
            raise ValueError("finite source data required")
        return data
    with matplotlib.rc_context({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
        "font.size":8,"axes.labelsize":9,"legend.fontsize":8,"pdf.fonttype":42,"svg.fonttype":"none",
        "svg.hashsalt":"t13b-development","axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False}):
        def export(fig,name,panels):
            fig.text(.5,.035,"ID validation + OOD development; dF16/dT4; formal test sealed.",ha="center",fontsize=7)
            fig.canvas.draw()
            require_matplotlib_panel_alignment(fig,json_out=output/f"{name}.alignment.json",strict=True,require_panel_labels=panels>1)
            fig.savefig(output/f"{name}.png",dpi=600)
            fig.savefig(output/f"{name}.pdf",metadata={"CreationDate":None,"ModDate":None})
            fig.savefig(output/f"{name}.svg",metadata={"Date":None})
            # Matplotlib emits whitespace at the end of SVG path lines. Retain
            # newline token separators while making generated text diff-clean.
            svg_path=output/f"{name}.svg"
            svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf8").splitlines())+"\n",encoding="utf8")
            paths.append(output/f"{name}.png")
            records.append({"name":name,"panels":panels,"source_csv":PREFIX+".csv","no_data_exclusions":True,
                "uncertainty":"none: pooled descriptive validation/development; independent unit is trajectory",
                "visual_self_check":"pending"})
            plt.close(fig)
        fig,axes=plt.subplots(1,3,sharey=True,figsize=(7.204724409,3.8))
        fig.subplots_adjust(left=.09,right=.975,bottom=.23,top=.73,wspace=.30)
        fig.suptitle("Fixed robust prior: ID cost and OOD recovery",y=.97,fontsize=10)
        all_y=values([{ "v":r[f"{m}_nmse_db"]} for r in rows for m,*_ in methods],"v")
        limits=(np.floor(all_y.min()/5)*5-1,max(0,np.ceil(all_y.max()/5)*5))
        for ax,c,title,label in zip(axes,("ID","speed120","DS300"),
                ("ID: 30 km/h, 100 ns","Speed: 120 km/h, 100 ns","Delay: 30 km/h, 300 ns"),("a","b","c")):
            part=group(c); snr=values(part,"snr_db")
            for m,text,color,style,marker in methods:
                ax.plot(snr,values(part,f"{m}_nmse_db"),label=text,color=color,linestyle=style,marker=marker,
                    markerfacecolor="none",linewidth=1.5,markersize=4)
            ax.set_title(title,fontsize=7.5,pad=12); ax.set_xlabel("SNR (dB)"); ax.set_xticks(snr)
            ax.set_ylim(*limits); ax.grid(alpha=.2)
            ax.text(-.05,1.04,label,transform=ax.transAxes,fontsize=9,fontweight="bold")
        axes[0].set_ylabel("Pooled data-RE NMSE (dB)")
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.52,.92),ncol=3)
        export(fig,FIGURE_NAMES[0],3)
        fig,ax=plt.subplots(figsize=(7.204724409,3.8))
        fig.subplots_adjust(left=.12,right=.97,bottom=.23,top=.73)
        fig.suptitle("Unclipped recovery / frozen R4-to-Oracle gap",y=.97,fontsize=10)
        for c,text,color,marker in (("speed120","Speed shift","#0072B2","o"),("DS300","Delay shift","#D55E00","s")):
            part=group(c)
            ax.plot(values(part,"snr_db"),values(part,"robust_closure"),label=text,color=color,
                marker=marker,markerfacecolor="none",linewidth=1.5,markersize=5)
        ax.axhline(.25,color="#777777",linestyle=":",linewidth=.8)
        ax.axhline(.50,color="#777777",linestyle="--",linewidth=.8)
        ax.axhline(0,color="#999999",linewidth=.6)
        ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Robust closure C = Q / G")
        ax.set_xticks([5,10,20]); ax.grid(alpha=.2)
        ax.legend(loc="lower center",bbox_to_anchor=(.5,1.05),ncol=2)
        export(fig,FIGURE_NAMES[1],1)
    return paths,records
