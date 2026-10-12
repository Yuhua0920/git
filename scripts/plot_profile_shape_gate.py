"""T14A two CSV-only pooled-development figures with common broad NMSE scales."""
import csv
import os
import tempfile
from pathlib import Path

import numpy as np

PREFIX="t14a_profile_shape_gate"
FIGURE_NAMES=(PREFIX+"_nmse",PREFIX+"_model_class_gap")
METHODS=(("R3","R3 LS-Linear","#777777","--","o"),
    ("R4","R4 frozen","#0072B2","-.","s"),
    ("R8","R8 A-family oracle","#CC79A7","-","D"),
    ("R5","R5 profile oracle","#D55E00","-","^"),
    ("B0","B0 mixed best57","#009E73",":","v"))


def plot_results(input_dir,output_dir,require_matplotlib_panel_alignment):
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    with (Path(input_dir)/(PREFIX+".csv")).open(encoding="utf8",newline="") as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=6 or {(r["profile"],int(r["snr_db"])) for r in rows}!={(p,s) for p in ("B","C") for s in (5,10,20)}:
        raise ValueError("complete B/C x3 SNR pooled CSV required")
    with tempfile.TemporaryDirectory(prefix="t14a_mpl_",dir=output) as cache:
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
    paths=[]; records=[]
    values=[float(r[m+"_nmse_db"]) for r in rows for m,*_ in METHODS]
    if not np.all(np.isfinite(values)):
        raise ValueError("finite pooled NMSE required")
    limits=(min(-5.,np.floor(min(values)/5)*5-1),max(0.,np.ceil(max(values)/5)*5))
    with matplotlib.rc_context({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
        "font.size":8,"axes.labelsize":9,"legend.fontsize":7,"pdf.fonttype":42,"svg.fonttype":"none",
        "svg.hashsalt":"t14a-development","axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False}):
        def export(fig,name):
            fig.text(.5,.035,"200 independent trajectories/profile x 5 continuous frames; dF16/dT4; development; formal test sealed.",ha="center",fontsize=7)
            fig.canvas.draw()
            require_matplotlib_panel_alignment(fig,json_out=output/f"{name}.alignment.json",strict=True,require_panel_labels=True)
            fig.savefig(output/f"{name}.png",dpi=600)
            fig.savefig(output/f"{name}.pdf",metadata={"CreationDate":None,"ModDate":None})
            fig.savefig(output/f"{name}.svg",metadata={"Date":None})
            svg=output/f"{name}.svg"
            svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf8").splitlines())+"\n",encoding="utf8")
            paths.append(output/f"{name}.png")
            records.append({"name":name,"panels":2,"source_csv":PREFIX+".csv","no_data_exclusions":True,"common_NMSE_limits_db":limits,
                "uncertainty":"pooled error/truth energy ratios; no inferential CI; paired trajectory quartiles saved separately",
                "visual_self_check":"pending"})
            plt.close(fig)

        fig,axes=plt.subplots(1,2,sharey=True,figsize=(7.204724409,4.1))
        fig.subplots_adjust(left=.10,right=.975,bottom=.23,top=.70,wspace=.24)
        fig.suptitle("Unseen PDP shape: frozen and oracle-projected channel estimates",y=.98,fontsize=10)
        for ax,p,label in zip(axes,("B","C"),("a","b")):
            group=sorted((r for r in rows if r["profile"]==p),key=lambda r:int(r["snr_db"]))
            for m,text,color,style,marker in METHODS:
                ax.plot([int(r["snr_db"]) for r in group],[float(r[m+"_nmse_db"]) for r in group],label=text,
                    color=color,linestyle=style,marker=marker,markerfacecolor="none",linewidth=1.4,markersize=4)
            ax.set_title(f"TDL-{p}: 30 km/h, nominal 100 ns",fontsize=8.5,pad=14)
            ax.set_xticks([5,10,20]); ax.set_xlabel("SNR (dB)"); ax.set_ylim(*limits); ax.grid(alpha=.2)
            ax.text(-.04,1.07,label,transform=ax.transAxes,fontsize=9,fontweight="bold")
        axes[0].set_ylabel("Pooled data-RE NMSE (dB)")
        h,l=axes[0].get_legend_handles_labels(); fig.legend(h,l,loc="upper center",bbox_to_anchor=(.5,.925),ncol=3)
        export(fig,FIGURE_NAMES[0])

        fig,axes=plt.subplots(1,2,sharey=True,figsize=(7.204724409,4.1))
        fig.subplots_adjust(left=.10,right=.975,bottom=.23,top=.72,wspace=.24)
        fig.suptitle("20-dB model-class gap after oracle TDL-A delay adjustment",y=.97,fontsize=10)
        fig.text(.5,.875,"R4 frozen -> R8 A-family KL projection -> R5 matched profile covariance",ha="center",fontsize=8)
        colors={m:c for m,_,c,*_ in METHODS}
        for ax,p,label in zip(axes,("B","C"),("a","b")):
            r=next(r for r in rows if r["profile"]==p and int(r["snr_db"])==20)
            ys=[float(r[m+"_nmse_db"]) for m in ("R4","R8","R5")]
            ax.plot([0,1,2],ys,color=".6",linewidth=1)
            for x,m,y in zip((0,1,2),("R4","R8","R5"),ys):
                ax.scatter([x],[y],color=colors[m],s=30,zorder=3)
            residual=float(r["in_family_residual_db"])
            ax.set_title(f"TDL-{p}; in-family residual {residual:.3f} dB",fontsize=8.5,pad=14)
            ax.set_xticks([0,1,2],["R4 frozen","R8 A-family","R5 profile"])
            ax.set_xlim(-.3,2.3); ax.set_ylim(*limits); ax.grid(axis="y",alpha=.2)
            ax.text(-.04,1.07,label,transform=ax.transAxes,fontsize=9,fontweight="bold")
        axes[0].set_ylabel("Pooled data-RE NMSE (dB)")
        export(fig,FIGURE_NAMES[1])
    return paths,records
