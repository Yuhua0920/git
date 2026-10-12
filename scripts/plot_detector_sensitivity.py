"""T11C-B CSV-only figures: bounded detector sensitivity and conditioning.

Figure contract: quantitative grid, six fixed pilot/SNR strata of paired pooled
BER; a separate Perfect-CSI condition-bin mechanism diagnostic. All trajectories
and REs retained. No smoothing, intervals or significance statements. Style-only
inheritance from accepted project Python figures; source fields are new.
"""

import csv
import os
from pathlib import Path
import tempfile

import numpy as np

FIGURE_NAMES = ("t11c_b_zf_mmse_sensitivity", "t11c_b_condition_number_detector_sensitivity")
COLORS = ("#0072B2", "#D55E00")
LABELS = ("ZF", "Linear-MMSE")
CONDITIONS = tuple((df,snr) for df in (2,16) for snr in (0,10,20))
BIN_LABELS = ("[1,2)","[2,4)","[4,8)","[8,16)","[16,inf)")


def read_rows(path):
    with Path(path).open(encoding="utf8",newline="") as stream:
        return list(csv.DictReader(stream))


def display_ber(rows,field):
    raw=np.array([float(r[field]) for r in rows])
    bits=np.array([int(r.get("total_bits",r.get("num_bits"))) for r in rows])
    if np.any(raw<0) or np.any(raw>1) or not np.all(np.isfinite(raw)) or np.any(bits<=0):
        raise ValueError("finite raw BER and positive bit denominators required")
    zero=raw==0
    y=raw.copy(); y[zero]=0.5/bits[zero]
    if np.any(y<=0): raise ValueError("log display values must be positive")
    return y,zero


def plot_results(aggregate_csv,bins_csv,output_dir,require_matplotlib_panel_alignment):
    rows=read_rows(aggregate_csv); bins=read_rows(bins_csv)
    expected={(df,snr,e) for df,snr in CONDITIONS for e in ("R0","R3","R4")}
    if len(rows)!=18 or {(int(r["dF"]),int(r["snr_db"]),r["estimator"]) for r in rows}!=expected:
        raise ValueError("exact eighteen fixed-condition estimator rows required")
    n=int(rows[0]["num_trajectories"])
    if any(int(r["num_trajectories"])!=n or int(r["frames_per_trajectory"])!=10 for r in rows):
        raise ValueError("matching independent trajectory metadata required")
    if len(bins)!=5 or tuple(r["condition_bin"] for r in bins)!=BIN_LABELS:
        raise ValueError("five frozen condition bins required")
    perfect=next(r for r in rows if (int(r["dF"]),int(r["snr_db"]),r["estimator"])==(16,10,"R0"))
    if sum(int(r["num_RE"]) for r in bins)*4!=int(perfect["total_bits"]):
        raise ValueError("all df16 data REs must remain in source bins")
    with tempfile.TemporaryDirectory(prefix="t11c_b_mpl_") as cache:
        previous=os.environ.get("MPLCONFIGDIR"); os.environ["MPLCONFIGDIR"]=cache
        try:
            return _render(rows,bins,n,aggregate_csv,bins_csv,output_dir,require_matplotlib_panel_alignment)
        finally:
            if previous is None: os.environ.pop("MPLCONFIGDIR",None)
            else: os.environ["MPLCONFIGDIR"]=previous


def _render(rows,bins,n,aggregate_csv,bins_csv,output_dir,require_matplotlib_panel_alignment):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    directory=Path(output_dir); directory.mkdir(exist_ok=True)
    paths=[]; records=[]
    with matplotlib.rc_context({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
        "font.size":9,"pdf.fonttype":42,"svg.fonttype":"none","svg.hashsalt":"t11c-b-sensitivity",
        "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.8,"legend.frameon":False}):
        handles=[Line2D([0],[0],color=c,marker=m,linestyle="none",label=l,
            markerfacecolor="white" if i else c) for i,(c,m,l) in enumerate(zip(COLORS,("o","s"),LABELS))]
        for name in FIGURE_NAMES:
            main=name==FIGURE_NAMES[0]; zero_count=0
            fig,axes=plt.subplots(2,3,sharey=True,figsize=(7.2,5.7)) if main else plt.subplots(figsize=(7.2,4.5))
            fig.subplots_adjust(left=0.12,right=0.98,bottom=0.24,top=0.78,
                                **({"wspace":0.22,"hspace":0.48} if main else {}))
            fig.suptitle("Held-out detector sensitivity" if main else "Perfect CSI: condition-number diagnostic",y=0.975,fontsize=12)
            fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(0.5,0.935),ncols=2,fontsize=9)
            if main:
                display=[display_ber(rows,f)[0] for f in ("zf_ber","mmse_ber")]
                limits=(min(v.min() for v in display)*0.6,max(v.max() for v in display)*1.7)
                for index,(axis,(df,snr)) in enumerate(zip(axes.flat,CONDITIONS)):
                    group=[next(r for r in rows if (int(r["dF"]),int(r["snr_db"]),r["estimator"])==(df,snr,e)) for e in ("R0","R3","R4")]
                    axis.set(title=f"dF={df}, SNR={snr} dB",xticks=(0,1,2),xticklabels=("R0","R3","R4"),xlim=(-0.4,2.4),ylim=limits)
                    axis.annotate(chr(97+index),(0,1),xycoords="axes fraction",xytext=(-10,12),textcoords="offset points",
                                  fontsize=10,fontweight="bold",annotation_clip=False)
                    for i,(field,color,marker) in enumerate(zip(("zf_ber","mmse_ber"),COLORS,("o","s"))):
                        y,zero=display_ber(group,field); xpos=np.arange(3)+(-0.08 if i==0 else 0.08)
                        axis.plot(xpos,y,linestyle="none",color=color,marker=marker,markersize=5,
                                  markerfacecolor="white" if i else color)
                        if np.any(zero): axis.scatter(xpos[zero],y[zero],marker="v",color=color,s=22)
                        zero_count+=int(zero.sum())
                    axis.set_yscale("log"); axis.grid(axis="y",alpha=0.18)
                fig.supylabel("Data-bit BER",x=0.018,fontsize=10)
                fig.supxlabel("Channel estimator: R0 Perfect, R3 Linear, R4 Train",y=0.135,fontsize=9)
            else:
                axis=axes
                axis.set(xticks=np.arange(5),xticklabels=BIN_LABELS,xlim=(-0.25,4.25),ylabel="Data-bit BER",xlabel="Condition number of true H on data REs")
                for i,(field,color,marker) in enumerate(zip(("zf_ber","mmse_ber"),COLORS,("o","s"))):
                    y,zero=display_ber(bins,field)
                    axis.plot(np.arange(5),y,color=color,marker=marker,linewidth=1.3,markersize=5,
                              markerfacecolor="white" if i else color)
                    if np.any(zero): axis.scatter(np.arange(5)[zero],y[zero],marker="v",color=color,s=22)
                    zero_count+=int(zero.sum())
                axis.set_yscale("log"); axis.grid(axis="y",alpha=0.18)
                axis.set_title("dF=16, SNR=10 dB; all data REs retained",fontsize=10,pad=10)
            fig.text(0.5,0.072,f"{n} independent trajectories; 10 correlated frames each; pooled BER; no intervals.",ha="center",fontsize=8)
            fig.text(0.5,0.03,"Zero errors: downward marks at 0.5/total bits; raw CSV remains zero." if zero_count else
                     "Validation sensitivity only; main detector remains ZF.",ha="center",fontsize=8)
            fig.canvas.draw()
            require_matplotlib_panel_alignment(fig,json_out=directory/f"{name}.alignment.json",
                                                strict=True,require_panel_labels=main)
            path=directory/f"{name}.png"
            fig.savefig(path,dpi=300)
            fig.savefig(directory/f"{name}.pdf",metadata={"CreationDate":None,"ModDate":None})
            fig.savefig(directory/f"{name}.svg",metadata={"Date":None})
            plt.close(fig); paths.append(path)
            records.append({"name":name,"source_csv":Path(aggregate_csv if main else bins_csv).name,
                "panels":6 if main else 1,"raw_zero_BER_points":zero_count,"all_observations_retained":True,
                "uncertainty":"none; descriptive validation, no CI or significance claims",
                "panel_roles":"six fixed pilot/SNR strata" if main else "Perfect-CSI conditioning mechanism"})
    return paths,records
