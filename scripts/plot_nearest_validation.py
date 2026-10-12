"""CSV-only R0-R5 validation figures, stratified by pilot density.

Contract: pooled data NMSE / integer-error BER / correct-bit-rate proxy.
Raw SNR points, no smoothing or exclusions. R1 is pilot-only and never plotted.
No intervals: this incremental replay reports descriptive pooled ratios only.
Same method encodings and axes across the four density strata in each figure.
Style adapted from the project's accepted T10D Python figures.
"""

import os
import tempfile
from pathlib import Path

import numpy as np

from src.nearest_validation import read_csv

FIGURE_NAMES = ("t11b_r0_r5_nmse_validation","t11b_r0_r5_ber_validation",
    "t11b_r0_r5_correct_bit_rate_validation")
METHOD_STYLE = {
    "r0":("R0 Perfect","#4C4C4C",":","^"),
    "r2":("R2 Nearest","#009E73","-","v"),
    "r3":("R3 Linear","#8A8A8A","--","D"),
    "r4":("R4 Train","#0072B2","-","o"),
    "r5":("R5 Oracle","#D55E00","--","s"),
}
LABEL = {"r0":"perfect","r2":"nearest","r3":"linear","r4":"train","r5":"oracle"}


def read_plot_data(csv_path,expected_trajectories=400):
    rows = read_csv(csv_path)
    if (len(rows)!=20 or {(int(r["dF"]),int(r["snr_db"])) for r in rows}
            !={(df,snr) for df in (2,4,8,16) for snr in (0,5,10,15,20)}):
        raise ValueError("twenty unique raw conditions required; no curve exclusions")
    if any(int(r["num_trajectories"])!=expected_trajectories or int(r["num_correlated_frames"])!=10*expected_trajectories
            or r["r1_data_ber"]!="N/A" for r in rows):
        raise ValueError("matching trajectory metadata and R1 BER N/A required")
    return {df:sorted((r for r in rows if int(r["dF"])==df),key=lambda r:int(r["snr_db"])) for df in (2,4,8,16)}


def plot_results(csv_path,output_dir,require_matplotlib_panel_alignment,expected_trajectories=400):
    with tempfile.TemporaryDirectory(prefix="t11b_matplotlib_") as cache:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            return _plot_results(csv_path,output_dir,require_matplotlib_panel_alignment,expected_trajectories)
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR",None)
            else:
                os.environ["MPLCONFIGDIR"] = previous


def _plot_results(csv_path,output_dir,require_matplotlib_panel_alignment,n):
    groups = read_plot_data(csv_path,n)
    output_dir = Path(output_dir); output_dir.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    with matplotlib.rc_context({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
        "font.size":9,"pdf.fonttype":42,"svg.fonttype":"none","svg.hashsalt":"t11b-validation",
        "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.8,"legend.frameon":False}):
        paths,qa = [],[]
        for name,metric in zip(FIGURE_NAMES,("nmse","ber","proxy")):
            methods = ("r2","r3","r4","r5") if metric=="nmse" else ("r0","r2","r3","r4","r5")
            title,ylabel = {
                "nmse":("Held-out ID: R2-R5 channel estimation","Data-RE NMSE (dB)"),
                "ber":("Held-out ID: R0-R5 data detection with ZF","Data-bit BER"),
                "proxy":("Held-out ID: correct-bit-rate proxy","Correct-bit-rate proxy (Mbit/s)"),
            }[metric]
            fig,axes = plt.subplots(2,2,sharex=True,sharey=True,figsize=(7.2,5.8))
            fig.subplots_adjust(left=0.12,right=0.98,bottom=0.21,top=0.77,wspace=0.16,hspace=0.36)
            fig.suptitle(title,y=0.98,fontsize=12)
            handles = [Line2D([0],[0],color=METHOD_STYLE[m][1],linestyle=METHOD_STYLE[m][2],
                marker=METHOD_STYLE[m][3],markerfacecolor="white" if m=="r5" else METHOD_STYLE[m][1],
                label=METHOD_STYLE[m][0]) for m in methods]
            fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(0.5,0.935),ncols=3,fontsize=9)
            fig.supxlabel("SNR (dB)",y=0.12,fontsize=10)
            fig.supylabel(ylabel,x=0.018,fontsize=10)
            zeros = 0
            for index,(axis,(df,group)) in enumerate(zip(axes.flat,groups.items())):
                axis.set(xlim=(-0.5,20.5),xticks=(0,5,10,15,20))
                axis.grid(alpha=0.18)
                axis.set_title(f"dF={df}, pilot ratio=1/{round(1/float(group[0]['pilot_ratio']))}",fontsize=10,pad=8)
                axis.annotate(chr(97+index),(0,1),xycoords="axes fraction",xytext=(-12,12),
                    textcoords="offset points",fontsize=10,fontweight="bold",annotation_clip=False)
                for method in methods:
                    field = f"{method}_{LABEL[method]}_nmse_db" if metric=="nmse" else (
                        f"{method}_{LABEL[method]}_ber" if metric=="ber" else f"{method}_proxy_bps")
                    x = np.array([float(r["snr_db"]) for r in group])
                    y = np.array([float(r[field]) for r in group])/(1e6 if metric=="proxy" else 1)
                    if not np.all(np.diff(x)>0) or not np.all(np.isfinite(y)):
                        raise ValueError("finite raw values and increasing SNR required")
                    zero = np.zeros(5,dtype=bool)
                    if metric=="ber":
                        if np.any(y<0) or np.any(y>1):
                            raise ValueError("raw BER must be in [0,1]")
                        zero = y==0
                        y[zero] = [0.5/int(group[i]["total_data_bits"]) for i in np.flatnonzero(zero)]
                        if np.any(y<=0):
                            raise ValueError("log BER display must be positive")
                    _,color,style,marker = METHOD_STYLE[method]
                    axis.plot(x,y,color=color,linestyle=style,marker=marker,linewidth=1.35,markersize=4,
                        markerfacecolor="white" if method=="r5" else color)
                    if np.any(zero):
                        axis.scatter(x[zero],y[zero],marker="v",color=color,s=24)
                    zeros += int(zero.sum())
                if metric=="ber":
                    axis.set_yscale("log")
            caption = {"nmse":"Full-grid estimators only; R1 pilot-domain MSE is not on this axis.",
                "ber":"R1 data BER is N/A; all plotted receivers share the frozen random experiment.",
                "proxy":"Proxy = data bits/frame x (1 - BER) / frame duration including CP."}[metric]
            if zeros:
                caption = "Zero BER: downward markers at 0.5/total bits; raw CSV remains zero."
            fig.text(0.5,0.065,caption,ha="center",fontsize=8)
            fig.text(0.5,0.03,f"{n} independent trajectories; 10 correlated frames each; pooled ratios; no intervals.",ha="center",fontsize=8)
            fig.canvas.draw()
            require_matplotlib_panel_alignment(fig,json_out=output_dir/f"{name}.alignment.json",
                strict=True,require_panel_labels=True)
            path = output_dir/f"{name}.png"
            # PNG is a 300-dpi preview; PDF/SVG preserve editable text.
            fig.savefig(path,dpi=300)
            fig.savefig(output_dir/f"{name}.pdf",metadata={"CreationDate":None,"ModDate":None})
            fig.savefig(output_dir/f"{name}.svg",metadata={"Date":None})
            paths.append(path)
            qa.append({"name":name,"source_csv":Path(csv_path).name,"panels":4,"methods":list(methods),
                "raw_zero_BER_points":zeros,"no_smoothing":True,"no_data_exclusions":True,
                "uncertainty":"none: descriptive pooled ratios; no interval or significance claim",
                "panel_role":"four pilot-density strata of the same receiver comparison"})
            plt.close(fig)
    return paths,qa
