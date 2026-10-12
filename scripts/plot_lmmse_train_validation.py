"""CSV-only T10D figures: pooled validation ratios, raw points, no intervals.

Figure contract: describe the four receivers across fixed pilot-density strata;
show the small paired Train/Oracle NMSE difference without a winner claim.
Panels stratify one comparison by density and share axes/method encodings.
Uncertainty exemption: this integration task explicitly excludes CI/significance.
Style-only adaptation of the existing project plotting workflow; no sampling.
"""

import csv
import os
import tempfile
from pathlib import Path

import numpy as np

FIGURE_NAMES = ("t10d_nmse_baseline_comparison","t10d_ber_baseline_comparison",
    "t10d_train_oracle_gap","t10d_correct_bit_rate_comparison")
METHOD_STYLE = {"perfect":("Perfect CSI","#4C4C4C",":","^"),
    "ls":("LS-LIN","#8A8A8A","--","D"),
    "train":("LMMSE-Train","#0072B2","-","o"),
    "oracle":("LMMSE-Oracle","#D55E00","--","s")}


def read_plot_data(csv_path):
    with Path(csv_path).open(encoding="utf-8",newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {(df,snr) for df in (2,4,8,16) for snr in (0,5,10,15,20)}
    if len(rows)!=20 or {(int(r["dF"]),int(r["snr_db"])) for r in rows}!=expected:
        raise ValueError("all 20 unique CSV conditions are required; no curve exclusions")
    if any(int(r["num_trajectories"])!=400 or int(r["num_correlated_frames"])!=4000 for r in rows):
        raise ValueError("figure source must describe 400 independent trajectories / 4000 correlated frames")
    return {df:sorted((r for r in rows if int(r["dF"])==df),key=lambda r:int(r["snr_db"])) for df in (2,4,8,16)}


def plot_results(csv_path,output_dir,require_matplotlib_panel_alignment):
    # Set a writable cache before the first Matplotlib import on Windows.
    with tempfile.TemporaryDirectory(prefix="t10d_matplotlib_") as cache:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            return _plot_results(csv_path,output_dir,require_matplotlib_panel_alignment)
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR",None)
            else:
                os.environ["MPLCONFIGDIR"] = previous


def _plot_results(csv_path,output_dir,require_matplotlib_panel_alignment):
    groups = read_plot_data(csv_path)
    output_dir = Path(output_dir); output_dir.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    with matplotlib.rc_context({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
        "font.size":9,"pdf.fonttype":42,"svg.fonttype":"none","svg.hashsalt":"t10d-validation",
        "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.8,"legend.frameon":False}):
        paths,qa = [],[]

        def finish(fig,name,panels,caption,zero_count=0):
            fig.text(0.5,0.065,caption,ha="center",fontsize=8)
            fig.text(0.5,0.03,"400 independent trajectories; 10 correlated frames each; pooled ratios; no intervals.",ha="center",fontsize=8)
            fig.canvas.draw()
            require_matplotlib_panel_alignment(fig,json_out=output_dir/f"{name}.alignment.json",strict=True,
                require_panel_labels=panels==4)
            # PNG is a 300-dpi preview; PDF/SVG retain editable text.
            png = output_dir/f"{name}.png"
            fig.savefig(png,dpi=300)
            fig.savefig(output_dir/f"{name}.pdf",metadata={"CreationDate":None,"ModDate":None})
            fig.savefig(output_dir/f"{name}.svg",metadata={"Date":None})
            paths.append(png)
            qa.append({"name":name,"source_csv":Path(csv_path).name,"panels":panels,
                "raw_zero_BER_points":zero_count,"no_smoothing":True,"no_data_exclusions":True,
                "uncertainty":"none: held-out integration; CI and significance excluded by task",
                "panel_role":"pilot-density strata of the same paired receiver comparison" if panels==4 else "paired Train minus Oracle difference",
                "curve_overlap_note":"Train/Oracle may overlap; method styles and the separate gap figure expose the difference."})
            plt.close(fig)

        def panel_canvas(title,ylabel,methods):
            fig,axes = plt.subplots(2,2,sharex=True,sharey=True,figsize=(7.2,5.6))
            fig.subplots_adjust(left=0.12,right=0.98,bottom=0.21,top=0.82,wspace=0.16,hspace=0.35)
            fig.suptitle(title,y=0.985,fontsize=12)
            handles = [Line2D([0],[0],color=METHOD_STYLE[m][1],linestyle=METHOD_STYLE[m][2],
                marker=METHOD_STYLE[m][3],markerfacecolor="white" if m=="oracle" else METHOD_STYLE[m][1],
                label=METHOD_STYLE[m][0]) for m in methods]
            fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(0.5,0.945),ncols=len(methods),fontsize=9)
            fig.supxlabel("SNR (dB)",y=0.12,fontsize=10)
            fig.supylabel(ylabel,x=0.018,fontsize=10)
            for index,(axis,(df,group)) in enumerate(zip(axes.flat,groups.items())):
                axis.set(xlim=(-0.5,20.5),xticks=(0,5,10,15,20))
                axis.grid(alpha=0.18)
                axis.set_title(f"dF={df}, pilot ratio=1/{round(1/float(group[0]['pilot_ratio']))}",fontsize=10,pad=8)
                axis.annotate(chr(97+index),(0,1),xycoords="axes fraction",xytext=(-12,12),textcoords="offset points",
                    fontsize=10,fontweight="bold",annotation_clip=False)
            return fig,axes

        def plot_panel(axis,group,method,field,scale=1,ber=False):
            x = np.array([float(r["snr_db"]) for r in group])
            y = np.array([float(r[field]) for r in group])/scale
            if not np.all(np.diff(x)>0) or not np.all(np.isfinite(y)):
                raise ValueError("finite raw values and increasing SNR required")
            zeros = np.zeros(5,dtype=bool)
            if ber:
                if np.any(y<0) or np.any(y>1):
                    raise ValueError("raw BER must lie in [0,1]")
                zeros = y==0
                y[zeros] = [0.5/int(group[i]["total_data_bits"]) for i in np.flatnonzero(zeros)]
                if np.any(y<=0):
                    raise ValueError("displayed BER must be positive on log axes")
            _,color,style,marker = METHOD_STYLE[method]
            axis.plot(x,y,color=color,linestyle=style,marker=marker,linewidth=1.35,markersize=4,
                markerfacecolor="white" if method=="oracle" else color)
            if np.any(zeros):
                axis.scatter(x[zeros],y[zeros],marker="v",color=color,s=24)
            return int(zeros.sum())

        fig,axes = panel_canvas("Held-out ID: channel-estimation comparison","Data-RE NMSE (dB)",("ls","train","oracle"))
        for axis,group in zip(axes.flat,groups.values()):
            for method in ("ls","train","oracle"):
                plot_panel(axis,group,method,f"{method}_nmse_db")
        finish(fig,FIGURE_NAMES[0],4,"Same axes for all densities; raw SNR points connected without smoothing.")

        fig,axes = panel_canvas("Held-out ID: data-bit BER with ZF","Data-bit BER",("perfect","ls","train","oracle"))
        zeros = 0
        for axis,group in zip(axes.flat,groups.values()):
            for method in ("perfect","ls","train","oracle"):
                zeros += plot_panel(axis,group,method,f"{method}_ber",ber=True)
            axis.set_yscale("log")
        finish(fig,FIGURE_NAMES[1],4,"Zero BER: downward markers at 0.5/total bits; raw CSV stays zero." if zeros else
            "All four receivers use the same observations; no raw zero BER points.",zeros)

        fig,axis = plt.subplots(figsize=(7.2,4.8))
        fig.subplots_adjust(left=0.15,right=0.98,bottom=0.23,top=0.78)
        fig.suptitle("Held-out ID: Train minus Oracle NMSE",y=0.975,fontsize=12)
        density_handles = []
        for (df,group),color,marker in zip(groups.items(),("#0072B2","#D55E00","#009E73","#CC79A7"),("o","s","^","D")):
            x = np.array([float(r["snr_db"]) for r in group])
            y = np.array([float(r["train_minus_oracle_nmse_db"]) for r in group])
            if not np.all(np.isfinite(y)) or not np.all(np.diff(x)>0):
                raise ValueError("finite unsmoothed paired gaps required")
            axis.plot(x,y,color=color,marker=marker,linewidth=1.35,markersize=4)
            density_handles.append(Line2D([0],[0],color=color,marker=marker,label=f"dF={df}"))
        axis.axhline(0,color="#888888",linewidth=0.7,linestyle=":")
        axis.set(xlabel="SNR (dB)",ylabel="Train NMSE - Oracle NMSE (dB)",xticks=(0,5,10,15,20),xlim=(-0.5,20.5))
        axis.ticklabel_format(axis="y",style="sci",scilimits=(0,0),useOffset=False)
        axis.grid(alpha=0.18)
        fig.legend(handles=density_handles,loc="upper center",bbox_to_anchor=(0.5,0.92),ncols=4,fontsize=9)
        finish(fig,FIGURE_NAMES[2],1,"Small signed validation differences; no significance or winner claim.")

        fig,axes = panel_canvas("Held-out ID: correct-bit-rate proxy","Correct-bit-rate proxy (Mbit/s)",("perfect","ls","train","oracle"))
        for axis,group in zip(axes.flat,groups.values()):
            for method in ("perfect","ls","train","oracle"):
                plot_panel(axis,group,method,f"{method}_proxy_bps",scale=1e6)
        finish(fig,FIGURE_NAMES[3],4,"Proxy = data bits per frame x (1 - BER) / frame duration including CP.")
    return paths,qa
