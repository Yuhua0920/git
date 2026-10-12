"""Three deterministic T13D figures rebuilt from two raw CSVs only."""
import csv
import os
import tempfile
from pathlib import Path

import numpy as np

FIGURE_NAMES=("t13d_statistics_crlb_relative","t13d_statistics_kl_surface","t13d_statistics_nonlocal_kl")
POINTS=(("P0",30,100),("P1",120,100),("P2",30,300),("P3",120,300))


def plot_results(input_dir,output_dir,require_matplotlib_panel_alignment):
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    def read(name):
        with (Path(input_dir)/(name+".csv")).open(encoding="utf8",newline="") as f:
            return list(csv.DictReader(f))
    crlb=read("t13d_statistics_crlb"); kl=read("t13d_statistics_kl_grid")
    if len(crlb)!=48 or len(kl)!=21576:
        raise ValueError("48 CRLB rows and 21576 KL cells required")
    with tempfile.TemporaryDirectory(prefix="t13d_mpl_",dir=output) as cache:
        old=os.environ.get("MPLCONFIGDIR"); os.environ["MPLCONFIGDIR"]=cache
        try:
            return render(crlb,kl,output,require_matplotlib_panel_alignment)
        finally:
            if old is None:
                os.environ.pop("MPLCONFIGDIR",None)
            else:
                os.environ["MPLCONFIGDIR"]=old


def render(crlb,kl,output,require_matplotlib_panel_alignment):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    paths=[]; records=[]
    styles=((16,1,"dF16; 1 frame","#0072B2","--","o"),
        (16,5,"dF16; 5 frames","#0072B2","-","s"),
        (8,1,"dF8; 1 frame","#D55E00","--","^"),
        (8,5,"dF8; 5 frames","#D55E00","-","D"))
    with matplotlib.rc_context({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
        "font.size":8,"axes.labelsize":8,"legend.fontsize":7,"pdf.fonttype":42,"svg.fonttype":"none",
        "svg.hashsalt":"t13d-information","axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False}):
        def export(fig,name,panels,source,footer,exclude_axes=()):
            fig.text(.5,.03,footer,ha="center",fontsize=7)
            fig.canvas.draw()
            require_matplotlib_panel_alignment(fig,json_out=output/f"{name}.alignment.json",strict=True,
                require_panel_labels=panels>1,exclude_axes=exclude_axes)
            fig.savefig(output/f"{name}.png",dpi=600)
            fig.savefig(output/f"{name}.pdf",metadata={"CreationDate":None,"ModDate":None})
            fig.savefig(output/f"{name}.svg",metadata={"Date":None})
            svg=output/f"{name}.svg"
            svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf8").splitlines())+"\n",encoding="utf8")
            paths.append(output/f"{name}.png")
            records.append({"name":name,"panels":panels,"source_csv":source,"no_data_exclusions":True,
                "uncertainty":"deterministic local unbiased-estimator lower bounds / distribution KL, no sample CI",
                "auxiliary_axis_exclusions":"KL colourbar only" if exclude_axes else "none","visual_self_check":"pending"})
            plt.close(fig)

        fig,axes=plt.subplots(2,4,sharex=True,sharey="row",figsize=(7.204724409,5.5))
        fig.subplots_adjust(left=.09,right=.98,bottom=.16,top=.77,wspace=.20,hspace=.40)
        fig.suptitle("Sparse-pilot channel statistics: relative CRLB standard deviation",y=.97,fontsize=10)
        for i,parameter in enumerate(("speed","delay")):
            values=np.array([float(r[f"relative_{parameter}_std_percent"]) for r in crlb])
            for j,(p,v,ds) in enumerate(POINTS):
                ax=axes[i,j]
                for d,w,label,color,style,marker in styles:
                    group=sorted((r for r in crlb if r["point"]==p and int(r["dF"])==d and int(r["window_frames"])==w),key=lambda r:int(r["snr_db"]))
                    ax.plot([int(r["snr_db"]) for r in group],[float(r[f"relative_{parameter}_std_percent"]) for r in group],
                        label=label,color=color,linestyle=style,marker=marker,markerfacecolor="none",linewidth=1.2,markersize=3)
                ax.axhline(10,color=".55",linewidth=.5,linestyle=":")
                ax.axhline(25,color=".55",linewidth=.5,linestyle=":")
                ax.set_yscale("log"); ax.set_ylim(min(1,values.min()/1.15),values.max()*1.25)
                ax.set_xticks([5,10,20]); ax.grid(alpha=.18)
                ax.set_title(f"{p}: {v} km/h, {ds} ns",fontsize=7.5,pad=14)
                ax.text(-.06,1.07,chr(97+i*4+j),transform=ax.transAxes,fontsize=9,fontweight="bold")
                if i==1:
                    ax.set_xlabel("SNR (dB)")
            axes[i,0].set_ylabel(("Speed" if i==0 else "Delay scale")+" relative std (%)")
        h,l=axes[0,0].get_legend_handles_labels()
        fig.legend(h,l,loc="upper center",bbox_to_anchor=(.5,.915),ncol=2)
        export(fig,FIGURE_NAMES[0],8,"t13d_statistics_crlb.csv",
            "Four independent links; continuous windows; dotted guides: prescribed 10% / 25%; no attainability claim.")

        group=[r for r in kl if r["reference_point"]=="P0" and int(r["window_frames"])==5 and int(r["snr_db"])==10]
        speeds=sorted({int(r["candidate_speed"]) for r in group}); delays=sorted({int(r["candidate_ds"]) for r in group})
        matrix=np.empty((len(speeds),len(delays)))
        for r in group:
            matrix[speeds.index(int(r["candidate_speed"])),delays.index(int(r["candidate_ds"]))]=float(r["KL_total"])
        fig,ax=plt.subplots(figsize=(7.204724409,4.6))
        fig.subplots_adjust(left=.12,right=.81,bottom=.19,top=.83)
        fig.suptitle("Global covariance diagnostic: P0, dF16, five frames, 10 dB",y=.96,fontsize=10)
        image=ax.pcolormesh(delays,speeds,np.log1p(matrix),shading="nearest",cmap="viridis",rasterized=True)
        ax.plot(100,30,marker="x",color="white",markersize=7,markeredgewidth=1.4)
        ax.add_patch(Rectangle((50,10),100,40,fill=False,edgecolor="white",linestyle="--",linewidth=.8))
        ax.set_xlabel("Candidate nominal delay scale (ns)"); ax.set_ylabel("Candidate speed (km/h)")
        ax.set_xlim(50,350); ax.set_ylim(10,150)
        cb=fig.colorbar(image,ax=ax,fraction=.045,pad=.04); cb.set_label("log(1 + four-link KL)",fontsize=8)
        # Colourbar is auxiliary, not a second scientific panel.
        export(fig,FIGURE_NAMES[1],1,"t13d_statistics_kl_grid.csv",
            "White cross: reference; dashed box: local-region boundary. No low-KL threshold; finite grid only.",(cb.ax,))

        fig,axes=plt.subplots(1,2,sharey=True,figsize=(7.204724409,3.8))
        fig.subplots_adjust(left=.11,right=.975,bottom=.23,top=.73,wspace=.26)
        fig.suptitle("Minimum nonlocal covariance KL under dF16 pilots",y=.97,fontsize=10)
        for ax,w,letter in zip(axes,(1,5),("a","b")):
            for (p,v,ds),color,marker in zip(POINTS,("#0072B2","#D55E00","#009E73","#CC79A7"),("o","s","^","D")):
                values=[]
                for snr in (5,10,20):
                    group=[r for r in kl if r["reference_point"]==p and int(r["window_frames"])==w and int(r["snr_db"])==snr
                        and (abs(float(r["candidate_speed"])-v)>=20 or abs(float(r["candidate_ds"])-ds)>=50)]
                    values.append(min(float(r["KL_total"]) for r in group))
                if min(values)<=0:
                    raise ValueError("nonlocal KL log display requires strictly positive minima")
                ax.plot([5,10,20],values,label=p,color=color,marker=marker,markerfacecolor="none",linewidth=1.3,markersize=4)
            ax.set_yscale("log"); ax.set_xticks([5,10,20]); ax.set_xlabel("SNR (dB)")
            ax.set_title(f"{w} continuous frame"+("s" if w>1 else ""),fontsize=9,pad=14)
            ax.grid(alpha=.2); ax.text(-.04,1.07,letter,transform=ax.transAxes,fontsize=9,fontweight="bold")
        axes[0].set_ylabel("Minimum nonlocal four-link KL")
        h,l=axes[0].get_legend_handles_labels(); fig.legend(h,l,loc="upper center",bbox_to_anchor=(.5,.91),ncol=4)
        export(fig,FIGURE_NAMES[2],2,"t13d_statistics_kl_grid.csv",
            "Nonlocal: speed difference >=20 km/h OR delay difference >=50 ns. No arbitrary innovation threshold.")
    return paths,records
