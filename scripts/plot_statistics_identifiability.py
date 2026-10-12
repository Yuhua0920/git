"""Two CSV-only T13C figures: joint selection counts and pooled NMSE."""
import csv
import os
import tempfile
from pathlib import Path

import numpy as np

SELECTION="t13c_statistics_identifiability"
PERFORMANCE="t13c_bank_selection_lmmse"
FIGURE_NAMES=(SELECTION+"_confusion",PERFORMANCE+"_nmse")
CONDITIONS=("ID","speed120","DS300")


def plot_results(input_dir,output_dir,require_matplotlib_panel_alignment):
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    def read(name):
        with (Path(input_dir)/(name+".csv")).open(encoding="utf8",newline="") as f:
            return list(csv.DictReader(f))
    selection=read(SELECTION); performance=read(PERFORMANCE)
    if len(selection)!=3600 or len(performance)!=9:
        raise ValueError("3600 selector and nine aggregate observations required")
    with tempfile.TemporaryDirectory(prefix="t13c_mpl_",dir=output) as cache:
        old=os.environ.get("MPLCONFIGDIR"); os.environ["MPLCONFIGDIR"]=cache
        try:
            return render(selection,performance,output,require_matplotlib_panel_alignment)
        finally:
            if old is None:
                os.environ.pop("MPLCONFIGDIR",None)
            else:
                os.environ["MPLCONFIGDIR"]=old


def render(selection,performance,output,require_matplotlib_panel_alignment):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths,records=[],[]
    with matplotlib.rc_context({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
        "font.size":8,"axes.labelsize":9,"legend.fontsize":8,"pdf.fonttype":42,"svg.fonttype":"none",
        "svg.hashsalt":"t13c-development","axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False}):
        def export(fig,name,source,footer):
            fig.text(.5,.035,footer,ha="center",fontsize=7)
            fig.canvas.draw()
            require_matplotlib_panel_alignment(fig,json_out=output/f"{name}.alignment.json",strict=True,require_panel_labels=True)
            fig.savefig(output/f"{name}.png",dpi=600)
            fig.savefig(output/f"{name}.pdf",metadata={"CreationDate":None,"ModDate":None})
            fig.savefig(output/f"{name}.svg",metadata={"Date":None})
            svg=output/f"{name}.svg"
            svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf8").splitlines())+"\n",encoding="utf8")
            paths.append(output/f"{name}.png")
            records.append({"name":name,"panels":3,"source_csv":source+".csv","no_data_exclusions":True,
                "uncertainty":"none: descriptive validation/development; independent unit is trajectory-class sample",
                "visual_self_check":"pending"})
            plt.close(fig)

        fig,axes=plt.subplots(1,3,sharey=True,figsize=(7.204724409,3.8))
        fig.subplots_adjust(left=.09,right=.975,bottom=.23,top=.73,wspace=.30)
        fig.suptitle("Five-frame pilot statistics identification",y=.97,fontsize=10)
        fig.text(.5,.875,"Rows: true regime; columns: selected regime. Common colour range: 0 to 200.",ha="center",fontsize=8)
        for ax,snr,label in zip(axes,(5,10,20),("a","b","c")):
            group=[r for r in selection if int(r["window_frames"])==5 and int(r["snr_db"])==snr]
            if len(group)!=600 or len({(r["true_condition"],int(r["trajectory_id"])) for r in group})!=600:
                raise ValueError("600 unique balanced samples per confusion matrix required")
            matrix=np.zeros((3,3),int)
            for r in group:
                matrix[CONDITIONS.index(r["true_condition"]),CONDITIONS.index(r["selected_condition"])]+=1
            if not np.array_equal(matrix.sum(1),[200]*3):
                raise ValueError("200 per true class required")
            ax.imshow(matrix,vmin=0,vmax=200,cmap="Blues",aspect="auto",interpolation="nearest")
            for i in range(3):
                for j in range(3):
                    ax.text(j,i,str(matrix[i,j]),ha="center",va="center",fontsize=9,color="white" if matrix[i,j]>=110 else "black")
            ax.set_xticks([0,1,2],["ID","TEMP","FREQ"],fontsize=7.5)
            ax.set_yticks([0,1,2],["ID","TEMP","FREQ"],fontsize=7.5)
            ax.set_title(f"{snr} dB; accuracy {np.trace(matrix)/600:.1%}",fontsize=8,pad=12)
            ax.set_xlabel("Selected regime")
            ax.text(-.05,1.04,label,transform=ax.transAxes,fontsize=9,fontweight="bold")
        axes[0].set_ylabel("True regime")
        export(fig,FIGURE_NAMES[0],SELECTION,"200 trajectories/class; dF16/dT4; validation + OOD development; formal test sealed.")

        methods=(("R4","R4 frozen","#CC79A7","-.","s"),
            ("R7_BankML","R7-BankML","#0072B2","-","o"),
            ("R5","R5 matched Oracle","#D55E00",":","^"))
        fig,axes=plt.subplots(1,3,sharey=True,figsize=(7.204724409,3.8))
        fig.subplots_adjust(left=.09,right=.975,bottom=.23,top=.73,wspace=.30)
        fig.suptitle("Selected covariance LMMSE: estimation performance",y=.97,fontsize=10)
        vals=np.array([float(r[m+"_nmse_db"]) for r in performance for m,*_ in methods])
        if not np.all(np.isfinite(vals)):
            raise ValueError("finite pooled NMSE required")
        limits=(np.floor(vals.min()/5)*5-1,max(0,np.ceil(vals.max()/5)*5))
        for ax,c,title,label in zip(axes,CONDITIONS,("ID: 30 km/h, 100 ns","TEMP: 120 km/h, 100 ns","FREQ: 30 km/h, 300 ns"),("a","b","c")):
            group=sorted((r for r in performance if r["condition"]==c),key=lambda r:int(r["snr_db"]))
            if [int(r["snr_db"]) for r in group]!=[5,10,20]:
                raise ValueError("three SNRs per performance condition required")
            for m,text,color,style,marker in methods:
                ax.plot([int(r["snr_db"]) for r in group],[float(r[m+"_nmse_db"]) for r in group],
                    label=text,color=color,linestyle=style,marker=marker,markerfacecolor="none",linewidth=1.5,markersize=4)
            ax.set_title(title,fontsize=7.5,pad=12); ax.set_xlabel("SNR (dB)"); ax.set_xticks([5,10,20])
            ax.set_ylim(*limits); ax.grid(alpha=.2)
            ax.text(-.05,1.04,label,transform=ax.transAxes,fontsize=9,fontweight="bold")
        axes[0].set_ylabel("Pooled data-RE NMSE (dB)")
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.52,.92),ncol=3)
        export(fig,FIGURE_NAMES[1],PERFORMANCE,"200 trajectories/class x 5 frames; candidate-bank feasibility; formal test sealed.")
    return paths,records
