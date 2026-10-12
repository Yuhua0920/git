"""Vector information-flow schematic; no scientific data or estimators are run."""
import argparse
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from paper_support import ROOT, json_write, sha256

QA = Path("C:/Users/yuhua/.codex/skills/nature-figure/scripts")
sys.path.insert(0, str(QA))
from audit_panel_alignment import require_matplotlib_panel_alignment


def build(output=None):
    output = Path(output) if output else ROOT / "paper/figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family":"Arial", "font.size":8.6,
                         "pdf.fonttype":42, "svg.fonttype":"none", "svg.hashsalt":"method-workflow"})
    fig = plt.figure(figsize=(180/25.4, 66/25.4))
    ax = fig.add_axes([0,0,1,1]); ax.set(xlim=(0,180), ylim=(0,66)); ax.axis("off")
    labels, boxes = [], []
    def box(x,y,w,h,text,offline=True):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.3,rounding_size=1",linewidth=.6,
                                    facecolor="#f2f4f5" if offline else "white",edgecolor="#333333"))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=8.6,linespacing=1.35)
        labels.append(text); boxes.append(dict(x=x,y=y,width=w,height=h,label=text))
    def route(points, dashed=False):
        style = "--" if dashed else "-"
        for p,q in zip(points[:-2],points[1:-1]):
            ax.plot([p[0],q[0]],[p[1],q[1]],color="#333333",lw=.65,ls=style)
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle="-|>",mutation_scale=6,
                                    color="#333333",lw=.65,linestyle=style,shrinkA=0,shrinkB=1))
    ax.text(2,64,"OFFLINE STATISTICAL PREPARATION",fontsize=8,fontweight="bold",va="center")
    box(2,53,53,8,"Clean source statistics")
    box(63,53,53,8,"Clean profile-origin\nstatistics")
    box(124,53,53,8,"Clean mobility-origin\nstatistics")
    # Source assets use source statistics only. Development statistics feed corrections and predictors.
    route([(28,53),(28,47)])
    for x in (90,150): ax.plot([x,x],[53,50],color="#333333",lw=.65)
    ax.plot([90,150],[50,50],color="#333333",lw=.65)
    for x in (90,150): route([(x,50),(x,47)])
    box(2,34,53,13,"Source means\n"+r"Full Source operator $W_s$"+"\n"+r"Source covariance $R_{pp,s}$")
    box(63,34,53,13,"Exact cross-pilot predictors\n"+r"$K_{s,f}$, $\Delta K_{P,f}$, $\Delta K_{M,f}$"+"\nFour fixed partitions")
    box(124,34,53,13,r"Differences $B_P$, $B_M$"+"\nWeighted rank-4 compression\n"+r"Frozen factors $A_{m,4}$, $Q_{m,4}$")
    ax.plot([2,177],[32,32],color="#aaaaaa",lw=.5)
    route([(55,40),(63,40)])
    route([(55,45),(58,45),(58,48),(150,48),(150,47)])
    ax.text(32,29.5,"ONLINE ANALYTICAL ADAPTATION",fontsize=8,fontweight="bold",va="center")
    box(2,15,26,11,"Received pilots\nPilot-LS\nSource centering",False)
    box(35,15,27,11,"Causal buffer\nCurrent + 4 past\nTrajectory reset",False)
    box(69,15,37,11,"Cross-pilot fitting\nReal two-coordinate LS",False)
    box(113,15,20,11,"Coordinates\n"+r"$\widehat\alpha_j$, $\widehat\beta_j$",False)
    box(140,10,37,16,"Full Source +\nrank-4 corrections\nChannel estimate\n"+r"$\widehat h_{d,j}$",False)
    for left,right in ((28,35),(62,69),(106,113),(133,140)):
        route([(left,20.5),(right,20.5)])
    route([(90,34),(90,26)])
    route([(150,34),(150,26)])
    route([(15,34),(15,26)],True)
    route([(4,34),(0.6,34),(0.6,6),(158,6),(158,10)],True)
    ax.text(45,3.0,"Solid: preparation / inference     Dashed: frozen Source assets",fontsize=7.5,va="center")
    fig.canvas.draw()
    alignment = require_matplotlib_panel_alignment(fig,json_out=ROOT/"outputs/t20e_workflow_alignment.json")
    metadata = {"Creator":"Matplotlib", "CreationDate":None, "ModDate":None}
    fig.savefig(output/"method_workflow.pdf",metadata=metadata)
    fig.savefig(output/"method_workflow.svg",metadata={"Date":None})
    fig.savefig(output/"method_workflow.png",dpi=300)
    plt.close(fig)
    report = dict(kind="schematic_no_data", intended_width_mm=180, height_mm=66,
        minimum_nominal_font_pt=7.5, diagram_labels=labels, boxes=boxes,
        single_canvas_alignment_audit=alignment["verdict"],
        clean_offline_dependency_disclosed=True, exact_pilot_fitting_separate_from_compression=True,
        new_scientific_data_used=False, scientific_code_called=False,
        source_sha256=sha256(Path(__file__)),
        generated_sha256={p.name:sha256(p) for p in sorted(output.glob("method_workflow.*"))})
    json_write(output/"method_workflow_provenance.json",report)
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path)
    print(build(parser.parse_args().output))
