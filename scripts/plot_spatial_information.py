"""T14B CSV-only deterministic theoretical spatial-information figure."""
import csv
import os
import tempfile
from pathlib import Path

import numpy as np

PREFIX = "t14b_spatial_information_gate"
FIGURE_NAMES = (PREFIX+"_gain",)


def plot_results(input_dir, output_dir, require_matplotlib_panel_alignment):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    with (Path(input_dir)/(PREFIX+".csv")).open(encoding="utf8", newline="") as f:
        rows = list(csv.DictReader(f))
    key = {(r["mode"], float(r["rho"]), int(r["snr_db"])) for r in rows}
    if len(rows) != 36 or key != {(m, r, s) for m in ("TX-only", "RX-only", "BOTH") for r in (0., .3, .6, .9) for s in (5, 10, 20)}:
        raise ValueError("complete 3 modes x4 rho x3 SNR theory CSV required")
    with tempfile.TemporaryDirectory(prefix="t14b_mpl_", dir=output) as cache:
        old = os.environ.get("MPLCONFIGDIR"); os.environ["MPLCONFIGDIR"] = cache
        try:
            return render(rows, output, require_matplotlib_panel_alignment)
        finally:
            if old is None: os.environ.pop("MPLCONFIGDIR", None)
            else: os.environ["MPLCONFIGDIR"] = old


def render(rows, output, require_matplotlib_panel_alignment):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gains = np.array([float(r["spatial_gain_db"]) for r in rows])
    if not np.all(np.isfinite(gains)) or gains.min() < -1e-10:
        raise ValueError("finite nonnegative theoretical information gains required")
    upper = max(1.5, np.ceil(gains.max()*2)/2+.25)
    with matplotlib.rc_context({"font.family":"sans-serif", "font.sans-serif":["Arial","DejaVu Sans"],
        "font.size":8, "axes.labelsize":9, "legend.fontsize":8, "pdf.fonttype":42, "svg.fonttype":"none",
        "svg.hashsalt":"t14b-theory", "axes.spines.top":False, "axes.spines.right":False, "legend.frameon":False}):
        fig, axes = plt.subplots(1, 3, sharey=True, figsize=(7.204724409, 3.5))
        fig.subplots_adjust(left=.10, right=.975, bottom=.23, top=.72, wspace=.27)
        fig.suptitle("Spatial correlation information value under sparse orthogonal pilots", y=.97, fontsize=10)
        styles = ((5,"#0072B2","--","o"), (10,"#D55E00","-.","s"), (20,"#009E73","-","^"))
        for ax, mode, letter in zip(axes, ("TX-only","RX-only","BOTH"), ("a","b","c")):
            for snr, color, line, marker in styles:
                group = sorted((r for r in rows if r["mode"]==mode and int(r["snr_db"])==snr), key=lambda r:float(r["rho"]))
                ax.plot([float(r["rho"]) for r in group], [float(r["spatial_gain_db"]) for r in group],
                    label=f"{snr} dB", color=color, linestyle=line, marker=marker,
                    markerfacecolor="none", linewidth=1.4, markersize=4)
            ax.set_title(mode, fontsize=9, pad=13); ax.set_xlabel("Correlation rho")
            ax.set_xticks([0,.3,.6,.9]); ax.set_xlim(-.03,.93); ax.set_ylim(-.02,upper); ax.grid(alpha=.2)
            ax.text(-.04, 1.08, letter, transform=ax.transAxes, fontsize=9, fontweight="bold")
        axes[0].set_ylabel("R5 - R9 theoretical NMSE (dB)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5,.9), ncol=3)
        fig.text(.5,.04,"2 x 2; TDL-A 30 km/h, 100 ns; dF16/dT4; theory only; formal test sealed.", ha="center", fontsize=7)
        fig.canvas.draw(); name = FIGURE_NAMES[0]
        require_matplotlib_panel_alignment(fig, json_out=output/(name+".alignment.json"), strict=True, require_panel_labels=True)
        fig.savefig(output/(name+".png"), dpi=600)
        fig.savefig(output/(name+".pdf"), metadata={"CreationDate":None,"ModDate":None})
        fig.savefig(output/(name+".svg"), metadata={"Date":None})
        svg = output/(name+".svg")
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf8").splitlines())+"\n", encoding="utf8")
        plt.close(fig)
    return [output/(name+".png")], [{"name":name, "panels":3, "source_csv":PREFIX+".csv",
        "common_gain_limits_db":[-.02,float(upper)], "no_data_exclusions":True,
        "uncertainty":"deterministic theoretical traces; no stochastic performance claim", "visual_self_check":"pending"}]
