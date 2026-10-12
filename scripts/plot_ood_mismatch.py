"""CSV-only T13A paired mechanism figures; no simulated production observations.

Quantitative grid: compare temporal/frequency perturbations and isolate penalty.
Style-only inheritance; all6 pooled conditions,200 independent trajectories each,
5 correlated frames/unit. No CI, smoothing, p-values or formal-test observations.
"""
import csv
import os
import tempfile
from pathlib import Path

import numpy as np

PREFIX = "t13a_ood_mismatch"
FIGURE_NAMES = (PREFIX+"_nmse", PREFIX+"_penalty")


def plot_results(input_dir, output_dir, require_matplotlib_panel_alignment):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    with (input_dir/f"{PREFIX}.csv").open(encoding="utf8", newline="") as f:
        rows = list(csv.DictReader(f))
    expected = {(shift, snr) for shift in ("speed120", "DS300") for snr in (5, 10, 20)}
    if len(rows) != 6 or {(r["shift"], int(r["snr_db"])) for r in rows} != expected:
        raise ValueError("all two-shift three-SNR observations required")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="t13a_mpl_", dir=output_dir) as cache:
        previous = os.environ.get("MPLCONFIGDIR"); os.environ["MPLCONFIGDIR"] = cache
        try:
            return render(rows, output_dir, require_matplotlib_panel_alignment)
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous


def render(rows, output, require_matplotlib_panel_alignment):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = [("R3_nmse_db", "R3 LS-Linear", "#777777", "--", "D"),
        ("R4_frozen_nmse_db", "R4 frozen", "#CC79A7", "-.", "s"),
        ("B0_nmse_db", "B0 mixed best 57", "#0072B2", "-", "o"),
        ("R5_ood_oracle_nmse_db", "R5 OOD Oracle", "#D55E00", ":", "^")]
    paths, records = [], []
    def values(group, column):
        result = np.array([float(r[column]) for r in group])
        if not np.all(np.isfinite(result)):
            raise ValueError("finite quantitative source data required")
        return result
    with matplotlib.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 9, "legend.fontsize": 8, "pdf.fonttype": 42,
        "svg.fonttype": "none", "svg.hashsalt": "t13a-development", "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False}):
        def export(fig, name, panels):
            fig.text(.5, .035, "OOD development: 200 trajectories/shift, 5 frames/unit; formal test sealed.", ha="center", fontsize=7)
            fig.canvas.draw()
            require_matplotlib_panel_alignment(fig, json_out=output/f"{name}.alignment.json", strict=True,
                require_panel_labels=panels > 1)
            fig.savefig(output/f"{name}.png", dpi=600)
            fig.savefig(output/f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
            fig.savefig(output/f"{name}.svg", metadata={"Date": None})
            paths.append(output/f"{name}.png")
            records.append({"name": name, "panels": panels, "source_csv": PREFIX+".csv",
                "all_six_conditions_used": True, "no_data_exclusions": True,
                "uncertainty": "none: pooled development descriptive results; trajectory quartiles in JSON",
                "visual_self_check": "pending"})
            plt.close(fig)
        fig, axes = plt.subplots(1, 2, sharey=True, figsize=(7.204724409, 4.0))
        fig.subplots_adjust(left=.10, right=.97, bottom=.22, top=.72, wspace=.25)
        fig.suptitle("Low-pilot channel estimation under statistics mismatch", y=.975, fontsize=10)
        all_values = np.array([float(r[m[0]]) for r in rows for m in methods])
        if not np.all(np.isfinite(all_values)):
            raise ValueError("all method/condition NMSE values required")
        lo = np.floor(all_values.min()/5)*5-1; hi = max(0, np.ceil(all_values.max()/5)*5)
        for ax, shift, title, label in zip(axes, ("speed120", "DS300"),
                ("Speed 120 km/h; DS 100 ns", "DS 300 ns; speed 30 km/h"), ("a", "b")):
            group = sorted((r for r in rows if r["shift"] == shift), key=lambda r: int(r["snr_db"]))
            snrs = np.array([int(r["snr_db"]) for r in group]); assert np.all(np.diff(snrs) > 0)
            for column, text, color, style, marker in methods:
                ax.plot(snrs, values(group, column), label=text, color=color, linestyle=style, marker=marker,
                    markerfacecolor="none", linewidth=1.6 if column == "R4_frozen_nmse_db" else 1.2, markersize=4)
            ax.set_title(title, fontsize=8, pad=12); ax.set_xlabel("SNR (dB)"); ax.set_xticks(snrs)
            ax.set_ylim(lo, hi); ax.grid(alpha=.2)
            ax.text(-.05, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight="bold")
        axes[0].set_ylabel("Pooled data-RE NMSE (dB)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.52, .92), ncol=4)
        export(fig, FIGURE_NAMES[0], 2)
        fig, ax = plt.subplots(figsize=(7.204724409, 4.0))
        fig.subplots_adjust(left=.12, right=.96, bottom=.22, top=.72)
        fig.suptitle("Frozen R4 minus condition-matched R5: mismatch penalty", y=.965, fontsize=10)
        for shift, text, color, marker in (("speed120", "Speed shift", "#0072B2", "o"), ("DS300", "Delay-spread shift", "#D55E00", "s")):
            group = sorted((r for r in rows if r["shift"] == shift), key=lambda r: int(r["snr_db"]))
            snrs = np.array([int(r["snr_db"]) for r in group]); assert np.all(np.diff(snrs) > 0)
            ax.plot(snrs, values(group, "R4_mismatch_penalty_db"), label=text, color=color,
                marker=marker, markerfacecolor="none", linewidth=1.5, markersize=5)
        ax.set_xlabel("SNR (dB)"); ax.set_ylabel("R4 - R5 OOD NMSE (dB)"); ax.set_xticks([5, 10, 20]); ax.grid(alpha=.2)
        ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.05), ncol=2)
        export(fig, FIGURE_NAMES[1], 1)
    return paths, records
