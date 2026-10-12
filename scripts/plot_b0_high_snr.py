"""CSV-only T12E development diagnostics; all raw observations, no smoothing.

Figure contract: convergence, dedicated capacity at20dB, and cross-SNR cost.
Three quantitative single-axis plots; one development seed,400 trajectories,
ten correlated frames/trajectory; no CI/p-value, no formal-test results.
Style-only inheritance from the project's Python plots; explicit new mapping.
"""
import csv
import os
import tempfile
from pathlib import Path
import numpy as np

PREFIX = "t12e_b0_df16_20db"
FIGURE_NAMES = tuple(PREFIX+s for s in ("_training_curve", "_primary_nmse", "_cross_snr_nmse"))


def read_csv(path):
    with Path(path).open(encoding="utf8", newline="") as file:
        return list(csv.DictReader(file))


def plot_results(input_dir, output_dir, require_matplotlib_panel_alignment):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    log = read_csv(input_dir/f"{PREFIX}_training_log.csv")
    primary = read_csv(input_dir/f"{PREFIX}_primary_comparison.csv")
    cross = read_csv(input_dir/f"{PREFIX}_validation.csv")
    if ([int(r["epoch"]) for r in log] != list(range(61)) or len(primary) != 6
            or [int(r["snr_db"]) for r in cross] != [0, 5, 10, 15, 20]
            or {r["checkpoint"] for r in cross} != {"global_best"}):
        raise ValueError("all61 epochs, six primary methods, five global-best SNRs required")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="t12e_mpl_", dir=output_dir) as cache:
        previous = os.environ.get("MPLCONFIGDIR"); os.environ["MPLCONFIGDIR"] = cache
        try:
            return render(log, primary, cross, output_dir, require_matplotlib_panel_alignment)
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous


def render(log, primary, cross, output, require_matplotlib_panel_alignment):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths, records = [], []
    with matplotlib.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8, "pdf.fonttype": 42,
        "svg.fonttype": "none", "svg.hashsalt": "t12e-development", "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False}):
        def canvas(title, ylabel):
            fig, ax = plt.subplots(figsize=(7.204724409, 4.330708661))
            fig.subplots_adjust(left=.12, right=.96, bottom=.22, top=.70)
            fig.suptitle(title, y=.965, fontsize=11)
            fig.text(.5, .035, "Capacity probe; one development seed; validation only; formal test sealed.", ha="center", fontsize=8)
            ax.set_ylabel(ylabel); ax.grid(alpha=.2)
            return fig, ax
        def export(fig, ax, name, source, has_legend=True):
            if has_legend:
                ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.055), ncol=3)
            fig.canvas.draw()
            require_matplotlib_panel_alignment(fig, json_out=output/f"{name}.alignment.json", strict=True, require_panel_labels=False)
            fig.savefig(output/f"{name}.png", dpi=600)
            fig.savefig(output/f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
            fig.savefig(output/f"{name}.svg", metadata={"Date": None})
            paths.append(output/f"{name}.png")
            records.append({"name": name, "source_csv": source, "no_data_exclusions": True,
                "uncertainty": "none: single development seed; pooled400 validation trajectories",
                "panels": 1, "alignment": "N/A: one plot area", "visual_self_check": "pending"})
            plt.close(fig)
        epochs = np.array([int(r["epoch"]) for r in log])
        val = np.array([float(r["val_mse_20db"]) for r in log])
        train = np.array([float(r["train_mse_20db"]) for r in log[1:]])
        if not np.all(np.isfinite(val)) or not np.all(np.isfinite(train)) or np.any(val <= 0) or np.any(train <= 0):
            raise ValueError("positive finite raw MSE required for log axis")
        best = int(np.argmin(val)); fig, ax = canvas("B0 dF=16: dedicated 20-dB training", "20-dB data-RE complex MSE")
        ax.plot(epochs[1:], train, color="#777777", linestyle="--", label="Train MSE", linewidth=1.3)
        ax.plot(epochs, val, color="#0072B2", label="Validation MSE", marker="o", markersize=3, linewidth=1.7)
        ax.axvline(12, color="#D55E00", linestyle=":", label="Exposure matched: epoch 12", linewidth=1.2)
        ax.axvline(best, color="#009E73", linestyle="-.", label=f"Global best: epoch {best}", linewidth=1.2)
        ax.axvline(60, color="#777777", linestyle=":", label="Epoch 60", linewidth=1.2)
        ax.set_xlabel("Epoch"); ax.set_xlim(0, 61); ax.set_yscale("log")
        export(fig, ax, FIGURE_NAMES[0], f"{PREFIX}_training_log.csv")
        values = np.array([float(r["nmse_db"]) for r in primary])
        if not np.all(np.isfinite(values)):
            raise ValueError("finite primary observations required")
        labels = ["R3", "Mixed B0\nbest 57", "20-dB B0\nepoch 12", "20-dB B0\nbest", "R4", "R5 Oracle"]
        expected = ["R3", "mixed_B0_best57", "B0_20db_epoch12", "B0_20db_best", "R4", "R5"]
        if [r["method"] for r in primary] != expected:
            raise ValueError("primary method mapping/order changed")
        fig, ax = canvas("20-dB capacity probe: frozen references and dedicated B0", "Data-RE NMSE (dB)")
        for index, (value, color) in enumerate(zip(values, ["#777777", "#56B4E9", "#009E73", "#0072B2", "#CC79A7", "#D55E00"])):
            ax.plot(index, value, marker="o", color=color, markersize=6)
        ax.set_xticks(np.arange(6), labels); ax.set_xlim(-.5, 5.5)
        # Show the entire interval from0dB, avoiding an exaggerated cropped comparison.
        ax.set_ylim(np.floor(values.min()/5)*5-1, 0)
        export(fig, ax, FIGURE_NAMES[1], f"{PREFIX}_primary_comparison.csv", False)
        fig, ax = canvas("20-dB specialization: global-best cross-SNR validation", "Pooled data-RE NMSE (dB)")
        snrs = np.array([int(r["snr_db"]) for r in cross]); assert np.all(np.diff(snrs) > 0)
        methods = [("r3_nmse_db", "R3 LS-Linear", "#777777", "--", "D"),
            ("mixed_B0_best57_nmse_db", "Mixed B0 best 57", "#56B4E9", "--", "o"),
            ("b0_20db_only_nmse_db", "20-dB B0 best", "#0072B2", "-", "o"),
            ("r4_nmse_db", "R4 LMMSE-Train", "#CC79A7", "-.", "s"),
            ("r5_nmse_db", "R5 Oracle", "#D55E00", ":", "^")]
        for column, label, color, style, marker in methods:
            values = np.array([float(r[column]) for r in cross])
            if not np.all(np.isfinite(values)):
                raise ValueError("all five-SNR observations must be finite")
            ax.plot(snrs, values, label=label, color=color, linestyle=style, marker=marker,
                markerfacecolor="none", linewidth=1.8 if column == "b0_20db_only_nmse_db" else 1.2, markersize=5)
        ax.set_xlabel("SNR (dB)"); ax.set_xticks(snrs)
        export(fig, ax, FIGURE_NAMES[2], f"{PREFIX}_validation.csv")
    return paths, records
