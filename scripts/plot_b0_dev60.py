"""CSV-only 60-epoch development diagnostics; raw points, one seed, no smoothing.

Figure contract: test whether extra budget improves convergence and which SNRs
benefit. Three independent quantitative single-axis diagnostics; no paper claim.
Style-only inheritance from existing project figures; new explicit column mapping.
All61 epoch rows/five SNR rows used; train epoch0 is undefined in the source.
No confidence interval/p-value: one development seed, 400 independent validation
trajectories, ten correlated frames each. Formal test remains sealed.
"""
import csv
import os
import tempfile
from pathlib import Path
import numpy as np

PREFIX = "t12d_b0_df16_dev60"
FIGURE_NAMES = tuple(PREFIX+suffix for suffix in ("_training_curve", "_nmse", "_nmse_improvement"))


def read_csv(path):
    with Path(path).open(encoding="utf8", newline="") as file:
        return list(csv.DictReader(file))


def plot_results(input_dir, output_dir, require_matplotlib_panel_alignment):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    log = read_csv(input_dir/f"{PREFIX}_training_log.csv")
    results = read_csv(input_dir/f"{PREFIX}_validation.csv")
    if [int(r["epoch"]) for r in log] != list(range(61)) or [int(r["snr_db"]) for r in results] != [0, 5, 10, 15, 20]:
        raise ValueError("all61 epochs and five SNR rows required")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="t12d_mpl_", dir=output_dir) as cache:
        previous = os.environ.get("MPLCONFIGDIR"); os.environ["MPLCONFIGDIR"] = cache
        try:
            return render(log, results, output_dir, require_matplotlib_panel_alignment)
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous


def render(log, rows, output, require_matplotlib_panel_alignment):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths, records = [], []
    with matplotlib.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8,
        "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "t12d-development",
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False}):
        def canvas(title, ylabel):
            fig, ax = plt.subplots(figsize=(7.204724409, 4.330708661))
            fig.subplots_adjust(left=.12, right=.96, bottom=.22, top=.70)
            fig.suptitle(title, y=.965, fontsize=11)
            fig.text(.5, .035, "Development diagnosis; one seed; validation only; formal test sealed.", ha="center", fontsize=8)
            ax.set_ylabel(ylabel); ax.grid(alpha=.2)
            return fig, ax
        def export(fig, ax, name, curves, scale, source):
            ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.055), ncol=3)
            fig.canvas.draw()
            require_matplotlib_panel_alignment(fig, json_out=output/f"{name}.alignment.json", strict=True, require_panel_labels=False)
            fig.savefig(output/f"{name}.png", dpi=600)
            fig.savefig(output/f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
            fig.savefig(output/f"{name}.svg", metadata={"Date": None})
            paths.append(output/f"{name}.png")
            records.append({"name": name, "source_csv": source, "curves": curves, "y_scale": scale,
                "no_data_exclusions": True, "panels": 1, "alignment": "N/A: one plot area",
                "uncertainty": "none: one development seed; descriptive validation",
                "visual_self_check": "pending"})
            plt.close(fig)
        epochs = np.array([int(r["epoch"]) for r in log]); val = np.array([float(r["val_mse_pooled"]) for r in log])
        train = np.array([float(r["train_mse"]) for r in log[1:]])
        if not np.all(np.isfinite(val)) or not np.all(np.isfinite(train)) or np.any(val <= 0) or np.any(train <= 0):
            raise ValueError("raw log MSE must be finite/positive")
        best = int(np.argmin(val)); fig, ax = canvas("B0 dF=16: continuous 60-epoch development run", "Data-RE complex MSE")
        ax.plot(epochs[1:], train, color="#777777", linestyle="--", label="Train MSE", linewidth=1.3)
        ax.plot(epochs, val, color="#0072B2", label="Validation MSE", marker="o", markersize=3, linewidth=1.7)
        ax.axvline(30, color="#D55E00", linestyle=":", label="Epoch 30", linewidth=1.2)
        ax.axvline(best, color="#009E73", linestyle="-.", label=f"Global best: epoch {best}", linewidth=1.2)
        ax.set_xlabel("Epoch"); ax.set_xlim(0, 60); ax.set_yscale("log")
        export(fig, ax, FIGURE_NAMES[0], ["Train MSE epochs1..60", "Validation MSE epochs0..60", "Epoch30", "Global best"], "log", f"{PREFIX}_training_log.csv")
        snrs = np.array([int(r["snr_db"]) for r in rows]); assert np.all(np.diff(snrs) > 0)
        methods = [("r3_nmse_db", "R3 LS-Linear", "#777777", "--", "D"),
                   ("b0_epoch30_nmse_db", "B0 epoch 30", "#56B4E9", "--", "o"),
                   ("b0_best60_nmse_db", "B0 best 60", "#0072B2", "-", "o"),
                   ("r4_nmse_db", "R4 LMMSE-Train", "#CC79A7", "-.", "s"),
                   ("r5_oracle_nmse_db", "R5 Oracle", "#D55E00", ":", "^")]
        fig, ax = canvas("B0 dF=16: frozen epoch 30 vs global best", "Pooled data-RE NMSE (dB)")
        for column, label, color, style, marker in methods:
            values = np.array([float(r[column]) for r in rows])
            if not np.all(np.isfinite(values)):
                raise ValueError("all requested NMSE observations must be finite")
            ax.plot(snrs, values, label=label, color=color, linestyle=style, marker=marker,
                markerfacecolor="none", linewidth=1.8 if column == "b0_best60_nmse_db" else 1.2, markersize=5)
        ax.set_xlabel("SNR (dB)"); ax.set_xticks(snrs)
        export(fig, ax, FIGURE_NAMES[1], [m[1] for m in methods], "linear", f"{PREFIX}_validation.csv")
        gain = np.array([float(r["epoch30_to_best60_gain_db"]) for r in rows])
        expected = np.array([float(r["b0_epoch30_nmse_db"])-float(r["b0_best60_nmse_db"]) for r in rows])
        if not np.array_equal(gain, expected) or not np.all(np.isfinite(gain)):
            raise ValueError("raw paired NMSE improvement mismatch")
        fig, ax = canvas("Extra training: per-SNR validation NMSE improvement", "Epoch 30 minus best 60 NMSE (dB)")
        ax.plot(snrs, gain, label="B0 epoch 30 to best 60", color="#0072B2", linewidth=1.8, marker="o", markersize=5)
        ax.axhline(0, color="#777777", linestyle=":", linewidth=1)
        ax.set_xlabel("SNR (dB)"); ax.set_xticks(snrs)
        export(fig, ax, FIGURE_NAMES[2], ["B0 epoch30 minus best60 NMSE"], "linear", f"{PREFIX}_validation.csv")
    return paths, records
