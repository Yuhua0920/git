"""CSV-only development figures, all raw points and all requested baselines.

Figure contract: diagnose learning and describe gaps for one development seed;
400 independent validation trajectories each contain ten correlated frames.
No confidence intervals or significance tests are requested. No smoothing,
normalization, sampling, outlier removal, or source-data modification occurs.
"""

import csv
import os
import tempfile
from pathlib import Path

import numpy as np

PREFIX = "t12b_b0_df16_dev"
FIGURE_NAMES = tuple(PREFIX+suffix for suffix in ("_training_curve", "_nmse", "_ber"))
STYLE = {"r3": ("R3 LS-Linear", "#777777", "--", "D"),
         "b0": ("B0 dev", "#0072B2", "-", "o"),
         "r4": ("R4 LMMSE-Train", "#CC79A7", "-.", "s"),
         "r5": ("R5 Oracle", "#D55E00", ":", "^"),
         "perfect": ("Perfect CSI", "#111111", "--", "v")}


def read_csv(path):
    with Path(path).open(encoding="utf8", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_results(input_dir, output_dir, require_matplotlib_panel_alignment):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    log = read_csv(input_dir/f"{PREFIX}_training_log.csv")
    aggregate = read_csv(input_dir/f"{PREFIX}_validation.csv")
    if ([int(r["epoch"]) for r in log] != list(range(31))
            or [int(r["snr_db"]) for r in aggregate] != [0, 5, 10, 15, 20]):
        raise ValueError("complete 31-epoch and five-SNR CSV sources required")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="t12b_mpl_") as cache:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            return _render(log, aggregate, output_dir, require_matplotlib_panel_alignment)
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous


def _render(log, aggregate, output_dir, require_matplotlib_panel_alignment):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths, records = [], []
    with matplotlib.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8,
        "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "t12b-development",
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False}):
        def canvas(title, ylabel):
            # 183 x 110 mm, expressed in inches for Matplotlib and source QA.
            fig, ax = plt.subplots(figsize=(7.204724409, 4.330708661))
            fig.subplots_adjust(left=0.12, right=0.96, bottom=0.22, top=0.70)
            fig.suptitle(title, y=0.965, fontsize=11)
            fig.text(0.5, 0.035, "One development seed; validation only; formal test sealed.", ha="center", fontsize=8)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)
            return fig, ax

        def export(fig, ax, name, curves, scale, source):
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.055), ncol=3)
            fig.canvas.draw()
            require_matplotlib_panel_alignment(fig, json_out=output_dir/f"{name}.alignment.json", strict=True,
                                               require_panel_labels=False)
            png = output_dir/f"{name}.png"
            fig.savefig(png, dpi=300)
            fig.savefig(output_dir/f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
            fig.savefig(output_dir/f"{name}.svg", metadata={"Date": None})
            paths.append(png)
            records.append({"name": name, "source_csv": source, "axes": {"x": ax.get_xlabel(), "y": ax.get_ylabel()},
                "y_scale": scale, "curves": curves, "panels": 1, "alignment": "not applicable: single plot area",
                "no_data_exclusions": True, "uncertainty": "none; single development seed, descriptive validation",
                "visual_self_check": "pending", "overlap_note": "R4/R5 are both plotted; markers and line styles retain identity"})
            plt.close(fig)

        epochs = np.array([int(r["epoch"]) for r in log])
        val = np.array([float(r["val_mse_pooled"]) for r in log])
        train = np.array([float(r["train_mse"]) for r in log[1:]])
        if not np.all(np.isfinite(val)) or not np.all(np.isfinite(train)) or np.any(val <= 0) or np.any(train <= 0):
            raise ValueError("MSE source must be finite and positive")
        fig, ax = canvas("B0 dF=16 development training", "Data-RE complex MSE")
        ax.plot(epochs[1:], train, label="Train MSE", color="#777777", linestyle="--", linewidth=1.3)
        ax.plot(epochs, val, label="Validation MSE", color="#0072B2", linewidth=1.7, marker="o", markersize=3)
        # Log is useful when the actual source spans over one order of magnitude.
        scale = "log" if max(np.max(val), np.max(train))/min(np.min(val), np.min(train)) > 10 else "linear"
        ax.set_yscale(scale); ax.set_xlabel("Epoch"); ax.set_xlim(0, 30)
        export(fig, ax, FIGURE_NAMES[0], ["Train MSE (epochs 1..30)", "Validation MSE (epochs 0..30)"], scale,
               f"{PREFIX}_training_log.csv")

        snrs = np.array([int(r["snr_db"]) for r in aggregate])
        assert np.all(np.diff(snrs) > 0)
        for suffix, methods, ylabel in (("nmse", ("r3", "b0", "r4", "r5"), "Pooled data-RE NMSE (dB)"),
                                        ("ber", ("perfect", "r3", "b0", "r4", "r5"), "Data BER with ZF")):
            fig, ax = canvas(f"B0 dF=16 development validation: {suffix.upper()}", ylabel)
            labels = []
            for method in methods:
                column = ("r5_oracle_nmse_db" if method == "r5" else f"{method}_nmse_db") if suffix == "nmse" else f"{method}_ber"
                values = np.array([float(r[column]) for r in aggregate])
                if not np.all(np.isfinite(values)) or (suffix == "ber" and np.any(values <= 0)):
                    raise ValueError("finite NMSE and strictly positive raw BER required; no silent log shift")
                label, color, line, marker = STYLE[method]
                ax.plot(snrs, values, label=label, color=color, linestyle=line, marker=marker,
                        linewidth=1.7 if method == "b0" else 1.2, markersize=5, markerfacecolor="none")
                labels.append(label)
            ax.set_xlabel("SNR (dB)"); ax.set_xticks(snrs)
            if suffix == "ber":
                ax.set_yscale("log")
            export(fig, ax, PREFIX+"_"+suffix, labels, "log" if suffix == "ber" else "linear",
                   f"{PREFIX}_validation.csv")
    return paths, records
