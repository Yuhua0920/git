"""Rebuild both T07A calibration figures using only the aggregate CSV."""

import csv
import os
import tempfile
from pathlib import Path

# Keep the plotting cache in a writable temporary location on restricted hosts.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mimo_ofdm_matplotlib_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ber_plot_values(raw_ber: np.ndarray, total_bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Display zero errors at half of one-error resolution; leave raw values intact."""
    values = np.asarray(raw_ber, dtype=float).copy()
    bits = np.asarray(total_bits, dtype=float)
    if values.shape != bits.shape or np.any(bits <= 0) or np.any(values < 0) or np.any(values > 1):
        raise ValueError("BER and positive bit counts must have matching shapes")
    zero = values == 0
    values[zero] = 0.5 / bits[zero]
    return values, zero


def plot_results(csv_path: Path, output_dir: Path) -> list[Path]:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 20 or len({(row["dF"], row["snr_db"]) for row in rows}) != 20:
        raise ValueError("T07A plotting requires 20 unique aggregate conditions")
    if any(int(row["num_frames"]) != 200 for row in rows):
        raise ValueError("T07A performance figures require 200 frames/condition")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    markers = ("o", "s", "^", "D")
    figures = []
    for metric, filename in (("nmse", "t07a_nmse_vs_snr.png"), ("ber", "t07a_ber_vs_snr.png")):
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        fig.subplots_adjust(left=0.13, right=0.97, top=0.83, bottom=0.31)
        fig.suptitle("Fixed-channel AWGN calibration", fontsize=12, y=0.96)
        ax.set_title("200 frames/condition; deterministic four-path 2×2 MIMO", fontsize=9)
        any_zero = False
        for df, denominator, color, marker in zip((2, 4, 8, 16), (4, 8, 16, 32), colors, markers):
            selected = sorted((row for row in rows if int(row["dF"]) == df), key=lambda row: float(row["snr_db"]))
            snr = np.array([float(row["snr_db"]) for row in selected])
            if not np.array_equal(snr, [0, 5, 10, 15, 20]):
                raise ValueError("SNR points must be [0,5,10,15,20]")
            label = f"LS-LIN, ρ=1/{denominator}"
            if metric == "nmse":
                values = [float(row["nmse_data_db"]) for row in selected]
                ax.plot(snr, values, color=color, marker=marker, linewidth=1.5, markersize=4, label=label)
            else:
                counts = np.array([int(row["total_data_bits"]) for row in selected])
                for field, style, width, curve_label in (
                    ("ls_lin_ber", "-", 1.5, label),
                    ("perfect_csi_ber", "--", 1.0, f"Perfect CSI, ρ=1/{denominator}"),
                ):
                    raw = np.array([float(row[field]) for row in selected])
                    values, zero = ber_plot_values(raw, counts)
                    any_zero |= bool(np.any(zero))
                    ax.plot(snr, values, color=color, linestyle=style, marker=marker,
                            linewidth=width, markersize=4, label=curve_label)
                    if np.any(zero):
                        ax.scatter(snr[zero], values[zero], color=color, marker="v", s=35, zorder=4)
        ax.set_xlabel("SNR (dB)")
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.grid(True, which="major", alpha=0.22)
        if metric == "nmse":
            ax.set_ylabel("Data-RE NMSE (dB)")
        else:
            ax.set_ylabel("Data-RE BER")
            ax.set_yscale("log")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8)
        if metric == "ber":
            note = "Dashed: Perfect CSI; masks differ across ρ, so reference BER can differ."
            if any_zero:
                note += "\nZero-error points are visualization-adjusted only to 0.5/Nbits; raw CSV remains BER=0."
        else:
            note = "LS-LIN only; energies pooled over frames and data REs before ratio and dB."
        note += "\nCalibration experiment; no TDL-A or confidence intervals. Source: aggregate CSV."
        fig.text(0.5, 0.015, note, ha="center", va="bottom", fontsize=8)
        path = output_dir / filename
        fig.savefig(path, dpi=300)
        plt.close(fig)
        figures.append(path)
    return figures


if __name__ == "__main__":
    for path in plot_results(PROJECT_ROOT / "outputs/t07a_awgn_pilot_snr_scan.csv", PROJECT_ROOT / "outputs"):
        print(f"Plot saved: {path}")
