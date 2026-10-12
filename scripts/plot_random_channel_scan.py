"""Regenerate the three T07B figures from the unmodified aggregate CSV."""

import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plot_awgn_pilot_snr_scan import ber_plot_values  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_results(csv_path: Path, output_dir: Path) -> list[Path]:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 20 or len({(r["dF"], r["snr_db"]) for r in rows}) != 20:
        raise ValueError("T07B figures require 20 unique conditions")
    if any(int(r["num_trajectories"]) != 200 for r in rows):
        raise ValueError("T07B figures require 200 independent trajectories/condition")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    colors, markers = ("#0072B2", "#D55E00", "#009E73", "#CC79A7"), ("o", "s", "^", "D")
    paths = []
    for metric, filename in (("nmse", "t07b_nmse_vs_snr.png"), ("ber", "t07b_ber_vs_snr.png"),
                             ("proxy", "t07b_correct_bit_rate_vs_snr.png")):
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        fig.subplots_adjust(left=0.14, right=0.97, top=0.83, bottom=0.31)
        fig.suptitle("Random tapped-fading Monte Carlo calibration", fontsize=12, y=0.96)
        ax.set_title("200 independent channel trajectories; Gaussian taps; no Doppler", fontsize=9)
        any_zero = False
        for df, denominator, color, marker in zip((2, 4, 8, 16), (4, 8, 16, 32), colors, markers):
            selected = sorted((r for r in rows if int(r["dF"]) == df), key=lambda r: float(r["snr_db"]))
            snr = np.array([float(r["snr_db"]) for r in selected])
            if not np.array_equal(snr, [0, 5, 10, 15, 20]):
                raise ValueError("expected five SNR points")
            label = f"LS-LIN, ρ=1/{denominator}"
            if metric != "ber":
                field = "nmse_data_db" if metric == "nmse" else "correct_bit_rate_proxy_bps"
                values = np.array([float(r[field]) for r in selected])
                if metric == "proxy":
                    values = values / 1e6
                ax.plot(snr, values, color=color, marker=marker, markersize=4, linewidth=1.5, label=label)
            else:
                bits = np.array([int(r["total_data_bits"]) for r in selected])
                for field, style, width, display in (("ls_lin_ber", "-", 1.5, label),
                                                     ("perfect_csi_ber", "--", 1.0, f"Perfect CSI, ρ=1/{denominator}")):
                    raw = np.array([float(r[field]) for r in selected])
                    values, zero = ber_plot_values(raw, bits)
                    any_zero |= bool(np.any(zero))
                    ax.plot(snr, values, color=color, linestyle=style, marker=marker, linewidth=width, markersize=4, label=display)
                    if np.any(zero):
                        ax.scatter(snr[zero], values[zero], color=color, marker="v", s=35, zorder=4)
        ax.set_xlabel("SNR (dB)")
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.grid(True, alpha=0.22)
        if metric == "nmse":
            ax.set_ylabel("Data-RE NMSE (dB)")
            note = "LS-LIN only; pooled data-RE error/truth energies, then ratio and dB."
        elif metric == "ber":
            ax.set_ylabel("Data-RE BER")
            ax.set_yscale("log")
            note = "Dashed: Perfect CSI with noisy Y; data masks differ across ρ."
            if any_zero:
                note += "\nZero-error points are visualization-adjusted only to 0.5/Nbits; raw CSV remains BER=0."
        else:
            ax.set_ylabel("Correct-bit-rate proxy (Mbit/s)")
            note = "LS-LIN proxy = data bits/frame × (1 − BER) / frame duration (0.6667 ms)."
        note += "\nRandom-channel calibration; not 3GPP TDL-A. Source: aggregate CSV."
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8)
        fig.text(0.5, 0.015, note, ha="center", va="bottom", fontsize=8)
        path = output_dir / filename
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(path)
    return paths


if __name__ == "__main__":
    for output in plot_results(PROJECT_ROOT / "outputs/t07b_random_channel_scan.csv", PROJECT_ROOT / "outputs"):
        print(f"Plot saved: {output}")
