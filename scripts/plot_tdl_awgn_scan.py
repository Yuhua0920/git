"""Rebuild the three T09B diagnostic figures solely from aggregate CSV."""

import csv
import os
import tempfile
from pathlib import Path

import numpy as np


def plot_results(csv_path: Path, output_dir: Path) -> list[Path]:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {(dF, snr) for dF in (2, 4, 8, 16) for snr in (0, 5, 10, 15, 20)}
    if len(rows) != 20 or {(int(row["dF"]), int(row["snr_db"])) for row in rows} != expected:
        raise ValueError("T09B plotting requires exactly 20 unique layout/SNR rows")
    groups = {dF: sorted((row for row in rows if int(row["dF"]) == dF), key=lambda row: float(row["snr_db"]))
              for dF in (2, 4, 8, 16)}
    with tempfile.TemporaryDirectory(prefix="t09b_matplotlib_") as cache:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D

            colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
            markers = ("o", "s", "^", "D")
            def canvas(title, ylabel):
                fig, axis = plt.subplots(figsize=(9, 5.6))
                fig.subplots_adjust(left=0.12, right=0.98, bottom=0.20, top=0.80)
                fig.suptitle(title, y=0.97)
                axis.set(xlabel="SNR (dB)", ylabel=ylabel, xticks=(0, 5, 10, 15, 20), xlim=(-0.5, 20.5))
                axis.grid(alpha=0.25)
                return fig, axis
            def density_label(dF):
                ratio = float(groups[dF][0]["pilot_ratio"])
                return f"rho = 1/{int(round(1/ratio))} (dF={dF})"
            def finish(fig, axis, name, caption):
                handles, labels = axis.get_legend_handles_labels()
                fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncols=4, fontsize=9)
                fig.text(0.5, 0.035, caption, ha="center", fontsize=9)
                path = output_dir / name
                fig.savefig(path, dpi=180)
                plt.close(fig)
                return path

            fig, axis = canvas("Dynamic TDL-A AWGN: LS-LIN channel estimation", "Data-RE NMSE (dB)")
            for (dF, group), color, marker in zip(groups.items(), colors, markers):
                x = [float(row["snr_db"]) for row in group]
                axis.plot(x, [float(row["nmse_data_db"]) for row in group], color=color, marker=marker,
                          linewidth=1.4, markersize=4, label=density_label(dF))
                floors = [float(row["noiseless_floor_nmse_db"]) for row in group]
                if not np.allclose(floors, floors[0], rtol=0, atol=1e-12):
                    raise ValueError("one layout must have one T09A floor across SNR")
                axis.axhline(floors[0], color=color, linestyle="--", linewidth=1, alpha=0.8)
            paths = [finish(fig, axis, "t09b_nmse_vs_snr.png",
                            "Dashed horizontal lines indicate T09A noiseless interpolation floors.")]

            fig, axis = canvas("Dynamic TDL-A AWGN: ZF bit error rate", "Data-bit BER")
            fig.subplots_adjust(top=0.73)
            has_zero = False
            density_handles = []
            for (dF, group), color, marker in zip(groups.items(), colors, markers):
                x = np.array([float(row["snr_db"]) for row in group])
                density_handles.append(Line2D([0], [0], color=color, marker=marker, label=density_label(dF)))
                for prefix, style in (("ls_lin", "-"), ("perfect_csi", "--")):
                    raw = np.array([float(row[f"{prefix}_ber"]) for row in group])
                    zeros = raw == 0
                    visual = raw.copy()
                    visual[zeros] = [0.5 / int(group[i]["total_data_bits"]) for i in np.flatnonzero(zeros)]
                    axis.plot(x, visual, color=color, linestyle=style, marker=marker, markersize=4, linewidth=1.4)
                    if np.any(zeros):
                        has_zero = True
                        axis.scatter(x[zeros], visual[zeros], color=color, marker="v", s=35)
            axis.set_yscale("log")
            fig.legend(handles=density_handles, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncols=4, fontsize=9)
            fig.legend(handles=[Line2D([0], [0], color="black", linestyle="-", label="LS-LIN + ZF"),
                                Line2D([0], [0], color="black", linestyle="--", label="Perfect CSI + ZF")],
                       loc="upper center", bbox_to_anchor=(0.5, 0.865), ncols=2, fontsize=9)
            caption = "Zero raw BER: shown at 0.5/total_data_bits with downward markers; CSV remains zero." \
                if has_zero else "Solid: LS-LIN + ZF. Dashed: Perfect CSI + ZF. No zero raw BER points."
            fig.text(0.5, 0.035, caption, ha="center", fontsize=9)
            path = output_dir / "t09b_ber_vs_snr.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            paths.append(path)

            fig, axis = canvas("Dynamic TDL-A AWGN: LS-LIN correct-bit-rate proxy", "Correct-bit-rate proxy (Mbit/s)")
            for (dF, group), color, marker in zip(groups.items(), colors, markers):
                axis.plot([float(row["snr_db"]) for row in group],
                          [float(row["ls_lin_correct_bit_rate_proxy_bps"]) / 1e6 for row in group],
                          color=color, marker=marker, linewidth=1.4, markersize=4, label=density_label(dF))
            paths.append(finish(fig, axis, "t09b_correct_bit_rate_vs_snr.png",
                                "Proxy = data bits per frame x (1 - BER) / frame duration."))
            return paths
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous
