"""Four T10B figures rebuilt from all 20 aggregate CSV conditions.

Figures show pooled energy/count ratios for the fixed paired protocol; no
confidence intervals or significance claims. No smoothing or data exclusions.
Density uses color/marker, estimator uses line style. BER includes all methods.
"""

import csv
import os
import tempfile
from pathlib import Path

import numpy as np


FIGURE_NAMES = ("t10b_nmse_comparison", "t10b_ber_comparison",
                "t10b_oracle_theory_vs_empirical", "t10b_correct_bit_rate_comparison")


def plot_results(csv_path: Path, output_dir: Path, alignment_gate=None) -> list[Path]:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {(d_f, snr) for d_f in (2, 4, 8, 16) for snr in (0, 5, 10, 15, 20)}
    if len(rows) != 20 or {(int(r["dF"]), int(r["snr_db"])) for r in rows} != expected:
        raise ValueError("T10B plotting requires all 20 unique dF/SNR conditions")
    if any(r["all_checks_passed"].lower() != "true" for r in rows):
        raise ValueError("comparison figures require passed regressions/calibration")
    groups = {d_f: sorted((r for r in rows if int(r["dF"]) == d_f), key=lambda r: float(r["snr_db"]))
              for d_f in (2, 4, 8, 16)}
    output_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="t10b_matplotlib_") as cache:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D

            with matplotlib.rc_context({"font.family": "sans-serif",
                "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 10,
                "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
                "axes.spines.right": False, "axes.linewidth": 0.8, "legend.frameon": False}):
                colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
                markers = ("o", "s", "^", "D")
                density_handles = [Line2D([0], [0], color=color, marker=marker,
                    label=f"dF={d_f}, rho=1/{int(round(1/float(groups[d_f][0]['pilot_ratio'])))}")
                    for d_f, color, marker in zip(groups, colors, markers)]
                def canvas(title, ylabel):
                    fig, axis = plt.subplots(figsize=(7.2, 4.8))
                    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.21, top=0.75)
                    fig.suptitle(title, y=0.975, fontsize=13)
                    axis.set(xlabel="SNR (dB)", ylabel=ylabel, xticks=(0, 5, 10, 15, 20), xlim=(-0.5, 20.5))
                    axis.grid(alpha=0.22)
                    fig.legend(handles=density_handles, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncols=4, fontsize=9)
                    return fig, axis
                def finish(fig, axis, name, methods, caption):
                    handles = [Line2D([0], [0], color="black", linestyle=style, label=label) for label, style in methods]
                    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.86), ncols=len(handles), fontsize=10)
                    fig.text(0.5, 0.07, caption, ha="center", fontsize=9)
                    fig.text(0.5, 0.035, "200 paired trajectories; pooled energy/count ratios; no confidence intervals.", ha="center", fontsize=9)
                    fig.canvas.draw()
                    if alignment_gate is not None:
                        alignment_gate(fig, json_out=output_dir / f"{name}.alignment.json", strict=True)
                    # Vector export is also the source for final rendered text/collision QA.
                    png = output_dir / f"{name}.png"
                    fig.savefig(png, dpi=300)
                    fig.savefig(output_dir / f"{name}.svg")
                    fig.savefig(output_dir / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
                    plt.close(fig)
                    return png
                def curves(axis, fields):
                    zero_count = 0
                    for (d_f, group), color, marker in zip(groups.items(), colors, markers):
                        x = np.array([float(r["snr_db"]) for r in group])
                        assert np.all(np.diff(x) > 0)
                        for field, style, unit_scale in fields:
                            raw = np.array([float(r[field]) for r in group]) / unit_scale
                            if not np.all(np.isfinite(raw)):
                                raise ValueError(f"nonfinite plotting values: {field}")
                            visual = raw.copy()
                            zeros = np.zeros(raw.shape, dtype=bool)
                            if field.endswith("_ber"):
                                if np.any(raw < 0) or np.any(raw > 1):
                                    raise ValueError("BER must lie in [0,1]")
                                zeros = raw == 0
                                visual[zeros] = [0.5 / int(group[i]["total_data_bits"]) for i in np.flatnonzero(zeros)]
                                zero_count += int(np.count_nonzero(zeros))
                                if np.any(visual <= 0):
                                    raise ValueError("visual BER must be strictly positive on the log axis")
                            axis.plot(x, visual, color=color, marker=marker, linestyle=style,
                                      linewidth=1.4, markersize=4, markerfacecolor="white" if style == ":" else color)
                            if np.any(zeros):
                                axis.scatter(x[zeros], visual[zeros], marker="v", color=color, s=35)
                    return zero_count

                fig, axis = canvas("Dynamic TDL-A: channel-estimation comparison", "Data-RE NMSE (dB)")
                curves(axis, (("ls_lin_nmse_db", "--", 1), ("oracle_lmmse_nmse_db", "-", 1)))
                paths = [finish(fig, axis, FIGURE_NAMES[0], (("LS-LIN", "--"), ("Oracle LMMSE", "-")),
                    "All four pilot densities; raw SNR points connected without smoothing.")]

                fig, axis = canvas("Dynamic TDL-A: data-bit BER with ZF", "Data-bit BER")
                zero_count = curves(axis, (("perfect_csi_ber", ":", 1), ("ls_lin_ber", "--", 1), ("oracle_lmmse_ber", "-", 1)))
                axis.set_yscale("log")
                caption = "Raw zero BER shown at 0.5/total_data_bits with downward markers; CSV remains zero." \
                    if zero_count else "All 12 method/density curves retained; no raw zero BER points."
                paths.append(finish(fig, axis, FIGURE_NAMES[1], (("Perfect CSI", ":"), ("LS-LIN", "--"), ("Oracle LMMSE", "-")), caption))

                fig, axis = canvas("Oracle LMMSE: empirical and posterior-theory calibration", "Data-RE NMSE (dB)")
                curves(axis, (("oracle_lmmse_nmse_db", "-", 1), ("oracle_theoretical_nmse_db", "--", 1)))
                paths.append(finish(fig, axis, FIGURE_NAMES[2], (("Empirical Oracle", "-"), ("T10A Posterior Theory", "--")),
                    "Solid: empirical Oracle. Dashed: exact matched-statistics posterior prediction."))

                fig, axis = canvas("Dynamic TDL-A: correct-bit-rate proxy comparison", "Correct-bit-rate proxy (Mbit/s)")
                curves(axis, (("ls_lin_correct_bit_rate_proxy_bps", "--", 1e6), ("oracle_lmmse_correct_bit_rate_proxy_bps", "-", 1e6)))
                paths.append(finish(fig, axis, FIGURE_NAMES[3], (("LS-LIN", "--"), ("Oracle LMMSE", "-")),
                    "Proxy = data bits per frame x (1 - BER) / frame duration including CP."))
                return paths
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous
