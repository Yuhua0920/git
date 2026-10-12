"""Plot only final MR-01 development CSVs; no experiment or historical reads.

Figure contract: compare absolute pooled NMSE across the five prespecified
conditions, then show the paired Pooled-minus-LR4 difference and its exploratory
uncertainty. No direction of improvement is presumed before reading the data.
Archetype: quantitative grid; condition panels provide prespecified stratification.
The six-method figure shows point estimates only; the paired figure carries the
primary difference uncertainty. All conditions, SNR points and effect signs are
retained. The source-centered diagnostic is retained in the input provenance.

Python backend; 180 x 108 mm; text >=7 pt; PNG 600 dpi plus editable PDF/SVG.
Only connect the three observed SNR points: no smoothing or interpolation.
Root runs rendered PDF text/collision QA and final-size inspection after export.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import ScaledTranslation
import numpy as np


CONDITIONS = ("ID", "Profile", "Mobility", "Composed", "Boundary")
SNRS = (10, 20, 30)
MAIN_METHODS = ("Source", "Pooled", "Hard", "Convex", "Dense", "LR4")
ALL_METHODS = MAIN_METHODS + ("Pooled-source-centered",)
CONTRASTS = ("Source-Pooled", "Pooled-LR4", "Source-LR4", "Pooled-Dense", "Dense-LR4", "Natural-SourceCentered")
STYLES = {
    "Source": dict(color="#666666", marker="D", linestyle="--", linewidth=1.0),
    "Pooled": dict(color="#0072B2", marker="o", linestyle="-", linewidth=1.3),
    "Hard": dict(color="#E69F00", marker="v", linestyle=":", linewidth=1.0),
    "Convex": dict(color="#009E73", marker="^", linestyle="-.", linewidth=1.0),
    "Dense": dict(color="#CC79A7", marker="P", linestyle=(0, (5, 1, 1, 1)), linewidth=1.0),
    "LR4": dict(color="#D55E00", marker="s", linestyle=(0, (3, 1)), linewidth=1.3),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(required).issubset(reader.fieldnames):
            raise ValueError(f"CSV schema is missing required fields: {Path(path).name}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty CSV: {Path(path).name}")
    return rows


def number(row, field):
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"nonfinite {field}: {row}")
    return value


def load_final_tables(out):
    absolute_path = out / "tables/nmse_all_methods.csv"
    paired_path = out / "tables/paired_contrasts.csv"
    absolute_rows = read_csv(absolute_path, ("condition", "snr_db", "method", "nmse_db", "CI_lower", "CI_upper", "trajectory_count", "scored_frames"))
    paired_rows = read_csv(paired_path, ("condition", "snr_db", "contrast", "first_method", "second_method", "estimate_db", "CI_lower", "CI_upper", "trajectory_count", "bootstrap_resamples", "interval_identity"))
    absolute, paired = {}, {}
    for row in absolute_rows:
        key = (row["condition"], number(row, "snr_db"), row["method"])
        if key in absolute:
            raise ValueError(f"duplicate absolute row: {key}")
        for field in ("nmse_db", "CI_lower", "CI_upper"):
            number(row, field)
        if number(row, "CI_lower") > number(row, "CI_upper"):
            raise ValueError(f"reversed absolute interval: {key}")
        if number(row, "trajectory_count") != 100 or number(row, "scored_frames") != 600:
            raise ValueError(f"absolute sampling unit/count drift: {key}")
        absolute[key] = row
    expected_absolute = {(c, s, m) for c in CONDITIONS for s in SNRS for m in ALL_METHODS}
    if set(absolute) != expected_absolute:
        raise ValueError("absolute CSV must contain all 105 prespecified condition/SNR/method cells")
    for row in paired_rows:
        key = (row["condition"], number(row, "snr_db"), row["contrast"])
        if key in paired:
            raise ValueError(f"duplicate paired row: {key}")
        for field in ("estimate_db", "CI_lower", "CI_upper"):
            number(row, field)
        if number(row, "CI_lower") > number(row, "CI_upper"):
            raise ValueError(f"reversed paired interval: {key}")
        if number(row, "trajectory_count") != 100 or number(row, "bootstrap_resamples") != 2000:
            raise ValueError(f"paired sampling unit/count drift: {key}")
        identity = row["interval_identity"].lower()
        if "unadjusted" not in identity or "95%" not in identity or "exploratory" not in identity:
            raise ValueError(f"paired uncertainty identity drift: {key}")
        if row["contrast"] == "Pooled-LR4" and (row["first_method"], row["second_method"]) != ("Pooled", "LR4"):
            raise ValueError(f"Pooled-LR4 sign drift: {key}")
        paired[key] = row
    expected_paired = {(c, s, contrast) for c in CONDITIONS for s in SNRS for contrast in CONTRASTS}
    if set(paired) != expected_paired:
        raise ValueError("paired CSV must contain all 90 prespecified condition/SNR/contrast cells")
    for condition in CONDITIONS:
        for snr in SNRS:
            expected = number(absolute[condition, snr, "Pooled"], "nmse_db") - number(absolute[condition, snr, "LR4"], "nmse_db")
            got = number(paired[condition, snr, "Pooled-LR4"], "estimate_db")
            if not np.isclose(got, expected, rtol=0, atol=1e-10):
                raise ValueError(f"paired sign/value disagrees with absolute CSV: {condition}, {snr}")
    return absolute, paired, {
        "inputs": {"tables/nmse_all_methods.csv": sha256(absolute_path), "tables/paired_contrasts.csv": sha256(paired_path)},
        "absolute_input_rows": len(absolute_rows), "absolute_plotted_rows": 90,
        "absolute_exclusion": {"method": "Pooled-source-centered", "rows": 15, "reason": "Requested absolute figure compares six main methods; diagnostic centering control remains in source CSV"},
        "paired_input_rows": len(paired_rows), "paired_plotted_rows": 15,
        "paired_selection": "Prespecified Pooled-LR4 contrast only; all five conditions, all three SNRs and all signs retained",
        "data_geometry": "Actual SNR 10,20,30 dB points only, straight connecting segments; no interpolation/smoothing",
    }


def configure_style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0, "axes.labelsize": 8.0, "axes.titlesize": 8.0,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "legend.fontsize": 7.0,
        "axes.linewidth": 0.7, "axes.spines.right": False, "axes.spines.top": False,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
        "text.usetex": False, "axes.unicode_minus": True, "savefig.facecolor": "white",
    })


def make_grid(title, ylabel, ylim):
    width_mm = 180
    height_mm = 108
    fig, axes = plt.subplots(2, 3, figsize=(width_mm / 25.4, height_mm / 25.4))
    fig.subplots_adjust(left=.092, right=.979, bottom=.126, top=.866, wspace=.43, hspace=.69)
    fig.text(.092, .961, title, fontsize=9.0, fontweight="bold", ha="left", va="top")
    panels = list(axes.flat[:5])
    for index, (ax, condition) in enumerate(zip(panels, CONDITIONS)):
        label = chr(ord("a") + index)
        transform = ax.transAxes + ScaledTranslation(-13 / 72, 8 / 72, fig.dpi_scale_trans)
        ax.text(0, 1, label, transform=transform, fontsize=8.0, fontweight="bold", va="bottom", ha="left")
        ax.set_title(condition, pad=8)
        ax.set_xlim(8, 32)
        ax.set_xticks(SNRS)
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(direction="out", length=2.5, width=.6, pad=2)
        ax.grid(axis="y", color="#E6E6E6", linewidth=.45, zorder=0)
        ax.set_xlabel("Source-referenced SNR (dB)", labelpad=3)
        if index in (0, 3):
            ax.set_ylabel(ylabel, labelpad=3)
    legend_ax = axes.flat[5]
    legend_ax.set_axis_off()
    return fig, panels, legend_ax


def padded_limits(values, include_zero=False):
    low, high = min(values), max(values)
    if include_zero:
        low, high = min(low, 0), max(high, 0)
    padding = max((high - low) * .09, .15)
    return low - padding, high + padding


def save_figure(fig, panels, legend_ax, filename, alignment):
    fig.canvas.draw()
    require_matplotlib_panel_alignment = alignment
    require_matplotlib_panel_alignment(
        fig, json_out=filename.with_suffix(".alignment.json"),
        overlay_svg=filename.with_suffix(".alignment.svg"),
        axes=panels, panel_ids=["a", "b", "c", "d", "e"], exclude_axes=[legend_ax],
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True,
    )
    # Preserve exact physical dimensions: do not crop with bbox_inches='tight'.
    fig.savefig(filename.with_suffix(".pdf"))
    fig.savefig(filename.with_suffix(".svg"))
    fig.savefig(filename.with_suffix(".png"), dpi=600)
    plt.close(fig)


def render_absolute(absolute, directory, alignment):
    values = [number(absolute[c, s, m], "nmse_db") for c in CONDITIONS for s in SNRS for m in MAIN_METHODS]
    fig, panels, legend_ax = make_grid("MR-01 development: absolute estimation error", "Pooled NMSE (dB)", padded_limits(values))
    handles = []
    for condition, ax in zip(CONDITIONS, panels):
        for method in MAIN_METHODS:
            y = [number(absolute[condition, s, method], "nmse_db") for s in SNRS]
            line, = ax.plot(SNRS, y, markersize=3.5, markeredgewidth=.65, zorder=3, **STYLES[method])
            if condition == CONDITIONS[0]:
                handles.append(line)
    legend_ax.legend(handles, MAIN_METHODS, loc="upper left", borderaxespad=0,
                     ncol=2, handlelength=1.7, handletextpad=.45, columnspacing=.7, labelspacing=.6)
    fig.text(.092, .024, "100 trajectories/condition, frames 4–9; point estimates only. Lines connect tested SNRs.", fontsize=7.0, va="bottom")
    save_figure(fig, panels, legend_ax, directory / "mr01_absolute_nmse", alignment)


def render_paired(paired, directory, alignment):
    values = [number(paired[c, s, "Pooled-LR4"], field) for c in CONDITIONS for s in SNRS for field in ("estimate_db", "CI_lower", "CI_upper")]
    fig, panels, legend_ax = make_grid("MR-01 development: paired Pooled−LR4 difference", "Pooled−LR4 NMSE (dB)", padded_limits(values, include_zero=True))
    for condition, ax in zip(CONDITIONS, panels):
        point = np.array([number(paired[condition, s, "Pooled-LR4"], "estimate_db") for s in SNRS])
        lower = np.array([number(paired[condition, s, "Pooled-LR4"], "CI_lower") for s in SNRS])
        upper = np.array([number(paired[condition, s, "Pooled-LR4"], "CI_upper") for s in SNRS])
        ax.axhline(0, color="#777777", linewidth=.8, linestyle="--", zorder=1)
        # fmt='none' draws literal CI endpoints; a percentile interval need not
        # contain the point estimate. Its midpoint is not drawn as a statistic.
        ax.errorbar(SNRS, (lower + upper) / 2, yerr=(upper - lower) / 2, fmt="none",
                    ecolor="#0072B2", elinewidth=1.0, capsize=2.5, capthick=.8, zorder=2)
        ax.plot(SNRS, point, color="#0072B2", marker="o", markersize=4.0, linewidth=1.2, zorder=3)
    handle = Line2D([], [], color="#0072B2", marker="o", markersize=4.0, linewidth=1.2)
    legend_ax.legend([handle], ["Pooled−LR4"], loc="upper left", borderaxespad=0, handlelength=2.7)
    legend_ax.text(0, .75, "Positive: LR4 has lower NMSE.\nNegative: Pooled has lower NMSE.\n\nBars: paired percentile 95% CI.\n2,000 trajectory resamples.\nUnadjusted; exploratory.\n\n100 trajectories per condition;\nframes 4–9 scored.", fontsize=7.0, ha="left", va="top", linespacing=1.35)
    fig.text(.092, .024, "Exploratory development intervals; no multiple-comparison adjustment or confirmatory inference.", fontsize=7.0, va="bottom")
    save_figure(fig, panels, legend_ax, directory / "mr01_pooled_lr4_difference", alignment)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="MR-01 output root containing the two final tables")
    parser.add_argument("--figure-skill-dir", type=Path, default=Path.home() / ".codex/skills/nature-figure", help="Directory supplying scripts/audit_panel_alignment.py")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    helper_dir = args.figure_skill_dir.resolve() / "scripts"
    if not (helper_dir / "audit_panel_alignment.py").is_file():
        raise FileNotFoundError("nature-figure alignment helper is required; supply --figure-skill-dir")
    sys.path.insert(0, str(helper_dir))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    absolute, paired, provenance = load_final_tables(out)
    directory = out / "figures" / "final"
    directory.mkdir(parents=True, exist_ok=True)
    configure_style()
    render_absolute(absolute, directory, require_matplotlib_panel_alignment)
    render_paired(paired, directory, require_matplotlib_panel_alignment)
    captions = (
        "Absolute NMSE. Pooled error-energy/truth-energy ratios in dB for the six main methods "
        "in each of the five prespecified conditions at 10, 20 and 30 dB Source-referenced SNR. "
        "Each condition contains 100 independent physical trajectories, with frames 4–9 scored "
        "within each trajectory. Points are estimates only; lines connect the actual three SNR "
        "points. Paired uncertainty for the primary comparison is displayed in the difference figure.\n\n"
        "Paired Pooled−LR4 difference. Difference of the pooled NMSEs in dB, with supplied unadjusted "
        "95% paired trajectory-percentile intervals from 2,000 resamples. Resampling preserves "
        "paired methods, SNRs and the six scored frames within each physical trajectory. Positive "
        "values indicate lower NMSE for LR4; negative values indicate lower NMSE for Pooled. "
        "Every condition and effect sign is retained. These are exploratory development intervals "
        "without multiple-comparison adjustment.\n"
    )
    (directory / "captions.txt").write_text(captions, encoding="utf-8")
    provenance.update({
        "script_sha256": sha256(__file__), "backend": "python-matplotlib", "width_mm": 180, "height_mm": 108,
        "minimum_configured_font_pt": 7, "png_dpi": 600, "formats": ["PDF", "SVG", "PNG"],
        "uncertainty": "2,000 paired trajectory-percentile resamples; exploratory unadjusted 95% CI; no bootstrap recomputation",
        "alignment": "Measured final five plot areas with 1.5 pt tolerance; sixth grid cell is legend-only",
        "remaining_rendered_QA": "Root must run PDF glyph-size and collision audits and inspect each panel at final size",
    })
    (directory / "plot_provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "EXPORTED", "figures": ["mr01_absolute_nmse", "mr01_pooled_lr4_difference"], "dimensions_mm": [180, 108], "dpi": 600}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
