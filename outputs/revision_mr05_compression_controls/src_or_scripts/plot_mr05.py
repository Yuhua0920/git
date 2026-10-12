"""CSV-only figures for same-coordinate compression controls.

No inference, coordinate fitting, bootstrap, metric recomputation, or model
selection occurs here. Existing tables are read without changing their bytes.
The separate source/PDF/glyph/collision audits and human image review are run
by the caller after these exports. Panel alignment is mandatory before export.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl

CONDITIONS = ("ID", "Profile", "Mobility", "Composed", "Boundary")
SNRS = (10, 20, 30)
METHODS = (
    "SOURCE", "POOLED_NATURAL", "DENSE_TW5", "CORR_ORIGIN_R4",
    "CORR_I_R4", "CORR_ORIGIN_M2_R4", "CORR_POOL_M2_R4",
    "WHOLE_POOL_M2_R23", "WHOLE_POOL_M2_R24",
)
CONTROLS = METHODS[3:]
OTHER_CORRECTIONS = CONTROLS[1:4]
WHOLES = CONTROLS[4:]
RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9.0,
    "axes.labelsize": 9.0,
    "axes.titlesize": 10.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.unicode_minus": True,
}
STYLES = {
    "CORR_ORIGIN_R4": ("Original correction", "#222222", "o", "-"),
    "CORR_I_R4": ("Identity-weighted correction", "#0072B2", "s", "--"),
    "CORR_ORIGIN_M2_R4": ("Origin-second-moment correction", "#E69F00", "D", "-.",),
    "CORR_POOL_M2_R4": ("Pooled-second-moment correction", "#009E73", "^", "-"),
    "WHOLE_POOL_M2_R23": ("Whole bank, rank 23", "#CC79A7", "v", "--"),
    "WHOLE_POOL_M2_R24": ("Whole bank, rank 24", "#D55E00", "P", "-.",),
    "DENSE_TW5": ("Dense, shared coordinates", "#666666", "x", "-."),
    "POOLED_NATURAL": ("Fixed Pooled", "#999999", "+", ":"),
}
DISPLAY = {
    "CORR_I_R4": "Identity weight",
    "CORR_ORIGIN_M2_R4": "Origin second moment",
    "CORR_POOL_M2_R4": "Pooled second moment",
    "WHOLE_POOL_M2_R23": "Whole rank 23 (324,288 MACs)",
    "WHOLE_POOL_M2_R24": "Whole rank 24 (336,384 MACs)",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        answer = list(csv.DictReader(stream))
    if not answer:
        raise ValueError(f"Empty source table: {path.name}")
    return answer


def finite_number(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {key}: {row}")
    return value


def cell(row: dict[str, str]) -> tuple[str, int]:
    condition, snr = row["condition"], int(row["snr_db"])
    if condition not in CONDITIONS or snr not in SNRS:
        raise ValueError(f"Unexpected cell: {condition}/{snr}")
    return condition, snr


def source_tables(output: Path) -> tuple[dict, dict, dict, dict]:
    paths = {name: output / "tables" / f"{name}.csv"
             for name in ("nmse", "contrasts", "decomposition")}
    before = {name: sha256(path) for name, path in paths.items()}
    nmse, contrasts, decomposition = {}, {}, {}
    for row in rows(paths["nmse"]):
        key = (*cell(row), row["method"])
        if key in nmse or key[2] not in METHODS:
            raise ValueError(f"Duplicate/unknown NMSE identity: {key}")
        if int(row["trajectory_count"]) != 100:
            raise ValueError("Diagnostic sample count is not 100")
        if finite_number(row, "nmse_linear") <= 0:
            raise ValueError("Nonpositive linear NMSE")
        finite_number(row, "nmse_db")
        nmse[key] = row
    expected = {(condition, snr, method) for condition in CONDITIONS
                for snr in SNRS for method in METHODS}
    if set(nmse) != expected or len(nmse) != 135:
        raise ValueError("NMSE table does not cover exactly 135 method cells")
    for row in rows(paths["contrasts"]):
        key = (*cell(row), row["contrast"], row["first"], row["second"])
        if key in contrasts:
            raise ValueError(f"Duplicate contrast identity: {key}")
        if int(row["trajectory_count"]) != 100 or int(row["bootstrap"]) != 2000:
            raise ValueError("Contrast sample/replicate count differs from protocol")
        finite_number(row, "estimate_db")
        lower, upper = (finite_number(row, k) for k in ("CI_lower", "CI_upper"))
        if lower > upper:
            raise ValueError(f"Reversed interval: {key}")
        contrasts[key] = row
    for row in rows(paths["decomposition"]):
        key = (*cell(row), row["method"])
        if key in decomposition:
            raise ValueError(f"Duplicate risk identity: {key}")
        for field in ("A_over_truth", "B_over_truth", "sum_over_truth",
                      "delta_nmse_linear", "identity_scaled_error"):
            finite_number(row, field)
        decomposition[key] = row
    for condition in CONDITIONS:
        for snr in SNRS:
            for method in CONTROLS:
                if (condition, snr, method) not in decomposition:
                    raise ValueError("A/B table misses a requested control cell")
    return nmse, contrasts, decomposition, {"paths": paths, "sha256": before}


def panel_label(fig, axis, label: str) -> None:
    from matplotlib.transforms import ScaledTranslation
    transform = axis.transAxes + ScaledTranslation(
        -24.0 / 72.0, 8.0 / 72.0, fig.dpi_scale_trans)
    axis.text(0, 1, label, transform=transform, fontweight="bold",
              fontsize=10, ha="left", va="bottom", clip_on=False)
    axis.set_gid(label)


def five_panels(plt, title: str, ylabel: str):
    fig, array = plt.subplots(2, 3, figsize=(11.8, 6.9))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.17, top=0.80,
                        wspace=0.36, hspace=0.68)
    axes = list(array.flat[:5])
    blank = array.flat[5]
    blank.set_axis_off()
    for index, (condition, axis) in enumerate(zip(CONDITIONS, axes)):
        panel_label(fig, axis, chr(ord("a") + index))
        axis.set_title(condition, pad=9)
        axis.set_xticks(SNRS)
        axis.set_xlim(8, 32)
        axis.set_xlabel("Source-referenced SNR (dB)")
        axis.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
        axis.grid(axis="y", color="#e6e6e6", linewidth=0.5, zorder=0)
    fig.suptitle(title, y=0.982, fontsize=12, fontweight="bold")
    fig.text(0.012, 0.51, ylabel, rotation=90, rotation_mode="anchor",
             ha="center", va="center", fontsize=9.5)
    return fig, axes, blank


def signed_limits(axis, values: list[float]) -> None:
    lower, upper = min([0.0, *values]), max([0.0, *values])
    span = upper - lower
    margin = max(0.12 * span, 0.015)
    axis.set_ylim(lower - margin, upper + margin)


def interval_series(axis, x, estimates, lower, upper, method, label=None):
    """Draw actual percentile endpoints even if the point lies outside them."""
    name, color, marker, line = STYLES[method]
    artist, = axis.plot(x, estimates, color=color, marker=marker,
                        linestyle=line, markersize=4.5,
                        label=label or name, zorder=4)
    axis.vlines(x, lower, upper, color=color, linewidth=1.0, zorder=3)
    cap = 0.25
    axis.hlines(lower, np.asarray(x)-cap, np.asarray(x)+cap,
                color=color, linewidth=1.0, zorder=3)
    axis.hlines(upper, np.asarray(x)-cap, np.asarray(x)+cap,
                color=color, linewidth=1.0, zorder=3)
    return artist


def save(fig, stem: str, output: Path, require_matplotlib_panel_alignment, *, exclude_axes=()):
    figure_dir, qa_dir = output / "figures", output / "reports" / "figure_alignment"
    figure_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    alignment = require_matplotlib_panel_alignment(
        fig, json_out=qa_dir / f"{stem}.json",
        overlay_svg=qa_dir / f"{stem}.svg", tolerance_pt=1.5,
        gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True,
        exclude_axes=list(exclude_axes))
    paths = {}
    export_paths = {"pdf": figure_dir / f"{stem}.pdf", "svg": figure_dir / f"{stem}.svg", "png": figure_dir / f"{stem}.png"}
    for suffix, path in export_paths.items():
        fig.savefig(path, format=suffix, dpi=600, facecolor="white")
        paths[suffix] = {"path": path.relative_to(output).as_posix(),
                         "sha256": sha256(path), "bytes": path.stat().st_size}
    return {"stem": stem, "size_inches": list(map(float, fig.get_size_inches())),
            "formats": paths, "png_dpi": 600, "alignment": alignment,
            "PDF_text_collision_and_human_review": "PENDING_CALLER_QA"}


def plot_weights(plt, contrasts, output, require_alignment):
    fig, axes, blank = five_panels(
        plt, "Correction weights at fixed rank four and shared coordinates",
        "Δ Pooled-energy NMSE (dB)")
    handles = []
    for condition, axis in zip(CONDITIONS, axes):
        extent = []
        axis.axhline(0, color="#333333", linewidth=0.9, zorder=2)
        for offset, method in zip((-0.45, 0.0, 0.45), OTHER_CORRECTIONS):
            records = [contrasts[(condition, snr, "original_minus_control",
                                  "CORR_ORIGIN_R4", method)] for snr in SNRS]
            point, lower, upper = [np.array([float(row[field]) for row in records])
                                   for field in ("estimate_db", "CI_lower", "CI_upper")]
            artist = interval_series(axis, np.array(SNRS)+offset, point,
                                     lower, upper, method, DISPLAY[method])
            if condition == CONDITIONS[0]:
                handles.append(artist)
            extent.extend(np.r_[point, lower, upper].tolist())
        signed_limits(axis, extent)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.93),
               ncol=3, columnspacing=1.8, handlelength=2.4, frameon=False)
    fig.text(0.09, 0.065,
             "Difference = original LR4 − alternative. Positive favors the alternative weight; all 15 cells shown; panel y-ranges vary.",
             fontsize=8.5)
    fig.text(0.09, 0.028,
             "Disclosed development data; 100 trajectories/condition; unadjusted exploratory 95% paired intervals (2,000 resamples).",
             fontsize=8.0)
    record = save(fig, "mr05_same_rank_weights", output, require_alignment,
                  exclude_axes=[blank])
    record.update(question="Does preparation-only weighting change risk at fixed rank and fitted coordinates?",
                  source_table="tables/contrasts.csv", plotted_contrasts=45,
                  display_subset="Three alternative correction weights versus the original; original-minus-self is structurally zero.",
                  uncertainty="Existing unadjusted exploratory 95% paired-trajectory intervals; no new draws.")
    plt.close(fig)
    return record


def plot_location_absolute(plt, nmse, output, require_alignment):
    fig, axes, blank = five_panels(
        plt, "Compression placement under a common pooled second-moment weight",
        "Pooled-energy NMSE (dB)")
    displayed = ("POOLED_NATURAL", "DENSE_TW5", "CORR_POOL_M2_R4", *WHOLES)
    values = [float(nmse[(condition, snr, method)]["nmse_db"])
              for condition in CONDITIONS for snr in SNRS for method in displayed]
    lower, upper = min(values), max(values)
    margin = 0.07 * max(upper-lower, 1.0)
    handles = []
    for condition, axis in zip(CONDITIONS, axes):
        axis.set_ylim(lower-margin, upper+margin)
        for method in displayed:
            label, color, marker, line = STYLES[method]
            if method == "CORR_POOL_M2_R4":
                label = "Correction rank 4 (332,288 MACs)"
            elif method in WHOLES:
                label = DISPLAY[method]
            point = [float(nmse[(condition, snr, method)]["nmse_db"]) for snr in SNRS]
            artist, = axis.plot(SNRS, point, color=color, marker=marker,
                                linestyle=line, markersize=4.5,
                                linewidth=1.7 if method == "CORR_POOL_M2_R4" else 1.3,
                                label=label, zorder=3)
            if condition == CONDITIONS[0]:
                handles.append(artist)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.944),
               ncol=3, columnspacing=1.7, handlelength=2.3, frameon=False)
    fig.text(0.09, 0.065,
             "Point estimates from the same 100 paired development trajectories/condition; shared coordinates and current observations.",
             fontsize=8.0)
    fig.text(0.09, 0.028,
             "Dominant MAC proxies bracket, rather than equal, the correction budget; paired uncertainty is shown separately.",
             fontsize=8.0)
    record = save(fig, "mr05_near_budget_location_absolute", output, require_alignment,
                  exclude_axes=[blank])
    record.update(question="How does whole-bank versus correction-only deployment change absolute risk near the same arithmetic budget?",
                  source_table="tables/nmse.csv", plotted_method_cells=75,
                  uncertainty="Point estimates only; paired location intervals are in the separate location-contrast figure.",
                  budgets=dict(correction_rank4=332288, whole_rank23=324288, whole_rank24=336384),
                  display_subset="Same-weight deployment arms with Dense and fixed Pooled backgrounds; full 135-row table retained.")
    plt.close(fig)
    return record


def plot_location_paired(plt, contrasts, output, require_alignment):
    fig, axes, blank = five_panels(
        plt, "Paired risk change from correction-only to whole-bank compression",
        "Δ Pooled-energy NMSE (dB)")
    handles = []
    for condition, axis in zip(CONDITIONS, axes):
        extent = []
        axis.axhline(0, color="#333333", linewidth=0.9, zorder=2)
        for offset, method in zip((-0.25, 0.25), WHOLES):
            records = [contrasts[(condition, snr, "whole_minus_corrpool",
                                  method, "CORR_POOL_M2_R4")] for snr in SNRS]
            point, lower, upper = [np.array([float(row[field]) for row in records])
                                   for field in ("estimate_db", "CI_lower", "CI_upper")]
            artist = interval_series(axis, np.array(SNRS)+offset, point,
                                     lower, upper, method, DISPLAY[method])
            if condition == CONDITIONS[0]:
                handles.append(artist)
            extent.extend(np.r_[point, lower, upper].tolist())
        signed_limits(axis, extent)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.93),
               ncol=2, columnspacing=2.0, handlelength=2.4, frameon=False)
    fig.text(0.09, 0.065,
             "Difference = whole − CorrPool. Positive favors correction-only; same weight, unequal near budgets; panel y-ranges vary.",
             fontsize=8.0)
    fig.text(0.09, 0.028,
             "Disclosed development data; 100 trajectories/condition; unadjusted exploratory 95% paired intervals (2,000 resamples).",
             fontsize=8.0)
    record = save(fig, "mr05_near_budget_location_paired", output, require_alignment,
                  exclude_axes=[blank])
    record.update(question="Do paired whole-bank-minus-correction contrasts have a consistent direction at the two prespecified near budgets?",
                  source_table="tables/contrasts.csv", plotted_contrasts=30,
                  uncertainty="Existing unadjusted exploratory 95% paired-trajectory intervals; no new draws.",
                  budgets=dict(correction_rank4=332288, whole_rank23=324288, whole_rank24=336384))
    plt.close(fig)
    return record


def plot_risk(plt, decomposition, output, require_alignment):
    fig, array = plt.subplots(3, 5, figsize=(13.0, 8.0))
    fig.subplots_adjust(left=0.065, right=0.987, bottom=0.18, top=0.84,
                        wspace=0.55, hspace=0.60)
    handles = []
    fields = ("A_over_truth", "B_over_truth", "sum_over_truth")
    colors = ("#0072B2", "#D55E00", "#333333")
    names = ("Output perturbation A/Q", "Residual cross term B/Q", "Signed total (A+B)/Q")
    for row_index, snr in enumerate(SNRS):
        for column_index, condition in enumerate(CONDITIONS):
            axis = array[row_index, column_index]
            panel_label(fig, axis, chr(ord("a")+row_index*5+column_index))
            axis.set_title(f"{condition} · {snr} dB", fontsize=9, pad=8)
            x = np.arange(len(CONTROLS))
            for offset, field, color, name in zip((-0.24, 0, 0.24), fields, colors, names):
                values = [float(decomposition[(condition, snr, method)][field])
                          for method in CONTROLS]
                bars = axis.bar(x+offset, values, width=0.22, color=color,
                                edgecolor=color, linewidth=0.35, label=name,
                                zorder=3)
                if row_index == 0 and column_index == 0:
                    handles.append(bars)
            axis.axhline(0, color="#333333", linewidth=0.6, zorder=2)
            axis.set_xticks(x, [str(k+1) for k in range(len(CONTROLS))])
            axis.set_xlim(-0.6, len(CONTROLS)-0.4)
            axis.tick_params(axis="both", labelsize=7.5)
            axis.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
            axis.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda value, _: f"{value:.1e}"))
            axis.margins(y=0.15)
            axis.grid(axis="y", color="#e8e8e8", linewidth=0.4, zorder=0)
    fig.suptitle("Signed compression-risk decomposition relative to Dense",
                 y=0.986, fontsize=12, fontweight="bold")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.944),
               ncol=3, columnspacing=2.0, frameon=False)
    fig.text(0.009, 0.51, "Linear-risk terms / total truth energy Q",
             rotation=90, rotation_mode="anchor", fontsize=9,
             ha="center", va="center")
    fig.text(0.065, 0.110,
             "Control key: 1 Origin covariance; 2 Identity; 3 Origin second moment; 4 Pooled second moment; 5 Whole rank 23; 6 Whole rank 24.",
             fontsize=8.0)
    fig.text(0.065, 0.072,
             "Each panel shows all six controls at one condition/SNR. Panel scales differ; signed cross terms are retained.",
             fontsize=8.0)
    fig.text(0.065, 0.035,
             "Same 100 development trajectories/condition; pooled energy sums. No uncertainty shown for this exact observed risk bookkeeping.",
             fontsize=8.0)
    record = save(fig, "mr05_signed_risk_terms", output, require_alignment)
    record.update(question="How do perturbation energy and the signed Dense-residual cross term combine for each compressed deployment?",
                  source_table="tables/decomposition.csv", plotted_control_cells=90,
                  component_bars=270,
                  controls={str(k+1): method for k, method in enumerate(CONTROLS)},
                  uncertainty="Observed pooled energy identities, no interval; not a claim of an independent additive error model or a physical denoising mechanism.")
    plt.close(fig)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qa-scripts", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    qa_scripts = args.qa_scripts.resolve()
    if not (qa_scripts / "audit_panel_alignment.py").is_file():
        raise FileNotFoundError("Panel alignment helper is required")
    sys.path.insert(0, str(qa_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    nmse, contrasts, decomposition, sources = source_tables(output)
    with mpl.rc_context(RC):
        records = [
            plot_weights(plt, contrasts, output, require_matplotlib_panel_alignment),
            plot_location_absolute(plt, nmse, output, require_matplotlib_panel_alignment),
            plot_location_paired(plt, contrasts, output, require_matplotlib_panel_alignment),
            plot_risk(plt, decomposition, output, require_matplotlib_panel_alignment),
        ]
    if any(sha256(path) != sources["sha256"][name]
           for name, path in sources["paths"].items()):
        raise RuntimeError("Source tables changed during figure rendering")
    report = {
        "source_csv_sha256": sources["sha256"],
        "source_counts": {"nmse": len(nmse), "contrasts": len(contrasts),
                          "decomposition": len(decomposition)},
        "script_sha256": sha256(Path(__file__).resolve()),
        "backend": "Python/Matplotlib", "source_csv_modified": False,
        "new_scientific_arrays_or_bootstrap_draws": 0,
        "scope": "Disclosed development data, 15 cells, 100 trajectories/condition",
        "figures": records,
        "remaining_QA": "Caller must run source, exported-PDF glyph/collision audits and inspect every final image.",
    }
    path = output / "reports" / "plot_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"figures": len(records), "formats_per_figure": 3,
                      "CSV_unchanged": True, "alignment_gate": "PASSED_BEFORE_EXPORT",
                      "human_visual_QA": "PENDING"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
