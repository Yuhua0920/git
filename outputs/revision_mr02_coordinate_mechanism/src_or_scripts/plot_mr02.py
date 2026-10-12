"""Render MR-02 diagnostic figures from persisted small sufficient statistics.

Figure contracts (Python backend, no scientific computation or new random draws):
1. Both output spaces: show online, fixed pilot/data references and truth-assisted
   oracle relative to the same Pooled baseline in every condition/SNR. Panels
   stratify by condition and output space; all intervals are supplied exploratory
   paired trajectory intervals, conditional on the independently fitted references.
2. Four prespecified coordinate cases: retain all online and oracle coordinates,
   including tails, alongside the fixed references and Convex triangle. Coordinates
   are a descriptive display; Euclidean separation is not output-risk evidence.
3. Compression: display signed linear A/truth, B/truth and their sum. Never take
   logarithms of the possibly negative cross term; the arithmetic identity is not
   a causal mechanism. All fifteen cells are retained.

Development-only presentation: editable PDF/SVG, 600 dpi PNG; no manuscript edits.
Root completes final PDF collision/glyph checks and actual image inspection.
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
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import ScaledTranslation
import numpy as np

CONDITIONS = ("ID", "Profile", "Mobility", "Composed", "Boundary")
SNRS = (10, 20, 30)
SPACES = ("Dense", "LR4")
CASES = (("Composed", 10), ("Composed", 20), ("Composed", 30), ("Boundary", 30))
KINDS = ("online", "cp", "cd", "oracle")
COLORS = {"online": "#0072B2", "cp": "#E69F00", "cd": "#009E73", "oracle": "#CC79A7"}
STYLES = {
    "online": dict(color=COLORS["online"], marker="o", linestyle="-"),
    "cp": dict(color=COLORS["cp"], marker="s", linestyle="-."),
    "cd": dict(color=COLORS["cd"], marker="D", linestyle="-"),
    "oracle": dict(color=COLORS["oracle"], marker="^", linestyle="--"),
}
DISPLAY = {"online": "Original online", "cp": "Fixed pilot reference", "cd": "Fixed data reference", "oracle": "Per-frame oracle (truth-assisted)"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path, keys):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(keys).issubset(reader.fieldnames or []):
            raise ValueError("Missing CSV schema: " + str(path))
        result = list(reader)
    if not result:
        raise ValueError("Empty CSV: " + str(path))
    return result


def num(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError("Nonfinite plotting field: " + key)
    return value


def configure():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5,
        "svg.fonttype": "none", "pdf.fonttype": 42, "text.usetex": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "lines.linewidth": 1.0, "legend.frameon": False,
        "savefig.dpi": 600,
    })


def panel_label(fig, ax, label):
    transform = ax.transAxes + ScaledTranslation(-8 / 72, 5 / 72, fig.dpi_scale_trans)
    ax.text(0, 1, label, transform=transform, fontsize=8, fontweight="bold", ha="right", va="bottom", clip_on=False)


def axes_style(ax):
    ax.grid(axis="y", color="#DDDDDD", linewidth=.45, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(length=2.5, width=.65, pad=2.5)


def limits(values, zero=False, minimum_margin=.05):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite axis extent")
    lo, hi = float(values.min()), float(values.max())
    if zero:
        lo, hi = min(lo, 0), max(hi, 0)
    margin = max((hi - lo) * .08, minimum_margin)
    return lo - margin, hi + margin


def save(fig, axes, stem, require_matplotlib_panel_alignment):
    labels = list("abcdefghijklmnopqrstuvwxyz")[:len(axes)]
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=stem.with_suffix(".alignment.json"),
              overlay_svg=stem.with_suffix(".alignment.svg"), axes=axes,
              panel_ids=labels, tolerance_pt=1.5, gutter_tolerance_pt=1.5,
              require_panel_labels=True, strict=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


def load_risk(out):
    nr = rows(out / "tables/nmse.csv", ("condition", "snr_db", "method", "nmse_db", "CI_lower", "CI_upper", "trajectory_count"))
    cr = rows(out / "tables/contrasts.csv", ("condition", "snr_db", "space", "contrast", "estimate_db", "CI_lower", "CI_upper"))
    nmse = {(r["condition"], int(num(r, "snr_db")), r["method"]): r for r in nr}
    contrasts = {(r["condition"], int(num(r, "snr_db")), r["space"], r["contrast"]): r for r in cr}
    if len(nmse) != len(nr) or len(contrasts) != len(cr):
        raise ValueError("Duplicate plotting identity")
    selected = {}
    for condition in CONDITIONS:
        for snr in SNRS:
            pooled = num(nmse[condition, snr, "Pooled"], "nmse_db")
            for space in SPACES:
                for kind in KINDS:
                    method = f"{space}-{kind}"
                    row = contrasts[condition, snr, space, kind + "-Pooled"]
                    estimate = num(row, "estimate_db")
                    if not np.isclose(estimate, num(nmse[condition, snr, method], "nmse_db") - pooled, rtol=0, atol=1e-10):
                        raise ValueError("Paired sign/value is not method minus Pooled")
                    if num(row, "CI_lower") > num(row, "CI_upper"):
                        raise ValueError("Reversed paired interval")
                    selected[condition, snr, space, kind] = row
    return selected


def plot_risk(out, directory, alignment):
    table = load_risk(out)
    ylimits = limits([num(r, f) for r in table.values() for f in ("estimate_db", "CI_lower", "CI_upper")], zero=True)
    fig, array = plt.subplots(2, 5, figsize=(8.66141732, 5.51181102), sharey=True)  # 220×140 mm development figure
    fig.subplots_adjust(left=.07, right=.985, bottom=.23, top=.875, hspace=.47, wspace=.30)
    fig.suptitle("Output-space attainability versus the fixed Pooled baseline", y=.97, fontsize=9)
    axes = list(array.ravel())
    handles = []
    for ri, space in enumerate(SPACES):
        for ci, condition in enumerate(CONDITIONS):
            ax = array[ri, ci]
            panel_label(fig, ax, "abcdefghij"[ri * 5 + ci])
            axes_style(ax)
            ax.set_title(f"{condition} · {space}", pad=11)
            ax.axhline(0, color="#777777", linestyle=":", linewidth=.8, zorder=1)
            ax.set_ylim(*ylimits); ax.set_xlim(8, 32); ax.set_xticks(SNRS)
            if ri == 1:
                ax.set_xlabel("SNR (dB)", labelpad=3)
            if ci == 0:
                ax.set_ylabel("Method−Pooled NMSE (dB)", labelpad=5)
            for ki, kind in enumerate(KINDS):
                rr = [table[condition, s, space, kind] for s in SNRS]
                y = np.array([num(r, "estimate_db") for r in rr])
                lower = np.array([num(r, "CI_lower") for r in rr])
                upper = np.array([num(r, "CI_upper") for r in rr])
                # Literal percentile endpoints; do not assume they bracket point estimate.
                ax.errorbar(SNRS, (lower + upper) / 2, yerr=(upper - lower) / 2,
                            fmt="none", ecolor=COLORS[kind], elinewidth=.65, capsize=1.6, zorder=2)
                line, = ax.plot(SNRS, y, markersize=3.2, markeredgewidth=.5, zorder=3, **STYLES[kind])
                if ri == ci == 0:
                    handles.append(line)
    fig.legend(handles, [DISPLAY[k] for k in KINDS], loc="lower center", bbox_to_anchor=(.5, .085), ncol=2,
               handlelength=2.2, columnspacing=1.6, labelspacing=.7)
    fig.text(.07, .047, "Negative: lower NMSE than Pooled. Bars: exploratory unadjusted 95% paired trajectory intervals.", fontsize=7.0)
    fig.text(.07, .017, "100 evaluation trajectories/condition; frames 4–9. Intervals condition on independently fitted fixed references.", fontsize=7.0)
    save(fig, axes, directory / "mr02_attainable_risk", alignment)


def load_coordinates(out):
    """Accept the explicitly documented final summary axes, not sampled examples."""
    source = out / "data/eval_summary.npz"
    with np.load(source, allow_pickle=False) as z:
        online = z["online"].copy()
        oracle = z["oracle"].copy()
    if online.shape != (5, 100, 3, 6, 2) or oracle.shape != (5, 100, 3, 6, 2, 2):
        raise ValueError("Coordinate shape: conditions, trajectories, SNR, six frames, [space], two real coordinates")
    if not np.isfinite(online).all() or not np.isfinite(oracle).all():
        raise ValueError("Nonfinite coordinates; no silent exclusion")
    with np.load(out / "data/fixed_references.npz", allow_pickle=False) as z:
        cp, cd = z["cp"].copy(), z["cd"].copy()
    if cp.shape != (5, 3, 2) or cd.shape != (5, 3, 2, 2) or not np.isfinite(cp).all() or not np.isfinite(cd).all():
        raise ValueError("Fifteen fixed pilot references and two separately fitted data references are required")
    fixed = {(c, s): dict(alpha_cp=cp[ci, si, 0], beta_cp=cp[ci, si, 1],
                         alpha_cd_D=cd[ci, si, 0, 0], beta_cd_D=cd[ci, si, 0, 1],
                         alpha_cd_L=cd[ci, si, 1, 0], beta_cd_L=cd[ci, si, 1, 1])
             for ci, c in enumerate(CONDITIONS) for si, s in enumerate(SNRS)}
    return online, oracle, fixed


def plot_coordinates(out, directory, alignment):
    online, oracle, fixed = load_coordinates(out)
    fig, array = plt.subplots(2, 2, figsize=(7.08661417, 6.18110236))  # 180×157 mm
    fig.subplots_adjust(left=.10, right=.975, bottom=.28, top=.90, wspace=.28, hspace=.38)
    fig.suptitle("Pilot/data coordinate diagnostics in four prespecified cases", y=.976, fontsize=9)
    markers = [("alpha_cp", "beta_cp", "*", "#E69F00", "Fixed pilot reference"),
               ("alpha_cd_D", "beta_cd_D", "D", "#009E73", "Fixed Dense data reference"),
               ("alpha_cd_L", "beta_cd_L", "s", "#D55E00", "Fixed LR4 data reference")]
    for ci, ((condition, snr), ax) in enumerate(zip(CASES, array.ravel())):
        condition_id = CONDITIONS.index(condition); si = SNRS.index(snr)
        on = online[condition_id, :, si].reshape(-1, 2)
        od = oracle[condition_id, :, si, :, 0].reshape(-1, 2)
        ol = oracle[condition_id, :, si, :, 1].reshape(-1, 2)
        points = np.vstack([on, od, ol, [[0, 0], [1, 0], [0, 1]]])
        ref = fixed[condition, snr]
        points = np.vstack([points, [[num(ref, x), num(ref, y)] for x, y, _, _, _ in markers]])
        if len(on) != 600 or len(od) != 600 or len(ol) != 600:
            raise ValueError("Six frame coordinates for every evaluation trajectory must be visible")
        ax.scatter(on[:, 0], on[:, 1], s=3.0, color="#666666", alpha=.26, linewidths=0, rasterized=True, zorder=2)
        ax.scatter(od[:, 0], od[:, 1], s=3.5, marker="^", color="#0072B2", alpha=.24, linewidths=0, rasterized=True, zorder=2)
        ax.scatter(ol[:, 0], ol[:, 1], s=3.5, marker="v", color="#CC79A7", alpha=.24, linewidths=0, rasterized=True, zorder=2)
        ax.plot([0, 1, 0, 0], [0, 0, 1, 0], color="#333333", linestyle="--", linewidth=.8, zorder=1)
        for x, y, marker, color, _ in markers:
            ax.scatter([num(ref, x)], [num(ref, y)], marker=marker, color=color,
                       s=43 if marker == "*" else 24, edgecolors="black", linewidths=.55, zorder=4)
        panel_label(fig, ax, "abcd"[ci]); axes_style(ax)
        ax.set_xlim(*limits(points[:, 0])); ax.set_ylim(*limits(points[:, 1]))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_title(f"{condition}, {snr} dB", pad=11)
        ax.set_xlabel("α coordinate", labelpad=3); ax.set_ylabel("β coordinate", labelpad=4)
    handles = [Line2D([], [], marker="o", color="#666666", linestyle="none", markersize=3, label="Original online coordinates"),
               Line2D([], [], marker="^", color="#0072B2", linestyle="none", markersize=3, label="Dense oracle (truth-assisted)"),
               Line2D([], [], marker="v", color="#CC79A7", linestyle="none", markersize=3, label="LR4 oracle (truth-assisted)")]
    handles += [Line2D([], [], marker=m, color=color, markeredgecolor="black", markeredgewidth=.5, linestyle="none", markersize=6, label=label)
                for _, _, m, color, label in markers]
    handles += [Line2D([], [], color="#333333", linestyle="--", linewidth=.8, label="Convex triangle")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .07), ncol=2, handlelength=1.5,
               columnspacing=1.7, labelspacing=.65)
    fig.text(.10, .037, "Each cloud retains all 600 frame coordinates; no tail trimming or sample selection.", fontsize=7.0)
    fig.text(.10, .012, "Coordinate separation is descriptive; output risk uses the corresponding data Gram and gradient.", fontsize=7.0)
    save(fig, list(array.ravel()), directory / "mr02_coordinate_cases", alignment)


def plot_compression(out, directory, alignment):
    cr = rows(out / "tables/compression.csv", ("condition", "snr_db", "A_over_truth", "B_over_truth", "sum_over_truth", "LR4_minus_Dense_linear", "identity_error"))
    table = {(r["condition"], int(num(r, "snr_db"))): r for r in cr}
    if len(table) != 15:
        raise ValueError("All fifteen compression cells required")
    fig, array = plt.subplots(3, 5, figsize=(8.66141732, 6.49606299))  # 220×165 mm development figure
    fig.subplots_adjust(left=.085, right=.985, bottom=.16, top=.88, wspace=.40, hspace=.63)
    fig.suptitle("Original rank-four compression: energy and signed risk cross term", y=.974, fontsize=9)
    for ri, snr in enumerate(SNRS):
        extents = [num(table[c, snr], f) for c in CONDITIONS for f in ("A_over_truth", "B_over_truth", "sum_over_truth")]
        yl = limits(extents, zero=True, minimum_margin=1e-12)
        for ci, condition in enumerate(CONDITIONS):
            row = table[condition, snr]; ax = array[ri, ci]
            a, b, total = (num(row, f) for f in ("A_over_truth", "B_over_truth", "sum_over_truth"))
            if not np.isclose(a + b, total, rtol=1e-10, atol=1e-12) or not np.isclose(total, num(row, "LR4_minus_Dense_linear"), rtol=1e-10, atol=1e-12):
                raise ValueError("Compression identity is not closed")
            ax.bar([0, 1, 2], [a, b, total], width=.65, color=["#0072B2", "#E69F00", "#666666"],
                   edgecolor="#333333", linewidth=.45, zorder=3)
            ax.axhline(0, color="#555555", linestyle="-", linewidth=.8, zorder=2)
            axes_style(ax); panel_label(fig, ax, "abcdefghijklmno"[ri * 5 + ci])
            ax.set_ylim(*yl); ax.set_xlim(-.6, 2.6); ax.set_xticks([0, 1, 2], ["A", "B", "A+B"])
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=False)
            ax.set_title(f"{condition}, {snr} dB", pad=13)
            if ci == 0:
                ax.set_ylabel("Signed energy / truth energy", labelpad=5)
    handles = [Patch(facecolor="#0072B2", edgecolor="#333333", label="A: correction-change energy"),
               Patch(facecolor="#E69F00", edgecolor="#333333", label="B: signed risk cross term"),
               Patch(facecolor="#666666", edgecolor="#333333", label="A+B: LR4−Dense linear NMSE")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .058), ncol=2,
               columnspacing=1.5, labelspacing=.75)
    fig.text(.085, .015, "Same online coordinates in Dense and LR4; point energy decomposition only. Cross term B can be negative.", fontsize=7.0)
    save(fig, list(array.ravel()), directory / "mr02_compression_terms", alignment)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--figure-skill-dir", type=Path, default=Path.home() / ".codex/skills/nature-figure")
    a = p.parse_args(argv)
    out = a.out.resolve(); helper = a.figure_skill_dir.resolve() / "scripts"
    sys.path.insert(0, str(helper))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    configure(); directory = out / "figures"; directory.mkdir(parents=True, exist_ok=True)
    plot_risk(out, directory, require_matplotlib_panel_alignment)
    plot_coordinates(out, directory, require_matplotlib_panel_alignment)
    plot_compression(out, directory, require_matplotlib_panel_alignment)
    sources = ["tables/nmse.csv", "tables/contrasts.csv", "tables/compression.csv", "data/eval_summary.npz", "data/fixed_references.npz"]
    provenance = dict(backend="Python-matplotlib", source_hashes={f: sha(out / f) for f in sources}, script_sha256=sha(__file__),
                      formats=["PDF", "SVG", "600 dpi PNG"], source_rows_excluded=0,
                      coordinate_cases="Prespecified Composed 10/20/30 and Boundary30; all online/oracle frame coordinates in each case retained",
                      uncertainty="Exploratory unadjusted 95% paired intervals from saved MR01 trajectory draws, conditional on fixed references",
                      compression_uncertainty="Not displayed: exact empirical point-energy identity, not an effect hypothesis test",
                      alignment="Final rendered plot-area bounds; tolerance1.5pt; no exemptions",
                      remaining_QA="Run final exported PDF glyph/collision QA and view every complete figure/panel")
    (directory / "plot_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    captions = (
        "Attainable risk. Differences of pooled-energy NMSEs in dB relative to the same Natural Pooled baseline. "
        "Rows distinguish Dense and rank-four output spaces; columns show all five prespecified conditions. "
        "Fixed pilot/data coordinates are fitted only on independent D_ref. Truth-assisted per-frame oracles are "
        "post hoc reachability diagnostics, not deployable methods or Bayes bounds. Bars show supplied exploratory "
        "unadjusted 95% paired trajectory-percentile intervals from 2000 saved MR01 resamples of the 100 D_eval "
        "trajectories per condition, preserving six frames and all methods/SNRs. These intervals exclude reference "
        "fitting uncertainty. Positive differences mean higher NMSE than Pooled.\n\n"
        "Coordinate diagnostics. Four prespecified condition/SNR cases retain all 600 online coordinates and "
        "600 truth-assisted oracle coordinates from each output space. The shared pilot reference, separate fixed "
        "Dense/LR4 data references and Convex triangle are shown. Full coordinate tails determine each axis extent; "
        "no samples are clipped or selected. This descriptive scatter does not substitute for corresponding "
        "data-Gram risk or establish a causal mechanism.\n\n"
        "Compression identity. For the same original online coordinates, A is the squared Dense-to-LR4 estimate "
        "change, B is twice its real inner product with the Dense estimation error, and A+B is the linear "
        "LR4-minus-Dense error-energy difference. Each component is summed over the same scored positions and "
        "divided by their common truth-energy total. Values are signed linear quantities; B can be negative. "
        "Point decompositions do not prove denoising or a regularization mechanism. All15 cells are displayed.\n"
    )
    (directory / "captions.txt").write_text(captions, encoding="utf8")
    print(json.dumps(dict(status="EXPORTED", figures=["mr02_attainable_risk", "mr02_coordinate_cases", "mr02_compression_terms"])))


if __name__ == "__main__":
    main()
