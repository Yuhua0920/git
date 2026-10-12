"""T16A two CSV-only figures, scoped Python style, deterministic exports."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
ROOT = Path(__file__).resolve().parents[1]
SKILL = Path("C:/Users/yuhua/.codex/skills/nature-figure/scripts")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/".pytest_cache/matplotlib-t16a"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment
from scripts.run_cdl_mismatch import sha
STYLE = {"font.family":"sans-serif", "font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
         "font.size":10, "legend.fontsize":10, "svg.fonttype":"none", "svg.hashsalt":"T16A-fixed",
         "pdf.fonttype":42, "axes.spines.right":False, "axes.spines.top":False, "legend.frameon":False}
SHIFTS = ("CDL-B-30", "CDL-C-120")
STEMS = ("t16a_lowrank_gap_closure", "t16a_lowrank_weighted_energy")
CURVES = ((5,"#0072B2","o","-"), (10,"#D55E00","s","--"), (20,"#009E73","^","-."))


def plot(input_dir, output_dir):
    directory = Path(output_dir); directory.mkdir(parents=True, exist_ok=True)
    with (Path(input_dir)/"t16a_lowrank_filter_correction.csv").open(encoding="utf8") as f: metrics = list(csv.DictReader(f))
    with (Path(input_dir)/"t16a_lowrank_singular_spectrum.csv").open(encoding="utf8") as f: spectrum = list(csv.DictReader(f))
    with matplotlib.rc_context(STYLE):
        for index, stem in enumerate(STEMS):
            fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.8), sharey=True)
            fig.subplots_adjust(left=.12, right=.97, bottom=.28, top=.72, wspace=.22)
            for ax, name, letter, title in zip(axes, SHIFTS, ("a","b"), ("CDL-B / 30 km/h", "CDL-C / 120 km/h")):
                for s, color, marker, linestyle in CURVES:
                    if index == 0:
                        v = sorted((r for r in metrics if r["shift"] == name and int(r["snr_db"]) == s and r["means"] == "target" and r["rank"] != ""), key=lambda r:int(r["rank"]))
                        x = [int(r["rank"]) for r in v]; y = [float(r["closure_r"]) for r in v]
                    else:
                        v = sorted((r for r in spectrum if r["shift"] == name and int(r["snr_db"]) == s), key=lambda r:int(r["singular_index"]))
                        x = [0]+[int(r["singular_index"]) for r in v]; y = [0.]+[float(r["cumulative_weighted_energy"]) for r in v]
                    ax.plot(x, y, color=color, marker=marker, linestyle=linestyle, markerfacecolor="white",
                            linewidth=1.5, markersize=4, markevery=None if index == 0 else [0,4,8,16,32,48,64], label=f"{s} dB")
                ax.set_title(title, fontsize=11, pad=12); ax.set_xticks([0,8,16,32,48,64]); ax.set_xlabel("Correction rank r")
                ax.set_xlim(-1,65); ax.grid(axis="y", color="0.9", linewidth=.6); ax.set_axisbelow(True)
                ax.axvline(8, color="0.7", linewidth=.8, linestyle=":")
                if index == 0: ax.axhline(.8, color="0.7", linewidth=.8, linestyle=":")
                ax.annotate(letter, xy=(0,1), xycoords="axes fraction", xytext=(-20,12), textcoords="offset points", fontweight="bold", fontsize=11)
            axes[0].set_ylabel("Frozen-to-matched gap closure" if index == 0 else "Cumulative weighted energy fraction")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(.55,.90), columnspacing=2.)
            fig.suptitle("Oracle low-rank correction: validation gap recovery" if index == 0 else "Target-covariance-weighted correction spectrum", fontsize=12, y=.98)
            fig.text(.12,.105,"100 existing validation trajectories/shift; target means; dF=16", fontsize=9)
            fig.text(.12,.045,"Closure not clipped; dotted r=8 and closure=0.8; pooled ratios, no CI" if index == 0 else "SVD of (W target - W source)L; dotted r=8; all 64 modes", fontsize=9)
            require_matplotlib_panel_alignment(fig, json_out=str(directory/f"{stem}.alignment.json"), strict=True)
            fig.savefig(directory/f"{stem}.png", dpi=300)
            fig.savefig(directory/f"{stem}.pdf", metadata={"CreationDate":None,"ModDate":None})
            fig.savefig(directory/f"{stem}.svg", metadata={"Date":None})
            plt.close(fig)
    return STEMS


def audit(input_dir, output_dir):
    directory = Path(output_dir); plot(input_dir, directory); checks = {}
    source = subprocess.run([sys.executable, str(SKILL/"validate_figure.py"), str(Path(__file__)), "--json"], capture_output=True, text=True)
    for stem in STEMS:
        text = subprocess.run([sys.executable, str(SKILL/"audit_pdf_text.py"), str(directory/f"{stem}.pdf"), "--min-pt","5","--json"], capture_output=True, text=True)
        collision = subprocess.run([sys.executable, str(SKILL/"audit_figure_collisions.py"), str(directory/f"{stem}.pdf"), "--json-out", str(directory/f"{stem}.collision-audit.json")], capture_output=True, text=True)
        checks[stem] = dict(text_returncode=text.returncode, collision_returncode=collision.returncode,
                           text_output=text.stdout+text.stderr, collision_output=collision.stdout+collision.stderr)
    with tempfile.TemporaryDirectory(dir=ROOT/".pytest_cache") as d:
        temp = Path(d)
        for name in ("t16a_lowrank_filter_correction.csv", "t16a_lowrank_singular_spectrum.csv"):
            shutil.copy2(Path(input_dir)/name, temp)
        plot(temp, temp/"rebuilt")
        rebuilt = {f"{stem}.{ext}":sha(directory/f"{stem}.{ext}") == sha(temp/"rebuilt"/f"{stem}.{ext}") for stem in STEMS for ext in ("png","pdf","svg")}
    result = dict(source_returncode=source.returncode, source_output=source.stdout+source.stderr,
        rendered_checks=checks, CSV_only_rebuild=rebuilt, visual_review="pending", all_checks_passed=False,
        figure_contract=dict(backend="python", archetype="paired quantitative grid", width_mm=182.88,
            conclusion="Test low-rank representation capacity and inspect weighted spectral concentration",
            uncertainty="pooled ratios; independent-trajectory dispersion saved separately; no CI claimed",
            source_data=["t16a_lowrank_filter_correction.csv", "t16a_lowrank_singular_spectrum.csv"],
            exclusions="baselines and source-mean sensitivity intentionally not in primary closure plot; all primary ranks included",
            exports=["PNG300dpi", "editable PDF", "editable SVG"]))
    (directory/"t16a_lowrank_figure_audit.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf8")
    if source.returncode or any(v["text_returncode"] or v["collision_returncode"] for v in checks.values()) or not all(rebuilt.values()):
        raise ValueError("T16A figure gate failed; numerical CSV protected")
    print("T16A figure source/alignment/text/collision and CSV-only byte rebuild PASS; visual review pending")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--input-dir", default=str(ROOT/"outputs")); p.add_argument("--output-dir", default=str(ROOT/"outputs")); p.add_argument("--audit", action="store_true")
    a = p.parse_args()
    if a.audit: audit(a.input_dir, a.output_dir)
    else: plot(a.input_dir, a.output_dir)
