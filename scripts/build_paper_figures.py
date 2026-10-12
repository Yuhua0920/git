"""Appearance-only replay of frozen CSV figure geometry, never an experiment."""
import argparse
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_support import ROOT, json_write, require_sources, sha256

SOURCES = ("outputs/t20b_formal_bootstrap.csv", "scripts/plot_formal_test.py")
STYLE = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
         "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
         "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
         "svg.fonttype": "none", "pdf.fonttype": 42, "svg.hashsalt": "T20C-paper-figures"}

ENCODING = {
    'Perfect CSI': ('Perfect', 'x', ':', '#777777'),
    'LS+Linear': ('LS-LIN', '+', '--', '#aaaaaa'),
    'Source': ('Source', 's', '--', '#444444'),
    'Hard': ('Hard', 'v', ':', '#d6a945'),
    'Convex': ('Convex', '^', '-.', '#0072b2'),
    'Dense affine': ('Dense Affine', 'D', (0, (5, 2)), '#56b4e9'),
    'Final LR4 affine': ('LR4 Affine', 'o', '-', '#d55e00'),
    'Matched reference': ('Matched', '*', (0, (1, 1)), '#009e73'),
}
PANEL_TITLES = ('Source-matched\nC / 30 / 100', 'Profile\nB / 30 / 100',
                'Mobility\nC / 120 / 100', 'Composed\nB / 120 / 100', 'Boundary\nC / 30 / 200')


def style_metric_axes(fig):
    """Only encoding and text change; numeric marks and uncertainty stay intact."""
    for ax, title in zip(fig.axes, PANEL_TITLES):
        ax.set_title(title, fontsize=8, pad=14)
        for line in ax.lines:
            label, marker, linestyle, color = ENCODING[line.get_label()]
            line.set_label(label); line.set_marker(marker); line.set_linestyle(linestyle)
            line.set_color(color); line.set_markersize(3.5)
            if label in ('Dense Affine', 'Convex'):
                line.set_markerfacecolor('none'); line.set_markersize(4.5)
            if label == 'Matched':
                line.set_markersize(4.5)
            line.set_linewidth(1.3 if label == 'LR4 Affine' else .8)
    for legend in list(fig.legends):
        legend.remove()
    handles, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.53, .055),
               ncol=4, frameon=False, columnspacing=1., fontsize=8)
    fig.subplots_adjust(top=.77)


def original(root):
    spec = importlib.util.spec_from_file_location("paper_frozen_plot", Path(root) / SOURCES[1])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def geometry(fig):
    """Capture numeric marks/CI endpoints; typography and canvas coordinates excluded."""
    return [{"lines": [{"x": np.asarray(line.get_xdata(), dtype=float).tolist(),
                       "y": np.asarray(line.get_ydata(), dtype=float).tolist()}
                      for line in ax.lines],
             "intervals": [[np.asarray(segment, dtype=float).tolist()
                            for segment in collection.get_segments()]
                           for collection in ax.collections if hasattr(collection, "get_segments")],
             "yscale": ax.get_yscale()} for ax in fig.axes]


def clean_svg_export(path):
    """Remove export whitespace only after exact SVG drawing-token comparison."""
    path=Path(path);before=path.read_text(encoding='utf8')
    after='\n'.join(line.rstrip(' \t') for line in before.splitlines())+'\n'
    def semantics(text):
        return [(node.tag, {key:(' '.join(value.split()) if key in ('d','points') else value)
                            for key,value in node.attrib.items()},
                 node.text if node.text and not node.text.isspace() else None,
                 node.tail if node.tail and not node.tail.isspace() else None)
                for node in ET.fromstring(text).iter()]
    if semantics(before)!=semantics(after):
        raise ValueError('SVG whitespace cleanup changed drawing or text semantics')
    path.write_text(after,encoding='utf8',newline='\n')
    return dict(svg_drawing_tokens_exact=True, meaningful_text_exact=True,
                removed_trailing_whitespace_lines=sum(a!=b for a,b in zip(before.splitlines(),after.splitlines())))


def capture_marks(root=ROOT, paper_style=False):
    module = original(root)
    marks = {}
    if paper_style:
        module.STYLE.update(STYLE)
    def capture(fig, path):
        marks[Path(path).name.replace("t20b_", "")] = geometry(fig)
        plt.close(fig)
    module.export = capture
    module.plot(Path(root) / "outputs")
    return marks


def build(root=ROOT, output=None):
    root = Path(root)
    output = Path(output) if output else root / "paper/figures"
    output.mkdir(parents=True, exist_ok=True)
    hashes = require_sources(root, SOURCES)
    before = capture_marks(root, paper_style=False)
    after = capture_marks(root, paper_style=True)
    if before != after:
        raise ValueError("Appearance revision changed scientific line/CI geometry")
    module = original(root)
    module.STYLE.update(STYLE)
    produced = [];svg_cleanup={}
    def export(fig, old_path):
        path = output / Path(old_path).name.replace("t20b_", "")
        if geometry(fig) != before[path.name]:
            raise ValueError("Final rendering changed frozen plotted geometry")
        if path.name in ("formal_nmse", "formal_ber"):
            style_metric_axes(fig)
            if path.name == "formal_ber":
                fig.subplots_adjust(left=.105)
        else:
            for ax, text in zip(fig.axes, ('Source − LR4\nNMSE (dB)', 'Convex − LR4\nNMSE (dB)', 'LR4 − Dense\nNMSE (dB)')):
                ax.set_ylabel(text)
                ax.tick_params(axis='x', labelsize=8)
            fig.texts[0].set_text('Grey: paired 95% interval; blue: one-sided Bonferroni bound')
            fig.texts[1].set_text('Dashed: zero gain / predefined 0.50-dB compression margin')
        if geometry(fig) != before[path.name]:
            raise ValueError("Encoding revision changed frozen geometry")
        fig.canvas.draw()
        module.require_matplotlib_panel_alignment(fig, json_out=str(path) + ".alignment.json",
            overlay_svg=str(path) + ".alignment.svg", tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
        fig.savefig(str(path) + ".pdf", metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(str(path) + ".svg", metadata={"Date": None})
        fig.savefig(str(path) + ".png", dpi=600)
        for ext in ('.svg','.alignment.svg'):
            svg_cleanup[path.name+ext]=clean_svg_export(str(path)+ext)
        produced.extend(path.name + ext for ext in (".pdf", ".svg", ".png", ".alignment.json", ".alignment.svg"))
        plt.close(fig)
    module.export = export
    module.plot(root / "outputs")
    report = {"task": "T20F", "source_experiment": "T20B frozen evaluation",
              "source_sha256": hashes, "generation_script": "scripts/build_paper_figures.py",
              "generation_script_sha256": sha256(root / "scripts/build_paper_figures.py"),
              "formatting_only": True, "original_figure_data_geometry_exact": before == after,
              "csv_rows_read": 220, "numeric_series_rows_rendered": 180,
              "declared_perfect_csi_nmse_na_rows": 10,
              "secondary_ber_difference_rows_not_plotted": 30,
              "numeric_geometry": after, "rendered_base_font_pt": 9,
              "method_encoding": ENCODING, "target_width_mm": 174,
              "scientific_data_recomputed": False,
              "svg_export_whitespace_cleanup": svg_cleanup,
              "generated_sha256": {p: sha256(output / p) for p in sorted(produced)}}
    json_write(output / "t20f_provenance.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    r = build(output=args.output)
    print({k: r[k] for k in ("formatting_only", "original_figure_data_geometry_exact", "csv_rows_read")})
