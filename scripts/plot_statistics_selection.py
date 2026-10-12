"""T16F CSV-only paired-subset performance and candidate-weight geometry."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SKILL = Path('C:/Users/yuhua/.codex/skills/nature-figure/scripts')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.pytest_cache/matplotlib-t16f'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment

STYLE = {
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10, 'legend.fontsize': 9, 'svg.fonttype': 'none',
    'svg.hashsalt': 'T16F-fixed', 'pdf.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
}
STEMS = ('t16f_selection_mixing_nmse', 't16f_b120_weight_geometry')
DATA_FILES = ('t16f_selection_mixing.csv', 't16f_weight_geometry.csv')
METHODS = (
    ('source', 'Source', '0.5', 's', '--'),
    ('hard', 'Hard', '#CC79A7', '^', ':'),
    ('convex', 'Convex', '#E69F00', 'D', '-.'),
    ('affine', 'Affine', '#0072B2', 'o', '-'),
    ('matched', 'Matched', '#009E73', 'v', '--'),
)
SNRS = (5, 10, 20)
SHIFTS = ('CDL-C-75', 'CDL-B-120')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path, required):
    with Path(path).open(encoding='utf8', newline='') as handle:
        reader = csv.DictReader(handle)
        if not set(required).issubset(reader.fieldnames or ()):
            raise ValueError(f'{Path(path).name}: required fields missing')
        return list(reader)


def number(row, field):
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f'non-finite {field}; rows must not be silently discarded')
    return value


def performance_series(rows):
    """Map all five methods to the same causal-five-frame evaluation subset."""
    selected = [r for r in rows if int(r['window']) == 5 and r['subset'] == 'last6'
                and r['shift'] in SHIFTS and r['method'] in {m[0] for m in METHODS}]
    mapping = {}
    for shift in SHIFTS:
        for method, *_ in METHODS:
            group = sorted((r for r in selected if r['shift'] == shift and r['method'] == method),
                           key=lambda r: int(r['snr_db']))
            if [int(r['snr_db']) for r in group] != list(SNRS):
                raise ValueError('exactly one paired last6 row per method/shift/SNR is required')
            mapping[(shift, method)] = [number(r, 'nmse_db') for r in group]
    return mapping, len(selected)


def weight_summary(rows):
    """Use every B120 five-frame window; IQR describes correlated window weights."""
    selected = [r for r in rows if r['shift'] == 'CDL-B-120' and int(r['window']) == 5
                and r['method'] in ('convex', 'affine')]
    mapping, groups = {}, {}
    for method in ('convex', 'affine'):
        for snr in SNRS:
            group = [r for r in selected if r['method'] == method and int(r['snr_db']) == snr]
            keys = [(int(r['trajectory_id']), int(r['frame_id'])) for r in group]
            if not keys or len(keys) != len(set(keys)):
                raise ValueError('geometry windows missing or duplicated')
            trajectories = sorted({key[0] for key in keys})
            if set(keys) != {(tid, frame) for tid in trajectories for frame in range(4, 10)}:
                raise ValueError('causal five-frame geometry must include all frames 4..9')
            weights = np.array([[number(r, key) for key in ('alpha', 'beta', 'lambda0')]
                                for r in group], dtype=np.float64)
            if not np.allclose(weights[:, 2], 1 - weights[:, 0] - weights[:, 1],
                               rtol=0, atol=1e-12):
                raise ValueError('lambda0 must equal 1-alpha-beta')
            if method == 'convex' and np.min(weights) < -1e-12:
                raise ValueError('Convex weights are outside the simplex')
            mapping[(method, snr)] = np.quantile(weights, [.25, .5, .75], axis=0)
            groups[(method, snr)] = set(keys)
    for snr in SNRS:
        if groups[('convex', snr)] != groups[('affine', snr)]:
            raise ValueError('Convex and Affine geometry must use identical windows')
    if len({frozenset(keys) for keys in groups.values()}) != 1:
        raise ValueError('all geometry SNRs must use identical frozen trajectory/window IDs')
    sample_keys = groups[('convex', SNRS[0])]
    return mapping, len(selected), len({key[0] for key in sample_keys}), len(sample_keys)


def export(fig, stem, directory):
    require_matplotlib_panel_alignment(fig, json_out=str(directory / f'{stem}.alignment.json'), strict=True)
    fig.savefig(directory / f'{stem}.png', dpi=300)
    fig.savefig(directory / f'{stem}.pdf', metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(directory / f'{stem}.svg', metadata={'Date': None})
    plt.close(fig)


def panel(ax, letter):
    ax.annotate(letter, xy=(0, 1), xycoords='axes fraction', xytext=(-20, 12),
                textcoords='offset points', fontsize=11, fontweight='bold')
    ax.set_xticks(SNRS)
    ax.set_xlim(4, 21)
    ax.grid(axis='y', color='0.9', linewidth=.6)
    ax.set_axisbelow(True)


def plot(input_dir, output_dir):
    input_dir, directory = Path(input_dir), Path(output_dir)
    performance_rows = read_csv(input_dir / DATA_FILES[0], ('shift', 'snr_db', 'window', 'subset', 'method', 'nmse_db'))
    geometry_rows = read_csv(input_dir / DATA_FILES[1], ('shift', 'snr_db', 'window', 'trajectory_id', 'frame_id', 'method',
                                                       'alpha', 'beta', 'lambda0'))
    performance, performance_count = performance_series(performance_rows)
    geometry, geometry_count, trajectories, windows = weight_summary(geometry_rows)
    directory.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.8), sharey=True)
        fig.subplots_adjust(left=.12, right=.97, bottom=.27, top=.73, wspace=.22)
        for ax, shift, title, letter in zip(axes, SHIFTS, ('CDL-C / 75 km/h', 'CDL-B / 120 km/h'), ('a', 'b')):
            for method, label, color, marker, style in METHODS:
                ax.plot(SNRS, performance[(shift, method)], label=label, color=color, marker=marker,
                        linestyle=style, markersize=4, linewidth=1.8 if method == 'affine' else 1.2)
            panel(ax, letter)
            ax.set_xlabel('Source deployment SNR label (dB)')
            ax.set_title(title, fontsize=11, pad=12)
        axes[0].set_ylabel('Actual validation NMSE (dB)')
        fig.suptitle('Discrete selection, convex mixing and affine composition', fontsize=12, y=.98)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=5, loc='upper center', bbox_to_anchor=(.55, .91), columnspacing=1.2)
        fig.text(.12, .105, 'Causal 5-frame pilots; all methods evaluated on paired frame4..9; dF=16', fontsize=9)
        fig.text(.12, .045, 'Deployments: source means; Matched: frozen target means; pooled NMSE, no CI', fontsize=9)
        export(fig, STEMS[0], directory)

        fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.8), sharey=True)
        fig.subplots_adjust(left=.12, right=.97, bottom=.27, top=.73, wspace=.24)
        for j, (ax, title, letter) in enumerate(zip(axes, ('Alpha', 'Beta', 'Lambda0 = 1 - alpha - beta'), ('a', 'b', 'c'))):
            for method, label, color, marker, style in (m for m in METHODS if m[0] in ('convex', 'affine')):
                quantiles = np.array([geometry[(method, snr)][:, j] for snr in SNRS])
                median = quantiles[:, 1]
                ax.errorbar(SNRS, median, yerr=np.stack((median - quantiles[:, 0], quantiles[:, 2] - median)),
                            color=color, marker=marker, linestyle=style, label=label,
                            markersize=4, linewidth=1.8 if method == 'affine' else 1.2, capsize=3)
            ax.axhline(0, color='0.6', linestyle=':', linewidth=.7)
            panel(ax, letter)
            ax.set_xlabel('Source SNR label (dB)', fontsize=9)
            ax.set_title(title, fontsize=10, pad=12)
        axes[0].set_ylabel('Coefficient value')
        fig.suptitle('B120 causal five-frame candidate-weight geometry', fontsize=12, y=.98)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=2, loc='upper center', bbox_to_anchor=(.55, .91), columnspacing=1.5)
        fig.text(.12, .105, f'{trajectories} original trajectories; {windows} paired windows/SNR/method; frames4..9', fontsize=9)
        fig.text(.12, .045, 'Median with Q25-Q75; correlated windows; descriptive IQR, no CI or clipping', fontsize=9)
        export(fig, STEMS[1], directory)
    return dict(stems=STEMS, source_data_selection={
        DATA_FILES[0]: dict(input_rows=len(performance_rows), plotted_rows=performance_count,
                            rule='window=5, subset=last6, both shifts, Source/Hard/Convex/Affine/Matched',
                            reason='primary five-frame matched-subset comparison; other numeric rows remain in original CSV'),
        DATA_FILES[1]: dict(input_rows=len(geometry_rows), plotted_rows=geometry_count,
                            rule='B120, window=5, Convex/Affine, all frame4..9 windows',
                            reason='candidate-hull geometry of composed shift; raw Hard/C75/one-frame rows remain in original CSV'),
    })


def audit(input_dir, output_dir):
    directory = Path(output_dir)
    data_selection = plot(input_dir, directory)
    checks = {}
    source = subprocess.run([sys.executable, str(SKILL / 'validate_figure.py'), str(Path(__file__)), '--json'],
                            capture_output=True, text=True)
    for stem in STEMS:
        text = subprocess.run([sys.executable, str(SKILL / 'audit_pdf_text.py'), str(directory / f'{stem}.pdf'),
                               '--min-pt', '5', '--json'], capture_output=True, text=True)
        collision = subprocess.run([sys.executable, str(SKILL / 'audit_figure_collisions.py'),
                                    str(directory / f'{stem}.pdf'), '--json-out',
                                    str(directory / f'{stem}.collision-audit.json')], capture_output=True, text=True)
        checks[stem] = dict(text_returncode=text.returncode, collision_returncode=collision.returncode,
                           text_output=text.stdout + text.stderr, collision_output=collision.stdout + collision.stderr)
    (ROOT / '.pytest_cache').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / '.pytest_cache') as temporary:
        temporary = Path(temporary)
        for filename in DATA_FILES:
            shutil.copy2(Path(input_dir) / filename, temporary / filename)
        plot(temporary, temporary / 'rebuilt')
        rebuilt = {f'{stem}.{extension}': sha(directory / f'{stem}.{extension}') ==
                   sha(temporary / 'rebuilt' / f'{stem}.{extension}')
                   for stem in STEMS for extension in ('png', 'pdf', 'svg')}
    report = dict(source_returncode=source.returncode, source_output=source.stdout + source.stderr,
                  rendered_checks=checks, CSV_only_rebuild=rebuilt, visual_review='pending',
                  all_checks_passed=False, source_data_selection=data_selection['source_data_selection'],
                  source_data_hashes={filename: sha(Path(input_dir) / filename) for filename in DATA_FILES},
                  figure_contract=dict(
                      backend='python', archetype='quantitative comparison',
                      scientific_question='Does unrestricted affine composition add capacity beyond candidate selection or convex mixing?',
                      panel_roles={'performance': ['C75 single-axis mismatch comparator', 'B120 composed-shift comparison'],
                                   'geometry': ['Profile correction alpha', 'Mobility correction beta', 'Source coefficient lambda0']},
                      source_data=list(DATA_FILES), subset='all performance methods frames4..9; all B120 five-frame geometry windows',
                      uncertainty='pooled NMSE without CI; weight median and Q25-Q75 across correlated windows, descriptive IQR',
                      exports=['PNG300dpi', 'editable PDF', 'editable SVG'],
                      adaptation='style-only; data mapping, simplex/paired-window guards and geometry summaries built anew',
                      final_size_inches=[7.2, 4.8], editable_text=True,
                  ), reviewed_warnings=['PNG300dpi/no TIFF: research exports with editable vectors; no journal submission requested'])
    (directory / 't16f_figure_audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    if source.returncode or not all(rebuilt.values()) or any(
            c['text_returncode'] or c['collision_returncode'] for c in checks.values()):
        raise ValueError('T16F figure gates failed; numerical CSVs protected')
    print('T16F source/alignment/text/collision/CSV-only byte rebuild PASS; visual review pending')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', default=str(ROOT / 'outputs'))
    parser.add_argument('--output-dir', default=str(ROOT / 'outputs'))
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    if args.audit:
        audit(args.input_dir, args.output_dir)
    else:
        plot(args.input_dir, args.output_dir)
