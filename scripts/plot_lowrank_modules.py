"""T16G CSV-only performance and paired dense-to-low-rank compression penalty."""
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
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.pytest_cache/matplotlib-t16g'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment

STYLE = {
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10, 'legend.fontsize': 9, 'svg.fonttype': 'none',
    'svg.hashsalt': 'T16G-fixed', 'pdf.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
}
DATA_FILE = 't16g_lowrank_affine.csv'
STEMS = ('t16g_lowrank_affine_nmse', 't16g_compression_penalty')
SNRS = (5, 10, 20)
SHIFTS = ('CDL-C-75', 'CDL-B-120')
RANKS = (1, 2, 4, 8, 16)
METHODS = (
    ('affine', 0, 'Dense Affine', '0.25', 'o', '-'),
    ('lr-affine', 2, 'LR-r2', '#E69F00', '^', ':'),
    ('lr-affine', 4, 'LR-r4', '#0072B2', 'D', '--'),
    ('lr-affine', 8, 'LR-r8', '#CC79A7', 's', '-.'),
    ('matched', 0, 'Matched', '#009E73', 'v', '--'),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    required = ('shift', 'snr_db', 'window', 'subset', 'method', 'rank', 'nmse_db',
                'compression_penalty_db', 'recovery_preservation', 'lowrank_affine_vs_convex_gain_db')
    with Path(path).open(encoding='utf8', newline='') as handle:
        reader = csv.DictReader(handle)
        if not set(required).issubset(reader.fieldnames or ()):
            raise ValueError('T16G performance CSV: required fields missing')
        return list(reader)


def number(row, field):
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f'non-finite {field}; rows must not be silently discarded')
    return value


def primary_rows(rows):
    return [row for row in rows if int(row['window']) == 5 and row['subset'] == 'last6'
            and row['shift'] in SHIFTS]


def performance_series(rows):
    """Map the requested five curves without substituting secondary frame subsets."""
    selected = primary_rows(rows)
    mapping, count = {}, 0
    for shift in SHIFTS:
        for method, rank, *_ in METHODS:
            group = sorted((row for row in selected if row['shift'] == shift
                            and row['method'] == method and int(row['rank']) == rank),
                           key=lambda row: int(row['snr_db']))
            if [int(row['snr_db']) for row in group] != list(SNRS):
                raise ValueError('exactly one paired last6 row per method/rank/shift/SNR is required')
            mapping[(shift, method, rank)] = [number(row, 'nmse_db') for row in group]
            count += len(group)
    return mapping, count


def penalty_series(rows):
    """Use all five frozen ranks at all four primary conditions, with no clipping."""
    selected = primary_rows(rows)
    mapping, count = {}, 0
    for shift in SHIFTS:
        for snr in (10, 20):
            dense = [row for row in selected if row['shift'] == shift and int(row['snr_db']) == snr
                     and row['method'] == 'affine' and int(row['rank']) == 0]
            if len(dense) != 1:
                raise ValueError('exactly one paired Dense Affine baseline is required')
            dense_nmse = number(dense[0], 'nmse_db')
            group = sorted((row for row in selected if row['shift'] == shift and int(row['snr_db']) == snr
                            and row['method'] == 'lr-affine'), key=lambda row: int(row['rank']))
            if [int(row['rank']) for row in group] != list(RANKS):
                raise ValueError('all five frozen ranks must appear exactly once per primary condition')
            values = []
            for row in group:
                penalty = number(row, 'compression_penalty_db')
                if not math.isclose(penalty, number(row, 'nmse_db') - dense_nmse, rel_tol=0, abs_tol=1e-10):
                    raise ValueError('compression penalty must equal LR NMSE minus Dense Affine NMSE')
                values.append(penalty)
            mapping[(shift, snr)] = values
            count += len(group)
    return mapping, count


def export(fig, stem, directory):
    require_matplotlib_panel_alignment(fig, json_out=str(directory / f'{stem}.alignment.json'), strict=True)
    fig.savefig(directory / f'{stem}.png', dpi=300)
    fig.savefig(directory / f'{stem}.pdf', metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(directory / f'{stem}.svg', metadata={'Date': None})
    plt.close(fig)


def plot(input_dir, output_dir):
    input_dir, directory = Path(input_dir), Path(output_dir)
    rows = read_rows(input_dir / DATA_FILE)
    performance, performance_count = performance_series(rows)
    penalty, penalty_count = penalty_series(rows)
    directory.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.8), sharey=True)
        fig.subplots_adjust(left=.12, right=.97, bottom=.27, top=.73, wspace=.22)
        for ax, shift, title, letter in zip(axes, SHIFTS, ('CDL-C / 75 km/h', 'CDL-B / 120 km/h'), ('a', 'b')):
            for method, rank, label, color, marker, style in METHODS:
                ax.plot(SNRS, performance[(shift, method, rank)], label=label, color=color,
                        marker=marker, linestyle=style, markersize=4,
                        linewidth=1.8 if rank == 4 else 1.2)
            ax.annotate(letter, xy=(0, 1), xycoords='axes fraction', xytext=(-20, 12),
                        textcoords='offset points', fontsize=11, fontweight='bold')
            ax.set_xticks(SNRS)
            ax.set_xlim(4, 21)
            ax.grid(axis='y', color='0.9', linewidth=.6)
            ax.set_axisbelow(True)
            ax.set_xlabel('Source deployment SNR label (dB)')
            ax.set_title(title, fontsize=11, pad=12)
        axes[0].set_ylabel('Actual validation NMSE (dB)')
        fig.suptitle('Low-rank compression of affine mismatch modules', fontsize=12, y=.98)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=5, loc='upper center', bbox_to_anchor=(.55, .91), columnspacing=1.2)
        fig.text(.12, .105, 'Frozen causal 5-frame coefficients; paired frame4..9; dF=16', fontsize=9)
        fig.text(.12, .045, 'Affine: source means; Matched: frozen target means; pooled NMSE, no CI', fontsize=9)
        export(fig, STEMS[0], directory)

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        fig.subplots_adjust(left=.14, right=.97, bottom=.27, top=.73)
        for shift, color, short in (('CDL-C-75', '#0072B2', 'C75'), ('CDL-B-120', '#CC79A7', 'B120')):
            for snr, marker, style in ((10, 'o', '--'), (20, 's', '-')):
                ax.plot(RANKS, penalty[(shift, snr)], color=color, marker=marker, linestyle=style,
                        markersize=4, linewidth=1.5, label=f'{short} / {snr} dB')
        ax.axhline(.5, color='0.4', linestyle=':', linewidth=1, label='0.5 dB reference')
        ax.set_xscale('log', base=2)
        ax.set_xticks(RANKS, [str(rank) for rank in RANKS])
        ax.set_xlim(.9, 17.5)
        ax.grid(axis='y', color='0.9', linewidth=.6)
        ax.set_axisbelow(True)
        ax.set_xlabel('Rank per module (log2 spacing)')
        ax.set_ylabel('LR minus Dense Affine NMSE (dB)')
        fig.suptitle('Compression penalty at the four primary conditions', fontsize=12, y=.98)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, ncol=3, loc='upper center', bbox_to_anchor=(.55, .91),
                   columnspacing=1.5, handlelength=2.3)
        fig.text(.14, .105, 'Paired frame4..9; unchanged weights at every rank; lower penalty is better', fontsize=9)
        fig.text(.14, .045, 'All frozen ranks shown; signed penalty without clipping; no CI', fontsize=9)
        export(fig, STEMS[1], directory)
    return dict(stems=STEMS, source_data_selection={DATA_FILE: dict(
        input_rows=len(rows), figure1_plotted_rows=performance_count, figure2_plotted_rows=penalty_count,
        figure1_rule='window=5, subset=last6, both shifts, Dense Affine/LR-r2/LR-r4/LR-r8/Matched',
        figure2_rule='window=5, subset=last6, C75/B120 at 10/20dB, all five frozen ranks',
        reason='requested primary paired-subset comparisons; secondary and other method/rank rows remain in original CSV',
    )})


def audit(input_dir, output_dir):
    directory = Path(output_dir)
    data_selection = plot(input_dir, directory)
    source = subprocess.run([sys.executable, str(SKILL / 'validate_figure.py'), str(Path(__file__)), '--json'],
                            capture_output=True, text=True)
    checks = {}
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
        shutil.copy2(Path(input_dir) / DATA_FILE, temporary / DATA_FILE)
        plot(temporary, temporary / 'rebuilt')
        rebuilt = {f'{stem}.{extension}': sha(directory / f'{stem}.{extension}') ==
                   sha(temporary / 'rebuilt' / f'{stem}.{extension}')
                   for stem in STEMS for extension in ('png', 'pdf', 'svg')}
    report = dict(source_returncode=source.returncode, source_output=source.stdout + source.stderr,
                  rendered_checks=checks, CSV_only_rebuild=rebuilt, visual_review='pending',
                  all_checks_passed=False, source_data_selection=data_selection['source_data_selection'],
                  source_data_hashes={DATA_FILE: sha(Path(input_dir) / DATA_FILE)},
                  figure_contract=dict(
                      backend='python', archetype='quantitative comparison',
                      scientific_question='Can development-only low-rank correction retain the dense affine validation advantage?',
                      panel_roles={'performance': ['C75 single-axis compression comparison', 'B120 composed-shift compression comparison'],
                                   'penalty': ['Four primary paired conditions against the pre-registered 0.5dB penalty reference']},
                      source_data=[DATA_FILE], subset='all plotted methods use causal-five-frame frame4..9',
                      uncertainty='pooled NMSE and paired pooled-NMSE differences without CI',
                      exports=['PNG300dpi', 'editable PDF', 'editable SVG'],
                      adaptation='style-only; rank mapping and signed paired-penalty guards built anew',
                      final_size_inches=[7.2, 4.8], editable_text=True,
                  ), reviewed_warnings=['PNG300dpi/no TIFF: research exports with editable vectors; no journal submission requested'])
    (directory / 't16g_figure_audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    if source.returncode or not all(rebuilt.values()) or any(
            check['text_returncode'] or check['collision_returncode'] for check in checks.values()):
        raise ValueError('T16G figure gates failed; numerical CSVs protected')
    print('T16G source/alignment/text/collision/CSV-only byte rebuild PASS; visual review pending')
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
