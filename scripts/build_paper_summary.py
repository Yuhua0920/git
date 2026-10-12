"""Format existing frozen comparisons; no metric estimation or experiment calls."""
from decimal import Decimal, ROUND_CEILING
from paper_support import ROOT, csv_read, json_write, require_sources, sha256
from build_paper_tables import CONDITIONS, CONDITION_NAMES, index_unique, longtable

SOURCES = ('outputs/t20b_formal_nmse.csv', 'outputs/t20b_formal_bootstrap.csv')
PAIR = {'source_gain': ('Source LMMSE', 'Final LR4 Affine'),
        'convex_gain': ('Convex', 'Final LR4 Affine'),
        'compression_penalty': ('Final LR4 Affine', 'Dense Affine'),
        'matched_gap': ('Final LR4 Affine', 'Matched LMMSE reference')}


def conservative_bound(values):
    if not values:
        raise ValueError('A bound requires every single-shift cell')
    return str(max(map(Decimal, map(str, values))).quantize(Decimal('.01'), rounding=ROUND_CEILING))


def products(rows):
    indexed = index_unique(rows, ('environment', 'snr_db', 'method'))
    output, records, gaps = [], [], []
    for environment in CONDITIONS:
        for snr in ('10', '20'):
            columns = [CONDITION_NAMES[environment], snr]
            for comparison, methods in PAIR.items():
                included = (comparison in ('source_gain', 'compression_penalty') or
                            comparison == 'convex_gain' and environment == 'CDL-B-120-100' or
                            comparison == 'matched_gap' and environment == 'CDL-C-30-200')
                raw = [indexed[(environment, snr, m)]['nmse_db'] for m in methods]
                value = float(raw[0])-float(raw[1])
                if comparison == 'matched_gap' and environment in ('CDL-B-30-100', 'CDL-C-120-100'):
                    gaps.append(dict(environment=environment, snr_db=snr, raw_operands=raw, gap_db=value))
                if included:
                    display = f'{value:.3f}'
                    records.append(dict(environment=environment, snr_db=snr, comparison=comparison,
                                        methods=list(methods), raw_operands=raw, value_db=value, display=display))
                else:
                    display = '--'
                columns.append(display)
            output.append(columns)
    headers = ['Condition', 'SNR', 'Source--LR4', 'Convex--LR4', 'LR4--Dense', 'LR4--Matched']
    caption = 'Key NMSE differences across all evaluation conditions (dB)'
    note = ('Positive Source--LR4 and Convex--LR4 values favor LR4. '
            'Positive LR4--Dense values denote compression loss. '
            'LR4--Matched reports the delay-boundary gap; -- denotes a comparison omitted from this summary. '
            'Paired uncertainty is shown in Fig.~\\ref{fig:endpoints}; full values and intervals for all eight methods appear in the appendix.')
    table = longtable(headers, output, caption, 'tab:summary', 'lrrrrr', note)
    bound = conservative_bound([g['gap_db'] for g in gaps])
    macros = '% Reporting bound from all four frozen single-shift cells, rounded upward.\n'
    macros += '\\newcommand{\\SingleShiftMatchedBound}{'+bound+'}\n'
    return {'06_summary.tex': table, 'summary_numbers.tex': macros}, records, gaps, bound


def build():
    sources = require_sources(ROOT, SOURCES)
    output, records, gaps, bound = products(csv_read(ROOT/SOURCES[0]))
    for name, text in output.items():
        (ROOT/'paper/tables'/name).write_text(text, encoding='utf8', newline='\n')
    report = dict(source_sha256=sources, cells=records, row_count=10,
                  all_five_conditions_and_both_snrs_retained=True, single_shift_gaps=gaps,
                  single_shift_upper_reporting_bound_db=bound,
                  reporting_arithmetic_only=True, new_metric_computed=False,
                  generated_table_sha256={name: sha256(ROOT/'paper/tables'/name) for name in output})
    json_write(ROOT/'paper/tables/t20g_provenance.json', report)
    print(dict(rows=10, cells=len(records), single_shift_reporting_bound_db=bound))


if __name__ == '__main__':
    build()
