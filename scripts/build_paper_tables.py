"""Deterministically format frozen CSV/JSON into tables; never recompute metrics."""
import argparse
import re
from pathlib import Path

from paper_support import ROOT, accepted_results, csv_read, json_read, json_write, require_sources, sha256

SOURCES = ("outputs/t20a_results.json", "outputs/t20b_formal_results.json",
           "outputs/t20b_formal_nmse.csv", "outputs/t20b_formal_ber.csv",
           "outputs/t20b_formal_bootstrap.csv")
NAMES = {"Perfect CSI": "Perfect", "R3 LS+linear": "LS-LIN",
         "Source LMMSE": "Source", "Hard": "Hard", "Convex": "Convex",
         "Dense Affine": "Dense Affine", "Final LR4 Affine": "LR4 Affine",
         "Matched LMMSE reference": "Matched"}


def escape(text):
    return str(text).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def number(value, metric):
    if value is None or value == "":
        return "N/A"
    return f"{float(value):.3f}" if metric == "nmse_db" else f"{float(value):.5f}"


def longtable(headers, rows, caption, label, spec, note):
    body = [r"\begingroup\small", r"\setlength{\tabcolsep}{4pt}",
            r"\begin{longtable}{" + spec + "}",
            r"\caption{" + caption + r"}\label{" + label + r"}\\",
            r"\toprule", " & ".join(headers) + r" \\", r"\midrule\endfirsthead",
            r"\toprule", " & ".join(headers) + r" \\", r"\midrule\endhead"]
    body += [" & ".join(map(str, row)) + r" \\" for row in rows]
    body += [r"\bottomrule", r"\end{longtable}", note, r"\endgroup", ""]
    return "\n".join(body)


def metric_table(rows, intervals, metric):
    display, cells = [], []
    for row in rows:
        key = (row["environment"], int(row["snr_db"]), row["method"], metric)
        interval = intervals[key]
        value = number(row[metric], metric)
        ci = "N/A" if not interval["ci95_lower"] else (
            "[" + number(interval["ci95_lower"], metric) + ", " + number(interval["ci95_upper"], metric) + "]")
        display.append((row["environment"].replace("CDL-", ""), row["snr_db"],
                        NAMES[row["method"]], value, ci))
        cells.append({"environment": row["environment"], "snr_db": row["snr_db"],
                      "method": row["method"], "metric": metric, "raw_value": row[metric],
                      "raw_ci95_lower": interval["ci95_lower"],
                      "raw_ci95_upper": interval["ci95_upper"], "display_value": value, "display_ci": ci})
    return display, cells


def endpoint_rows(rows):
    display = []
    for row in rows:
        family = row["primary_endpoint"]
        if not family:
            continue
        upper = family == "C"
        bound = row["primary_upper" if upper else "primary_lower"]
        display.append((family, row["environment"].replace("CDL-", ""), row["snr_db"],
                        f"{float(row['estimate']):.3f}", f"{float(bound):.3f}",
                        "upper < 0.50" if upper else "lower > 0", row["endpoint_cell_pass"]))
    return display



CONDITIONS = ("CDL-C-30-100", "CDL-B-30-100", "CDL-C-120-100", "CDL-B-120-100", "CDL-C-30-200")
CONDITION_NAMES = dict(zip(CONDITIONS, ("ID", "Profile", "Mobility", "Composed", "Boundary")))
CORE_METHODS = ("Source LMMSE", "Convex", "Dense Affine", "Final LR4 Affine", "Matched LMMSE reference")


def index_unique(rows, fields):
    result = {}
    for row in rows:
        key = tuple(row[f] for f in fields)
        if key in result:
            raise ValueError("Duplicate numeric source cell: " + str(key))
        result[key] = row
    return result


def compact_table(rows, metric, conditions=CONDITIONS, snrs=(10, 20), methods=CORE_METHODS):
    lookup = index_unique(rows, ("environment", "snr_db", "method"))
    display, cells = [], []
    for name in conditions:
        for snr in snrs:
            values = []
            for method in methods:
                row = lookup[name, str(snr), method]
                value = number(row[metric], metric)
                values.append(value)
                cells.append(dict(environment=name, snr_db=str(snr), method=method, metric=metric,
                                  raw_value=row[metric], display_value=value))
            display.append((CONDITION_NAMES.get(name, name), str(snr), *values))
    caption = ("Pooled data-RE NMSE (dB; lower is better)" if metric == "nmse_db" else
               "Uncoded data-RE BER with ZF detection (lower is better)")
    label = "tab:nmse-compact" if metric == "nmse_db" else "tab:ber-compact"
    text = longtable(["Condition", "SNR", *[NAMES[m] for m in methods]], display, caption, label,
                     "ll" + "r" * len(methods),
                     "Core comparisons across every evaluation condition; full eight-estimator values and 95\\% intervals appear in the appendix.")
    return text, cells


def build(root=ROOT, output=None):
    root = Path(root)
    output = Path(output) if output else root / "paper/tables"
    output.mkdir(parents=True, exist_ok=True)
    sources = require_sources(root, (*SOURCES, "outputs/t15a_cdl_results.json"))
    formal = accepted_results(root)
    a = json_read(root / SOURCES[0]); cdl = json_read(root / "outputs/t15a_cdl_results.json")
    frozen, protocol = a["frozen_components"], a["formal_protocol"]
    sys, array = frozen["system"], cdl["array_geometry"]
    assert tuple(formal["formal_conditions"]) == CONDITIONS
    assert sys["sample_rate_hz"] == sys["nfft"] * sys["delta_f_hz"]
    params = [(r"$N_t,N_r$", f"{sys['nt']}, {sys['nr']}"),
              (r"$K,T$; active subcarriers", f"{sys['nfft']}, {sys['num_symbols']}; {sys['active_carriers']}"),
              (r"$\Delta f$; $F_s$", f"{sys['delta_f_hz']/1000:g} kHz; {sys['sample_rate_hz']/1e6:g} MHz"),
              ("CP length; FFT normalization", f"{sys['cp_len']} samples; {sys['fft_norm']}"),
              ("Carrier frequency", f"{cdl['system']['carrier_frequency_hz']/1e9:g} GHz"),
              ("Modulation; total expected power/active RE", "Gray QPSK; " + str(sys['total_power'])),
              ("Data-stream scaling", r"$1/\sqrt{2}$"),
              ("Carrier order", "fftshift, ascending physical frequency"),
              (r"$d_F,d_T$; union pilot fraction", f"{frozen['dF']}, {frozen['dT']}; $1/32$"),
              ("Transmit-antenna offsets; pilot symbols", "[0,8]; [0,4,8,12]"),
              ("Pilot mapping", "Unit-modulus owner pilot; other antenna zero"),
              ("Transmit/receive arrays", f"{array['rows']} by {array['columns']}, vertically polarized; half-wavelength spacing"),
              ("Array orientations (azimuth, downtilt, slant)", r"BS $(0,0,0)$; UT $(\pi,0,0)$ radians"),
              ("Mobility direction", "Global positive x-axis"),
              (r"$P_{\rm rx,ref}$", f"{frozen['source_noise_reference']:.12f}"),
              ("Source-referenced SNR", ", ".join(map(str, formal['snr'])) + " dB"),
              (r"$\sigma^2$ at 10/20 dB", "; ".join(f"{frozen['noise_variances'][str(s)]:.12f}" for s in formal['snr']))]
    products = {"01_system.tex": longtable(["Parameter", "Value"], params,
        "System configuration and pilot pattern", "tab:system", "p{.45\\linewidth}p{.49\\linewidth}",
        "Noise uses the fixed source ensemble reference. The CDL frequency responses retain their original amplitudes.")}
    condition_rows = []
    for name in CONDITIONS:
        _, profile, speed, delay = name.split("-")
        condition_rows.append((CONDITION_NAMES[name], "CDL-" + profile, speed, delay))
    products['evaluation_conditions.tex'] = longtable(
        ["Condition", "Profile", "Speed (km/h)", "Nominal DS (ns)"], condition_rows,
        "Independent evaluation conditions", "tab:conditions", "llrr",
        f"{formal['trajectories_per_condition']} independent trajectories per condition; "
        f"{formal['frames_per_trajectory']} frames per trajectory; scored frames "
        f"{min(protocol['scored_frames'])}--{max(protocol['scored_frames'])}.")
    boot = csv_read(root / SOURCES[4])
    intervals = {(r["environment"], int(r["snr_db"]), r["comparison"], r["metric"]): r for r in boot}
    cells, compact_cells, metrics = [], [], {}
    expected = {(n, str(s), m) for n in CONDITIONS for s in formal['snr'] for m in formal['methods']}
    for metric, source, filename in (("nmse_db", SOURCES[2], "02_nmse.tex"), ("ber", SOURCES[3], "03_ber.tex")):
        raw = csv_read(root/source)
        lookup = index_unique(raw, ("environment", "snr_db", "method"))
        if set(lookup) != expected:
            raise ValueError("Full eight-method coverage differs from frozen protocol")
        metrics[metric] = lookup
        rows, ledger = metric_table(raw, intervals, metric); cells += ledger
        caption = ("Full pooled data-RE NMSE (dB)" if metric == "nmse_db" else "Full uncoded data-RE BER with ZF detection")
        products[filename] = longtable(["Condition", "SNR", "Estimator", "Estimate", "95\\% CI"], rows,
            caption, "tab:" + metric, "lllrr",
            "All eight methods and all five conditions are retained. Intervals use paired trajectory percentile bootstrap. "
            "Perfect CSI has zero linear NMSE (dB N/A); Matched uses evaluation-only target statistics.")
        compact, ledger = compact_table(raw, metric); compact_cells += ledger
        products[filename.replace('.tex', '_compact.tex')] = compact
    macros = {}
    def put(name, raw, display, source, selector, operation="direct frozen field"):
        macros[name] = dict(raw_value=raw, display_value=display, source_artifact=source,
                            selector=selector, operation=operation)
    for name in CONDITIONS:
        stem = CONDITION_NAMES[name]
        for snr, word in ((10, 'Ten'), (20, 'Twenty')):
            for method, short in zip(CORE_METHODS, ('Source','Convex','Dense','Low','Matched')):
                for metric, prefix, source in (('nmse_db','Nmse',SOURCES[2]), ('ber','Ber',SOURCES[3])):
                    raw = metrics[metric][name,str(snr),method][metric]
                    put(prefix+stem+short+word, raw, number(raw, metric), source,
                        dict(environment=name,snr_db=snr,method=method,field=metric))
            for comparison, prefix in (('source_gain','Gain'), ('convex_gain','AffineGain'), ('compression_penalty','Penalty')):
                row = intervals[name,snr,comparison,'nmse_difference_db']
                put(prefix+stem+word, row['estimate'], f"{float(row['estimate']):.3f}", SOURCES[4],
                    dict(environment=name,snr_db=snr,comparison=comparison,metric='nmse_difference_db',field='estimate'))
                if row['primary_endpoint']:
                    field = 'primary_upper' if comparison=='compression_penalty' else 'primary_lower'
                    put(('Upper' if field=='primary_upper' else 'Lower')+prefix+stem+word,
                        row[field], f"{float(row[field]):.3f}", SOURCES[4],
                        dict(environment=name,snr_db=snr,comparison=comparison,metric='nmse_difference_db',field=field))
            low = float(metrics['nmse_db'][name,str(snr),'Final LR4 Affine']['nmse_db'])
            matched = float(metrics['nmse_db'][name,str(snr),'Matched LMMSE reference']['nmse_db'])
            put('Gap'+stem+word, [str(low),str(matched)], f"{low-matched:.3f}", SOURCES[2],
                dict(environment=name,snr_db=snr,methods=['Final LR4 Affine','Matched LMMSE reference'],field='nmse_db'),
                'reporting subtraction: LR4 minus Matched; no new aggregation')
    for snr,word in ((10,'Ten'),(20,'Twenty')):
        value=intervals[CONDITIONS[0],snr,'source_gain','nmse_difference_db']['estimate']
        put('IDLoss'+word,value,f"{-float(value):.3f}",SOURCES[4],
            dict(environment=CONDITIONS[0],snr_db=snr,comparison='source_gain',metric='nmse_difference_db',field='estimate'),'negate source-minus-LR4 to express loss')
    put('CostSaving', formal['complexity']['saving_vs_dense_affine_percent'],
        f"{formal['complexity']['saving_vs_dense_affine_percent']:.2f}",SOURCES[1],{'path':'complexity.saving_vs_dense_affine_percent'})
    for macro,path in [('NumTrajectories','trajectories_per_condition'),('NumFrames','frames_per_trajectory'),
                       ('NumBootstrap','bootstrap_resamples'),('NumComparisons','primary_cell_comparisons'),
                       ('CompressionMargin','compression_margin_db')]:
        value=protocol[path]; put(macro,value,str(value),SOURCES[0],{'path':'formal_protocol.'+path})
    put('ScoredFrames', protocol['scored_frames'], f"{min(protocol['scored_frames'])}--{max(protocol['scored_frames'])}",
        SOURCES[0], {'path':'formal_protocol.scored_frames'})
    put('EvaluationSnrs', formal['snr'], ' and '.join(map(str,formal['snr'])), SOURCES[1], {'path':'snr'})
    put('WindowFrames', frozen['causal_window'], str(frozen['causal_window']), SOURCES[0], {'path':'frozen_components.causal_window'})
    put('RankValue', frozen['rank_per_module'], str(frozen['rank_per_module']), SOURCES[0], {'path':'frozen_components.rank_per_module'})
    confidence = re.search(r'percentile(\d+)%', protocol['descriptive_CI'])[1]
    put('ConfidenceLevel', protocol['descriptive_CI'], confidence, SOURCES[0], {'path':'formal_protocol.descriptive_CI'})
    for macro, value in zip(('BonfLowerQuantile','BonfUpperQuantile'),
                            re.search(r';([\d.]+)/([\d.]+)',protocol['primary_CI']).groups()):
        put(macro, protocol['primary_CI'], value, SOURCES[0], {'path':'formal_protocol.primary_CI'})
    for condition in CONDITIONS:
        _, profile, speed, delay = condition.split('-')
        put(CONDITION_NAMES[condition]+'Channel',condition,f'CDL-{profile}/{speed} km/h/{delay} ns',SOURCES[1],
            {'path':'formal_conditions','condition':condition})
    put('SionnaVersion',cdl['sionna_provenance']['version'],cdl['sionna_provenance']['version'],
        'outputs/t15a_cdl_results.json',{'path':'sionna_provenance.version'})
    products['result_numbers.tex'] = '\n'.join('\\newcommand{\\'+key+'}{'+value['display_value']+'}' for key,value in macros.items())+'\n'
    for name, text in products.items():
        (output/name).write_text('% Generated from frozen evidence; edit the formatter, not cells.\n'+text,encoding='utf8',newline='\n')
    report = dict(task='T20F', source_sha256=sources, full_cells=cells, compact_cells=compact_cells,
        generation_script_sha256=sha256(root/'scripts/build_paper_tables.py'),
        number_macros=macros, full_nmse_rows=80,full_ber_rows=80,compact_nmse_rows=10,compact_ber_rows=10,
        compact_methods=list(CORE_METHODS),full_method_count=8,all_five_conditions_retained=True,
        scientific_metrics_recomputed=False,reporting_subtractions_only=True,
        historical_provenance_not_overwritten=True,
        generated_table_sha256={p:sha256(output/p) for p in sorted(products)})
    json_write(output/'t20f_provenance.json',report)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path)
    args=parser.parse_args();r=build(output=args.output)
    print({k:r[k] for k in ('full_nmse_rows','full_ber_rows','compact_nmse_rows','compact_ber_rows','scientific_metrics_recomputed')})
