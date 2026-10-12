"""Frozen T20A pooled metrics and paired trajectory bootstrap, no estimator."""
import numpy as np

CONDITIONS = ('CDL-C-30-100', 'CDL-B-30-100', 'CDL-C-120-100', 'CDL-B-120-100', 'CDL-C-30-200')
SNRS = (10, 20)
METHODS = ('Perfect CSI', 'R3 LS+linear', 'Source LMMSE', 'Hard', 'Convex',
           'Dense Affine', 'Final LR4 Affine', 'Matched LMMSE reference')
PAIRS = (('source_gain', 'Source LMMSE', 'Final LR4 Affine'),
         ('convex_gain', 'Convex', 'Final LR4 Affine'),
         ('compression_penalty', 'Final LR4 Affine', 'Dense Affine'))


def pooled(rows):
    error = sum(r['error_energy'] for r in rows)
    truth = sum(r['truth_energy'] for r in rows)
    errors = sum(r['bit_errors'] for r in rows)
    bits = sum(r['total_bits'] for r in rows)
    if truth <= 0 or bits <= 0 or error < 0 or not np.isfinite([error, truth]).all():
        raise ValueError('Invalid pooled energies/counts')
    return dict(error_energy=error, truth_energy=truth, nmse_linear=error/truth,
                nmse_db=float(10*np.log10(error/truth)) if error else None,
                bit_errors=errors, total_bits=bits, ber=errors/bits)


def aggregate(rows, conditions=CONDITIONS, snrs=SNRS):
    result = []
    for name in conditions:
        for snr in snrs:
            for method in METHODS:
                group = [r for r in rows if (r['environment'], r['snr_db'], r['method']) == (name, snr, method)]
                ids = [r['trajectory_id'] for r in group]
                if not ids or len(set(ids)) != len(ids):
                    raise ValueError('Missing or duplicate trajectory metric')
                result.append(dict(environment=name, snr_db=snr, method=method,
                                   trajectories=len(group), scored_frames=6, **pooled(group)))
    return result


def bootstrap(rows, *, conditions=CONDITIONS, snrs=SNRS, resamples=10000, seed=20262001):
    """One RNG, table-order environments, shared trajectory draws across all cells."""
    rng = np.random.default_rng(seed)
    result = []
    for name in conditions:
        ids = sorted({r['trajectory_id'] for r in rows if r['environment'] == name})
        indexed = {(r['snr_db'], r['method'], r['trajectory_id']): r
                   for r in rows if r['environment'] == name}
        if len(indexed) != len(ids)*len(snrs)*len(METHODS) or not ids:
            raise ValueError('Incomplete paired bootstrap population')
        # Generate the entire specified draw table once; splitting matrix sums
        # below does not change RNG consumption or paired resample membership.
        draws = rng.integers(0, len(ids), size=(resamples, len(ids)))
        energy, denominator, errors, bits = {}, {}, {}, {}
        for snr in snrs:
            for method in METHODS:
                group = [indexed[snr, method, i] for i in ids]
                key = snr, method
                values = [np.array([r[k] for r in group]) for k in
                          ('error_energy', 'truth_energy', 'bit_errors', 'total_bits')]
                totals = [np.empty(resamples) for _ in values]
                for start in range(0, resamples, 100):
                    local = draws[start:start+100]
                    for dest, source in zip(totals, values):
                        dest[start:start+len(local)] = source[local].sum(axis=1)
                energy[key], denominator[key], errors[key], bits[key] = totals
                linear = totals[0]/totals[1]
                ber = totals[2]/totals[3]
                nmse = 10*np.log10(linear) if method != 'Perfect CSI' else None
                observed = pooled(group)
                for metric, dist, point in (('nmse_db', nmse, observed['nmse_db']), ('ber', ber, observed['ber'])):
                    lower, upper = np.quantile(dist, [.025, .975], method='linear') if dist is not None else (None, None)
                    result.append(dict(environment=name, snr_db=snr, comparison=method, metric=metric,
                        estimate=point, ci95_lower=float(lower) if lower is not None else None,
                        ci95_upper=float(upper) if upper is not None else None,
                        primary_endpoint='', primary_lower=None, primary_upper=None, endpoint_cell_pass=None))
            for label, first, second in PAIRS:
                a, b = (snr, first), (snr, second)
                dist = 10*np.log10(energy[a]/denominator[a])-10*np.log10(energy[b]/denominator[b])
                point = pooled([indexed[snr, first, i] for i in ids])['nmse_db']-pooled([indexed[snr, second, i] for i in ids])['nmse_db']
                lo, hi = np.quantile(dist, [.025, .975], method='linear')
                lower, upper = np.quantile(dist, [.005, .995], method='linear')
                family = ('A' if label == 'source_gain' and name in CONDITIONS[1:4] else
                          'B' if label == 'convex_gain' and name == CONDITIONS[3] else
                          'C' if label == 'compression_penalty' and name == CONDITIONS[3] else '')
                passed = bool(upper < .50 if family == 'C' else lower > 0) if family else None
                result.append(dict(environment=name, snr_db=snr, comparison=label, metric='nmse_difference_db',
                    estimate=point, ci95_lower=float(lo), ci95_upper=float(hi), primary_endpoint=family,
                    primary_lower=float(lower) if family else None, primary_upper=float(upper) if family else None,
                    endpoint_cell_pass=passed))
                dist_ber = errors[a]/bits[a]-errors[b]/bits[b]
                lo, hi = np.quantile(dist_ber, [.025, .975], method='linear')
                point_ber = pooled([indexed[snr, first, i] for i in ids])['ber']-pooled([indexed[snr, second, i] for i in ids])['ber']
                result.append(dict(environment=name, snr_db=snr, comparison=label, metric='ber_difference',
                    estimate=point_ber, ci95_lower=float(lo), ci95_upper=float(hi), primary_endpoint='',
                    primary_lower=None, primary_upper=None, endpoint_cell_pass=None))
    return result


def endpoints(rows):
    result = {}
    for family, count in (('A', 6), ('B', 2), ('C', 2)):
        cells = [r for r in rows if r['primary_endpoint'] == family]
        if len(cells) != count or len({(r['environment'], r['snr_db']) for r in cells}) != count:
            raise ValueError('Exactly ten preregistered endpoint cells required')
        result[f'endpoint_{family}_pass'] = all(r['endpoint_cell_pass'] is True for r in cells)
        result[f'endpoint_{family}_status'] = 'PASS' if result[f'endpoint_{family}_pass'] else 'NOT_SUPPORTED_BY_FORMAL_ENDPOINT'
    a, b, c = [result[f'endpoint_{f}_pass'] for f in 'ABC']
    result.update(paper_core_empirically_supported=a, paper_results_ready=a and b and c,
                  paper_results_ready_partial=a and (b != c))
    return result


def weight_summary(rows):
    result = []
    for name in CONDITIONS:
        for snr in SNRS:
            group = [r for r in rows if (r['environment'], r['snr_db']) == (name, snr)]
            record = dict(environment=name, snr_db=snr, scored_windows=len(group))
            for field in ('alpha', 'beta', 'lambda0'):
                x = np.array([r[field] for r in group])
                record[field] = dict(median=float(np.median(x)), q25=float(np.quantile(x,.25)),
                    q75=float(np.quantile(x,.75)), mean=float(np.mean(x)), std=float(np.std(x,ddof=0)))
            record['outside_simplex_fraction'] = float(np.mean([r['outside_simplex'] for r in group]))
            result.append(record)
    return result
