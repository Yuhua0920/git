"""Independent T16F figure-data mapping and export checks on small CSV fixtures."""
import csv
import json

import numpy as np
import pytest

from scripts.plot_statistics_selection import (METHODS, SHIFTS, SNRS, STEMS, audit,
                                               performance_series, weight_summary)


def rows():
    performance = [dict(shift=shift, snr_db=snr, window=5, subset='last6', method=method,
                        nmse_db=-snr / 2 - offset)
                   for shift in SHIFTS for snr in SNRS
                   for offset, (method, *_) in enumerate(METHODS)]
    geometry = []
    for snr in SNRS:
        for method in ('convex', 'affine'):
            for tid in range(3):
                for frame in range(4, 10):
                    alpha = .3 + .01 * tid if method == 'convex' else .9 + .01 * tid
                    beta = .6 - .01 * frame if method == 'convex' else 1.02 - .01 * frame
                    geometry.append(dict(shift='CDL-B-120', snr_db=snr, window=5, trajectory_id=tid,
                                         frame_id=frame, method=method, alpha=alpha, beta=beta,
                                         lambda0=1 - alpha - beta))
    return performance, geometry


def save(path, data):
    with path.open('w', encoding='utf8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def test_performance_uses_paired_five_frame_subset_and_rejects_missing_or_duplicate():
    performance, _ = rows()
    extra = dict(performance[0], window=1, subset='all10', nmse_db=999)
    mapped, count = performance_series(performance + [extra])
    assert count == 30
    assert mapped[('CDL-C-75', 'source')] == [-2.5, -5, -10]
    with pytest.raises(ValueError, match='exactly one'):
        performance_series(performance[:-1])
    with pytest.raises(ValueError, match='exactly one'):
        performance_series(performance + [performance[0]])
    performance[0]['nmse_db'] = 'nan'
    with pytest.raises(ValueError, match='non-finite'):
        performance_series(performance)


def test_geometry_all_windows_quantiles_and_unclipped_negative_source_weight():
    _, geometry = rows()
    summary, count, trajectories, windows = weight_summary(geometry)
    assert (count, trajectories, windows) == (108, 3, 18)
    affine = np.array([[r['alpha'], r['beta'], r['lambda0']] for r in geometry
                       if r['snr_db'] == 10 and r['method'] == 'affine'])
    np.testing.assert_array_equal(summary[('affine', 10)], np.quantile(affine, [.25, .5, .75], axis=0))
    assert np.all(summary[('affine', 10)][:, 2] < 0)


@pytest.mark.parametrize('fault', ['missing_window', 'duplicate_window', 'lambda_mismatch', 'convex_outside'])
def test_geometry_rejects_pairing_or_semantic_errors(fault):
    _, geometry = rows()
    if fault == 'missing_window':
        geometry.pop()
    elif fault == 'duplicate_window':
        geometry.append(dict(geometry[0]))
    elif fault == 'lambda_mismatch':
        geometry[0]['lambda0'] += .1
    else:
        geometry[0]['alpha'] = -.1
        geometry[0]['lambda0'] = 1 - geometry[0]['alpha'] - geometry[0]['beta']
    with pytest.raises(ValueError):
        weight_summary(geometry)


def test_two_figures_CSV_only_byte_rebuild_alignment_text_collision_and_scoped_style(tmp_path):
    import matplotlib
    performance, geometry = rows()
    save(tmp_path / 't16f_selection_mixing.csv', performance)
    save(tmp_path / 't16f_weight_geometry.csv', geometry)
    before = dict(matplotlib.rcParams)
    report = audit(tmp_path, tmp_path / 'figures')
    assert dict(matplotlib.rcParams) == before
    assert report['source_returncode'] == 0
    assert all(report['CSV_only_rebuild'].values())
    assert report['visual_review'] == 'pending'
    for stem in STEMS:
        checks = report['rendered_checks'][stem]
        assert checks['text_returncode'] == checks['collision_returncode'] == 0
        alignment = json.loads((tmp_path / 'figures' / f'{stem}.alignment.json').read_text())
        assert alignment['verdict'] == 'PASS'
