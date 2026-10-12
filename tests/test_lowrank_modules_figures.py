"""T16G independent plotting contract tests; fixture output is not scientific data."""
import csv
import json

import pytest

from scripts.plot_lowrank_modules import (DATA_FILE, METHODS, RANKS, SHIFTS, SNRS, STEMS,
                                         audit, penalty_series, performance_series)


def rows():
    result = []
    for shift in SHIFTS:
        for snr in SNRS:
            dense_nmse = -snr
            for method, rank in [('source', 0), ('hard', 0), ('convex', 0), ('affine', 0),
                                 ('matched', 0)] + [('lr-affine', rank) for rank in RANKS]:
                penalty = 2 / rank - .15 if method == 'lr-affine' else 0
                nmse = dense_nmse + penalty
                if method == 'matched':
                    nmse -= .5
                result.append(dict(shift=shift, snr_db=snr, window=5, subset='last6', method=method,
                                   rank=rank, nmse_db=nmse, compression_penalty_db=penalty,
                                   recovery_preservation=.95, lowrank_affine_vs_convex_gain_db=1.5))
    return result


def save(path, data):
    with path.open('w', encoding='utf8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def test_requested_performance_curves_use_fair_five_frame_subset():
    data = rows()
    extra = dict(data[0], window=1, subset='all10', nmse_db=999)
    mapping, count = performance_series(data + [extra])
    assert count == 30
    assert mapping[('CDL-C-75', 'affine', 0)] == [-5, -10, -20]
    assert mapping[('CDL-B-120', 'lr-affine', 4)] == [-4.65, -9.65, -19.65]


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'non_finite'])
def test_performance_rejects_missing_duplicate_or_nonfinite(fault):
    data = rows()
    target = next(row for row in data if row['method'] == 'matched')
    if fault == 'missing':
        data.remove(target)
    elif fault == 'duplicate':
        data.append(dict(target))
    else:
        target['nmse_db'] = 'nan'
    with pytest.raises(ValueError):
        performance_series(data)


def test_penalty_uses_all_frozen_ranks_and_preserves_negative_values():
    mapping, count = penalty_series(rows())
    assert count == 20
    assert mapping[('CDL-B-120', 20)] == [2 / rank - .15 for rank in RANKS]
    assert mapping[('CDL-B-120', 20)][-1] < 0


@pytest.mark.parametrize('fault', ['missing_rank', 'duplicate_rank', 'new_rank', 'wrong_formula', 'non_finite'])
def test_penalty_rejects_rank_grid_or_formula_errors(fault):
    data = rows()
    target = next(row for row in data if row['shift'] == 'CDL-B-120' and row['snr_db'] == 20
                  and row['method'] == 'lr-affine' and row['rank'] == 16)
    if fault == 'missing_rank':
        data.remove(target)
    elif fault == 'duplicate_rank':
        data.append(dict(target))
    elif fault == 'new_rank':
        target['rank'] = 12
    elif fault == 'wrong_formula':
        target['compression_penalty_db'] += .1
    else:
        target['compression_penalty_db'] = 'inf'
    with pytest.raises(ValueError):
        penalty_series(data)


def test_small_fixture_two_figures_alignment_text_collision_byte_rebuild_and_scoped_style(tmp_path):
    import matplotlib
    save(tmp_path / DATA_FILE, rows())
    before = dict(matplotlib.rcParams)
    report = audit(tmp_path, tmp_path / 'fixture_figures')
    assert dict(matplotlib.rcParams) == before
    assert report['source_returncode'] == 0
    assert all(report['CSV_only_rebuild'].values())
    assert report['visual_review'] == 'pending'
    assert report['source_data_selection'][DATA_FILE]['input_rows'] == 60
    for stem in STEMS:
        check = report['rendered_checks'][stem]
        assert check['text_returncode'] == check['collision_returncode'] == 0
        alignment = json.loads((tmp_path / 'fixture_figures' / f'{stem}.alignment.json').read_text())
        assert alignment['verdict'] == ('PASS' if stem == STEMS[0] else 'NOT APPLICABLE')
