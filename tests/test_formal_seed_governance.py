"""Seed metadata and immutable pre-unseal governance; no signal simulation."""
import ast
import json

import pytest

from scripts import assign_formal_seeds as seeds
from scripts.formal_governance_snapshot import sealed_snapshot


def test_canonical_serialization_and_sha256_known_answer():
    key = seeds.canonical_key('CDL-C-30-100', 0, 0, 'sionna_seed', -1)
    assert key == '["T20B_FORMAL_SEED_V2","CDL-C-30-100",0,0,"sionna_seed",-1]'
    assert [seeds.candidate(key, r) for r in range(3)] == [2414744325, 1329894834, 1453682773]


@pytest.mark.parametrize('args', [
    ('invalid', 0, 0, 'sionna_seed', -1),
    ('CDL-C-30-100', 1000, 0, 'sionna_seed', -1),
    ('CDL-C-30-100', -1, 0, 'sionna_seed', -1),
    ('CDL-C-30-100', 0, 1, 'sionna_seed', -1),
    ('CDL-C-30-100', 0, 10, 'noise_seed', -1),
    ('CDL-C-30-100', 0, 0, 'bits_seed_tx0', 1),
    ('CDL-C-30-100', 0, 0, 'noise_seed', 0),
])
def test_undeclared_item_or_stream_scope_rejected(args):
    with pytest.raises(ValueError, match='scopes'):
        seeds.canonical_key(*args)


@pytest.mark.parametrize('retry', [-1, 2**64, 0.0, True])
def test_retry_counter_encoding_range(retry):
    with pytest.raises(ValueError, match='unsigned64'):
        seeds.candidate('key', retry)


def test_collision_rejection_checks_history_and_entire_formal_prefix():
    items = list(seeds.random_items())[:3]
    proposals = {'sionna_seed': [7, 8, 9], 'torch_seed': [9, 10], 'numpy_seed': [10, 8, 11]}
    def derive(key, retry):
        return proposals[json.loads(key)[4]][retry]
    result = seeds.allocation({7, 8}, items, derive)
    assert [r['assigned_seed'] for r in result] == [9, 10, 11]
    assert [r['retry_counter'] for r in result] == [2, 1, 2]
    assert result == seeds.allocation({8, 7}, items, derive)


@pytest.mark.parametrize('value', [-1, 2**32, True, 1.0])
def test_out_of_range_candidate_rejected(value):
    with pytest.raises(ValueError, match='uint32'):
        seeds.allocation(set(), list(seeds.random_items())[:1], lambda *_: value)


def test_item_counts_and_trajectory_scoped_channel_consumption():
    items = list(seeds.random_items())
    assert len(items) == 165000
    assert len(set(items)) == 165000
    assert sum(r[3] == 'sionna_seed' for r in items) == 5000
    assert sum(r[3] == 'noise_seed' for r in items) == 50000
    assert all(r[2] == 0 for r in items if r[3] in seeds.frozen.ROLES[:3])
    assert items[:6] == [
        ('CDL-C-30-100', 0, 0, 'sionna_seed', -1),
        ('CDL-C-30-100', 0, 0, 'torch_seed', -1),
        ('CDL-C-30-100', 0, 0, 'numpy_seed', -1),
        ('CDL-C-30-100', 0, 0, 'bits_seed_tx0', 0),
        ('CDL-C-30-100', 0, 0, 'bits_seed_tx1', 1),
        ('CDL-C-30-100', 0, 0, 'noise_seed', -1)]


def test_historical_seed_fields_include_nested_smoke_lists_and_exclude_counts():
    found, labels = set(), set()
    seeds.record_values(dict(seed=7, channel_seed=8, bits_seed_tx1='9', bootstrap_seed=10,
        formal_seed_count_per_role={'noise_seed': 50000}, channel_seed_tuples=1200,
        smoke={'seeds': [11, 12], 'namespace': 'OLD_SMOKE'}, noise_seed=True,
        base_seed=13), found, labels)
    assert found == {7, 8, 9, 10, 11, 12, 13}
    assert labels == {'OLD_SMOKE'}
    assert seeds.source_seed_literals('config.seed=14\ntorch.manual_seed(15)\nrng=np.random.default_rng(16)\n') == {14, 15, 16}


def test_active_archived_and_smoke_record_formats(tmp_path, monkeypatch):
    (tmp_path / 'active.csv').write_text('namespace,bits_seed_tx0\nOLD_A,17\n', encoding='utf8')
    (tmp_path / 'archive.json').write_text('{"seed":18,"namespaces":["OLD_B"]}', encoding='utf8')
    (tmp_path / 'config.yaml').write_text('seed:\n  base_seed: 19\n', encoding='utf8')
    (tmp_path / 'smoke.py').write_text('config.seed=20\n', encoding='utf8')
    monkeypatch.setattr(seeds, 'ROOT', tmp_path)
    found, labels, sources = seeds.historical_records(['active.csv', 'archive.json', 'config.yaml', 'smoke.py'])
    assert found == {17, 18, 19, 20}
    assert labels == {'OLD_A', 'OLD_B'}
    assert len(sources) == 4


def miniature_plan():
    items = [(name, 0, 0, 'sionna_seed', -1) for name in seeds.original.CONDITIONS]
    return seeds.allocation(set(), items)


@pytest.mark.parametrize('failure', ['history', 'duplicate_seed', 'namespace', 'duplicate_key'])
def test_seed_and_namespace_gate_rejections(failure, monkeypatch):
    monkeypatch.setattr(seeds, 'ITEM_COUNT', 5)
    rows = miniature_plan()
    forbidden, labels = set(), set()
    if failure == 'history':
        forbidden.add(rows[0]['assigned_seed'])
    elif failure == 'duplicate_seed':
        rows[1]['assigned_seed'] = rows[0]['assigned_seed']
    elif failure == 'namespace':
        labels.add(rows[0]['namespace'])
    else:
        rows[1]['canonical_key'] = rows[0]['canonical_key']
    with pytest.raises(ValueError):
        seeds.inspect_manifest(rows, forbidden, labels)


def test_namespace_registrations_distinct_and_rows_deterministic(monkeypatch):
    monkeypatch.setattr(seeds, 'ITEM_COUNT', 5)
    rows = miniature_plan()
    assert seeds.csv_bytes(rows) == seeds.csv_bytes(miniature_plan())
    proof = seeds.inspect_manifest(rows, set(), set())
    assert proof['historical_namespace_intersection'] == 0
    assert proof['formal_namespace_duplicate'] == 0


def test_seed_only_contract_preserves_method_and_readiness():
    proof = seeds.immutable_contract()
    assert proof['original_protocol_sha256'] == seeds.original.PROTOCOL_SHA
    assert proof['readiness_amendment_sha256'] == seeds.READINESS_SHA
    assert proof['execution_level_numerical_changes'] == ['channel_bits_unit_noise_seed_assignment']
    assert proof['original_formal_contract']['trajectories_per_condition'] == 1000
    assert proof['original_formal_contract']['frames_per_trajectory'] == 10
    assert proof['original_formal_contract']['snr_db'] == [10, 20]
    assert proof['original_formal_contract']['bootstrap_seed'] == 20262001
    assert proof['frozen_method']['arrays_verified'] == 50
    assert len(proof['readiness_truth_table']) == 8


def test_immutable_contract_survives_exact_json_roundtrip():
    proof = seeds.immutable_contract()
    assert proof == json.loads(json.dumps(proof))
    assert proof['numerical_conventions']['noise_variances'] == {
        '5': 3.1436628565989504, '10': .9941134822523973, '20': .09941134822523973}


def test_no_scientific_calls_or_unseal_entrypoint():
    text = (seeds.ROOT / 'scripts/assign_formal_seeds.py').read_text(encoding='utf8')
    calls = {getattr(n.func, 'attr', getattr(n.func, 'id', ''))
             for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Call)}
    assert not calls & {'generate', 'prepare', 'fit', 'svd', 'default_rng', 'normal', 'integers', 'zf_detect'}
    assert sealed_snapshot()
    assert seeds.original.RESULT.exists()


def test_actual_seed_artifacts():
    if not (seeds.ROOT / seeds.AUDIT).exists():
        pytest.skip('Seed artifacts not generated yet; only independent preparation tests can run')
    result = seeds.validate_artifacts()
    assert result['formal_assigned_seed_count'] == 165000
    assert result['historical_seed_intersection'] == 0
    assert result['formal_duplicate_seed_count'] == 0
    assert result['historical_namespace_intersection'] == 0
    assert result['formal_namespace_duplicate'] == 0
    assert not result['formal_test_used'] and result['formal_test_sealed']
    assert result['endpoint_A'] == result['endpoint_B'] == result['endpoint_C'] == 'UNEVALUATED'
    assert not any(result[k] for k in ('H_generated', 'bits_generated', 'noise_generated',
                                     'received_grid_generated', 'nmse_evaluated', 'ber_evaluated', 'bootstrap_evaluated'))
    if result['formal_preflight_ready']:
        assert result['full_pytest_passed'] and result['all_checks_passed']
        assert seeds.frozen.passed_summary(seeds.ROOT / 'outputs/t20b_seed_full_pytest.txt', 1918) == result['full_pytest_summary']
