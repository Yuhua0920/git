"""Resource-stop regression; later-phase checks are explicitly unexecuted."""
import ast
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import sp_u3b_gate as g


def test_exact_author_and_prior_delivery_hashes():
    result = g.verify_u3a()
    assert result['passed'] and result['counts']['protected_sources'] == 539
    assert result['counts']['artifact_sha256'] == 67


def test_native_expression_eager_broadcast():
    proof = g.author_allocation()
    assert proof['eager_argument'] and not proof['chunking']
    assert 'zeroed_data[:, None, :] * latents[None, :, :].conj()' in proof['expression']


def test_exact_B_allocation_exceeds_cap_without_allocation():
    bound = g.allocation_bound(4, 16)
    assert bound['product_shape'] == [4096, 16, 12000]
    assert bound['temporary_bytes'] == 12582912000
    assert bound['temporary_GiB'] == 11.71875
    assert bound['necessary_process_array_GiB'] == 14.451171875
    assert bound['temporary_bytes'] > g.protocol()['max_process_bytes']


def test_A_lower_bound_is_not_a_resource_pass():
    row = g.load(g.OUT / 'sp_u3b_resource_gate.json')['sentinels'][0]
    assert row['allocation']['temporary_GiB'] == 2.9296875
    assert row['estimated_resource_gate'] == 'LOWER_BOUND_WITHIN_CAP_NOT_A_PASS'
    assert not row['fit_started'] and row['converged'] is None


def test_native_convergence_state_not_fabricated():
    assert 'no public converged_' in g.author_allocation()['convergence_state']
    for row in g.load(g.OUT / 'sp_u3b_resource_gate.json')['sentinels']:
        assert row['iterations'] is None and row['converged'] is None
        assert row['fit_wall_seconds'] is None and row['observed_peak_memory_bytes'] is None
        assert row['lower_bound_trace'] == [] and row['weights'] is None


def test_phase_order_and_global_stop():
    gate = g.load(g.OUT / 'sp_u3b_resource_gate.json')
    assert gate['gate'] == 'FAIL' and gate['stop']
    assert gate['phase_B'] == gate['phase_C'] == 'BLOCKED'
    assert gate['scientific_fits_executed'] == gate['failed_executed_fits'] == 0
    assert not gate['full_corpus_replayed'] and not gate['B120_generated']


def test_full_unexecuted_candidate_tables_no_silent_deletion():
    rows = g.load(g.OUT / 'sp_u3b_grid_status.json')['rows']
    assert len(rows) == 54 and rows == g.grid_status()
    for variant in g.protocol()['variants']:
        subset = [r for r in rows if r['variant'] == variant]
        assert len(subset) == 27
        assert all(r['status'] == 'NOT_RUN_PHASE_A_RESOURCE_FAIL' for r in subset)
        assert all(r['validation_NLL'] is None for r in subset)


def test_frozen_split_pool_budget_and_no_B120_training():
    cfg = g.protocol()
    assert cfg['fit_ids'] == [0, 399] and cfg['validation_ids'] == [400, 499]
    assert cfg['allowed_environments'] == ['CDL-C-30', 'CDL-B-30', 'CDL-C-120']
    assert 'CDL-B-120' in cfg['forbidden_training']
    assert 'labels excluded from EM input' in cfg['pool_order']
    assert 3 * 400 * cfg['frames_per_trajectory'] == 12000
    assert 3 * 100 * cfg['frames_per_trajectory'] == 3000


def test_protocol_bytes_frozen_in_actual_gate():
    assert g.load(g.OUT / 'sp_u3b_resource_gate.json')['protocol_sha256'] == g.sha(ROOT / 'configs/sp_u3b_protocol.json')


def test_no_algorithm_fit_or_generator_called_by_gate():
    tree = ast.parse((ROOT / 'scripts/sp_u3b_gate.py').read_text(encoding='utf8'))
    functions = [n.func for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert not any(isinstance(f, ast.Attribute) and f.attr in ('fit', 'generate', 'estimate') for f in functions)


@pytest.mark.parametrize('path', ['t20b_formal_outcomes.csv', 'sp_u2d_age_rows.csv',
                                  't16c_target_statistics_B120.npz'])
def test_firewall_denies_old_outcomes_before_open(path):
    with pytest.raises(PermissionError, match='SP-U3B firewall'):
        (g.OUT / path).open('rb')


def test_prior_outputs_read_only():
    with pytest.raises(PermissionError, match='SP-U3B firewall'):
        (g.OUT / 'sp_u3a_baseline_gate.json').open('a', encoding='utf8')


@pytest.mark.parametrize('requirement', [
    'actual corpus ID/split and balanced loading',
    'actual three-seed candidate scores and seed aggregation',
    'actual final full-data refit seed rule',
    'finite fitted model and component-weight sum',
    'actual clean validation NLL reconstruction',
    'generated B120 seed uniqueness',
    'actual paired observations across five estimators',
    'actual trajectory-level NMSE/BER aggregation',
    'production LR4 numerical replay on new B120 data'])
def test_later_phase_explicitly_unexecuted(requirement):
    pytest.skip('PHASE A estimated resource FAIL: ' + requirement + ' not executed; no scientific PASS claimed')
