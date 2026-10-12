"""Pre-unseal governance only; no formal signal or performance tests."""
import ast
import itertools
import json

import pytest

from scripts import audit_formal_preflight as audit
from scripts.formal_governance_snapshot import historical_hashes, sealed_snapshot


@pytest.mark.parametrize('a,b,c', list(itertools.product((False, True), repeat=3)))
def test_researcher_confirmed_readiness_truth_table(a, b, c):
    result = audit.readiness(a, b, c)
    assert result['paper_core_empirically_supported'] is a
    assert result['paper_results_ready'] is (a and b and c)
    assert result['paper_results_ready_partial'] is (a and (b != c))


def test_non_pass_is_not_contradicted():
    assert audit.endpoint_status(False) == 'NOT_SUPPORTED_BY_FORMAL_ENDPOINT'
    assert audit.endpoint_status(True) == 'FORMALLY_SUPPORTED'


@pytest.mark.parametrize('value', [0, 1, None, 'true'])
def test_readiness_rejects_missing_or_non_boolean_outcomes(value):
    with pytest.raises(ValueError, match='Boolean'):
        audit.readiness(value, True, True)


def test_original_protocol_and_committed_method_exact():
    proof = audit.provenance()
    assert proof['original_protocol_sha256'] == audit.PROTOCOL_SHA
    assert proof['historical_content_mismatches'] == 0
    assert proof['frozen_method']['arrays_verified'] == 50
    assert len(proof['frozen_method']['rank4_array_sha256']) == 12


def test_amendment_is_independent_and_has_no_seed_permission():
    text = (audit.ROOT / audit.AMENDMENT).read_text(encoding='utf8')
    assert audit.PROTOCOL_SHA in text
    assert 'paper_core_empirically_supported = A' in text
    assert 'NOT_SUPPORTED_BY_FORMAL_ENDPOINT' in text
    assert 'No endpoint, numerical threshold, environment, method, statistics, bootstrap,' in text
    assert 'detector, seed, factor, fold, mean policy or estimator changes' in text


def test_individual_seed_reservation_is_original_and_unique():
    index, role_values = audit.formal_reservation()
    assert len(index) == 165000
    assert len(role_values['sionna_seed']) == 5000
    assert len(role_values['noise_seed']) == 50000
    hit = index[2702061545]
    assert hit == dict(environment='CDL-C-120-100', trajectory_id=1215,
                       role='bits_seed_tx0', frame_id=5)


def test_same_role_single_seed_collision_rejected_without_tuple_match(tmp_path, monkeypatch):
    (tmp_path / 'old.csv').write_text(
        'trajectory_id,frame_id,bits_seed_tx0,noise_seed\n89,3,2702061545,17\n', encoding='utf8')
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    index = {2702061545: dict(role='bits_seed_tx0')}
    result = audit.manifest_overlap(['old.csv'], index)
    assert result['unique_same_role_overlapping_seed_values'] == 1
    with pytest.raises(ValueError, match='overlap'):
        audit.require_seed_independence(result)


def test_cross_role_seed_collision_also_rejected(tmp_path, monkeypatch):
    (tmp_path / 'old.csv').write_text('noise_seed\n42\n', encoding='utf8')
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    result = audit.manifest_overlap(['old.csv'], {42: dict(role='bits_seed_tx0')})
    assert result['unique_same_role_overlapping_seed_values'] == 0
    assert result['unique_overlapping_seed_values'] == 1
    with pytest.raises(ValueError, match='STOP'):
        audit.require_seed_independence(result)


def test_zero_overlap_gate_passes_without_relaxation(tmp_path, monkeypatch):
    (tmp_path / 'old.csv').write_text('noise_seed\n1\n', encoding='utf8')
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    result = audit.manifest_overlap(['old.csv'], {42: dict(role='noise_seed')})
    audit.require_seed_independence(result)
    assert result['all_individual_seeds_disjoint']


def test_preflight_has_no_scientific_execution_calls():
    source = (audit.ROOT / 'scripts/audit_formal_preflight.py').read_text(encoding='utf8')
    calls = {getattr(n.func, 'attr', getattr(n.func, 'id', ''))
             for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call)}
    assert not calls & {'generate', 'prepare', 'fit', 'svd', 'default_rng', 'normal', 'integers', 'zf_detect'}
    assert sealed_snapshot()


def test_preserved_preflight_evidence_if_generated():
    # Before running the audit, this test still verifies actual source/protocol
    # provenance through the independent tests above; no absent scan is passed.
    if not audit.RESULT.exists():
        pytest.skip('preflight artifact not generated yet; independent governance tests still execute')
    result = json.loads(audit.RESULT.read_text(encoding='utf8'))
    assert result['completion_state'] == 'BLOCKED_BEFORE_UNSEAL'
    assert not result['formal_test_used'] and result['formal_test_sealed']
    assert not result['all_checks_passed']
    assert result['independence_audit']['unique_overlapping_seed_values'] == 7
    assert result['independence_audit']['unique_same_role_overlapping_seed_values'] == 3
    historical_hashes(result['source_sha256'])
    historical_hashes(result['historical_sha256'])
