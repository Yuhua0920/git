"""T20A governance regression; no scientific numerical experiments."""
import pytest

from scripts import audit_final_method_freeze as audit


def test_history_tampering_is_rejected(tmp_path):
    p = tmp_path / 'historical.json'
    p.write_text('{"value":1}', encoding='utf8')
    original = audit.sha(p)
    assert audit.verify_hashes({'historical.json':original}, tmp_path) == 1
    p.write_text('{"value":2}', encoding='utf8')
    with pytest.raises(ValueError, match='content mismatch'):
        audit.verify_hashes({'historical.json':original}, tmp_path)


def test_full_passing_summary_required(tmp_path):
    p = tmp_path / 'log.txt'
    p.write_text('1864 passed, 1 failed in 1s\n', encoding='utf8')
    with pytest.raises(ValueError, match='passing pytest'):
        audit.passed_summary(p, 1864)
    p.write_text('1864 passed in 424.95s (0:07:04)\n', encoding='utf8')
    assert audit.passed_summary(p,1864).startswith('1864 passed')


def test_literature_schema_and_unread_fields():
    r = audit.literature()
    assert len(r['papers']) == 12
    assert len(r['fields']) == 20
    for p in r['papers']:
        if p['full_text'] == 'unavailable':
            assert all(p['values'][i]=='UNKNOWN' for i in audit.TECHNICAL_FIELDS)


@pytest.mark.parametrize('paper_id', ['A','B','C','D','E','F','G1','G2','G3','G4','G5','H'])
def test_matrix_contains_verified_reference_and_all_fields(paper_id):
    r = audit.literature()
    p = next(v for v in r['papers'] if v['id'] == paper_id)
    rendered = audit.render_matrix(r)
    assert p['source'] in rendered
    assert all(f'| {field} | {value} |' in rendered for field,value in zip(r['fields'],p['values']))


def test_governance_cannot_import_scientific_execution():
    proof = audit.no_experiment_audit()
    assert proof['generator_calls'] == proof['NMSE_evaluations'] == proof['BER_evaluations'] == 0
    assert proof['scientific_imports'] == []


def test_constants_and_permanent_claims():
    r = audit.constants_audit()
    assert r['module_count'] == 2 and r['rank_per_module'] == 4
    assert r['causal_window'] == 5 and r['means'] == 'source'
    assert audit.claim_states()['c12_claim_status'] == 'CONTRADICTED_BY_HELDOUT_FACTOR_TRANSFER'
    assert audit.claim_states()['t19a_rank4_out_of_factor_warning'] is True


def test_channel_seed_scope_and_frame_communication_scope():
    a,b = audit.seeds(0,600,0),audit.seeds(0,600,9)
    assert a[:3] == b[:3]
    assert all(x!=y for x,y in zip(a[3:],b[3:]))
    assert len(set(a)) == 6


@pytest.mark.parametrize('args', [(-1,600,0),(5,600,0),(0,599,0),(0,1600,0),(0,600,10)])
def test_undeclared_formal_allocation_rejected(args):
    with pytest.raises(ValueError,match='preregistered'):
        audit.seeds(*args)


def test_frozen_arrays_are_checked_not_reconstructed():
    r = audit.frozen_arrays()
    assert r['fold_sha256'] == audit.FOLD
    assert len(r['rank4_array_sha256']) == 12
    assert all(shape in ([3968,4],[4,64]) for shape in r['rank4_shapes'].values())
    assert r['svd_calls'] == r['estimator_evaluations'] == r['covariance_fits'] == 0


def test_namespace_is_disjoint_without_signal_generation():
    r = audit.namespace_audit()
    assert r['tuple_collision_count'] == 0 and r['signal_rng_calls'] == 0
    assert r['formal_manifest_materialized'] is False
    assert r['historical_manifests_checked']


def test_formal_design_required_controls_and_boundaries():
    text = (audit.ROOT / audit.DOCS[-1]).read_text(encoding='utf8')
    assert all(v in text for v in audit.METHODS)
    assert 'Delay boundary' in text and 'not compulsory success' in text
    assert 'Bonferroni' in text and '10000' in text and 'trajectory' in text
    assert 'D1/Linear-MMSE remains historical sensitivity evidence only' in text
    assert 'No method changes after opening' in text


@pytest.mark.parametrize('blocked', ['critical_collision', 'core_evidence_adequate',
    'executable_without_estimator_change', 'third_module_needed'])
def test_readiness_fails_when_any_freeze_requirement_fails(blocked):
    args = dict(critical_collision=False,core_evidence_adequate=True,protocol_complete=True,
                executable_without_estimator_change=True,third_module_needed=False)
    assert audit.readiness_gate(**args) == (True, True)
    args[blocked] = not args[blocked]
    assert audit.readiness_gate(**args) == (False, False)


def test_protocol_incomplete_prevents_paper_readiness():
    assert audit.readiness_gate(critical_collision=False,core_evidence_adequate=True,
        protocol_complete=False,executable_without_estimator_change=True,third_module_needed=False) == (True,False)


def test_actual_governance_artifact():
    r = audit.load(audit.OUT / 't20a_results.json')
    audit.verify_hashes(r['source_sha256'])
    assert r['method_components_frozen'] and r['literature_gate_completed']
    assert r['paper_core_ready_for_formal_test'] == (r['method_ready_to_freeze'] and r['formal_test_protocol_complete'] and not r['critical_novelty_collision'])
    assert r['potential_novelty_collision'] is True
    assert not r['new_channel_generated'] and not r['new_model_trained']
    assert r['formal_test_sealed'] and not r['formal_test_used']
    assert len(r['retained_contribution_candidates']) <= 3
    assert r['historical_content_mismatches'] == 0
    assert not r['formal_protocol']['formal_orchestration_implemented']
