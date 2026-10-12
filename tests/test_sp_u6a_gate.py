"""Only SP-U6A allowed source/interface tests; no historical performance reads."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('sp_u6a_audit', ROOT/'scripts/sp_u6a_audit.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)

def output(name):
    return json.loads(AUDIT.audit_read(ROOT/('outputs/sp_u6a_'+name+'.json')))

def test_source_freeze_bytes_and_latest_version():
    pin=output('source_freeze')
    for row in pin['sources']:
        assert AUDIT.sha(AUDIT.audit_read(AUDIT.SOURCES/row['path'])) == row['sha256']
    assert pin['commit'] == 'fa2647f7fd0411bdb6a33ad9705cadb134ab3793'
    assert 'this version, v5' in AUDIT.text_of_html(AUDIT.source('2506.00452_abs.html'))

def test_old_literature_source_equality_not_outcomes():
    ledger=output('ammse_version_ledger')
    assert ledger['same_commit']
    assert all(ledger['current_vs_u2a_bytes_equal'].values())

@pytest.mark.parametrize('forbidden', ['outputs/t20b_formal_results.json','outputs/sp_u2b_analysis.json','outputs/sp_u4b_summary.csv','outputs/sp_u5a_nmse.csv','outputs/sp_u5b_nmse.csv'])
def test_firewall_rejects_forbidden_before_open(forbidden):
    with pytest.raises(PermissionError,match='firewall rejected'):
        AUDIT.audit_read(ROOT/forbidden)

def test_decoder_parameter_bound_independent_of_batch():
    resource=output('resource_gate')
    q=2*64*4096
    assert resource['project']['kernel_parameters'] == 7*q*q
    assert resource['project']['float32_kernel_bytes'] == 7*2**40
    assert resource['project']['unconditional_five_square_kernel_bytes'] == 5*2**40
    assert resource['native']['kernel_parameters'] == 7*(2*72*1008)**2
    assert resource['STOP_D'] and not resource['project']['measured_allocation']

def test_joint_fdm_selector_and_permutation():
    geometry=output('geometry')
    indices=np.array(geometry['pilot_indices'])
    r,q,k,t=np.unravel_index(indices,(2,2,64,16),order='C')
    assert indices.size == 64 and np.unique(indices).size == 64
    assert np.all(k % 16 == q*8) and np.all(t%4 == 0)
    assert all(geometry['frozen_mask_comparisons'].values())
    # A coordinate permutation must preserve the complete joint linear map.
    rng=np.random.default_rng(6001)
    w=rng.normal(size=(11,6))+1j*rng.normal(size=(11,6))
    y=rng.normal(size=6)+1j*rng.normal(size=6)
    ph=rng.permutation(11); py=rng.permutation(6)
    np.testing.assert_allclose(w[np.ix_(ph,py)]@y[py], (w@y)[ph],rtol=1e-14,atol=1e-14)

def test_complex_packing_transpose_serialization_and_nmse_algebra():
    result=output('algebra_identity')
    assert result['passed'] and result['array_roundtrip_exact'] and result['pilot_roundtrip_exact']
    assert result['transpose_is_not_Hermitian']
    assert not result['author_network_executed']

def test_global_mode_and_loss_source_contract():
    main=AUDIT.source('A-MMSE__main.py'); core=AUDIT.source('A-MMSE__AMMSE.py')
    assert 'USE_TRANSFORMER_GLOBAL_FILTER = True' in main
    assert 'USE_TRAINABLE_SHARED_FILTER = False' in main
    assert 'self.process_batch(self.global_token, training=False)' in core
    assert 'loss=complex_mse' in main
    assert output('equation_inventory')['joint_project_candidate']['network_implemented'] is False

def test_fairness_split_and_no_target_exposure():
    fairness=output('fairness')
    assert fairness['future_trajectory_split']['Source'] == [4000,1000]
    assert fairness['future_trajectory_split']['Prepared'] == [12000,3000]
    assert fairness['B120_opened'] is False and fairness['noise_generated'] is False
    assert not fairness['labels_as_network_input']
    assert fairness['sigma2_10'] == 0.9941134822523973
    assert fairness['sigma2_20'] == 0.09941134822523973
    assert np.isclose(fairness['sigma2_10'],fairness['P_ref']/10,rtol=1e-15,atol=0)
    assert np.isclose(fairness['sigma2_20'],fairness['P_ref']/100,rtol=1e-15,atol=0)

def test_stop_d_does_not_promote_unrun_checks_to_pass():
    decision=output('decision'); resource=output('resource_gate')
    assert decision['engineering_verdict'] == 'FAIL / STOP-D'
    assert decision['project_forward_backward'] == 'NOT_RUN'
    assert decision['toy_author_network'] == 'NOT_RUN'
    assert resource['GPU_training_peak_bytes'] is None
    assert not decision['next_phase_authorized']
    assert resource['license']['status'] == 'LICENSE_OPEN_RISK'

@pytest.mark.parametrize('check', ['author_toy_identity','project_batch1_forward_backward','project_batch50_forward_backward','one_step_semantics'])
def test_neural_smokes_explicitly_unexecuted_after_stop_d(check):
    pytest.skip(check+': STOP-D established by source-derived batch-independent weight bound; no neural model executed')
