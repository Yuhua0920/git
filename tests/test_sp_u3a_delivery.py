"""Actual smoke, fairness and complete gate delivery checks."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import sp_u3a_common as c
import numpy as np

def test_actual_project_shape_smoke_and_mapping():
    s=c.load(c.OUT/'sp_u3a_project_shape_smoke.json')
    assert s['passed'] and s['train_shape']==[6,4096] and s['pilot_shape']==[2,64] and s['target_shape']==[2,3968]
    assert s['finite_output'] and s['responsibility_sum_error']<1e-12 and s['direct_selector_estimate_error']<1e-10
    assert s['mapping']['noiseless_error']<1e-12 and s['mapping']['data_extraction_exact']
    assert s['EM_iterations']==1 and not s['converged_scientific_model'] and s['EM_warnings']
    assert s['process_memory']['peak_working_set_bytes']>0 and s['resident_prior_array_bytes']>500000000
    assert all(a['frozen_source_exact'] for a in s['trajectory_replay'])

def test_smoke_source_cache_and_no_training_comparison():
    with np.load(c.OUT/'sp_u3a_source_smoke_cache.npz',allow_pickle=False) as f:
        assert f['clean'].shape==(6,4096) and np.isfinite(f['clean']).all()
    s=c.load(c.OUT/'sp_u3a_project_shape_smoke.json')
    assert s['no_target_or_formal_CSI'] and not s['full_MFA_training'] and not s['comparison_NMSE_computed'] and not s['BER_computed']

def test_fairness_and_gate_scope():
    gate=c.load(c.OUT/'sp_u3a_baseline_gate.json')
    assert gate['gate']=='PASS' and all(gate['criteria'].values()) and gate['stop'] and gate['theory']=='FREEZE'
    assert not gate['comparison'] and not gate['full_training'] and not gate['paper_edit'] and not gate['commit']
    matrix=c.load(c.OUT/'sp_u3a_mfa_fairness_matrix.json')
    assert not matrix['exact_information_equivalence'] and len(matrix['rows'])==4

def test_original_and_protocol_integrity():
    assert len(c.verify()['protected_sources'])==539
    pre=c.load(c.OUT/'sp_u3a_preparation_gate.json')
    assert c.sha(ROOT/'configs/sp_u3a_protocol.json')==pre['protocol_sha256']
    assert all(c.sha(c.OUT/p)==h for p,h in pre['old_input_sha256'].items())

def test_upstream_actual_tests_and_environment():
    test=c.load(c.OUT/'sp_u3a_tests_gate.json');assert test['passed'] and len(test['results'])==2
    assert all(a['exit_code']==0 for a in test['results'])
    env=c.load(c.OUT/'sp_u3a_environment_audit.json');assert env['dependency_installs']==env['environment_writes']==0
    assert all(c.sha(a['module_file'])==a['module_sha256'] for a in env['dependencies'])

def test_required_outputs_and_no_new_novelty_claim():
    for name in ('mfa_source_audit','mfa_equation_inventory','offline_information_ledger','mfa_fairness_matrix',
       'mfa_hyperparameter_protocol','mfa_toy_validation','project_shape_smoke','baseline_gate'):
        assert (c.OUT/f'sp_u3a_{name}.json').is_file()
    n=c.load(c.OUT/'sp_u3a_novelty_ledger.json');assert n['tier']=='Tier1 closest work'
    assert n['prohibited_novelty']=='observation-dependent LMMSE mixing itself'
