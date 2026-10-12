"""Faithful package equations, real protocol geometry and no-leakage contract."""
import inspect
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import sp_u3a_mfa_port as p
from scripts import sp_u3a_common as c
import numpy as np

def test_paper_and_software_pin():
    audit=c.load(c.OUT/'sp_u3a_mfa_source_audit.json')
    assert audit['paper']['manuscript']=='arXiv:2304.14809v2'
    assert [r['commit'] for r in audit['repositories']]==['9f9c22a7775076fbedd0fb73929c195b0dd2aa6b','d11915070e676579008be6f3a02b35dffb1fa189']
    for r in audit['repositories']:assert r['version']==r['PyPI_version']=='0.1.1'

def test_license_and_author_bytes():
    for name in ('cplx-mfa','MFA_estimator'):
        root=p.SOURCE/name;pin=c.load(root/'repository_pin.json')
        assert pin['license_metadata']['spdx_id']=='BSD-3-Clause'
        assert all(c.sha(root/file)==value for file,value in pin['files'].items())

@pytest.mark.parametrize('case',['rectangular_white','rectangular_colored','selector','identity'])
def test_author_formula_exact(case):
    toy=c.load(c.OUT/'sp_u3a_mfa_toy_validation.json')
    row=next(a for a in toy['fixtures'] if a['case']==case)
    assert row['finite'] and row['dtype']=='complex128'
    assert max(row[k] for k in ('responsibility_error','component_estimate_error','mixture_estimate_error','responsibility_sum_error'))<1e-10

def test_K1_and_complex_variance():
    n=c.load(c.OUT/'sp_u3a_mfa_toy_validation.json')['noise']
    assert n['K1_LMMSE_error']<1e-12 and n['wrong_half_variance_error']>.01
    assert abs(n['empirical_complex_variance']-.5)<.0075
    assert abs(n['empirical_real_variance']-.25)<.0075 and abs(n['empirical_imag_variance']-.25)<.0075

def test_actual_fit_deterministic_and_learned_formula():
    t=c.load(c.OUT/'sp_u3a_mfa_toy_validation.json');assert t['actual_fit']['fits']==2 and t['actual_fit']['deterministic_exact']
    assert t['learned_model_formula']['mixture_estimate_error']<1e-10

def test_full_mixture_requires_float_one():
    model=p.known_model();A=np.eye(8)[[0,3,7]];y=np.zeros((2,3),complex);Cn=.37*np.eye(3)
    full=model.estimate(y,Cn,A,1.0);default=model.estimate(y,Cn,A)
    assert p.error(full,default)>1e-3
    assert p.error(full,p.reference(model.means_,model.covariances_,model.weights_,y,A,Cn)[2])<1e-10

def test_selector_C_order_and_partition():
    g,A=p.mapping();H=np.arange(4096).reshape(2,2,64,16)
    assert np.array_equal(A@H.ravel(order='C'),H.ravel(order='C')[g.selected])
    assert A.shape==(64,4096) and np.count_nonzero(A)==64
    assert g.data.sum()==3968 and len(g.selected)+g.data.sum()==4032
    assert len(set(g.selected)&set(np.flatnonzero(g.data)))==0
    for index in g.selected:
        rx,tx,k,t=np.unravel_index(index,H.shape)
        assert index==(((rx*2+tx)*64+k)*16+t)

def test_actual_owner_division_and_noiseless_mapping():
    row=c.load(c.OUT/'sp_u3a_mfa_toy_validation.json')['mapping']
    assert row['noiseless_error']<1e-12 and row['owner_division_noise_error']<1e-12
    assert row['noise_covariance_error']<1e-12 and row['data_extraction_exact']

def test_allowed_offline_budget_and_C30_alias():
    ledger=c.load(c.OUT/'sp_u3a_offline_information_ledger.json')
    rows=[a for a in ledger['rows'] if a['used_final_method']]
    assert len(rows)==3 and sum(a['independent_trajectories'] for a in rows)==1500
    assert sum(a['frames_samples'] for a in rows)==15000
    assert all(a['source_and_two_corrections_bit_exact'] for a in ledger['operator_reconstruction'])
    assert not any(a['MFA_Prepared_allowed'] for a in ledger['rows'] if a['environment'] in ('CDL-B-120','CDL-C-75','T20B formal / held-out delay'))

def test_tuning_protocol_no_target_and_grouped_split():
    cfg=p.protocol();assert cfg['K']==[1,2,4] and cfg['latent_rank']==[4,8,16]
    assert cfg['estimate_summands']==1.0 and not cfg['next_task_automatic']
    assert 'trajectory' in cfg['tuning_split'] and '400..499' in cfg['tuning_split']
    assert 'B120 clean training' in cfg['forbidden'] and cfg['zero_mean'] is False

def test_online_entry_no_privileged_argument():
    assert list(inspect.signature(p.estimate_data).parameters)==['model','y','sigma2']
    source=inspect.getsource(p.estimate_data)
    assert 'n_summands_or_proba=1.0' in source

@pytest.mark.parametrize('file',['t20b_formal_outcomes.csv','sp_u2d_age_rows.csv','t16c_target_statistics_B120.npz'])
def test_formal_and_theory_outcome_firewall(file):
    with pytest.raises(PermissionError,match='firewall'):
        (c.OUT/file).open('rb')

def test_existing_source_files_unchanged():
    assert len(c.verify()['protected_sources'])==539

def test_toy_completed_before_smoke():
    pre=c.load(c.OUT/'sp_u3a_preparation_gate.json');assert pre['before_toy'] and pre['before_CDL_smoke']
    assert c.load(c.OUT/'sp_u3a_mfa_toy_validation.json')['passed']

def test_faithful_adapter_source_no_author_changes():
    assert p.MfaEstimator.__module__=='mfa_estimator.estimator'
    assert Path(inspect.getfile(p.MfaEstimator)).resolve()==(p.SOURCE/'MFA_estimator/src/mfa_estimator/estimator.py').resolve()
