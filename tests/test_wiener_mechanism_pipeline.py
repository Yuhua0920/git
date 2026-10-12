"""T17A frozen provenance, information boundaries, CSV and actual gate regression."""
import ast
import inspect
import json
from pathlib import Path
import numpy as np
import pytest
from src.wiener_perturbation import compare,mechanism_gate
from scripts import run_wiener_mechanism as run
from scripts.audit_wiener_mechanism import verify
from scripts.run_cdl_mismatch import load,sha,csvsave
from scripts.run_pilot_cv import readcsv


def test_accepted_frozen_provenance_and_exact_operator_sentinels():
    commit,_=run.accepted()
    assert commit=='37c6536096aca288a0cd1db3b030bfbb12789ea3'
    s,v,proof=run.load_frozen_statistics();w,h,q=run.prepare_operators(s,v)
    assert set(s)=={'C30','B30','C120','B120','C75'} and set(v)=={5,10,20}
    assert len(proof)==5 and len(h)==15 and len(q)==6
    assert sum(row['historical_raw_W_hash_exact'] is True for row in h)==9
    assert all(row['all_fields_exact'] for row in q)
    assert all(not a.flags.writeable and a.dtype==np.complex128 for a in w.values())


def test_no_generator_fit_inverse_and_preflight_no_mechanism():
    source=inspect.getsource(run)
    calls={getattr(n.func,'id',getattr(n.func,'attr','')) for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call)}
    assert not calls & {'generate','communication','fit','StreamingStatistics','inv','pinv'}
    assert 'evaluate(' not in inspect.getsource(run.formal_preflight)
    assert "corrected=ladd+lint+cross" in source
    assert "compare(delta_pm,value,lower)" in source
    assert "target_weight='B120'" in source


@pytest.mark.parametrize('first,second,expected',[(.8,.95,'SUPPORTED_IN_TESTED_B120_CONDITIONS'),
    (.4,.8,'PARTIALLY_SUPPORTED'),(-20,-100,'CONTRADICTED_BY_CURRENT_MECHANISM_TEST')])
def test_c6_status_is_only_the_preregistered_gate(first,second,expected):
    rows=[dict(snr_db=s,first_order_additive_closure=first,interaction_corrected_closure=second) for s in (10,20)]
    assert run.c6_status(mechanism_gate(rows))==expected


def test_common_numerical_CSV_and_JSON_serialization_before_main(tmp_path):
    a=np.array([[1,2j],[3,4]],complex);b=.6*a
    covariance=[run.covariance_additivity(a,b,s,'Rdp') for s in (5,10,20)]
    assert covariance[0]['additive_to_actual_norm_ratio']==pytest.approx(.6)
    assert covariance[0]['relative_interaction_norm']==pytest.approx(.4)
    data=[dict(snr_db=s,prediction=p,**compare(a,m*a,np.eye(2))) for s in (5,10,20)
          for p,m in [('exact_correction',1),('first_order_additive',3)]]
    path=tmp_path/'sentinel.csv';csvsave(path,data)
    loaded=readcsv(path);assert len(loaded)==6
    assert {float(r['closure']) for r in loaded}=={1.,-3.}
    json.dumps(dict(covariance=covariance,closure=data),allow_nan=False)


def test_preparation_and_permission_provenance_preserved_if_locked():
    if not run.FORMAL_PREFLIGHT.exists():
        # Preparation is an executable state, not a skipped actual-result test.
        assert 'UNCONFIRMED' in (run.ROOT/run.CLAIM).read_text(encoding='utf8')
        return
    p=load(run.FORMAL_PREFLIGHT)
    assert p['startup_full_pytest_passed'] and p['startup_full_pytest_count']==1610
    assert p['permission_recovery_completed'] and p['C6']=='UNCONFIRMED'
    for path,digest in p['preparation_artifact_sha256'].items():assert sha(run.ROOT/path)==digest,path
    for path,digest in p['source_sha256'].items():
        assert sha(run.ROOT/p['source_snapshot_root']/path)==digest,path


def test_actual_result_provenance_gate_and_C75_fields():
    if not run.RESULT.exists():
        assert not (run.OUT/'t17a_mechanism_closure.csv').exists()
        return
    result=load(run.RESULT);assert verify(result)
    assert result['formal_test_sealed'] and not result['formal_test_used']
    assert result['generator_calls']==result['coefficient_refits']==0
    assert not result['new_channel_generated'] and not result['new_statistics_fit']
    assert not result['new_model_trained'] and not result['novelty_claim_made']
    assert result['gate_reference']=='W_B120-W0'
    assert result['gate_stage1']=='L_add' and result['gate_stage2']=='L_add + L_interaction + Q_cross'
    assert result['scientific_gate_thresholds']==dict(strong_first=.70,strong_second=.90,contradicted_first=.30,contradicted_second=.50)
    assert len(result['mechanism_closure'])==21 and len(result['first_order_results'])==12
    assert all(r['target_weight']=='B120' for r in result['first_order_results'])
    for row in result['interaction_decomposition']:
        assert row['Q_cross_identity_relative_residual']<=1e-11
        for name,key in [('first_order_additive','first_order_additive_closure'),('interaction_corrected','interaction_corrected_closure')]:
            c=next(r for r in result['mechanism_closure'] if r['snr_db']==row['snr_db'] and r['prediction']==name)
            assert c['closure']==row[key] and c['closure']==1.-c['weighted_residual_fraction']
    assert len(result['c75_interpolation'])==3
    for row in result['c75_interpolation']:
        for suffix in ('D_weighted','P_weighted','D_fro','P_fro'):
            assert np.isfinite(row['cosine_'+suffix]) and -1.-1e-12<=row['cosine_'+suffix]<=1.+1e-12
        assert row['distinct_objectives'] and row['beta_model_classes_distinguished'] and not row['used_for_primary_gate']
    for name,length in [('t17a_covariance_additivity.csv',6),('t17a_perturbation_terms.csv',12),
        ('t17a_mechanism_closure.csv',21),('t17a_interaction_spectrum.csv',1536),('t17a_c75_interpolation.csv',3)]:
        assert len(readcsv(run.OUT/name))==length


def test_actual_singular_energy_complete_and_subspace_angles():
    if not run.RESULT.exists():return
    r=load(run.RESULT);assert len(r['rank_diagnostics'])==24 and len(r['subspace_diagnostics'])==48
    for snr in (5,10,20):
        for weighting in ('raw','target_weighted'):
            for op in ('R_exact','R_pred_1','Q_cross','R_pred_2'):
                rows=[v for v in r['interaction_spectrum'] if (v['snr_db'],v['weighting'],v['operator'])==(snr,weighting,op)]
                assert [v['mode'] for v in rows]==list(range(1,65))
                energies=np.array([v['singular_value']**2 for v in rows])
                np.testing.assert_allclose([v['cumulative_capture'] for v in rows],np.cumsum(energies)/energies.sum(),rtol=2e-14,atol=2e-15)
    for row in r['subspace_diagnostics']:
        if row['defined']:
            assert len(row['angles_degrees'])==row['top'] and all(0<=v<=90 for v in row['angles_degrees'])
