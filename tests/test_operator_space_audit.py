"""T19B independent exact identities, immutable provenance and scope gates."""
import ast
import inspect
import numpy as np
import pytest
from scipy.linalg import cho_factor, cho_solve
from src import operator_space_audit as p
from src.cdl_benchmark import OfficialCDL, StreamingStatistics
from src.lowrank_modules import WeightedModule
from scripts import run_operator_space_audit as run
from scripts import plot_operator_space_audit as plot


def matrices():
    k=np.arange(1,25).reshape(4,6)
    cols=[np.sin(k*j+.3)+1j*np.cos(k*(.31+j)) for j in (1,2,4)]
    source=cols[0]*.17;lower=np.diag(np.arange(1,7)*.3)
    return source,lower,cols


def test_complete_quadratic_and_centered_mean_sentinel():
    result=p.calibration()
    assert result['passed'] and result['maximum_absolute_error']<1e-11
    assert result['full_Rdd_available_in_synthetic_sentinel_only'] and not result['random_sampling']


@pytest.mark.parametrize('count',[2,3])
def test_joint_real_LS_known_complex_coefficients(count):
    source,lower,cols=matrices();coeff=np.array([.37,-.22,.63])[:count]
    target=source+sum(c*m for c,m in zip(coeff,cols))
    fit=p.operator_fit(source,target,lower,cols[:count])
    assert fit['coefficients']==pytest.approx(coeff,rel=1e-12,abs=1e-12)
    assert fit['excess']<1e-24 and fit['normal_relative_residual']<1e-12


def test_dependent_columns_stop_without_regularization():
    source,lower,cols=matrices()
    with pytest.raises(ValueError,match='ill-conditioned'):
        p.operator_fit(source,source+cols[0],lower,[cols[0],cols[0]*2])


def test_nonfinite_input_rejected():
    with pytest.raises(ValueError,match='finite'):
        p.energy([complex(float('nan'),0)])


def test_operator_LS_is_real_not_complex_fit():
    source,lower,cols=matrices();target=source+1j*cols[0]+cols[1]*.23
    fit=p.operator_fit(source,target,lower,cols[:2])
    c=fit['coefficients'];res=(target-source-sum(v*m for v,m in zip(c,cols)))@lower
    assert c.dtype==np.float64 and all(abs(np.vdot(m@lower,res).real)<1e-10 for m in cols[:2])
    assert fit['excess']>0


def test_mean_aware_stacked_LS_sign_and_joint_optimum():
    source,lower,cols=matrices();target=source+cols[0]*.8-cols[1]*.4
    dp=np.arange(1,7)*(.1+.3j);dd=np.arange(4)*(.2-.1j)
    fit=p.mean_aware_fit(source,target,lower,cols[:2],dp,dd)
    c=fit['coefficients'];w=source+sum(v*m for v,m in zip(c,cols))
    budget=p.mismatch(w,target,lower,dp,dd)
    assert budget['E_total']==pytest.approx(fit['excess'],rel=1e-12)
    # Independent derivative of operator and mean response, both signs checked.
    er=(w-target)@lower;bias=p.mean_bias(w,dp,dd)
    for m in cols[:2]:
        derivative=2*(np.vdot(m@lower,er).real-np.vdot(m@dp,bias).real)
        assert abs(derivative)<1e-10
    op=p.operator_fit(source,target,lower,cols[:2]);opw=source+sum(v*m for v,m in zip(op['coefficients'],cols))
    assert budget['E_total']<=p.mismatch(opw,target,lower,dp,dd)['E_total']


def test_matched_operator_source_mean_floor():
    source,lower,cols=matrices();dp=np.ones(6)*(1+.3j);dd=np.arange(4)*.2
    b=p.mismatch(source,source,lower,dp,dd)
    assert b['E_op']==0 and b['E_total']==b['E_mean']==pytest.approx(np.linalg.norm(dd-source@dp)**2)


def test_rank4_audit_reconstruction_leaves_factors_untouched():
    source,lower,cols=matrices()
    a=np.arange(16).reshape(4,4)*.1+np.eye(4)*1j
    q=np.arange(24).reshape(4,6)*.04+np.sin(np.arange(24).reshape(4,6))*1j
    a0=a.copy();q0=q.copy();modules=[a@q,cols[1]]
    fit=p.operator_fit(source,source+modules[0]*.4+modules[1]*.7,lower,modules)
    assert fit['coefficients']==pytest.approx([.4,.7],abs=1e-12)
    assert np.array_equal(a,a0) and np.array_equal(q,q0)


def test_saved_window_excess_orthogonality_and_efficiency_not_clipped():
    source,lower,cols=matrices();target=source+cols[0]*.23+cols[1]*.16+cols[2]*.21
    fit=p.operator_fit(source,target,lower,cols[:2]);coeff=np.array([2.,-3.])
    w=source+sum(c*m for c,m in zip(coeff,cols))
    ex=p.deployed_excess(coeff,fit['coefficients'],fit['gram'],fit['excess'])
    assert ex==pytest.approx(p.operator_excess(w,target,lower),rel=1e-12)
    source_ex=p.operator_excess(source,target,lower)
    assert p.deployed_efficiency(source_ex,ex,fit['excess'])==(source_ex-ex)/(source_ex-fit['excess'])
    assert p.deployed_efficiency(source_ex,ex,fit['excess'])<0
    assert p.deployed_efficiency(10,-1,2)>1


@pytest.mark.parametrize('denominator',[0.,-1.,float('nan'),float('inf')])
def test_invalid_ratio_denominator_explicit_null(denominator):
    assert p.ratio(1.,denominator) is None


@pytest.mark.parametrize('sign',[1.,-1.])
def test_common_source_weighted_signed_cosine(sign):
    source,lower,cols=matrices();d=cols[0];r=sign*d
    assert p.signed_cosine(d,r,lower)==pytest.approx(sign,abs=1e-12)
    other=cols[1];a=d@lower;b=other@lower
    expected=np.vdot(a,b).real/(np.linalg.norm(a)*np.linalg.norm(b))
    assert p.signed_cosine(d,other,lower)==pytest.approx(expected,abs=1e-12)
    assert p.signed_cosine(d,np.zeros_like(d),lower) is None


def test_three_variable_fit_is_joint_not_sequential():
    source,lower,cols=matrices();target=source+cols[0]*.3+cols[1]*.4+cols[2]*.7
    two=p.operator_fit(source,target,lower,cols[:2]);joint=p.operator_fit(source,target,lower,cols)
    assert joint['coefficients']==pytest.approx([.3,.4,.7],abs=1e-12)
    assert not np.allclose(joint['coefficients'][:2],two['coefficients'])
    assert joint['excess']<two['excess']


def dense_rows(value):
    return [dict(environment=n,snr_db=s,operator_span_capture_dense=value) for n in p.ENVIRONMENTS for s in p.SNRS]


@pytest.mark.parametrize('value,sufficient,deficient',[(.70,True,False),(.50,False,False),(.499,False,True)])
def test_dense_gate_exact_boundaries(value,sufficient,deficient):
    r=p.dense_gate(dense_rows(value))
    assert r['dense_pm_operator_space_delay_sufficient']==sufficient
    assert r['dense_pm_operator_space_delay_deficient']==deficient
    assert r['dense_pm_operator_space_delay_partial']==(not sufficient and not deficient)


def test_condition_specific_deficiency_and_combined_conditions_excluded():
    rows=dense_rows(.8);rows[0]['operator_span_capture_dense']=.49
    for r in rows[4:]:r['operator_span_capture_dense']=-2
    g=p.dense_gate(rows)
    assert g['condition_specific_operator_deficiency'] and len(g['deficient_conditions'])==1
    assert g['dense_pm_operator_space_delay_partial'] and not g['dense_pm_operator_space_delay_deficient']
    rows[0]['operator_span_capture_dense']=None
    assert p.dense_gate(rows)['dense_pm_operator_space_delay_partial']
    with pytest.raises(ValueError):p.dense_gate(rows[:-1])


def reuse_rows(capture=.7,gain=.1):
    return [dict(environment=n,snr_db=s,operator_span_capture_3d=capture,delay_direction_incremental_capture=gain) for n in p.HELDOUT for s in p.SNRS]


@pytest.mark.parametrize('capture,gain,state',[(.7,.1,'SUPPORTED_AS_ORACLE_DIRECTION_DIAGNOSTIC'),(.7,.049,'CONTRADICTED_BY_DIRECTION_REUSE_TEST'),(.7,.05,'INCONCLUSIVE'),(.699,.1,'INCONCLUSIVE')])
def test_reuse_gate_frozen_boundaries(capture,gain,state):
    assert p.reuse_gate(reuse_rows(capture,gain))['c10_claim_status']==state


def test_reuse_six_of_eight_and_secondary_exclusion():
    rows=reuse_rows();primary=[r for r in rows if r['environment'] in p.PRIMARY]
    for r in rows:
        if r['environment'] not in p.PRIMARY:r.update(operator_span_capture_3d=-10,delay_direction_incremental_capture=-2)
    primary[0]['delay_direction_incremental_capture']=.09;primary[1]['delay_direction_incremental_capture']=.09
    assert p.reuse_gate(rows)['delay_residual_direction_reusable']
    primary[2]['delay_direction_incremental_capture']=.09
    assert p.reuse_gate(rows)['delay_direction_reuse_partial']
    with pytest.raises(ValueError):p.reuse_gate(rows+[dict(environment=p.DEVELOPMENT,snr_db=10)])


def test_runtime_forbids_generator_refit_and_basis_reconstruction():
    with run.no_generation():
        for function in (lambda:OfficialCDL(),lambda:StreamingStatistics(None,None),lambda:WeightedModule.build(None,None,None),lambda:np.linalg.svd(np.eye(2))):
            with pytest.raises(RuntimeError):function()
        assert p.calibration()['passed']


def test_no_inverse_optimizer_receiver_or_new_direction_deployment_calls():
    for module in (p,run):
        tree=ast.parse(inspect.getsource(module));calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
        assert not any(x.endswith(('.inv','.pinv','.svd','.generate','.freeze')) or 'optimize' in x or 'deployment' in x or 'detect' in x for x in calls)
        assert not any(x in ('statistics.update','stats.update','StreamingStatistics.update') for x in calls)
    scientific=ast.parse(inspect.getsource(run.scientific));assert not any('communication' in ast.unparse(n.func) for n in ast.walk(scientific) if isinstance(n,ast.Call))


def test_T19A_committed_hashes_and_actual_mean_policy():
    with run.no_generation():
        r=run.accepted();e=run.mean_policy(r)
    assert e['policy']=='source_means' and e['prediction_metadata_rows']==16
    assert r['c9_claim_status']=='INCONCLUSIVE' and r['rank4_out_of_factor_warning']


def test_no_target_mean_or_label_guesses_in_provenance(monkeypatch):
    r=run.load(run.OUT/'t19a_results.json');r['source_means_for_all_nonmatched']=False
    with pytest.raises(ValueError):run.mean_policy(r)


def test_figure_complete_population_and_budget_controls(tmp_path):
    tables=[[] for _ in plot.DATA]
    for n in p.ENVIRONMENTS:
        for s in p.SNRS:
            for table in tables[:3]:table.append(dict(environment=n,snr_db=s))
            if n in p.HELDOUT:
                for table in tables[3:]:table.append(dict(environment=n,snr_db=s))
    for name,table in zip(plot.DATA,tables):run.csvsave(tmp_path/name,table)
    assert [len(i) for i in plot.series(tmp_path)]==[16,16,16,14,14]
    run.csvsave(tmp_path/plot.DATA[-1],tables[-1][:-1])
    with pytest.raises(ValueError):plot.series(tmp_path)


def test_actual_exact_matrix_results_regression():
    if not run.RESULT.exists():pytest.skip('T19B result not generated; preparation-only explicit skip')
    from scripts.audit_operator_space_audit import verify
    r=run.load(run.RESULT)
    assert verify(r)
    assert r['generator_calls']==r['statistics_refits']==0
    assert not r['new_channel_generated'] and not r['new_module_added'] and r['formal_test_sealed']
    assert len(r['operator_excess'])==16 and len(r['operator_span_capture_3d'])==14
