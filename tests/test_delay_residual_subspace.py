"""T19D independent real geometry, joint fits, exclusion and frozen regression."""
import ast
import inspect
import json
import numpy as np
import pytest
from src import delay_residual_subspace as p
from src import operator_space_audit as old
from scripts import run_delay_residual_subspace as run


def test_real_augmentation_exact_C_order_and_roundtrip():
    z=np.array([[1+2j,3-4j],[-5+6j,7+8j]],np.complex128)
    assert np.array_equal(p.augment(z),[1,3,-5,7,2,-4,6,8])
    assert np.array_equal(p.unaugment(p.augment(z),z.shape),z)
    assert p.unaugment(p.augment(z),z.shape).dtype==np.complex128


@pytest.mark.parametrize('value',[np.zeros(4),np.array([1.,np.nan]),np.array([1+1j])])
def test_invalid_or_zero_normalization_stops(value):
    with pytest.raises(ValueError):p.unit(value)


def test_deterministic_sign_largest_entry_tie_uses_first():
    v=np.array([-.5,-2.,2.,1.]);a,s=p.sign_rule(v);b,t=p.sign_rule(-v)
    assert np.array_equal(a,b) and s['pivot_index']==t['pivot_index']==1
    assert a[1]==2 and s['sign_multiplier']==-1 and t['sign_multiplier']==1


def synthetic_residuals(scale=1.):
    z=np.array([[1+.4j,.3+.9j],[.6-.8j,1.2+.2j],[.5+.1j,-.7+.4j]])
    lower=np.array([[.9,0],[.2+.4j,1.1]],np.complex128)
    residuals={(n,r):(z if n==p.DEVELOPMENT[0] else scale*1j*z)*(1+r*.13) for n in p.DEVELOPMENT for r in p.REPLICATES}
    return residuals,lower


def test_equal_direction_PCA_two_real_directions_not_one_complex_phase():
    residuals,lower=synthetic_residuals();basis=p.learn(residuals,lower)
    assert basis['explained_energy_ratio'][:2]==pytest.approx([.5,.5],abs=1e-14)
    assert basis['cumulative_energy'][1]==pytest.approx(1.,abs=1e-14)
    assert not basis['centered'] and basis['vectors'].dtype==np.float64
    assert all(p.projection_capture(r,lower,basis['vectors'],2)==pytest.approx(1.,abs=1e-13) for r in residuals.values())
    assert np.allclose(basis['vectors']@basis['vectors'].T,np.eye(2),atol=1e-13)
    assert all(p.augment(d@lower)[s['pivot_index']]>0 for d,s in zip(basis['directions'],basis['sign_sentinels']))


def test_normalization_prevents_large_amplitude_dominance():
    a,l=synthetic_residuals();b,_=synthetic_residuals(scale=1e6)
    x=p.learn(a,l);y=p.learn(b,l)
    assert y['explained_energy_ratio']==pytest.approx(x['explained_energy_ratio'],abs=1e-14)
    raw=np.stack([p.augment(r@l) for r in b.values()]);singular=np.linalg.svd(raw,compute_uv=False)
    assert singular[0]**2/np.sum(singular**2)>.999999  # contrasts the mandated normalized construction


def test_triangular_recovery_complex_right_solve_not_conjugate():
    r,l=synthetic_residuals();z=r[p.DEVELOPMENT[0],1]
    assert np.allclose(p.recover(z@l,l),z,rtol=1e-13,atol=1e-13)
    assert p.calibration()['passed']


@pytest.mark.parametrize('k',p.REPLICATES)
def test_LORO_excludes_exact_two_delay_pair(k):
    residuals,l=synthetic_residuals();training={key:value for key,value in residuals.items() if key[1]!=k}
    basis=p.learn(training,l,excluded_pair=k)
    assert len(basis['construction_keys'])==8 and all(key[1]!=k for key in basis['construction_keys'])
    for n in p.DEVELOPMENT:assert p.projection_capture(residuals[n,k],l,basis['vectors'],2)==pytest.approx(1.,abs=1e-13)
    with pytest.raises(ValueError):p.learn(residuals,l,excluded_pair=k)


@pytest.mark.parametrize('name',p.HELDOUT)
def test_heldout_context_cannot_enter_basis(name,monkeypatch):
    residuals,l=synthetic_residuals();residuals[name,1]=residuals.pop((p.DEVELOPMENT[0],1))
    def forbidden(*args,**kwargs):raise AssertionError('SVD must not run for contaminated basis inputs')
    monkeypatch.setattr(np.linalg,'svd',forbidden)
    with pytest.raises(ValueError,match='pure-delay'):p.learn(residuals,l)


@pytest.mark.parametrize('q',[1,2])
def test_joint_real_LS_matches_independent_lstsq_and_is_not_sequential(q):
    k=np.arange(36,dtype=np.float64).reshape(12,3)
    mods=[np.sin(k*a)+1j*np.cos(k*b) for a,b in ((.17,.29),(.43,.61),(.79,.97),(1.13,1.31))]
    ws=np.zeros_like(mods[0]);wt=sum(c*m for c,m in zip((.3,-.2,.7,-.4),mods));l=np.diag([.4,.7,1.2])
    fit=p.joint_fit(ws,wt,l,mods[:2],mods[2:],q)
    design=np.stack([p.augment(m@l) for m in mods[:2+q]],axis=1);response=p.augment(wt@l)
    independent=np.linalg.lstsq(design,response,rcond=None)[0]
    assert fit['coefficients']==pytest.approx(independent,abs=1e-12)
    assert fit['excess']==pytest.approx(np.sum((response-design@independent)**2),rel=1e-12,abs=1e-12)
    baseline=p.joint_fit(ws,wt,l,mods[:2],mods[2:],0)
    assert not np.allclose(fit['coefficients'][:2],baseline['coefficients'])
    if q==2:assert fit['coefficients']==pytest.approx([.3,-.2,.7,-.4],abs=1e-12)


def test_four_column_rank_failure_does_not_regularize():
    columns=[np.ones((2,2)),np.ones((2,2)),np.ones((2,2)),np.ones((2,2))]
    with pytest.raises(ValueError,match='ill-conditioned'):p.four_real_ls(columns[0],columns)
    with pytest.raises(ValueError):p.four_real_ls(columns[0],columns[:3])


def gate_rows(c0=.64,c1=.72,c2=.78):
    return [dict(environment=n,snr_db=s,capture_q0=c0,capture_q1=c1,capture_q2=c2,
        increment_q1=c1-c0,increment_q2=c2-c0,marginal_q2=c2-c1) for n in p.HELDOUT for s in p.SNRS]


def replace(row,c0,c1,c2):
    row.update(capture_q0=c0,capture_q1=c1,capture_q2=c2,increment_q1=c1-c0,increment_q2=c2-c0,marginal_q2=c2-c1)


@pytest.mark.parametrize('count',[8,9])
def test_single_direction_9_of_12_boundary(count):
    rows=gate_rows(.70,.71,.72)
    for row in rows[:count]:replace(row,.5,.72,.74)
    flags=p.transfer_gate(rows)
    assert flags['single_delay_direction_transfer_supported']==(count==9)
    assert sum(flags[k] for k in ('single_delay_direction_transfer_supported','two_direction_delay_subspace_transfer_supported','delay_residual_subspace_transfer_weak','delay_residual_subspace_transfer_partial'))==1


@pytest.mark.parametrize('count',[5,6])
def test_two_direction_6_of_12_marginal_boundary(count):
    rows=gate_rows(.64,.75,.79)
    for row in rows[:count]:replace(row,.64,.72,.78)
    flags=p.transfer_gate(rows)
    assert flags['q2_promising_count']==12 and flags['q2_marginal_at_least_005_count']==count
    assert flags['two_direction_delay_subspace_transfer_supported']==(count==6)


def test_weak_partial_increment_formulas_and_no_clipping():
    rows=gate_rows(.70,.71,.72)
    assert p.transfer_gate(rows)['delay_residual_subspace_transfer_weak']
    for row in rows[:4]:replace(row,.64,.72,.78)
    assert p.transfer_gate(rows)['delay_residual_subspace_transfer_partial']
    rows[0]['increment_q2']=0
    with pytest.raises(ValueError,match='formula'):p.transfer_gate(rows)
    rows=gate_rows(.8,.79,.78)
    assert p.transfer_gate(rows)['delay_residual_subspace_transfer_weak']  # raw negative values preserved


def test_gate_rejects_development_missing_and_nonfinite_rows():
    rows=gate_rows()
    with pytest.raises(ValueError):p.transfer_gate(rows[:-1])
    rows[0]['environment']=p.DEVELOPMENT[0]
    with pytest.raises(ValueError):p.transfer_gate(rows)
    rows=gate_rows();rows[0]['capture_q1']=np.nan
    with pytest.raises(ValueError):p.transfer_gate(rows)


def test_runtime_guard_blocks_generators_statistics_modules_PilotCV_and_BER():
    from src import pilot_cv,link
    with run.guard():
        for callback in (lambda:run.OfficialCDL(),lambda:run.StatisticsCDL.generate(None,0,split='train_dev'),
                         lambda:run.StreamingStatistics(None,None),lambda:run.WeightedModule.build(None,None,None),
                         lambda:pilot_cv.PilotCV.fit(None,None,modules=('P','M')),lambda:link.detect_data_zf(None,None,None)):
            with pytest.raises(RuntimeError):callback()
        assert p.calibration()['passed']


def test_static_scope_no_generation_inverse_new_rank_or_receiver():
    for module in (p,run):
        tree=ast.parse(inspect.getsource(module));calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
        assert not any(x.endswith(('.generate','.inv','.pinv','.fit','.build','.detect_data_zf','.update_statistics')) for x in calls)
    tree=ast.parse(inspect.getsource(run.compute));calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert not any('family' in x or 'factored' in x for x in calls)
    assert run.math.real_ls is old.real_ls


def test_T19C_committed_and_historical_q0_D200_regression():
    with run.guard():
        c=run.accepted();a,source,dev,held,proof=run.inputs();replay=run.regression(a,source,dev,held)
    assert c['all_checks_passed'] and c['c11_claim_status']=='SUPPORTED_IN_C30_200_10_STATISTICS_REPLICATION'
    assert replay['T19C_development_conditions_exact']==20 and replay['T19B_heldout_conditions_exact']==12
    assert proof['no_large_stat_or_original_pure_delay_in_basis']


def test_basis_metadata_and_calibration_JSON_roundtrip():
    residuals,l=synthetic_residuals();basis=p.learn(residuals,l)
    metadata=dict(construction_keys=[list(key) for key in basis['construction_keys']],
        sign_sentinels=basis['sign_sentinels'],construction=basis['construction'],centered=basis['centered'])
    assert json.loads(json.dumps(metadata))==metadata
    assert json.loads(json.dumps(p.calibration()))==p.calibration()


def test_plot_schema_complete_and_rejects_development(tmp_path):
    from scripts import plot_delay_residual_subspace as plot
    spectrum=[];transfer=gate_rows()
    for s in p.SNRS:
        for k in range(1,11):spectrum.append(dict(snr_db=s,component=k,explained_energy_ratio=.1,cumulative_energy=k*.1))
    for name,value in zip(plot.DATA,(spectrum,transfer)):run.csvsave(tmp_path/name,value)
    assert [len(v) for v in plot.series(tmp_path)]==[20,12]
    transfer[0]['environment']=p.DEVELOPMENT[0];run.csvsave(tmp_path/plot.DATA[1],transfer)
    with pytest.raises(ValueError):plot.series(tmp_path)


def test_actual_frozen_residual_subspace_results():
    if not run.RESULT.exists():pytest.skip('T19D actual operator study missing; explicit preparation-only skip')
    from scripts.audit_delay_residual_subspace import verify
    r=run.load(run.RESULT);assert verify(r)
    assert len(r['heldout_transfer'])==12 and len(r['loro_stability'])==40
    assert r['generator_calls']==r['statistics_refits']==0 and not r['new_module_added'] and r['formal_test_sealed']
