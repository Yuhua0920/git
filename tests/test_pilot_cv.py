"""T16E independent mathematical and leakage/causality sentinels."""
import inspect
import json
from pathlib import Path
import numpy as np
import pytest
from src.pilot_cv import (T16D_COMMIT,PilotCV,frozen_folds,cross_pilot_operators,
    causal_windows,deployment,recovery,primary_prediction,fitting_gate)
from src.cdl_benchmark import observation_geometry

ROOT=Path(__file__).resolve().parents[1]


def test_accepted_t16d_snapshot_and_all_frozen_hashes():
    import subprocess
    from scripts.run_cdl_mismatch import sha
    subprocess.check_call(['git','merge-base','--is-ancestor',T16D_COMMIT,'HEAD'],cwd=ROOT)
    assert subprocess.check_output(['git','show','-s','--format=%s',T16D_COMMIT],cwd=ROOT,text=True).strip()=="feat: add modular profile-mobility LMMSE correction composition gate"
    r=json.loads((ROOT/'outputs/t16d_modular_results.json').read_text(encoding='utf8'))
    assert r['all_checks_passed'] and r['completion_state']=='completed'
    for p,h in {**r['source_sha256'],**r['frozen_history_sha256'],**r['numeric_artifact_sha256']}.items():assert sha(ROOT/p)==h


def test_fold_geometry_partition_and_no_heldout_observed_overlap():
    g=observation_geometry();folds,d=frozen_folds(g.selected)
    assert frozen_folds(g.selected)[1]==d
    assert sorted(np.concatenate([h for _,h in folds]).tolist())==list(range(64))
    rx,tx,k,t=np.unravel_index(g.selected,(2,2,64,16))
    for observed,hold in folds:
        assert not set(observed)&set(hold) and len(observed)==48 and len(hold)==16
        for r in range(2):
            for a in range(2):
                ids=hold[(rx[hold]==r)&(tx[hold]==a)]
                assert len(ids)==len(set(k[ids]))==len(set(t[ids]))==4


def test_frozen_production_fold_and_operator_arrays_exact():
    from scripts.run_cdl_mismatch import sha
    from src.cdl_benchmark import array_hash
    report_path=ROOT/'outputs/t16e_pilot_cv_preflight.json'
    if not report_path.exists():pytest.skip('frozen T16E operator preflight not yet generated')
    r=json.loads(report_path.read_text(encoding='utf8'))
    folds,d=frozen_folds(observation_geometry().selected)
    assert d==r['pilot_fold_definition']
    assert json.loads((ROOT/'outputs/t16e_pilot_folds.json').read_text(encoding='utf8'))==d
    assert sha(ROOT/'outputs/t16e_pilot_folds.json')==r['fold_artifact_sha256']
    info=r['operator_artifact'];assert sha(ROOT/info['path'])==info['sha256']
    with np.load(ROOT/info['path'],allow_pickle=False) as f:
        for k,h in info['array_sha256'].items():assert array_hash(f[k])==h
        for s in (5,10,20):
            for arm in ('P','M'):
                np.testing.assert_array_equal(f[f'DeltaK{arm}_{s}'],f[f'K{arm}_{s}']-f[f'KSource_{s}'])
        assert f['source_mean_p'].shape==(64,) and f['source_mean_d'].shape==(3968,)
    assert r['generator_calls']==r['new_channel_realizations']==0 and r['source_means_only']


def test_frozen_cross_pilot_recomputed_regression():
    path=ROOT/'outputs/t16e_frozen_operators.npz'
    if not path.exists():pytest.skip('frozen T16E operators absent')
    from scripts.run_lowrank_filter import statistics
    _,source,development,variances=statistics()
    folds,_=frozen_folds(observation_geometry().selected)
    with np.load(path,allow_pickle=False) as f:
        for s in (5,10,20):
            for label,stats in (('Source',source),('P',development['CDL-B-30']),('M',development['CDL-C-120'])):
                operators,_=cross_pilot_operators(stats.rpp,variances[s],folds)
                np.testing.assert_array_equal(np.stack(operators),f[f'K{label}_{s}'])


def test_cross_pilot_K_rank_one_analytic_reference_without_inverse(monkeypatch):
    monkeypatch.setattr(np.linalg,'inv',lambda *a:pytest.fail('inverse forbidden'))
    monkeypatch.setattr(np.linalg,'pinv',lambda *a:pytest.fail('pinv forbidden'))
    a=(np.arange(64)+1)/64+1j*np.arange(64)[::-1]/128
    r=np.eye(64)+np.outer(a,a.conj());folds,_=frozen_folds(observation_geometry().selected)
    operators,info=cross_pilot_operators(r,.7,folds)
    for (observed,hold),k in zip(folds,operators):
        expected=np.outer(a[hold],a[observed].conj())/(1.7+np.vdot(a[observed],a[observed]).real)
        np.testing.assert_allclose(k,expected,atol=2e-16,rtol=1e-13)
    assert max(v['solve_relative_residual'] for v in info)<1e-13


def family():
    folds,_=frozen_folds(observation_geometry().selected)
    n=np.arange(16*48).reshape(16,48)
    base=tuple((.003*np.cos(n+f)+.002j*np.sin(n+f)) for f in range(4))
    profile=tuple(.002*np.sin(n/3+f)+.001j*np.cos(n/4+f) for f in range(4))
    mobility=tuple(.004*np.cos(n/5+f)+.003j*np.sin(n/7+f) for f in range(4))
    return PilotCV(folds,np.zeros(64,complex),base,profile,mobility)


@pytest.mark.parametrize('modules',[("M",),("P","M")])
def test_real_LS_independent_stacked_reference_and_brute_force(modules):
    cv=family();pilots=(np.arange(64)+1+1j*np.arange(64)[::-1])[None]/50
    coefficients,diag=cv.fit(pilots,modules=modules)
    designs=[];responses=[]
    for f,(observed,hold) in enumerate(cv.folds):
        z=pilots[0,observed]
        response=pilots[0,hold]-cv.source_k[f]@z
        columns=[cv.mobility_delta_k[f]@z] if len(modules)==1 else [cv.profile_delta_k[f]@z,cv.mobility_delta_k[f]@z]
        designs.append(np.column_stack(columns));responses.append(response)
    a=np.concatenate(designs);b=np.concatenate(responses)
    expected=np.linalg.lstsq(np.vstack([a.real,a.imag]),np.r_[b.real,b.imag],rcond=None)[0]
    np.testing.assert_allclose(coefficients,expected,rtol=1e-13,atol=1e-13)
    assert diag['fitted_cv_loss']==pytest.approx(float(np.sum(abs(a@coefficients-b)**2)),rel=1e-13)
    if len(modules)==1:
        grid=coefficients[0]+np.linspace(-1,1,201)
        losses=[np.sum(abs(a[:,0]*v-b)**2) for v in grid]
        assert grid[np.argmin(losses)]==coefficients[0]


def test_no_environment_or_oracle_features_in_fit_interfaces():
    for function in (PilotCV.fit,PilotCV.frame_objective,deployment):
        names=list(inspect.signature(function).parameters)
        assert not any('target' in v or 'oracle' in v or 'shift' in v or 'truth' in v for v in names)
    cv=family();z=np.linspace(0,1,64).astype(complex)[None]
    a=cv.fit(z,modules=('P','M'))[0]
    # Evaluation metadata cannot reach the estimator; same pilots/frozen models => same coefficients.
    np.testing.assert_array_equal(a,cv.fit(z.copy(),modules=('P','M'))[0])


def test_source_means_and_frozen_full_module_deployment():
    meanp=np.ones(64,complex)*(1+2j);meand=np.arange(3)+1j
    ws=np.ones((3,64),complex);dp=ws*(2-1j);dm=ws*(-1+3j)
    pilot=meanp+np.arange(64)
    for modules,coeff in ((('M',),np.array([-.5])),(('P','M'),np.array([2.,-.5]))):
        correction=coeff[0]*dm if len(coeff)==1 else coeff[0]*dp+coeff[1]*dm
        expected=meand+(pilot-meanp)@(ws+correction).T
        np.testing.assert_allclose(deployment(pilot,meanp,meand,ws,dp,dm,coeff,modules),expected)
        np.testing.assert_array_equal(deployment(meanp,meanp,meand,ws,dp,dm,coeff,modules),meand)


@pytest.mark.parametrize('window',[1,5])
def test_causal_windows_and_future_frame_perturbation(window):
    cv=family();x=np.arange(640).reshape(10,64).astype(complex)/640
    windows=causal_windows(x,window)
    assert [j for j,_ in windows]==list(range(window-1,10))
    for j,z in windows:
        changed=x.copy();changed[j+1:]+=1e6
        other=dict(causal_windows(changed,window))[j]
        np.testing.assert_array_equal(z,other)
        np.testing.assert_array_equal(cv.fit(z,modules=('P','M'))[0],cv.fit(other,modules=('P','M'))[0])


def test_unconstrained_extreme_coefficients_are_not_clipped():
    cv=family();z=(np.arange(64)+1).astype(complex)[None]
    tiny=PilotCV(cv.folds,cv.source_mean_p,cv.source_k,cv.profile_delta_k,tuple(a*1e-12 for a in cv.mobility_delta_k))
    a,d= tiny.fit(z,modules=('M',))
    expected=cv.fit(z,modules=('M',))[0]*1e12
    np.testing.assert_allclose(a,expected,rtol=1e-14)
    assert not d['clipped'] and not d['regularized']


def test_closure_efficiency_unclipped_and_zero_gap():
    r=recovery(-10,-20,-15,-8)
    assert r['closure']==-.2 and r['oracle_scalar_efficiency']==-.4
    assert recovery(-10,-10,-10,-8)['closure'] is None


def test_original_frame_energy_reduction_not_compensated_float_sum():
    import math
    from scripts.run_pilot_cv import trajectory_energy,checked_reference_energy
    energies=[1e16,1.,1.]
    assert trajectory_energy(energies)==1e16
    assert math.fsum(energies)==1e16+2.
    prior=dict(error_energy=1e16,truth_energy=3.,nmse_linear=1e16/3.)
    assert checked_reference_energy(energies,[1.,1.,1.],prior)==(1e16,3.)
    with pytest.raises(ValueError,match='not exact'):
        checked_reference_energy(energies,[1.,1.,1.],dict(prior,error_energy=1e16+2.))


def test_first_historical_trajectory_all_nine_reference_energies_exact():
    """Exercise the actual cached-input reader and energy gate before the scan."""
    from scripts.run_pilot_cv import frozen_operators,readcsv,checked_reference_energy
    from scripts.run_modular_correction import cached_H,cached_inputs,target_statistics
    from scripts.run_cdl_mismatch import load
    from src.cdl_benchmark import frames
    from src.link import estimate_ls_linear_channel
    old=load(ROOT/'outputs/t16d_modular_results.json');arrays,_=frozen_operators()
    info=old['validation_replay_gate']['cache']['CDL-C-75']
    h=cached_H(info,0);inputs=cached_inputs(info,500);g=observation_geometry()
    target=target_statistics(load(ROOT/'outputs/t16c_heldout_results.json'))['CDL-C-75']
    rows=readcsv(ROOT/'outputs/t16d_trajectory_metrics.csv')
    for si,s in enumerate((5,10,20)):
        matched=target.prepare(old['source_noise_variances'][str(s)])
        oracle=next(r for r in old['alpha_beta'] if r['shift']=='CDL-C-75' and r['snr_db']==s)
        errors=[[],[],[]];energy=[]
        for j,truth in enumerate(frames(h)):
            data=truth.ravel()[g.data];energy.append(float(np.sum(abs(data)**2)))
            sparse,_=estimate_ls_linear_channel(inputs['y'][j,si],g.layout)
            z=sparse.ravel()[g.selected]
            source=arrays['source_mean_d']+(z-arrays['source_mean_p'])@arrays[f'Wsource_{s}'].T
            x=z-target.mean_p
            scalar=target.mean_d+x@arrays[f'Wsource_{s}'].T+x@(oracle['beta']*arrays[f'DeltaWM_{s}']).T
            for e,prediction in zip(errors,(source,matched.estimate(z),scalar)):
                e.append(float(np.sum(abs(prediction-data)**2)))
        for e,method in zip(errors,('source','matched','scalar-real')):
            prior=next(r for r in rows if r['shift']=='CDL-C-75' and int(r['trajectory_id'])==500
                and int(r['snr_db'])==s and r['method']==method)
            error,total=checked_reference_energy(e,energy,prior)
            assert error==float(prior['error_energy']) and total==float(prior['truth_energy'])


def test_actual_results_when_available():
    path=ROOT/'outputs/t16e_pilot_cv_results.json'
    if not path.exists():pytest.skip('T16E actual validation not yet generated')
    r=json.loads(path.read_text(encoding='utf8'))
    assert not r['new_model_trained'] and not r['formal_test_used'] and r['formal_test_sealed']
    assert r['new_channel_realizations']==r['generator_calls']==r['BER_evaluations']==0
    assert r['historical_replay_gate_passed'] and r['source_means_only']
    assert not r['primary_estimator_uses_environment_label'] and r['primary_module_model']=='unified_two_real_coefficients'
    assert r['reporting_label_mutation_exact'] and r['label_assisted_one_module_diagnostic']
    assert not r['label_assisted_diagnostic_used_for_primary_gate']
    assert r['historical_replay_gate']['all_1800_reference_trajectory_rows_exact']
    assert r['historical_replay_gate']['all_18_reference_pooled_rows_exact']
    assert not r['prior_failed_attempt']['actual_PilotCV_started']
    assert len(r['validation_nmse'])==54 and len(r['actual_small_inference'])==6
    assert fitting_gate(r['validation_nmse'],False)=={k:r[k] for k in fitting_gate(r['validation_nmse'],False)}
    from scripts.run_pilot_cv import readcsv
    from scripts.run_cdl_mismatch import sha
    for p,h in {**r['source_sha256'],**r['frozen_history_sha256'],**r['numeric_artifact_sha256'],
        **r['prior_failed_attempt']['artifact_sha256']}.items():assert sha(ROOT/p)==h
    coeff=readcsv(ROOT/'outputs/t16e_pilot_cv_coefficients.csv');assert len(coeff)==9600
    assert sum(int(v['window'])==1 for v in coeff)==6000
    assert all(int(v['window_start'])==int(v['frame_id'])-int(v['window'])+1 for v in coeff)
    assert all(int(v['frame_id'])>=4 for v in coeff if int(v['window'])==5)
    tr=readcsv(ROOT/'outputs/t16e_pilot_cv_trajectory_metrics.csv');assert len(tr)==5400
    for row in r['validation_nmse']:
        selected=[x for x in tr if x['shift']==row['shift'] and int(x['snr_db'])==row['snr_db'] and x['method']==row['method'] and x['subset']==row['subset']]
        assert len(selected)==100
        assert sum(float(x['error_energy']) for x in selected)/sum(float(x['truth_energy']) for x in selected)==row['nmse_linear']
        assert all(int(x['frames'])==(10 if row['subset']=='all10' else 6) for x in selected)
    assert len(readcsv(ROOT/'outputs/t16e_label_assisted_coefficients.csv'))==4800


def test_reporting_label_mutation_cannot_change_primary_coefficients_predictions():
    from scripts.run_pilot_cv import fit_unlabelled,attach_reporting_label
    cv=family();n=np.arange(3*64).reshape(3,64)
    arrays={'source_mean_p':cv.source_mean_p,'source_mean_d':np.zeros(3,complex),
        'Wsource_10':np.sin(n),'DeltaWP_10':np.cos(n),'DeltaWM_10':np.sin(n/2),
        'KSource_10':np.stack(cv.source_k),'DeltaKP_10':np.stack(cv.profile_delta_k),
        'DeltaKM_10':np.stack(cv.mobility_delta_k)}
    x=np.arange(640).reshape(10,64).astype(complex)/640+1j*.3
    estimates,_,_=fit_unlabelled(x,arrays,10)
    a=attach_reporting_label(estimates,'CDL-C-75');b=attach_reporting_label(estimates,'CDL-B-120')
    assert len(a)==len(b)==16
    for u,v in zip(a,b):
        np.testing.assert_array_equal(u['coefficients'],v['coefficients'])
        np.testing.assert_array_equal(u['prediction'],v['prediction'])
        assert u['diagnostics']['real_ls_size']==v['diagnostics']['real_ls_size']==2
    assert 'label' not in inspect.signature(fit_unlabelled).parameters


def test_primary_disallows_oracle_target_inputs_and_always_two_modes(monkeypatch):
    cv=family();called=[];original=PilotCV.fit
    def checked(self,x,*,modules):
        called.append(modules);return original(self,x,modules=modules)
    monkeypatch.setattr(PilotCV,'fit',checked)
    z=np.linspace(0,1,64).astype(complex)[None]
    primary_prediction(cv,z,np.zeros(3),np.ones((3,64)),np.ones((3,64))*.5,np.ones((3,64))*2)
    assert called==[('P','M')]
    source=inspect.getsource(__import__('scripts.run_pilot_cv',fromlist=['fit_unlabelled']).fit_unlabelled)
    assert 'oracle' not in source.split('"""')[-1] and 'target' not in source.split('"""')[-1]


@pytest.mark.parametrize('values,expected',[
    ((.8,.9,1.,.8),'viable'),((.39,.2,.7,.7),'insufficient'),((.39,.4,.7,.7),'partial')])
def test_fitting_gate_fixed_boundaries_and_inherited_oracle_rule(values,expected):
    rows=[dict(shift=n,snr_db=s,method='pilotCV-5',oracle_scalar_efficiency=v,metric_type='actual_validation')
        for (n,s),v in zip(((n,s) for n in ('CDL-C-75','CDL-B-120') for s in (10,20)),values)]
    gate=fitting_gate(rows,False)
    assert gate[f'analytical_pilot_only_module_fitting_{expected}']
    assert gate['learning_based_coordinate_inference_may_be_justified']==(expected=='insufficient')
    assert not fitting_gate(rows,True)['learning_based_coordinate_inference_may_be_justified']
    diagnostic=[dict(shift='CDL-C-75',snr_db=10,method='label-assisted-M',oracle_scalar_efficiency=999,metric_type='diagnostic')]
    assert fitting_gate(rows+diagnostic,False)==gate


def test_frame_subset_aggregation_fairness_and_true_pooled_ratio():
    from scripts.run_pilot_cv import aggregate
    frames=[]
    for name in ('CDL-C-75','CDL-B-120'):
        for s in (5,10,20):
            for i in range(500,600):
                for j in range(10):
                    for method,scale in (('source',10.),('oracleScalar',3.),('matched',1.),('pilotCV-1',5.),('pilotCV-5',4.)):
                        if method=='pilotCV-5' and j<4:continue
                        frames.append(dict(shift=name,snr_db=s,trajectory_id=i,frame_id=j,method=method,
                            error_energy=scale*(j+1),truth_energy=float(i-499)))
    rows,tr=aggregate(frames)
    assert len(rows)==54 and len(tr)==5400
    for r in rows:
        assert r['frames']==(1000 if r['subset']=='all10' else 600)
    source_all=next(r for r in rows if r['method']=='source' and r['subset']=='all10')
    source_last=next(r for r in rows if r['method']=='source' and r['subset']=='last6')
    assert source_all['nmse_linear']!=source_last['nmse_linear']
    cv=next(r for r in rows if r['method']=='pilotCV-5')
    assert cv['closure']==pytest.approx(np.log10(10/4)/np.log10(10))
    assert cv['oracle_scalar_efficiency']==pytest.approx(np.log10(10/4)/np.log10(10/3))


def test_two_figures_preflight_QA_CSV_only_rebuild_no_global_style_drift(tmp_path):
    from scripts.plot_pilot_cv import audit
    from scripts.run_cdl_mismatch import csvsave
    import matplotlib
    before=dict(matplotlib.rcParams)
    rows=[dict(shift=n,snr_db=s,subset='last6',method=m,nmse_db=-s/2-offset,oracle_scalar_efficiency=eff)
        for n in ('CDL-C-75','CDL-B-120') for s in (5,10,20)
        for m,offset,eff in (('source',0,0),('pilotCV-1',1,.2),('pilotCV-5',2,.4),('oracleScalar',5,1),('matched',6,1.2))]
    csvsave(tmp_path/'t16e_pilot_cv_scan.csv',rows);audit(tmp_path,tmp_path/'figures')
    assert dict(matplotlib.rcParams)==before
