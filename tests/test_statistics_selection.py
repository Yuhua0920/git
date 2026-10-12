"""T16F provenance, fair oracle and actual-result regressions."""
import inspect
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scripts import run_statistics_selection as run
from scripts.run_cdl_mismatch import load,sha
from src.cdl_benchmark import observation_geometry,array_hash
from src.pilot_cv import frozen_folds
from src.statistics_mixing import scientific_gate
ROOT=Path(__file__).resolve().parents[1]


def test_accepted_t16e_historical_snapshot_and_all_hashes():
    e=run.accepted()
    assert e['all_checks_passed'] and e['formal_test_sealed']
    assert e['pilot_fold_definition']['sha256']==run.FOLD_HASH


def test_frozen_candidate_reconstruction_and_original_folds_exact():
    candidates,proof,_=run.build_candidates();arrays,_=run.frozen_operators()
    _,definition=frozen_folds(observation_geometry().selected)
    assert definition['sha256']==run.FOLD_HASH
    for s in run.SNRS:
        for label in ('P','M'):
            np.testing.assert_array_equal(candidates[f'W{label}_{s}']-arrays[f'Wsource_{s}'],arrays[f'DeltaW{label}_{s}'])
    assert len(proof)==3 and all(r['candidate_filters_T16D_exact'] and r['cross_pilot_T16E_exact'] for r in proof)


def test_frozen_persisted_candidate_file_and_array_hashes():
    if not (ROOT/'outputs/t16f_preflight.json').exists():pytest.skip('T16F candidate preflight absent')
    arrays,p=run.frozen_candidates()
    assert p['pilot_fold_sha256']==run.FOLD_HASH
    assert sha(ROOT/'outputs/t16e_frozen_operators.npz')==p['original_T16E_operator_file_sha256']
    for key,value in p['candidate_artifact']['array_sha256'].items():assert array_hash(arrays[key])==value


def toy_arrays():
    n=np.arange(16*48).reshape(16,48)
    k=np.stack([.003*np.cos(n+f)+.002j*np.sin(n+f) for f in range(4)])
    dp=np.stack([.002*np.sin(n/3+f)+.001j*np.cos(n/4+f) for f in range(4)])
    dm=np.stack([.004*np.cos(n/5+f)+.003j*np.sin(n/7+f) for f in range(4)])
    w=np.arange(3*64).reshape(3,64)/1000+1j*.01
    return dict(source_mean_p=np.zeros(64,complex),source_mean_d=np.arange(3)+2j,
        Wsource_10=w,DeltaWP_10=w*.5,DeltaWM_10=w*2,WP_10=w*1.5,WM_10=w*3,
        KSource_10=k,DeltaKP_10=dp,DeltaKM_10=dm)


def test_primary_reporting_label_mutation_and_no_oracle_target_inputs():
    arrays=toy_arrays();pilots=np.arange(640).reshape(10,64)/640+1j*.3
    original,_=run.fit_primary(pilots,arrays,10)
    first=run.with_reporting_label(original,'CDL-C-75');second=run.with_reporting_label(original,'CDL-B-120')
    assert len(first)==32
    for a,b in zip(first,second):
        np.testing.assert_array_equal(a['coefficients'],b['coefficients'])
        np.testing.assert_array_equal(a['prediction'],b['prediction'])
    assert not any(key in inspect.signature(run.fit_primary).parameters for key in ('label','shift','target','truth','oracle','H'))


@pytest.mark.parametrize('window',[1,5])
def test_hard_convex_causality_same_current_frame_prediction(window):
    arrays=toy_arrays();x=np.arange(640).reshape(10,64)/640+1j*.3
    before,_=run.fit_primary(x,arrays,10)
    for j in (4,7):
        changed=x.copy();changed[j+1:]+=1e6
        after,_=run.fit_primary(changed,arrays,10)
        for method in ('hard','convex'):
            a=next(r for r in before if (r['method'],r['window'],r['frame_id'])==(method,window,j))
            b=next(r for r in after if (r['method'],r['window'],r['frame_id'])==(method,window,j))
            np.testing.assert_array_equal(a['coefficients'],b['coefficients'])
            np.testing.assert_array_equal(a['prediction'],b['prediction'])
            assert a['window_start']==j-window+1


@pytest.mark.parametrize('method',['hard-oracle','convex-oracle','affine-oracle'])
def test_every_oracle_data_evaluation_uses_source_means(method):
    arrays=toy_arrays();arrays['source_mean_p']=np.ones(64,complex)*(1+2j)
    coeff=np.array([1.,0.]) if method=='hard-oracle' else np.array([.2,.3])
    np.testing.assert_array_equal(run.oracle_prediction(arrays['source_mean_p'],arrays,10,coeff,method),arrays['source_mean_d'])
    pilot=arrays['source_mean_p']+np.arange(64)/100
    expected=arrays['source_mean_d']+(pilot-arrays['source_mean_p'])@(
        arrays['Wsource_10']+coeff[0]*arrays['DeltaWP_10']+coeff[1]*arrays['DeltaWM_10']).T
    np.testing.assert_allclose(run.oracle_prediction(pilot,arrays,10,coeff,method),expected,atol=1e-13,rtol=1e-13)
    assert 'target' not in inspect.signature(run.oracle_prediction).parameters


def test_fair_two_module_oracle_independent_synthetic_reference(monkeypatch):
    """Orthogonal complex modules: known global C75/B120 two-mode optimum."""
    dp=np.zeros((3,64),complex);dm=dp.copy();dp[:,0]=[1,1j,-1];dm[:,1]=[1j,2,-1j]
    source=np.zeros_like(dp);arrays={}
    for s in run.SNRS:arrays.update({f'Wsource_{s}':source,f'DeltaWP_{s}':dp,f'DeltaWM_{s}':dm})
    desired={'CDL-C-75':np.array([-.3,.7]),'CDL-B-120':np.array([1.,1.])}
    targets={name:SimpleNamespace(rpp=np.eye(64),prepare=lambda sigma2,c=c:SimpleNamespace(weight=c[0]*dp+c[1]*dm)) for name,c in desired.items()}
    old=dict(source_noise_variances={str(s):.1 for s in run.SNRS},alpha_beta=[dict(shift=n,snr_db=s,
        alpha=None if n=='CDL-C-75' else 1.,beta=.7 if n=='CDL-C-75' else 1.) for n in run.TARGETS for s in run.SNRS])
    monkeypatch.setattr(run,'load',lambda p:old)
    monkeypatch.setattr(run,'target_statistics',lambda old:targets)
    controls,proof,_=run.oracle_models(arrays)
    for name in run.TARGETS:
        for s in run.SNRS:
            np.testing.assert_allclose(controls[name,s,'affine-oracle'],desired[name],atol=1e-15,rtol=1e-15)
            affine=next(r for r in proof if (r['shift'],r['snr_db'],r['method'])==(name,s,'affine-oracle'))
            assert affine['weighted_filter_excess_energy_per_frame']<1e-27
            assert affine['source_means'] and affine['affine_real_coordinate_count']==2
            if name=='CDL-C-75':assert affine['weighted_filter_excess_energy_per_frame']<affine['historical_single_module_excess_energy']
            else:assert affine['b120_coefficient_recompute_max_abs']<1e-15


def test_actual_frozen_target_oracle_regression_and_historical_t16d_hash_unchanged():
    path=ROOT/'outputs/t16d_modular_results.json';before=sha(path)
    arrays,_=run.frozen_operators();controls,proof,_=run.oracle_models(arrays)
    assert sha(path)==before
    for name in run.TARGETS:
        for s in run.SNRS:
            row=next(r for r in proof if (r['shift'],r['snr_db'],r['method'])==(name,s,'affine-oracle'))
            if name=='CDL-C-75':
                assert row['weighted_filter_excess_energy_per_frame']<=row['historical_single_module_excess_energy']
            else:
                old=next(r for r in load(path)['alpha_beta'] if (r['shift'],r['snr_db'])==(name,s))
                np.testing.assert_array_equal(controls[name,s,'affine-oracle'],[old['alpha'],old['beta']])
                assert row['b120_coefficient_recompute_max_abs']<1e-12
            errors={p['method']:p['weighted_filter_excess_energy_per_frame'] for p in proof if (p['shift'],p['snr_db'])==(name,s)}
            assert np.isfinite(list(errors.values())).all()
            tolerance=1e-12*max(1.,errors['hard-oracle'])
            assert errors['affine-oracle']<=errors['convex-oracle']+tolerance
            assert errors['convex-oracle']<=errors['hard-oracle']+tolerance


def test_paired_frame_aggregation_legacy_reduction_and_gain_scope():
    rows=[]
    for window,ids in ((1,range(10)),(5,range(4,10))):
        for i in range(500,600):
            for j in ids:
                for method,scale in (('source',5.),('hard',3.),('convex',2.),('affine',1.),('matched',.5)):
                    rows.append(dict(role='primary',shift='CDL-C-75',snr_db=10,window=window,method=method,
                        trajectory_id=i,frame_id=j,error_energy=scale*(j+1),truth_energy=i-499.))
    pooled,tr=run.aggregate(rows)
    assert len(pooled)==10 and len(tr)==1000
    for row in pooled:assert row['frames']==(1000 if row['window']==1 else 600)
    changed=rows+[dict(rows[-1])]
    with pytest.raises(ValueError,match='subset'):run.aggregate(changed)


def test_actual_result_replay_and_all_artifacts_when_available():
    path=ROOT/'outputs/t16f_results.json'
    if not path.exists():pytest.skip('T16F actual validation not yet generated')
    r=load(path)
    assert r['historical_replay_gate_passed'] and r['primary_source_means_only']
    assert r['oracle_controls_use_source_means'] and r['affine_oracle_matches_deployed_model_class']
    assert r['c75_affine_oracle_recomputed_as_two_module_target_weighted_optimum']
    assert r['b120_affine_oracle_reuses_t16d_coefficients'] and r['historical_t16d_c75_single_module_oracle_preserved']
    assert not r['primary_estimator_uses_environment_label'] and r['reporting_label_mutation_exact']
    assert r['generator_calls']==r['new_channel_realizations']==r['new_seed_ids']==r['BER_evaluations']==r['detector_calls']==0
    assert not r['new_model_trained'] and not r['lowrank_compression'] and not r['formal_test_used'] and r['formal_test_sealed']
    for p,h in {**r['source_sha256'],**r['frozen_history_sha256'],**r['numeric_artifact_sha256'],
        **r['preflight_failure_recovery']['artifact_sha256']}.items():assert sha(ROOT/p)==h
    gate=r['historical_replay_gate']
    assert gate['all_9600_affine_coefficients_and_diagnostics_exact'] and gate['affine_prediction_digest_exact']
    assert gate['all_3600_reference_trajectory_rows_exact'] and gate['all_36_reference_pooled_rows_exact']
    assert len(r['validation_nmse'])==60 and len(r['oracle_controls'])==36
    raw=run.readcsv(ROOT/'outputs/t16f_weight_geometry.csv');assert len(raw)==28800
    for row in raw:
        assert float(row['lambda0'])==1-float(row['alpha'])-float(row['beta'])
        if row['method']=='convex':assert min(float(row[k]) for k in ('alpha','beta','lambda0'))>=0
        assert int(row['window_start'])==int(row['frame_id'])-int(row['window'])+1
    trajectory=run.readcsv(ROOT/'outputs/t16f_trajectory_metrics.csv');assert len(trajectory)==9600
    for row in r['validation_nmse']+r['oracle_controls']:
        selected=[x for x in trajectory if (x['role'],x['shift'],int(x['snr_db']),int(x['window']),x['method'])==
            (row['role'],row['shift'],row['snr_db'],row['window'],row['method'])]
        assert len(selected)==100
        assert sum(float(x['error_energy']) for x in selected)/sum(float(x['truth_energy']) for x in selected)==row['nmse_linear']
    gains={(v['shift'],v['snr_db']):{k:v[k] for k in ('affine_vs_hard_gain_db','affine_vs_convex_gain_db')}
        for v in r['validation_nmse'] if v['method']=='affine' and v['window']==5 and v['snr_db'] in (10,20)}
    outside={(v['shift'],v['snr_db']):v['outside_any_fraction'] for v in r['outside_simplex_fraction'] if v['window']==5}
    assert scientific_gate(gains,outside)=={k:r[k] for k in scientific_gate(gains,outside)}
