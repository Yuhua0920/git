"""T16G actual frozen-file/information-boundary/aggregation regression gates."""
import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
from src.cdl_benchmark import array_hash
from src.lowrank_modules import WeightedModule,RANKS,scientific_gate,compression_metrics,minimum_practical_rank
from src.pilot_cv import PilotCV,deployment
from scripts import run_lowrank_modules as runner
from scripts.run_cdl_mismatch import load,sha
from scripts.run_statistics_selection import frozen_candidates
from scripts.run_pilot_cv import readcsv


def test_accepted_t16f_snapshot_and_frozen_modules():
    old=runner.accepted();arrays,pre=frozen_candidates()
    assert old['pilot_fold_sha256']=='16594a306157e0b6ac114100ddebbba262f8e4205a98c8f48726e5519e55ab18'
    historical=load(runner.OUT/'t16d_modular_results.json')
    for module,name in runner.DEVELOPMENT.items():
        for snr in runner.SNRS:
            row=next(v for v in historical['module_provenance']['development_operator_regression'] if v['shift']==name and v['snr_db']==snr)
            assert array_hash(arrays[f'DeltaW{module}_{snr}'])==row['DeltaW_sha256']
    for row in old['candidate_provenance']:
        for label,key in (('Source','K0'),('P','KP'),('M','KM')):
            assert array_hash(arrays[f'K{label}_{row["snr_db"]}'])==row['K_sha256'][key]


def test_development_only_svd_inputs_and_construction_boundary(monkeypatch):
    arrays,_=frozen_candidates();_,_,development,variances=runner.statistics()
    original=WeightedModule.build;seen=[]
    def observe(delta,covariance):
        matches=[(m,s) for m,name in runner.DEVELOPMENT.items() for s in runner.SNRS
            if np.array_equal(delta,arrays[f'DeltaW{m}_{s}']) and
            np.array_equal(covariance,development[name].rpp+variances[s]*np.eye(64))]
        assert len(matches)==1
        seen.extend(matches)
        return original(delta,covariance)
    monkeypatch.setattr(WeightedModule,'build',observe)
    def forbidden(*args,**kwargs):raise AssertionError('held-out statistics entered construction')
    monkeypatch.setattr(runner,'target_statistics',forbidden)
    factors,proof,spectrum,capture,_=runner.construct_development_factors(arrays)
    assert set(seen)=={(m,s) for m in ('P','M') for s in (5,10,20)} and len(seen)==6
    assert len(proof)==6 and len(spectrum)==384 and len(capture)==30
    assert all(not v['heldout_statistics_used'] and v['r64_restoration_relative_residual']<=1e-11 for v in proof)
    assert all(factors[f'A_{m}_{s}'].shape==(3968,16) and factors[f'Q_{m}_{s}'].shape==(16,64) for m,s in seen)


def test_no_coordinate_fit_or_dense_inference_reconstruction_paths():
    tree=ast.parse(Path(runner.__file__).read_text(encoding='utf8'))
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)]
    names=[getattr(node.func,'id',getattr(node.func,'attr','')) for node in calls]
    assert not {'fit','fit_unlabelled','primary_prediction','inv','pinv','communication'}&set(names)
    assert 'target_statistics' not in inspect.getsource(runner.construct_development_factors)
    # Only the evaluation diagnostic is allowed to read target covariance.
    for name in ('construct_development_factors','historical_gate'):
        assert 'target_statistics(' not in inspect.getsource(getattr(runner,name))


def test_mixed_baseline_lowrank_csv_serialization_preflight(tmp_path):
    frames=[]
    for row in readcsv(runner.OUT/'t16f_frame_metrics.csv'):
        if (row['role'],row['shift'],row['snr_db'],row['window'])!=('primary','CDL-C-75','10','5'):continue
        value=dict(role='primary',shift=row['shift'],snr_db=10,window=5,method=row['method'],rank=0,
            trajectory_id=int(row['trajectory_id']),frame_id=int(row['frame_id']),
            error_energy=float(row['error_energy']),truth_energy=float(row['truth_energy']))
        frames.append(value)
        if row['method']=='affine':frames.append(dict(value,method='lr-affine',rank=1,error_energy=value['error_energy']*1.1))
    pooled,_=runner.aggregate_ranked(frames);runner.attach_compression_metrics(pooled)
    assert len(pooled)==6 and all(set(row)==set(pooled[0]) for row in pooled)
    target=tmp_path/'mixed.csv';runner.csvsave(target,pooled)
    restored=readcsv(target)
    assert len(restored)==6
    assert all(v['dense_recovery_db']=='' and v['lowrank_recovery_db']=='' for v in restored if v['rank']=='0')
    low=next(v for v in restored if v['rank']=='1')
    assert float(low['compression_penalty_db'])==next(v['compression_penalty_db'] for v in pooled if v['rank']==1)


@pytest.mark.parametrize('window',(1,5))
def test_frozen_coordinate_sample_regression_without_refitting(monkeypatch,window):
    arrays,_=frozen_candidates();old=load(runner.OUT/'t16f_results.json')
    info=old['historical_replay_gate']['cache']['CDL-C-75/500']
    assert sha(runner.ROOT/info['path'])==info['sha256']
    with np.load(runner.ROOT/info['path'],allow_pickle=False) as f:
        pilots=f['pilots'];coeff=f['affine_coefficients']
    def forbidden(*args,**kwargs):raise AssertionError('coordinate refit forbidden')
    monkeypatch.setattr(PilotCV,'fit',forbidden)
    weights=readcsv(runner.OUT/'t16f_weight_geometry.csv')
    for k,(w,j) in enumerate(runner.WINDOW_ORDER):
        if w!=window:continue
        prior=next(v for v in weights if (v['shift'],v['snr_db'],v['trajectory_id'],v['window'],v['frame_id'],v['method'])==
            ('CDL-C-75','10','500',str(window),str(j),'affine'))
        assert np.array_equal(coeff[1,k],[float(prior['alpha']),float(prior['beta'])])
        pred=deployment(pilots[1,j],arrays['source_mean_p'],arrays['source_mean_d'],arrays['Wsource_10'],
            arrays['DeltaWP_10'],arrays['DeltaWM_10'],coeff[1,k],('P','M'))
        assert pred.shape==(3968,) and np.isfinite(pred).all()


def test_persisted_factors_if_present(monkeypatch):
    if not (runner.OUT/'t16g_factor_preflight.json').exists():pytest.skip('T16G factor artifact not generated yet')
    family,pre=runner.frozen_factors()
    assert pre['development_only'] and not pre['heldout_statistics_read_before_factor_freeze']
    assert pre['r64_math_only'] and pre['coordinate_refits']==0
    for module in ('P','M'):
        for snr in (5,10,20):
            full=family[module,snr,16]
            for rank in RANKS:
                f=family[module,snr,rank]
                assert np.array_equal(f.A,full.A[:,:rank]) and np.array_equal(f.Q,full.Q[:rank])
                assert not f.A.flags.writeable and not f.Q.flags.writeable
    def forbidden(*args,**kwargs):raise AssertionError('evaluation must not refit SVD')
    monkeypatch.setattr(np.linalg,'svd',forbidden)
    arrays,_=frozen_candidates();diagnostic=runner.heldout_diagnostic(arrays,family,pre)
    assert len(diagnostic)==60 and all(v['diagnostic_only'] and not v['used_for_factor_construction'] for v in diagnostic)


def test_actual_results_frozen_regression_and_scientific_gate_if_present():
    if not runner.RESULT.exists():pytest.skip('T16G actual validation result not generated yet')
    r=load(runner.RESULT)
    for field in ('historical_replay_gate_passed','primary_source_means_only','formal_test_sealed','compression_gate'):
        assert r[field]
    for field in ('coefficient_refits','generator_calls','new_channel_realizations','new_seed_ids','BER_evaluations','detector_calls'):
        assert r[field]==0
    for field in ('pilot_coordinate_inference_changed','pilot_K_compressed','source_W_compressed',
        'dense_Br_reconstructed_during_inference','heldout_covariance_used_for_factor_construction','formal_test_used',
        'new_model_trained','primary_estimator_uses_environment_label','r64_full_validation_performed'):
        assert not r[field]
    gate=r['historical_replay_gate']
    assert gate['all_9600_T16F_coefficients_exact'] and gate['coefficients_loaded_not_refitted']
    assert gate['all_200_H_hashes_exact'] and gate['all_200_paired_digests_exact']
    assert gate['dense_prediction_digest_exact'] and gate['dense_frame_errors_exact']
    assert gate['all_6000_reference_trajectory_rows_exact'] and gate['all_60_reference_pooled_rows_exact']
    assert r['rank_grid']==[1,2,4,8,16]
    for p,h in {**r['source_sha256'],**r['frozen_history_sha256'],**r['numeric_artifact_sha256'],**r['preparation_artifact_sha256'],**r['preflight_schema_recovery']['artifact_sha256']}.items():
        assert sha(runner.ROOT/p)==h,p
    rows=r['validation_nmse'];assert len(rows)==120
    focus={rank:{(v['shift'],v['snr_db']):v for v in rows if v['rank']==rank and v['window']==5 and v['snr_db'] in (10,20)} for rank in RANKS}
    assert all(r[k]==v for k,v in scientific_gate(focus).items())
    for row in rows:
        if row['method']!='lr-affine':continue
        get=lambda method:next(v['nmse_db'] for v in rows if (v['shift'],v['snr_db'],v['window'],v['method'])==
            (row['shift'],row['snr_db'],row['window'],method))
        assert all(row[k]==v for k,v in compression_metrics(get('source'),get('affine'),get('convex'),row['nmse_db']).items())
        assert row['frames']==(1000 if row['window']==1 else 600)
    for minimum in r['minimum_practical_ranks']:
        selected={v['rank']:v['compression_penalty_db'] for v in rows if v['method']=='lr-affine' and
            (v['shift'],v['snr_db'],v['window'])==(minimum['shift'],minimum['snr_db'],minimum['window'])}
        assert all(minimum[k]==v for k,v in minimum_practical_rank(selected).items())
    trajectories=readcsv(runner.OUT/'t16g_trajectory_metrics.csv');assert len(trajectories)==12000
    for row in rows:
        selected=[v for v in trajectories if (v['shift'],int(v['snr_db']),int(v['window']),v['method'],int(v['rank']))==
            (row['shift'],row['snr_db'],row['window'],row['method'],row['rank'])]
        assert len(selected)==100 and {int(v['trajectory_id']) for v in selected}==set(range(500,600))
        error=sum(float(v['error_energy']) for v in selected);energy=sum(float(v['truth_energy']) for v in selected)
        assert error==row['error_energy'] and energy==row['truth_energy'] and error/energy==row['nmse_linear']
