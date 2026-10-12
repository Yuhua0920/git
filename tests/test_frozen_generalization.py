"""T18A preflight boundaries, unchanged receiver and actual-data gates."""
import ast
import inspect
import json
from pathlib import Path
import numpy as np
import pytest
from src import frozen_affine_validation as p
from src.cdl_benchmark import array_hash,StreamingStatistics,OfficialCDL,FrozenStatistics
from src.lowrank_modules import WeightedModule
from src.modular_correction import ScalarComposition
from src.statistics_mixing import geometry,objective_moments,hard_selection,convex_mixing
from src.pilot_cv import frozen_folds,deployment
from scripts import run_frozen_generalization as run
from scripts import plot_frozen_generalization as plot
from scripts.run_pilot_cv import cv_family,readcsv


@pytest.fixture(scope='module')
def frozen():return run.frozen_method()


def test_T17B_committed_accepted_and_negative_claims():
    old=run.accepted()
    assert old['c6_claim_status']==p.C6 and old['c7_claim_status']==p.C7


def test_exact_operator_factor_fold_loading_without_refit(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('SVD/refit/generator prohibited during frozen method load')
    monkeypatch.setattr(WeightedModule,'build',forbidden)
    monkeypatch.setattr(OfficialCDL,'__init__',forbidden)
    monkeypatch.setattr(np.linalg,'svd',forbidden)
    arrays,family,proof=run.frozen_method()
    assert proof['fold']['sha256']=='16594a306157e0b6ac114100ddebbba262f8e4205a98c8f48726e5519e55ab18'
    assert len(proof['rank4_array_sha256'])==12
    for m in ('P','M'):
        for s in p.SNRS:
            assert family[m,s,4].rank==4
            assert np.array_equal(family[m,s,4].A,family[m,s,16].A[:,:4])
            assert np.array_equal(family[m,s,4].Q,family[m,s,16].Q[:4])
            assert not family[m,s,4].A.flags.writeable


def test_namespace_manifest_uniqueness_and_old_tuple_separation():
    rows=p.manifest_rows();proof=run.namespace_audit(rows)
    assert len(rows)==36000 and proof['train_validation_disjoint']
    assert len({r['namespace'] for r in rows})==6
    consumed=set()
    for name in p.ENVIRONMENTS:
        for split,ids in (('train_dev',range(500)),('validation_dev',range(500,600))):
            for i in ids:
                first=p.seeds(name,split,i)
                for j in range(10):
                    s=p.seeds(name,split,i,j)
                    assert all(s[k]==first[k] for k in p.ROLES[:3])
                    if j==0:
                        for key in p.ROLES[:3]:assert s[key] not in consumed;consumed.add(s[key])
                    for key in p.ROLES[3:]:assert s[key] not in consumed;consumed.add(s[key])
    assert all(0<=v<2**32 for v in consumed)


@pytest.mark.parametrize('args', [('CDL-A-60','train_dev',0,0),('CDL-C-60','train_dev',500,0),
    ('CDL-C-60','validation_dev',0,0),('CDL-C-60','formal_test',600,0),('CDL-C-60','train_dev',0,10)])
def test_reject_unapproved_namespace(args):
    with pytest.raises(ValueError):p.seeds(*args)


@pytest.mark.parametrize('name',p.ENVIRONMENTS)
def test_geometry_power_and_symbol_abstraction_fixed(name):
    d=p.definition(name)
    assert d['system']['nominal_delay_spread_seconds']==100e-9
    assert d['system']['sample_rate_hz']==d['system']['nfft']*d['system']['delta_f_hz']
    assert not d['ofdm_channel_normalize'] and not d['within_symbol_ICI']
    assert d['dF']==16 and d['spacing_time']==4
    assert d['array']['ut_orientation_radians']==[np.pi,0.,0.]


@pytest.mark.parametrize('s',p.SNRS)
def test_frozen_source_noise_variance(s):
    assert p.VARIANCES[s]==p.POWER/10**(s/10)
    assert p.VARIANCES[s]==run.load(run.OUT/'t16d_modular_results.json')['source_noise_variances'][str(s)]


def test_train_update_rejects_validation_before_mutation():
    g=run.observation_geometry();stream=StreamingStatistics(g.selected,g.data)
    with pytest.raises(ValueError):stream.update(np.zeros((10,4096),complex),[500])
    assert stream.n==0 and stream.ids==[]


def pilots_fixture(frozen):
    arrays,family,_=frozen
    info=run.load(run.OUT/'t16f_results.json')['historical_replay_gate']['cache']['CDL-C-75/500']
    with np.load(run.ROOT/info['path'],allow_pickle=False) as f:pilots=f['pilots'][1].copy()
    return arrays,family,pilots


def test_primary_signature_and_target_isolation(frozen,monkeypatch):
    assert set(inspect.signature(p.deployed).parameters)=={'pilots','arrays','snr','cv','profile_factor','mobility_factor'}
    def forbidden(*a,**k):raise AssertionError('target/oracle cannot enter deployment')
    monkeypatch.setattr(FrozenStatistics,'prepare',forbidden)
    monkeypatch.setattr(p,'oracle_coordinates',forbidden)
    arrays,family,pilots=pilots_fixture(frozen)
    estimates=p.deployed(pilots,arrays,10,cv_family(arrays,10),family['P',10,4],family['M',10,4])
    for r in estimates:
        assert r['dense_coefficients'] is r['lr4_coefficients']
        assert r['coefficients']['affine'] is r['dense_coefficients']
        assert set(r['predictions'])=={'source','hard','convex','dense-affine','lr4-affine'}
    reports=[dict(environment='wrong label',**r) for r in estimates]
    assert all(np.array_equal(a['dense_coefficients'],b['dense_coefficients']) and
        np.array_equal(a['predictions']['lr4-affine'],b['predictions']['lr4-affine']) for a,b in zip(estimates,reports))


def test_original_three_analytical_fits_exact_and_causal(frozen):
    arrays,family,pilots=pilots_fixture(frozen);cv=cv_family(arrays,10)
    estimates=p.deployed(pilots,arrays,10,cv,family['P',10,4],family['M',10,4])
    assert [r['frame_id'] for r in estimates]==list(range(4,10))
    for r in estimates:
        j=r['frame_id'];window=pilots[j-4:j+1];moments=objective_moments(cv,window)
        assert np.array_equal(r['coefficients']['hard'],hard_selection(*moments)[0])
        assert np.array_equal(r['coefficients']['convex'],convex_mixing(*moments)[0])
        assert np.array_equal(r['coefficients']['affine'],cv.fit(window,modules=('P','M'))[0])
    changed=pilots.copy();changed[5:]*=7
    alt=p.deployed(changed,arrays,10,cv,family['P',10,4],family['M',10,4])
    assert np.array_equal(estimates[0]['predictions']['dense-affine'],alt[0]['predictions']['dense-affine'])


def test_rank_change_fails(frozen):
    arrays,family,pilots=pilots_fixture(frozen)
    with pytest.raises(ValueError):p.deployed(pilots,arrays,10,cv_family(arrays,10),family['P',10,8],family['M',10,4])


def test_C75_B120_existing_receiver_frame_sentinels_exact(frozen):
    arrays,family,_=frozen
    assert run.historical_receiver_regression(arrays,family)['checked_windows']==36


def test_two_module_oracle_same_T16F_objective_without_SVD(frozen,monkeypatch):
    from scripts.run_modular_correction import target_statistics
    arrays,_,_=frozen
    target=target_statistics(run.load(run.OUT/'t16c_heldout_results.json'))['CDL-C-75']
    def forbidden(*a,**k):raise AssertionError('no SVD even in oracle rank audit')
    monkeypatch.setattr(np.linalg,'svd',forbidden)
    coefficients,proof=p.oracle_coordinates(target,arrays,10)
    saved=next(r for r in run.load(run.OUT/'t16f_results.json')['oracle_coefficient_optimization']
        if (r['shift'],r['snr_db'],r['method'])==('CDL-C-75',10,'affine-oracle'))
    np.testing.assert_allclose(coefficients,[saved['alpha'],saved['beta']],rtol=1e-12,atol=1e-12)
    assert proof['source_means_used_for_actual_evaluation'] and not proof['deployable']


def test_small_paired_link_and_reference_evaluation_before_main(frozen):
    arrays,family,_=frozen;source,_=run.source_statistics()
    matched={s:source.prepare(p.VARIANCES[s]) for s in p.SNRS}
    oracle={s:np.zeros(2) for s in p.SNRS}
    # Algebraic synthetic H sentinel, not a generated scientific environment.
    h=np.ones((2,2,64,160),np.complex128);h.flags.writeable=False
    before=array_hash(h)
    tr,raw,fr,pair,signal,noise,samples=run.evaluate('CDL-C-60',500,h,arrays,family,matched,oracle,run.observation_geometry())
    assert array_hash(h)==before and len(tr)==21 and len(raw)==54 and len(fr)==126 and len(pair)==30
    assert samples==6*992 and np.all(signal>0) and np.all(noise>0)
    for j in range(10):
        group=[r for r in pair if r['frame_id']==j]
        assert len({r['bits_sha256'] for r in group})==len({r['unit_noise_sha256'] for r in group})==1
        assert len({r['Y_sha256'] for r in group})==3
    for s in p.SNRS:
        for j in range(4,10):
            errors={r['method']:r['error_energy'] for r in fr if (r['snr_db'],r['frame_id'])==(s,j)}
            assert errors['source']==errors['matched']==errors['oracle-affine']


def test_recovery_compression_signed_db_formulas():
    m=p.metrics(-1.,-11.,-9.,-9.2,-7.,-11.)
    assert m['dense_affine_closure']==.8 and m['pilotcv_efficiency']==.8
    assert m['compression_penalty_db']==pytest.approx(-.2)
    assert m['lr4_vs_convex_gain_db']==pytest.approx(2.2)
    assert p.metrics(0.,0.,-1.,-1.,0.,0.)['pilotcv_efficiency'] is None
    assert p.metrics(0.,0.,-1.,-1.,0.,0.)['dense_affine_closure'] is None


@pytest.mark.parametrize('coeff,outside', [([0,1],False),([.5,.5],False),([-.001,0],True),([0,-.001],True),([.5,.50001],True)])
def test_strict_outside_simplex(coeff,outside):assert geometry(coeff)['outside_any']==outside


def gate_fixture():
    return [dict(environment=n,snr_db=s,pilotcv_efficiency=.8,compression_penalty_db=.5,dense_affine_closure=.7,
        lr4_vs_convex_gain_db=1.,lr4_vs_source_gain_db=1.,outside_simplex_fraction=.75) for n in p.ENVIRONMENTS for s in p.SNRS]


@pytest.mark.parametrize('field,value', [('pilotcv_efficiency',.7999),('compression_penalty_db',.50001),('dense_affine_closure',.6999)])
def test_interpolation_all_eight_frozen_thresholds(field,value):
    rows=gate_fixture();assert p.scientific_gate(rows)['frozen_pipeline_interpolation_supported']
    next(r for r in rows if (r['environment'],r['snr_db'])==('CDL-C-90',20))[field]=value
    assert not p.scientific_gate(rows)['frozen_pipeline_interpolation_supported']


def test_composed_and_stress_gates_literal_no_5db_leak():
    rows=gate_fixture()
    for r in rows:
        if r['snr_db']==5:r.update(pilotcv_efficiency=-20.,compression_penalty_db=100.,lr4_vs_source_gain_db=-10.,lr4_vs_convex_gain_db=-10.)
    gate=p.scientific_gate(rows)
    assert gate['frozen_pipeline_interpolation_supported'] and gate['affine_composition_beyond_convex_hull_replicated']
    assert gate['outside_convex_hull_geometry_replicated'] and gate['extrapolation_stress_promising']
    stress=next(r for r in rows if (r['environment'],r['snr_db'])==('CDL-B-180',20));stress['lr4_vs_source_gain_db']=-.01
    assert p.scientific_gate(rows)['extrapolation_failure_observed']
    assert not p.scientific_gate(rows)['extrapolation_stress_promising']
    for r in rows:
        if r['environment'] in ('CDL-B-60','CDL-B-90') and r['snr_db'] in (10,20):r['lr4_vs_convex_gain_db']=.24999
    assert p.scientific_gate(rows)['affine_composition_replication_weak']
    next(r for r in rows if (r['environment'],r['snr_db'])==('CDL-B-60',10))['lr4_vs_convex_gain_db']=.25
    assert p.scientific_gate(rows)['affine_composition_replication_partial']
    with pytest.raises(ValueError):p.scientific_gate(rows[:-1])


def test_no_unapproved_method_SVD_refit_detector_formal_path():
    for func in (p.deployed,p.oracle_coordinates,run.evaluate):
        calls={getattr(n.func,'id',getattr(n.func,'attr','')) for n in ast.walk(ast.parse(inspect.getsource(func))) if isinstance(n,ast.Call)}
        assert not calls & {'svd','build','inv','pinv','detect_data_zf','demodulate_data_bits','generate'}
    assert 'stream.update' not in inspect.getsource(run.evaluate)
    assert set(p.ENVIRONMENTS)=={'CDL-C-60','CDL-C-90','CDL-B-60','CDL-B-90','CDL-C-180','CDL-B-180'}
    assert run.load(run.OUT/'t17b_results.json')['formal_test_sealed']


def plot_fixture(directory):
    nmse=[];conditions=[]
    for n in p.ENVIRONMENTS:
        for s in p.SNRS:
            vals={'source':-1.,'hard':-2.,'convex':-4.,'dense-affine':-6.,'lr4-affine':-5.8,'matched':-7.}
            for m,v in vals.items():nmse.append(dict(environment=n,snr_db=s,method=m,nmse_db=v,subset='frames4..9'))
            conditions.append(dict(environment=n,snr_db=s,**p.metrics(-1.,-7.,-6.,-5.8,-4.,-6.5),outside_simplex_fraction=.9))
    run.csvsave(directory/plot.DATA[0],nmse);run.csvsave(directory/plot.DATA[1],conditions)
    return nmse,conditions


def test_plot_CSV_source_QA_byte_rebuild_and_scoped_style(tmp_path):
    plot_fixture(tmp_path);before=dict(plot.matplotlib.rcParams)
    report=plot.audit(tmp_path,tmp_path/'figures')
    assert report['source_returncode']==0 and all(report['CSV_only_rebuild'].values())
    assert dict(plot.matplotlib.rcParams)==before


def test_plot_rejects_missing_nonfinite_or_false_gain(tmp_path):
    nmse,conditions=plot_fixture(tmp_path)
    with pytest.raises(ValueError):plot.series(nmse[:-1],conditions)
    conditions[0]['lr4_vs_convex_gain_db']=100
    with pytest.raises(ValueError):plot.series(nmse,conditions)
    nmse[0]['nmse_db']=float('nan')
    with pytest.raises(ValueError):plot.series(nmse,conditions)


def test_actual_result_complete_paired_energy_weight_and_hash_regression():
    if not run.RESULT.exists():pytest.skip('T18A actual six-environment validation not generated yet')
    from scripts.audit_frozen_generalization import verify
    r=run.load(run.RESULT);assert verify(r)
    assert len(r['validation_nmse'])==126 and len(r['condition_metrics'])==18 and len(r['oracle_affine'])==18
    assert r['dataset']['primary_frames']==list(range(4,10))
    trajectory=readcsv(run.OUT/'t18a_generalization_trajectory.csv')
    parsed=[dict(environment=x['environment'],snr_db=int(x['snr_db']),trajectory_id=int(x['trajectory_id']),
        method=x['method'],frames=int(x['frames']),error_energy=float(x['error_energy']),truth_energy=float(x['truth_energy']),nmse_linear=float(x['nmse_linear'])) for x in trajectory]
    assert len(parsed)==12600 and run.aggregate(parsed)==r['validation_nmse']
    raw=readcsv(run.OUT/'t18a_generalization_weights.csv')
    assert len(raw)==32400
    parsed_weights=[dict(environment=x['environment'],snr_db=int(x['snr_db']),method=x['method'],
        alpha=float(x['alpha']),beta=float(x['beta']),lambda0=float(x['lambda0']),outside_any=(x['outside_any']=='True')) for x in raw]
    assert run.summarize_weights(parsed_weights)==r['weight_geometry']
    for x in raw:
        assert int(x['frame_id'])-int(x['window_start'])==4 and int(x['frame_id']) in range(4,10)
        assert geometry([float(x['alpha']),float(x['beta'])])['outside_any']==(x['outside_any']=='True')
    for row in r['condition_metrics']:
        values={x['method']:x['nmse_db'] for x in r['validation_nmse'] if (x['environment'],x['snr_db'])==(row['environment'],row['snr_db'])}
        expected=p.metrics(values['source'],values['matched'],values['dense-affine'],values['lr4-affine'],values['convex'],values['oracle-affine'])
        assert all(row[k]==v for k,v in expected.items())
    hashes=readcsv(run.OUT/'t18a_channel_hashes.csv');assert len(hashes)==3600
    assert len({x['H_sha256'] for x in hashes})==3600
    pairs=readcsv(run.OUT/'t18a_paired_input_hashes.csv');assert len(pairs)==18000
    for row in r['snr_audit']:assert row['sigma2']==p.VARIANCES[row['snr_db']] and not row['noise_recalibrated']
    assert r['complexity']['total']==332288 and r['complexity']['total_over_source_reported']==1.308468
    json.dumps(r,allow_nan=False)
