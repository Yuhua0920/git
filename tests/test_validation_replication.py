"""T18B independent trajectory uncertainty, immutable deployment and replay gates."""
import ast
import inspect
import textwrap
from pathlib import Path
import numpy as np
import pytest
from src import validation_replication as p
from src import frozen_affine_validation as old
from src.cdl_benchmark import StreamingStatistics,FrozenStatistics,OfficialCDL,array_hash
from src.lowrank_modules import WeightedModule
from scripts import run_validation_replication as run
from scripts import plot_validation_replication as plot


def test_T18A_snapshot_committed_and_materiality_immutable():
    r=run.accepted()
    assert r['affine_composition_replication_partial'] and not r['affine_composition_beyond_convex_hull_replicated']
    assert r['c6_claim_status']==old.C6 and r['c7_claim_status']==old.C7


def test_namespace_individual_roles_disjoint_and_scopes():
    rows=p.manifest_rows();proof=run.namespace_audit(rows)
    assert len(rows)==20000 and proof['unique_consumed_individual_seeds']==66000
    assert proof['t18a_all_individual_seeds_disjoint'] and proof['new_replicates_pairwise_disjoint']
    for n in p.ENVIRONMENTS:
        for r in p.REPLICATES:
            s0=p.seeds(n,r,0,0);s9=p.seeds(n,r,0,9)
            assert all(s0[k]==s9[k] for k in old.ROLES[:3])
            assert all(s9[k]-s0[k]==9 for k in old.ROLES[3:])


@pytest.mark.parametrize('args',[('CDL-C-180',1,0,0),('CDL-C-60',0,0,0),('CDL-C-60',6,0,0),('CDL-C-60',1,100,0),('CDL-C-60',1,0,10)])
def test_unapproved_namespace_rejected(args):
    with pytest.raises(ValueError):p.seeds(*args)


def test_frozen_bootstrap_size_and_eight_seeds():
    assert p.BOOTSTRAPS==10000 and p.BOOTSTRAP_SEED_BASE==20261802
    assert [p.bootstrap_seed(n,s) for n in p.ENVIRONMENTS for s in p.SNRS]==[20261812,20261822,20261912,20261922,20262012,20262022,20262112,20262122]
    assert p.SNRS==(10,20) and len(p.ENVIRONMENTS)==4


def fixture_energies(count=500):
    truth=np.linspace(1.,100.,count)
    error=np.column_stack([truth*x for x in (1.,.8,.7,.4,.41,.3,.35)])
    return error,truth


def test_energy_aggregation_before_log_not_trajectory_dB_mean():
    error=np.array([[1.],[100.]]);truth=np.array([1.,10.])
    numerator,denominator,db=p.pooled(error,truth)
    assert denominator==11 and numerator[0]==101
    assert db[0]==10*np.log10(101/11)
    assert db[0]!=np.mean(10*np.log10(error[:,0]/truth))


def test_paired_bootstrap_same_trajectory_indices_all_columns():
    e,t=fixture_energies(3);idx=np.array([[0,0,2],[1,2,1]],dtype=np.int64)
    got=p.paired_resamples(e,t,idx)
    for j,take in enumerate(idx):
        expected=10*np.log10(e[take].sum(axis=0)/t[take].sum())
        np.testing.assert_array_equal(got[j],expected)
    # Exact proportional method energies imply paired gain cancels draw randomness.
    np.testing.assert_allclose(got[:,0]-got[:,3],10*np.log10(1/.4),rtol=0,atol=1e-14)


def test_bootstrap_indices_frozen_repeat_and_percentiles():
    e,t=fixture_energies();rows,proof,db=p.bootstrap(e,t,p.ENVIRONMENTS[0],10)
    indices=np.random.default_rng(20261812).integers(0,500,size=(10000,500),dtype=np.int64)
    assert proof['indices_sha256']==array_hash(indices) and proof['shape']==[10000,500]
    assert db.shape==(10000,7) and len({v['shared_indices_sha256'] for v in rows})==1
    columns=p.metric_columns(db)
    for v in rows:
        assert (v['lower_2_5'],v['upper_97_5'])==tuple(float(x) for x in np.percentile(columns[v['metric']],[2.5,97.5]))
    repeat=p.bootstrap(e,t,p.ENVIRONMENTS[0],10)
    assert rows==repeat[0] and proof==repeat[1] and np.array_equal(db,repeat[2])


@pytest.mark.parametrize('count',[100,499,501,600])
def test_bootstrap_requires_exact_new500_trajectories(count):
    with pytest.raises(ValueError):p.bootstrap(*fixture_energies(count),p.ENVIRONMENTS[0],10)


def test_frame_or_window_index_cannot_replace_trajectory_index():
    e,t=fixture_energies(3)
    with pytest.raises(ValueError):p.paired_resamples(e,t,np.array([[0,1,2,0,1,2]]))
    with pytest.raises(ValueError):p.paired_resamples(e,t,np.array([[0,1,3]]))
    with pytest.raises(ValueError):p.paired_resamples(e,t,np.array([[0.,1.,2.]]))


def test_frozen_target_and_oracle_loading_without_fit_or_generator(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('target fitting/SVD/generation forbidden')
    monkeypatch.setattr(StreamingStatistics,'update',forbidden);monkeypatch.setattr(WeightedModule,'build',forbidden)
    monkeypatch.setattr(OfficialCDL,'__init__',forbidden);monkeypatch.setattr(old,'oracle_coordinates',forbidden)
    monkeypatch.setattr(np.linalg,'svd',forbidden)
    r=run.accepted();matched,oracle=run.references(r)
    arrays,family,proof=run.frozen_method()
    assert proof==r['frozen_method_provenance'] and set(matched)==set(p.ENVIRONMENTS)
    for n in p.ENVIRONMENTS:
        for s in p.SNRS:
            prior=next(v for v in r['oracle_affine'] if (v['environment'],v['snr_db'])==(n,s))
            assert np.array_equal(oracle[n][s],[prior['alpha'],prior['beta']])
            assert not oracle[n][s].flags.writeable and not matched[n][s].weight.flags.writeable
            assert matched[n][s].solve_relative_residual<=1e-11


def test_reference_artifact_retains_memory_layout_exact_prediction(tmp_path):
    r=run.accepted();matched,_=run.references(r)
    estimator=matched[p.ENVIRONMENTS[0]][10]
    rng=np.random.default_rng(18);z=rng.normal(size=64)+1j*rng.normal(size=64)
    path=tmp_path/'reference.npz';np.savez_compressed(path,W=estimator.weight)
    with np.load(path) as f:w=f['W'].copy(order='K')
    assert w.flags.f_contiguous==estimator.weight.flags.f_contiguous
    prediction=estimator.statistics.mean_d+(z-estimator.statistics.mean_p)@w.T
    assert array_hash(prediction)==array_hash(estimator.estimate(z))


def test_small_paired_link_two_SNRs_and_label_free_shared_coefficients():
    arrays,family,_=run.frozen_method()
    from scripts.run_cdl_mismatch import source_statistics
    source,_=source_statistics();matched={s:source.prepare(old.VARIANCES[s]) for s in p.SNRS};oracle={s:np.zeros(2) for s in p.SNRS}
    h=np.ones((2,2,64,160),np.complex128);before=array_hash(h)
    tr,raw,fr,pair=run.evaluate('CDL-C-60',0,h,arrays,family,matched,oracle,lambda j:p.seeds('CDL-C-60',1,0,j))
    assert array_hash(h)==before and len(tr)==14 and len(raw)==36 and len(fr)==84 and len(pair)==20
    for j in range(10):
        group=[v for v in pair if v['frame_id']==j]
        assert len({v['bits_sha256'] for v in group})==len({v['unit_noise_sha256'] for v in group})==1
        assert len({v['Y_sha256'] for v in group})==2
    for s in p.SNRS:
        for j in range(4,10):
            vals={v['method']:v['error_energy'] for v in fr if (v['snr_db'],v['frame_id'])==(s,j)}
            assert vals['source']==vals['matched']==vals['oracle-affine']
    assert all(v['dense_lr4_same_coefficients'] for v in raw if v['method']=='affine')
    assert set(inspect.signature(old.deployed).parameters)=={'pilots','arrays','snr','cv','profile_factor','mobility_factor'}


def test_all_historical_CSV_schema_comparisons_and_mismatch_rejection():
    for filename,key in [('t18a_generalization_trajectory.csv',('environment','snr_db','trajectory_id','method')),
        ('t18a_generalization_weights.csv',('environment','snr_db','trajectory_id','frame_id','method')),
        ('t18a_frame_metrics.csv',('environment','snr_db','trajectory_id','frame_id','method')),
        ('t18a_paired_input_hashes.csv',('environment','snr_db','trajectory_id','frame_id'))]:
        row=run.readcsv(run.OUT/filename)[0];index={tuple(row[k] for k in key):row}
        run.compare_rows([row],index,key)
        changed=dict(row);field=next(k for k in row if k not in key);changed[field]='invalid'
        with pytest.raises(ValueError):run.compare_rows([changed],index,key)


def gate_fixture():
    rep=[dict(environment=n,snr_db=s,replicate=r,dense_affine_closure=.8,pilotcv_efficiency=.9,compression_penalty_db=.1,lr4_vs_convex_gain_db=.8) for n in p.ENVIRONMENTS for s in p.SNRS for r in p.REPLICATES]
    point=[dict(environment=n,snr_db=s,dense_affine_closure=.8,pilotcv_efficiency=.9,compression_penalty_db=.1,lr4_vs_convex_gain_db=.8) for n in p.ENVIRONMENTS for s in p.SNRS]
    ci=[dict(environment=n,snr_db=s,metric=m,lower_2_5=.1,upper_97_5=.4) for n in p.ENVIRONMENTS for s in p.SNRS for m in ('lr4_vs_convex_gain_db','compression_penalty_db')]
    geo=[dict(environment=n,snr_db=s,replicate=r,outside_simplex_fraction=.9) for n in p.ENVIRONMENTS for s in p.SNRS for r in p.REPLICATES]
    return rep,point,ci,geo


def test_positive_support_new_hypothesis_does_not_reclassify_old_materiality():
    fixture=gate_fixture();g=p.scientific_gate(*fixture)
    assert g['positive_affine_advantage_statistically_replicated'] and g['c8_claim_status']=='SUPPORTED_IN_FIXED_ENVIRONMENTS'
    assert not g['post_replication_materiality_diagnostic'] and g['t18a_materiality_status_preserved']=='PARTIAL'
    fixture[0][-1]['lr4_vs_convex_gain_db']=0
    assert p.scientific_gate(*fixture)['replication_inconclusive']


def test_contradiction_three_failed_replicates_or_nonpositive_pooled():
    fixture=gate_fixture()
    for v in fixture[0][-5:-2]:v['lr4_vs_convex_gain_db']=0
    assert p.scientific_gate(*fixture)['replication_instability_observed']
    assert p.scientific_gate(*fixture)['c8_claim_status']=='CONTRADICTED_BY_REPLICATION'
    fixture=gate_fixture();fixture[1][-1]['lr4_vs_convex_gain_db']=0
    assert p.scientific_gate(*fixture)['replication_instability_observed']


def test_positive_CI_boundary_zero_is_inconclusive():
    fixture=gate_fixture();fixture[2][-2]['lower_2_5']=0
    assert p.scientific_gate(*fixture)['c8_claim_status']=='INCONCLUSIVE'


def test_geometry_secondary_not_in_C8_mapping():
    fixture=gate_fixture();fixture[3][-1]['outside_simplex_fraction']=.74
    g=p.scientific_gate(*fixture)
    assert not g['outside_simplex_geometry_stable'] and g['c8_claim_status']=='SUPPORTED_IN_FIXED_ENVIRONMENTS'


def test_compression_CI_warning_vetoes_C8_support_only():
    fixture=gate_fixture();fixture[2][1]['upper_97_5']=.50000001
    g=p.scientific_gate(*fixture)
    assert g['rank4_compression_uncertainty_warning'] and g['c8_claim_status']=='INCONCLUSIVE'
    assert not g['replication_instability_observed']


@pytest.mark.parametrize('field,value',[('dense_affine_closure',.6999),('pilotcv_efficiency',.7999),('compression_penalty_db',.5001)])
def test_multiple_performance_threshold_failures_contradict_C8(field,value):
    fixture=gate_fixture();fixture[0][0][field]=value
    assert p.scientific_gate(*fixture)['interpolation_threshold_stability']
    fixture[0][1][field]=value;g=p.scientific_gate(*fixture)
    assert not g['interpolation_threshold_stability'] and g['performance_replication_instability_observed']
    assert g['c8_claim_status']=='CONTRADICTED_BY_REPLICATION'


def test_pooled_only_interpolation_failure_is_inconclusive():
    fixture=gate_fixture();fixture[1][0]['dense_affine_closure']=.69;g=p.scientific_gate(*fixture)
    assert not g['interpolation_threshold_stability'] and not g['performance_replication_instability_observed']
    assert g['c8_claim_status']=='INCONCLUSIVE'


def test_no_refit_SVD_BER_detector_and_historical_gate_precedes_generation():
    for function in (run.evaluate,run.targets,run.references,run.run,p.ReplicationCDL.generate):
        calls={getattr(v.func,'id',getattr(v.func,'attr','')) for v in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(function)))) if isinstance(v,ast.Call)}
        assert not calls&{'svd','inv','pinv','oracle_coordinates','StreamingStatistics','detect_data_zf','demodulate_data_bits','freeze'}
    source=inspect.getsource(run.run)
    assert source.index("'historical replay gate failed: STOP'")<source.index('p.ReplicationCDL(n)')
    assert 'target_statistics_used_for_deployment=False' in source and 'formal_test_sealed=True' in source
    assert 'split=\'train_dev\'' not in source


def test_complete_mixed_CSV_and_result_json_serialization(tmp_path):
    row=dict(environment='CDL-C-60',snr_db=10,replicate=1,error_energy=1.,truth_energy=2.,method='source')
    run.csvsave(tmp_path/'energy.csv',[row,dict(row,method='matched')])
    run.save(tmp_path/'gate.json',p.scientific_gate(*gate_fixture()))
    assert len(run.readcsv(tmp_path/'energy.csv'))==2
    assert run.load(tmp_path/'gate.json')['t18a_materiality_status_preserved']=='PARTIAL'


def plot_fixture(directory):
    rep,points,ci,geo=gate_fixture()
    for v in points:v['lr4_vs_convex_gain_db']=.8
    intervals=[]
    for n in p.ENVIRONMENTS:
        for s in p.SNRS:
            for metric,value in [('lr4_vs_convex_gain_db',.8),('dense_affine_closure',.8),('compression_penalty_db',.1)]:
                intervals.append(dict(environment=n,snr_db=s,metric=metric,point_estimate=value,lower_2_5=value-.02,upper_97_5=value+.02,bootstrap_unit='trajectory',bootstrap_samples=10000))
    prior=[dict(environment=n,snr_db=s,lr4_vs_convex_gain_db=.7) for n in p.ENVIRONMENTS for s in p.SNRS]
    for name,rows in zip(plot.DATA,(rep,points,intervals,prior)):run.csvsave(directory/name,rows)


def test_figure_preflight_two_exports_CSV_only_and_no_global_style(tmp_path):
    plot_fixture(tmp_path);before=dict(plot.matplotlib.rcParams)
    report=plot.audit(tmp_path,tmp_path/'figures')
    assert report['source_returncode']==0 and all(report['CSV_only_rebuild'].values())
    assert dict(plot.matplotlib.rcParams)==before and len(plot.STEMS)==2


def test_plot_refuses_missing_or_nontrajectory_CI(tmp_path):
    plot_fixture(tmp_path);ci=run.readcsv(tmp_path/plot.DATA[2]);ci[0]['bootstrap_unit']='frame';run.csvsave(tmp_path/plot.DATA[2],ci)
    with pytest.raises(ValueError):plot.series(tmp_path)


def test_actual_complete_replay_energy_bootstrap_and_scientific_regression():
    if not run.RESULT.exists():pytest.skip('T18B new replication/uncertainty results not generated yet')
    from scripts.audit_validation_replication import verify
    r=run.load(run.RESULT);assert verify(r)
    assert len(r['replicate_nmse'])==280 and len(r['replicate_metrics'])==40 and len(r['weight_stability'])==40
    assert len(r['bootstrap_intervals'])==96
    assert r['historical_replay_gate']['comparisons']==dict(trajectory=5600,weights=14400,frames=33600,pairs=8000)
    assert r['historical_replay_gate']['all_saved_h_input_prediction_weights_energies_pooled_metrics_exact']
