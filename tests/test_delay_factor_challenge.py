"""T19A calibration first; actual result regression is gated on generation."""
import ast
import inspect
import textwrap
import numpy as np
import pytest
from src import delay_factor_challenge as p
from src import frozen_affine_validation as old
from src.cdl_benchmark import StreamingStatistics,FrozenStatistics,array_hash
from src.lowrank_modules import WeightedModule
from scripts import run_delay_factor_challenge as run
from scripts import plot_delay_factor_challenge as plot


@pytest.fixture(scope='module')
def frozen():return run.frozen_method()


def test_accepted_T18B_commit_and_permanent_claims():
    r=run.accepted()
    assert r['c8_claim_status']=='SUPPORTED_IN_FIXED_ENVIRONMENTS'
    assert r['c6_claim_status']==old.C6 and r['c7_claim_status']==old.C7
    assert r['t18a_materiality_status_preserved']=='PARTIAL'


def test_frozen_method_source_modules_factors_folds_and_K(frozen):
    _,_,proof=frozen
    assert proof==run.accepted()['frozen_method_provenance']
    assert proof['ranks']=={'P':4,'M':4} and proof['fold']['sha256']=='16594a306157e0b6ac114100ddebbba262f8e4205a98c8f48726e5519e55ab18'


def test_new_seed_roles_scopes_disjoint_from_all_history():
    rows=p.manifest_rows();proof=run.namespace_audit(rows)
    assert proof['allocated_individual_seeds']==158400 and proof['consumed_individual_seeds']==38400
    for n in p.ENVIRONMENTS:
        a=p.seeds(n,'train_dev',0,0);b=p.seeds(n,'train_dev',0,9);v=p.seeds(n,'validation_dev',500)
        assert all(a[k]==b[k] for k in old.ROLES[:3])
        assert all(b[k]-a[k]==9 for k in old.ROLES[3:])
        assert set(a.values()).isdisjoint(v.values())


@pytest.mark.parametrize('args',[(p.PURE[0],'train_dev',500,0),(p.PURE[0],'validation_dev',0,0),
    (p.PURE[0],'formal_test',500,0),(p.PURE[0],'validation_dev',600,0),(p.PURE[0],'train_dev',0,10),('CDL-C-30-100','train_dev',0,0)])
def test_forbidden_data_namespaces_rejected(args):
    with pytest.raises(ValueError):p.seeds(*args)


@pytest.mark.parametrize('n',p.ENVIRONMENTS)
def test_factorial_configuration_only_approved_changes(n):
    v=p.definition(n);profile,speed,delay=n.split('-')[1:]
    assert v['profile']==profile and v['system']['speed_kmh']==int(speed)
    assert v['system']['nominal_delay_spread_seconds']==int(delay)*1e-9
    assert v['system']['snr_db']==[10,20] and not v['ofdm_channel_normalize']
    assert v['dF']==16 and v['spacing_time']==4
    assert v['array']['bs_orientation_radians']==old.ARRAY['bs_orientation_radians']
    assert v['array']['ut_orientation_radians']==old.ARRAY['ut_orientation_radians']


@pytest.mark.parametrize('s',p.SNRS)
def test_source_variance_exact_no_target_calibration(s):
    assert old.POWER==9.941134822523972
    assert old.VARIANCES[s]==old.POWER/10**(s/10)
    assert old.VARIANCES[s]==run.load(run.OUT/'t16d_modular_results.json')['source_noise_variances'][str(s)]


def test_target_statistics_reject_validation_without_mutation():
    g=run.observation_geometry();obj=StreamingStatistics(g.selected,g.data)
    with pytest.raises(ValueError):obj.update(np.ones((10,4096),complex),[500])
    assert obj.n==0 and not obj.ids


def test_effective_delay_matches_T15A_pooled_CIR_energy_hand_value():
    obj=p.EffectiveDelayAudit();a=np.ones((2,2,2,160),np.complex128);a[:,:,1]*=np.sqrt(3.)
    for i in range(500):obj.update(i,a,np.array([0.,100e-9]))
    v=obj.freeze(50)
    np.testing.assert_allclose(v['normalized_empirical_path_powers'],[.25,.75],rtol=0,atol=1e-14)
    assert v['nominal_delay_spread_ns']==50
    np.testing.assert_allclose(v['empirical_array_weighted_rms_delay_ns'],100*np.sqrt(.25*.75),rtol=1e-14)
    assert v['empirical_array_weighted_rms_delay_ns']!=50


@pytest.mark.parametrize('i',[-1,500,600])
def test_delay_audit_rejects_nontrain_before_mutation(i):
    obj=p.EffectiveDelayAudit()
    with pytest.raises(ValueError):obj.update(i,np.ones((2,2,1,160)),np.zeros(1))
    assert obj.power is None and not obj.ids


def test_delay_audit_rejects_path_delay_drift_and_incomplete_training():
    obj=p.EffectiveDelayAudit();obj.update(0,np.ones((2,2,1,160)),np.zeros(1))
    with pytest.raises(ValueError):obj.update(1,np.ones((2,2,1,160)),np.ones(1))
    with pytest.raises(ValueError):obj.freeze(50)
    assert obj.ids==[0]


def test_Oracle2_complex_weighted_real_LS_correctness_without_SVD(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('no module/basis SVD')
    monkeypatch.setattr(np.linalg,'svd',forbidden)
    rng=np.random.default_rng(20261901)
    def cx(shape):return rng.normal(size=shape)+1j*rng.normal(size=shape)
    q=cx((64,64));r=q@q.conj().T/64+np.eye(64)
    w0,wp,wm=[cx((5,64)) for _ in range(3)];expected=np.array([-.3,1.7]);s=10
    z=r+old.VARIANCES[s]*np.eye(64);target=FrozenStatistics(np.zeros(64),np.zeros(5),r,(w0+expected[0]*wp+expected[1]*wm)@z,(0,),10)
    arrays={'Wsource_10':w0,'DeltaWP_10':wp,'DeltaWM_10':wm}
    got,proof=old.oracle_coordinates(target,arrays,s)
    np.testing.assert_allclose(got,expected,rtol=0,atol=2e-14)
    assert proof['weighted_filter_excess_energy_per_frame']<1e-24 and not proof['deployable']
    # An independent weighted normal-gradient check and displaced objectives.
    l=np.linalg.cholesky(z);cols=[wp@l,wm@l];response=(target.prepare(old.VARIANCES[s]).weight-w0)@l
    residual=response-got[0]*cols[0]-got[1]*cols[1]
    assert abs(np.vdot(cols[0],residual).real)<1e-10
    for shift in ([.1,0],[0,-.1],[.2,.2]):
        alt=response-(got[0]+shift[0])*cols[0]-(got[1]+shift[1])*cols[1]
        assert np.linalg.norm(alt)>np.linalg.norm(residual)


def test_primary_label_oracle_truth_isolation_and_rank4(frozen,monkeypatch):
    arrays,family,_=frozen
    assert set(inspect.signature(old.deployed).parameters)=={'pilots','arrays','snr','cv','profile_factor','mobility_factor'}
    def forbidden(*a,**k):raise AssertionError('target/oracle/refit leaked into deployment')
    monkeypatch.setattr(WeightedModule,'build',forbidden);monkeypatch.setattr(FrozenStatistics,'prepare',forbidden);monkeypatch.setattr(old,'oracle_coordinates',forbidden)
    z=np.random.default_rng(20261902).normal(size=(10,64)).astype(np.complex128)
    rows=old.deployed(z,arrays,10,run.cv_family(arrays,10),family['P',10,4],family['M',10,4])
    assert [r['frame_id'] for r in rows]==list(range(4,10))
    reports=[dict(environment='altered_reporting_label',**r) for r in rows]
    for a,b in zip(rows,reports):
        assert a['dense_coefficients'] is a['lr4_coefficients']
        np.testing.assert_array_equal(a['predictions']['lr4-affine'],b['predictions']['lr4-affine'])


def test_small_paired_link_source_means_and_oracle_separate(frozen):
    from scripts.run_cdl_mismatch import source_statistics
    arrays,family,_=frozen;source,_=source_statistics()
    matched={s:source.prepare(old.VARIANCES[s]) for s in p.SNRS};oracle={s:np.zeros(2) for s in p.SNRS}
    h=np.ones((2,2,64,160),np.complex128);h.flags.writeable=False;before=array_hash(h)
    tr,raw,fr,pairs,signal,noise,count=run.evaluate(p.PURE[0],500,h,arrays,family,matched,oracle,run.observation_geometry())
    assert before==array_hash(h) and (len(tr),len(raw),len(fr),len(pairs),count)==(14,36,84,20,6*992)
    assert np.all(signal>0) and np.all(noise>0)
    for j in range(10):
        a,b=pairs[j*2:j*2+2]
        assert all(a[k]==b[k] for k in ('H_sha256','bits_sha256','symbols_sha256','unit_noise_sha256'))
    for s in p.SNRS:
        for j in range(4,10):
            index={r['method']:r for r in fr if r['snr_db']==s and r['frame_id']==j}
            assert index['source']['prediction_sha256']==index['oracle2-affine']['prediction_sha256']
    assert all(r['dense_lr4_same_coefficients'] for r in raw if r['method']=='affine')
    assert all(r['coordinate_error_norm']==np.linalg.norm([r['alpha'],r['beta']]) for r in raw if r['method']=='affine')


def values():return dict(zip(p.METHODS,[0.,-.5,-1.,-2.,-1.8,-4.,-3.]))


def test_failure_decomposition_exact_unclipped_formulas():
    v=p.metric_columns(values())
    assert v['representation_closure']==.75 and v['coordinate_efficiency']==2/3
    assert v['lr4_closure']==.45 and v['compression_penalty_db']==pytest.approx(.2)
    assert v['dense_vs_convex_gain_db']==1 and v['lr4_vs_source_gain_db']==1.8
    z=values();z['dense-affine']=2;assert p.metric_columns(z)['coordinate_efficiency']<0
    z['dense-affine']=-6;assert p.metric_columns(z)['coordinate_efficiency']>1


@pytest.mark.parametrize('matched,oracle',[(0.,0.),(1.,1.),(1.,-3.),(-4.,1.)])
def test_nonpositive_ratio_denominators_flagged_not_clipped(matched,oracle):
    z=values();z['matched']=matched;z['oracle2-affine']=oracle;v=p.metric_columns(z)
    if matched>=0:assert v['representation_closure'] is None and v['lr4_closure'] is None
    if oracle>=0:assert v['coordinate_efficiency'] is None
    assert v['ratio_anomaly'] and 'nonpositive denominator' in v['ratio_anomaly_reasons']


def gate_rows(rep=.8,eff=.9,penalty=.2):
    return [dict(environment=n,snr_db=s,representation_closure=rep,coordinate_efficiency=eff,compression_penalty_db=penalty) for n in p.ENVIRONMENTS for s in p.SNRS]


@pytest.mark.parametrize('rep,eff,state',[(.7,.8,'SUPPORTED_IN_TESTED_DELAY_CONDITIONS'),(.49,.8,'CONTRADICTED_BY_DELAY_FACTOR_CHALLENGE'),(.6,.9,'INCONCLUSIVE')])
def test_pre_registered_gate_boundaries(rep,eff,state):
    assert p.scientific_gate(gate_rows(rep,eff))['c9_claim_status']==state


def test_sufficient_priority_and_independent_coordinate_bottleneck():
    g=p.scientific_gate(gate_rows(.7,.79))
    assert g['existing_pm_space_delay_representation_sufficient'] and g['delay_coordinate_inference_bottleneck']
    assert not g['delay_factor_gate_partial'] and g['c9_claim_status']=='SUPPORTED_IN_TESTED_DELAY_CONDITIONS'
    assert 'No delay module' in g['next_step_dependency']


def test_coordinate_only_and_three_same_conditions_rule():
    rows=gate_rows(.8,.79);rows[0]['representation_closure']=.6
    g=p.scientific_gate(rows)
    assert not g['existing_pm_space_delay_representation_sufficient'] and g['delay_coordinate_inference_bottleneck']
    assert g['c9_claim_status']=='INCONCLUSIVE_REPRESENTATION_DUE_TO_COORDINATE_BOTTLENECK'
    rows=gate_rows(.49,.8);rows[0]['coordinate_efficiency']=.79
    assert p.scientific_gate(rows)['delay_specific_representation_deficiency']
    rows[1]['coordinate_efficiency']=.79
    assert p.scientific_gate(rows)['delay_factor_gate_partial']


def test_combined_context_excluded_and_warning_strict_greater():
    rows=gate_rows();rows[-1].update(representation_closure=-10,coordinate_efficiency=-100,compression_penalty_db=9)
    assert p.scientific_gate(rows)['existing_pm_space_delay_representation_sufficient']
    assert not p.scientific_gate(rows)['rank4_out_of_factor_warning']
    rows[0]['compression_penalty_db']=1.;assert not p.scientific_gate(rows)['rank4_out_of_factor_warning']
    rows[0]['compression_penalty_db']=1.0001;assert p.scientific_gate(rows)['rank4_out_of_factor_warning']


def test_incomplete_grid_rejected_and_invalid_ratios_partial():
    with pytest.raises(ValueError):p.scientific_gate(gate_rows()[:-1])
    rows=gate_rows()
    for r in rows[:4]:r.update(representation_closure=None,coordinate_efficiency=None)
    assert p.scientific_gate(rows)['delay_factor_gate_partial']


def test_no_new_basis_or_detector_and_only_matched_training_in_runner():
    tree=ast.parse(inspect.getsource(run));attrs=[n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)]
    assert 'svd' not in attrs and 'pinv' not in attrs and 'inv' not in attrs and 'noise_variance' not in attrs
    assert not any(isinstance(n,ast.Call) and 'detect' in ast.unparse(n.func).lower() for n in ast.walk(tree))
    generator=ast.parse(textwrap.dedent(inspect.getsource(p.FactorCDL.generate)))
    call=next(n for n in ast.walk(generator) if isinstance(n,ast.Call) and ast.unparse(n.func)=='cir_to_ofdm_channel')
    assert next(k.value.value for k in call.keywords if k.arg=='normalize') is False


def fixture_tables():
    nmse=[];metrics=[]
    for j,n in enumerate(p.ENVIRONMENTS):
        for s in p.SNRS:
            v=values();v={m:x-j*.1-(s-10)/10 for m,x in v.items()}
            nmse.extend(dict(environment=n,snr_db=s,method=m,nmse_db=x) for m,x in v.items())
            metrics.append(dict(environment=n,snr_db=s,**p.metric_columns(v)))
    return nmse,metrics


def test_heterogeneous_CSV_schema_and_all_figure_conditions(tmp_path):
    n,m=fixture_tables();run.csvsave(tmp_path/'t19a_factor_nmse.csv',n);run.csvsave(tmp_path/'t19a_factor_metrics.csv',m)
    pure,points,invalid=plot.series(tmp_path)
    assert len(pure)==20 and len(points)==16 and not invalid
    m[0].update(representation_closure=None,ratio_anomaly=True,ratio_anomaly_reasons='representation_closure: nonpositive denominator')
    run.csvsave(tmp_path/'t19a_factor_metrics.csv',m)
    _,points,invalid=plot.series(tmp_path);assert len(points)==15 and len(invalid)==1


def test_actual_validation_results_regression():
    if not run.RESULT.exists():pytest.skip('T19A actual results not generated; preparation-only explicit skip')
    from scripts.audit_delay_factor_challenge import verify
    r=run.load(run.RESULT);assert verify(r)
    assert r['generator_calls']=={'preflight':32,'target_train':4000,'validation':800}
    assert len(r['effective_delay_audit'])==8 and len(r['factor_metrics'])==16
