"""T19C independent train statistics, frozen method, exact references and gates."""
import ast
import inspect
import numpy as np
import pytest
from src import target_statistics_replication as p
from src import operator_space_audit as math
from src.cdl_benchmark import StreamingStatistics
from scripts import run_target_statistics_replication as run
from scripts import plot_target_statistics_replication as plot


@pytest.mark.parametrize('name,replicate,trajectory,split',[
    ('CDL-B-30-50',1,0,'train_dev'),('CDL-C-30-50',0,0,'train_dev'),
    ('CDL-C-30-50',6,0,'train_dev'),('CDL-C-30-50',1,500,'train_dev'),
    ('CDL-C-30-200',1,0,'validation_dev'),('CDL-C-30-200',1,-1,'train_dev')])
def test_only_declared_train_scope(name,replicate,trajectory,split):
    with pytest.raises(ValueError):p.seeds(name,replicate,trajectory,split=split)


def test_namespace_manifest_unique_disjoint_and_actual_channel_scope():
    rows=p.manifest_rows();audit=run.namespace_audit(rows)
    assert audit['pairwise_disjoint'] and audit['historical_disjoint']
    assert audit['unique_trajectory_streams']==5000 and audit['unique_individual_channel_seeds']==15000
    assert len(rows)==50000 and all(r['communication_generated'] is False and r['split']=='train_dev' for r in rows)
    assert [tuple(v[k] for k in p.ROLES) for v in rows[:10]]==[tuple(rows[0][k] for k in p.ROLES)]*10
    assert all('bits_seed_tx0' not in r and 'noise_seed' not in r for r in rows)


def test_namespace_collision_and_corruption_stop(monkeypatch):
    rows=p.manifest_rows();bad=[dict(rows[0])]+rows[1:];bad[0]['sionna_seed']+=1
    with pytest.raises(ValueError,match='policy'):run.namespace_audit(bad)
    fake={k:str(rows[0][k]) for k in p.ROLES}
    monkeypatch.setattr(run,'readcsv',lambda _: [fake])
    with pytest.raises(ValueError,match='collision'):run.namespace_audit(rows)


def test_physical_generator_exact_except_declared_seed_call():
    proof=run.generator_semantics()
    assert proof['export_AST_identical_except_namespace_seed_call'] and proof['inherited_physical_constructor']
    assert proof['normalize'] is False and proof['profiles']==['C'] and proof['speed_kmh']==30


def test_original_statistics_estimator_and_unequal_weight_merge_identity():
    result=p.calibration()
    assert result['passed'] and result['max_absolute_error']<1e-11
    assert result['total_samples']==25 and result['unequal_sample_counts']==[3,4,5,6,7]
    assert not result['Rdd_fitted']


def synthetic_records():
    records=[];raw=[]
    for r in p.REPLICATES:
        k=np.arange((r+2)*5).reshape(r+2,5)
        x=np.sin(k*.17)+r*.2+1j*(np.cos(k*.13)-r*.1)
        stream=StreamingStatistics([0,2],np.array([False,True,False,True,True]),dimension=5)
        stream.update(x,[0]);records.append((p.namespace(p.ENVIRONMENTS[0],r),stream.freeze()));raw.append(x)
    return records,raw


def test_large_stat_correct_complex_mean_correction_not_average_covariance():
    records,raw=synthetic_records();merged=p.merge_sufficient_statistics(p.ENVIRONMENTS[0],records)
    x=np.concatenate(raw);xc=x-x.mean(axis=0);selected=[0,2];data=[1,3,4]
    assert merged.samples==25 and len(merged.train_trajectory_ids)==5
    assert np.allclose(merged.mean_p,x.mean(axis=0)[selected],rtol=1e-13,atol=1e-13)
    assert np.allclose(merged.rpp,xc[:,selected].T@xc[:,selected].conj()/len(x),rtol=1e-13,atol=1e-13)
    assert np.allclose(merged.rdp,xc[:,data].T@xc[:,selected].conj()/len(x),rtol=1e-13,atol=1e-13)
    naive=sum(st.rpp*st.samples/25 for _,st in records)
    assert not np.allclose(merged.rpp,naive)
    assert all(not a.flags.writeable for a in (merged.mean_p,merged.mean_d,merged.rpp,merged.rdp))


@pytest.mark.parametrize('corruption',['historical','duplicate','mixed_delay','missing'])
def test_large_stat_original_and_cross_delay_pooling_forbidden(corruption):
    records,_=synthetic_records()
    if corruption=='historical':records[0]=('T19A_CDL-C-30-50_v1',records[0][1])
    if corruption=='duplicate':records[0]=records[1]
    if corruption=='mixed_delay':records[0]=(p.namespace(p.ENVIRONMENTS[1],1),records[0][1])
    if corruption=='missing':records=records[:-1]
    with pytest.raises(ValueError):p.merge_sufficient_statistics(p.ENVIRONMENTS[0],records)


def test_frozen_operator_solver_complex_real_LS_and_capture_formula():
    k=np.arange(12).reshape(4,3)
    bp=np.sin(k)+1j*np.cos(k*.7);bm=np.cos(k*.3)+1j*np.sin(k*.8)
    ws=bp*.02;target=ws+bp*.3-bm*.6;lower=np.diag([.2,.7,1.5])
    fit=math.operator_fit(ws,target,lower,[bp,bm]);source=math.operator_excess(ws,target,lower)
    assert fit['coefficients']==pytest.approx([.3,-.6],abs=1e-12)
    assert 1-fit['excess']/source==pytest.approx(1.,abs=1e-12)
    assert run.math.operator_fit is math.operator_fit  # reuse, no new coefficient solver


def gate_rows(primary=.49):
    return [dict(environment=n,replicate=r,snr_db=s,operator_span_capture_dense=(primary if n==p.ENVIRONMENTS[1] and s==10 else .8))
            for n in p.ENVIRONMENTS for r in p.REPLICATES for s in p.SNRS]


@pytest.mark.parametrize('value,state',[(.499,'SUPPORTED_IN_C30_200_10_STATISTICS_REPLICATION'),(.50,'INCONCLUSIVE'),(.699,'INCONCLUSIVE'),(.70,'CONTRADICTED_BY_TARGET_STATISTICS_REPLICATION')])
def test_primary_strict_boundaries(value,state):
    assert p.scientific_gate(gate_rows(value))['c11_claim_status']==state


def test_four_of_five_gate_and_control_independence():
    rows=gate_rows();primary=[v for v in rows if v['environment']==p.ENVIRONMENTS[1] and v['snr_db']==10]
    primary[0]['operator_span_capture_dense']=.9
    for v in rows:
        if v['environment']==p.ENVIRONMENTS[0]:v['operator_span_capture_dense']=.1
    gate=p.scientific_gate(rows)
    assert gate['delay_operator_deficiency_replication_supported'] and not gate['delay50_control_stable']
    primary[1]['operator_span_capture_dense']=.9
    assert p.scientific_gate(rows)['delay_operator_deficiency_replication_inconclusive']


def test_control_count_per_SNR_and_paired_asymmetry_strict():
    rows=gate_rows()
    control=[v for v in rows if v['environment']==p.ENVIRONMENTS[0] and v['snr_db']==10]
    control[0]['operator_span_capture_dense']=.699
    assert p.scientific_gate(rows)['delay50_control_stable']
    control[1]['operator_span_capture_dense']=.699
    assert not p.scientific_gate(rows)['delay50_control_stable']
    assert p.scientific_gate(rows)['snr_asymmetry_direction_consistent']
    next(v for v in rows if (v['environment'],v['replicate'],v['snr_db'])==(p.ENVIRONMENTS[1],1,20))['operator_span_capture_dense']=.49
    assert not p.scientific_gate(rows)['snr_asymmetry_direction_consistent']


def test_summary_five_only_sample_std_and_original_external():
    rows=gate_rows();original={(n,s):.123 for n in p.ENVIRONMENTS for s in p.SNRS}
    for v in rows:v['operator_span_capture_dense']+=v['replicate']*.01
    summary=p.summarize(rows,original);x=np.array([.81,.82,.83,.84,.85])
    assert summary[0]['mean']==pytest.approx(x.mean()) and summary[0]['std']==pytest.approx(x.std(ddof=1))
    assert summary[0]['std_ddof']==1 and summary[0]['original_t19a_capture']==.123 and summary[0]['original_below_replicate_min']
    with pytest.raises(ValueError):p.summarize(rows+[dict(rows[0],replicate=0)],original)


def test_missing_or_nonfinite_scientific_row_stops():
    rows=gate_rows()
    with pytest.raises(ValueError):p.scientific_gate(rows[:-1])
    rows[-2]['operator_span_capture_dense']=float('nan')
    with pytest.raises(ValueError):p.scientific_gate(rows)


def test_no_validation_calls_new_solver_inverse_bootstrap_or_receiver():
    for module in (p,run):
        tree=ast.parse(inspect.getsource(module));calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
        assert not any(x.endswith(('.inv','.pinv','.svd','.fit','.detect_data_zf','.communication')) or 'bootstrap' in x for x in calls)
    tree=ast.parse(inspect.getsource(run))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='generate']
    assert len(calls)==2 and all(next(k.value.value for k in n.keywords if k.arg=='split')=='train_dev' for n in calls)
    assert 'StreamingStatistics(g.selected, g.data)' in ast.unparse(tree)


def test_runtime_guard_forbids_method_refit_PilotCV_noise_and_BER():
    from src import pilot_cv,link,numerics
    with run.method_guard():
        for callback in (lambda:pilot_cv.PilotCV.fit(None,None,modules=('P','M')),lambda:link.detect_data_zf(None,None,None),
                         lambda:run.WeightedModule.build(None,None,None),lambda:numerics.complex_gaussian_noise(None,None,None)):
            with pytest.raises(RuntimeError):callback()
        assert p.calibration()['passed']


def test_accepted_T19B_and_original_references_exact():
    r=run.accepted();arrays,family,_,targets,proof=run.frozen_inputs()
    original=run.original_reference(arrays,family,targets)
    assert len(original)==4 and all(v['exact_replay'] and not v['in_replication_aggregate'] for v in original)
    assert r['c9_claim_status']==r['c10_claim_status']=='INCONCLUSIVE'
    assert proof['frozen_method']['ranks']=={'P':4,'M':4}


def test_figure_population_pairing_and_CSV_schema(tmp_path):
    tables=[[],[],[]]
    for n in p.ENVIRONMENTS:
        for s in p.SNRS:
            for r in p.REPLICATES:tables[0].append(dict(environment=n,replicate=r,snr_db=s,operator_span_capture_dense=.8,statistics_sha256=f'{n}-{r}'))
            tables[1].append(dict(environment=n,snr_db=s,operator_span_capture_dense=.7))
            tables[2].append(dict(environment=n,snr_db=s,operator_span_capture_dense=.75))
    for name,table in zip(plot.DATA,tables):run.csvsave(tmp_path/name,table)
    assert [len(v) for v in plot.series(tmp_path)]==[20,4,4]
    tables[0][0]['statistics_sha256']='different population';run.csvsave(tmp_path/plot.DATA[0],tables[0])
    with pytest.raises(ValueError,match='same clean'):plot.series(tmp_path)


def test_actual_statistics_replication_results():
    if not run.RESULT.exists():pytest.skip('T19C actual statistics replication not generated; preparation-only explicit skip')
    from scripts.audit_target_statistics_replication import verify
    r=run.load(run.RESULT);assert verify(r)
    assert len(r['operator_capture_replicates'])==20 and len(r['large_stat_reference'])==4
    assert r['generator_calls']['validation']==0 and not r['validation_generated'] and r['formal_test_sealed']
