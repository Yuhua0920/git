"""U7A outcome firewall and preregistration validation; no scientific channels."""
import hashlib
import io
import json
import zipfile
import numpy as np
import pytest
from scripts import sp_u7a_metadata as md
from scripts import sp_u7a_gate as gate
from scripts import sp_u7a_statistics as st

def test_json_projection_discards_numerical_outcomes():
    p=md.project_json('{"profile":"C","noise_seed":123,"seed_count":91,"NMSE":-98765.4321,"BER":0.1234567,"nested":{"seeds":[2,3],"namespace":"old"}}')
    assert p['seeds']=={123,2,3}
    assert p['namespaces']=={'old'}
    assert 'NMSE' not in repr(p) and '98765' not in repr(p) and '0.1234567' not in repr(p)

def test_csv_projection_never_returns_metric_columns():
    p=md.csv_metadata('profile,condition,noise_seed,NMSE,BER\nC,CDL-C-120,145,-77777.7777,0.9876543\n')
    assert p['selected_columns']==['profile','condition','noise_seed']
    assert p['seeds']=={145}
    assert '77777' not in repr(p) and '0.9876543' not in repr(p)

def test_outcome_inside_seed_container_is_not_a_seed():
    p=md.project_json('{"seeds":{"channel_seed":123,"NMSE":999,"metrics":{"total_bits":888,"BER":777}},"noise_seed":456}')
    assert p['seeds']=={123,456}
    assert '999' not in repr(p) and '888' not in repr(p) and '777' not in repr(p)

def test_encoded_A_identity_is_visible_without_performance():
    p=md.project_json('{"profile":"A","speed_kmh":120,"delay_spread":1e-7,"NMSE":-55555}')
    assert ('profile','A') in p['identities']
    assert '55555' not in repr(p)

def test_npz_reads_only_metadata_and_headers(monkeypatch):
    b=io.BytesIO();np.savez(b,H=np.ones((2,3),dtype=complex),nmse=np.array([-55555.]),metadata_json=np.array('{"profile":"C","noise_seed":15}'))
    b.seek(0);old=np.lib.format.read_array;opened=[]
    def guarded(f,*args,**kwargs):
        opened.append(f.name)
        assert f.name=='metadata_json.npy'
        return old(f,*args,**kwargs)
    monkeypatch.setattr(np.lib.format,'read_array',guarded)
    p=md.archive_metadata(b)
    assert opened==['metadata_json.npy'] and p['seeds']=={15}
    assert p['members'][0]['shape']==[2,3]
    assert '55555' not in repr(p)

def test_unsafe_object_metadata_blocks():
    b=io.BytesIO();np.savez(b,metadata_json=np.array({'profile':'A'},dtype=object));b.seek(0)
    with pytest.raises(ValueError,match='unsafe object metadata'):md.archive_metadata(b)

def test_unknown_candidate_hit_blocks():
    h=md.classified_hit('outputs/unknown_results.json','{"condition":"CDL-A-120","NMSE":-77777}',md.project_json('{"condition":"CDL-A-120","NMSE":-77777}'))
    assert h['candidate_exposure']=='OPEN'
    assert '77777' not in repr(h)

def test_package_presence_does_not_imply_generation():
    text=json.dumps({'files':['X/Lib/site-packages/sionna/phy/channel/tr38901/models/v19_2/CDL-A.json']})
    h=md.classified_hit('outputs/t20b_preflight.json',text,md.project_json(text))
    assert h['candidate_exposure']=='E0' and h['kind']=='INSTALLED_MODEL_INVENTORY'

def test_seed_mapping_collision_rejects_only_metadata(monkeypatch):
    monkeypatch.setattr(gate,'seed_keys',lambda:[('EVAL',0,-1,'sionna_seed'),('BOOTSTRAP',-1,-1,'bootstrap_seed')])
    m=f'{gate.NAMESPACE}|EVAL|0|-1|sionna_seed|0';bad=int.from_bytes(hashlib.sha256(m.encode()).digest()[:4],'big')
    a,r=gate.allocate_seeds({bad});b,s=gate.allocate_seeds({bad})
    assert a==b and r==s and len(r)==1 and a[0]['counter']==1
    assert len({x['seed'] for x in a})==2 and bad not in {x['seed'] for x in a}

def test_seed_key_plan_individual_roles_and_reference_isolation():
    keys=gate.seed_keys()
    assert len(keys)==34501 and len(set(keys))==34501
    assert sum(k[0]=='EVAL' for k in keys)==33000
    assert sum(k[0]=='MATCHED_REF' for k in keys)==1500
    assert all(k[2]==-1 for k in keys if k[0]=='MATCHED_REF')

@pytest.mark.parametrize('key',gate.INTEGRITY_GATES)
def test_any_integrity_failure_stays_sealed(key):
    checks=dict.fromkeys(gate.INTEGRITY_GATES,True);checks[key]=False
    with pytest.raises(ValueError,match='STOP_SEALED'):gate.validate_unseal(checks)

def test_missing_gate_is_not_pass():
    with pytest.raises(ValueError):gate.validate_unseal({})
    assert gate.validate_unseal(dict.fromkeys(gate.INTEGRITY_GATES,True))

@pytest.mark.parametrize('point,lo,hi,result',[
    ([1,2],[.1,.2],[2,3],'A_STRONG_TRANSFER'),
    ([1,2],[0,.2],[2,3],'B_MIXED_OR_INCONCLUSIVE'),
    ([1,-1],[.1,-2],[2,-.1],'C_NONPOSITIVE_BOUNDARY'),
    ([0,0],[-1,-1],[1,1],'B_MIXED_OR_INCONCLUSIVE'),
    ([0,0],[-1,-1],[0,1],'C_NONPOSITIVE_BOUNDARY')])
def test_decision_boundary_exact_zero_and_mixed_precedence(point,lo,hi,result):
    assert st.decision(point,lo,hi)==result

def toy_panel(n=11):
    q=np.arange(1,n*2+1,dtype=float).reshape(n,2)
    e=np.arange(1,n*2*7+1,dtype=float).reshape(n,2,7)
    be=np.zeros_like(e,dtype=int);nb=np.full((n,2),100,dtype=int)
    return e,q,be,nb

def test_pooled_estimand_is_not_mean_trajectory_db():
    e,q,_,_=toy_panel();point=st.reduce_nmse(e,q)
    assert np.allclose(point,10*np.log10(e.sum(0)/q.sum(0)[:,None]))
    assert not np.allclose(point,np.mean(10*np.log10(e/q[:,:,None]),axis=0))

def test_reuses_formal_helper_draw_and_pooled_semantics_synthetic_only():
    from src import formal_evaluation as old
    e,q,be,nb=toy_panel();seed=48123
    new=st.paired_bootstrap(e,q,be,nb,seed,expected_n=11)
    rows=[]
    for i in range(11):
        for sidx,s in enumerate(st.SNRS):
            for j,name in enumerate(old.METHODS):
                rows.append({'environment':'synthetic_U7A_test','trajectory_id':i,'snr_db':s,'method':name,
                   'error_energy':0. if j==0 else e[i,sidx,j-1], 'truth_energy':q[i,sidx],
                   'bit_errors':0,'total_bits':100})
    original=old.bootstrap(rows,conditions=('synthetic_U7A_test',),snrs=st.SNRS,resamples=10000,seed=seed)
    for sidx,s in enumerate(st.SNRS):
        r=next(r for r in original if r['snr_db']==s and r['comparison']=='source_gain' and r['metric']=='nmse_difference_db')
        assert new['gain'][sidx]==pytest.approx(r['estimate'],abs=1e-12)
        assert new['primary_lcb'][sidx]==pytest.approx(r['ci95_lower'],abs=1e-12)
    rng=np.random.default_rng(seed);draw=rng.integers(0,11,size=(10000,11))
    assert new['indices_sha256']==hashlib.sha256(draw.tobytes()).hexdigest()

def test_incomplete_or_nonfinite_population_blocks_no_filtering():
    e,q,be,nb=toy_panel()
    with pytest.raises(ValueError):st.paired_bootstrap(e,q,be,nb,10)
    e[0,0,0]=np.nan
    with pytest.raises(ValueError):st.paired_bootstrap(e,q,be,nb,10,expected_n=11)

def test_no_new_simulation_entrypoint_and_preregistered_constants():
    p=gate.protocol(23)
    assert p['condition']['sample_rate_hz']==p['condition']['K']*p['condition']['spacing_hz']
    assert p['evaluation']['independent_trajectories']==1000
    assert p['statistics']['bootstrap']['primary_LCB_quantile']==.025
    assert 'margin' not in p['statistics']['secondary']
    assert p['deployable']['rank']==4 and p['deployable']['window']==5
    assert p['status']=='PREREGISTERED_SEALED_NOT_AUTHORIZED_TO_RUN'
    assert not hasattr(gate,'generate') and not hasattr(gate,'evaluate')

def test_real_qualification_manifests_metadata_only():
    a=gate.read_owned(md.OUT/'sp_u7a_exposure_audit.json')
    m=gate.read_owned(md.OUT/'sp_u7a_model_audit.json')
    s=gate.read_owned(md.OUT/'sp_u7a_seed_audit.json')
    assert a['verdict']=='PASS' and a['candidate_exposure']=='E0' and a['outcome_values_decoded']==0
    assert m['verdict']=='PASS' and m['channel_coefficient_calls']==0 and m['scientific_predictions']==0
    assert s['historical_overlap_count']==0 and s['total_unique_values']==34501
    assert md.sha(md.OUT/'sp_u7a_future_seed_manifest.csv')==s['manifest_sha256']

