"""Necessary R2 technical regressions only; all run receipts are synthetic."""
import ast,copy,json,os,subprocess,sys,types
from pathlib import Path
import pytest
import base as b
import control as c
import r2_gate as g
import science as s

@pytest.fixture
def fixture(tmp_path,monkeypatch):
    cfg=b.contract();freeze=copy.deepcopy(g.bindings())
    root=tmp_path/'synthetic Root';here=root/'deployment/b5c_paired_r2_v1';here.mkdir(parents=True)
    old=root/'deployment/b5c_paired_v1';old.mkdir();r1=root/'deployment/b5c_paired_r1_v1';r1.mkdir()
    def put(q,v):q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps(v),encoding='utf8')
    for q,key,v in [(old/'package_manifest.json','original_B5C_manifest_sha256',{'synthetic':True}),
                    (r1/'package_manifest.json','R1_manifest_sha256',{'synthetic':True}),
                    (old/'contract.json','contract_sha256',cfg),(old/'data_manifest.json','data_manifest_sha256',{'synthetic':True})]:
        put(q,v);freeze[key]=b.sha(q)
    put(here/'bindings.json',freeze)
    failed=root/'lab_runs/b5c_paired';put(failed/'STOP.json',{'SYNTHETIC':True});put(failed/'FORMAL_START_CONSUMED.json',{'SYNTHETIC':True})
    (failed/'formal').mkdir();(failed/'formal/stderr.log').write_text('SYNTHETIC_FAILURE_ONLY',encoding='utf8')
    monkeypatch.setattr(b,'ROOT',root);monkeypatch.setattr(b,'HERE',here);monkeypatch.setattr(b,'OUT',root/'lab_runs/b5c_paired_r2')
    job=root/'lab_runs/b5c_paired_r1/technical/identity_0123456789ab';job.mkdir(parents=True)
    prior=g.original_inventory()
    put(job/'original_preservation_before.json',prior)
    put(job/'original_preservation_after.json',{'PASS':True,'changed':[],'files':len(prior)})
    report={'PASS':True,'real_scientific_seed_mismatches':0,'formal_predictions':0,'formal_scientific_arrays_deserialized':0,
        'lab_failed_files_opaque_hashes_preserved':True,'original_H_noise_pilot_checks_unchanged':True,
        **freeze['required_R1_counts'],**freeze['input_identity_digests'],'cells':freeze['cells'],'SYNTHETIC_TEST_ONLY':True}
    put(job/'identity_report.json',report)
    put(job/'status.json',{'PASS':True,'state':'COMPLETE','formal_predictions':0,'corrected_proof_passes':5000,'SYNTHETIC_TEST_ONLY':True})
    return root,job,put,freeze

def test_synthetic_R1_complete_gate_passes(fixture):
    root,job,_,_=fixture;v=g.check();assert v['PASS'] and v['job']==job.relative_to(root).as_posix();assert g.receipt_unchanged(v)
def test_missing_R1_gate_denies_before_model_or_data(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'ROOT',tmp_path)
    with pytest.raises(RuntimeError,match='COMPLETE_PASS_REQUIRED'):g.select_job()
@pytest.mark.parametrize('state,passed',[('RUNNING',True),('FAILED',False),('STOP',False),('COMPLETE',False)])
def test_nonpassed_status_is_never_selected(fixture,state,passed):
    _,job,put,_=fixture;put(job/'status.json',{'state':state,'PASS':passed})
    with pytest.raises(RuntimeError,match='COMPLETE_PASS_REQUIRED'):g.check()
@pytest.mark.parametrize('field,value',[
    ('PASS',False),('real_scientific_seed_mismatches',1),('formal_predictions',1),('formal_scientific_arrays_deserialized',1),
    ('lab_failed_files_opaque_hashes_preserved',False),('original_H_noise_pilot_checks_unchanged',False),
    ('corrected_proof_passes',4999),('raw_trajectories_checked',4999),('full_input_identities_checked',349999),
    ('scored_input_identities',209999),('H_identity_digest','0'*64),('noise_identity_digest','0'*64),('pilot_truth_Y_identity_digest','0'*64),('cells',[])])
def test_corrupt_or_incomplete_R1_report_rejected(fixture,field,value):
    _,job,put,_=fixture;v=b.read(job/'identity_report.json');v[field]=value;put(job/'identity_report.json',v)
    with pytest.raises(RuntimeError):g.check()
@pytest.mark.parametrize('name',g.R1_FILES)
def test_missing_prerequisite_file_rejected(fixture,name):
    _,job,_,_=fixture;(job/name).unlink()
    with pytest.raises(RuntimeError):g.check()
def test_original_failed_file_change_rejected(fixture):
    root,_,_,_=fixture;(root/'lab_runs/b5c_paired/formal/stderr.log').write_text('changed',encoding='utf8')
    with pytest.raises(RuntimeError,match='CHANGED_AFTER_R1'):g.check()
def test_old_STOP_cannot_be_removed(fixture):
    root,_,_,_=fixture;(root/'lab_runs/b5c_paired/STOP.json').unlink()
    with pytest.raises(RuntimeError,match='FAILED_RECORD_MISSING'):g.check()
@pytest.mark.parametrize('name',['outputs/sp_u12b5c_r1/windows_qa/identity_0123456789ab','lab_runs/b5c_paired_r1/technical/pytest_0123456789ab','../escape'])
def test_foreign_or_synthetic_QA_receipt_path_rejected(fixture,name):
    with pytest.raises(RuntimeError):g.check(name)
def test_gate_receipt_change_after_preflight_rejected(fixture):
    _,job,put,_=fixture;receipt=g.check();v=b.read(job/'status.json');v['extra']='changed';put(job/'status.json',v)
    with pytest.raises(RuntimeError,match='CHANGED_AFTER_PREFLIGHT'):g.receipt_unchanged(receipt)
@pytest.mark.parametrize('folder',['r6_formal','r7_sequential','r8_selection','final_mfa_once','b5c_paired','b5c_paired_r1'])
def test_other_execution_lock_rejected(fixture,folder):
    root,_,put,_=fixture;put(root/'lab_runs'/folder/'execution.lock',{'SYNTHETIC':True})
    with pytest.raises(RuntimeError,match='CONCURRENT'):c.absent()
def test_running_legacy_worker_rejected(fixture):
    root,_,put,_=fixture;folder=root/'lab_runs/r6_formal';put(folder/'active_worker.json',{'job':'lab_runs/r6_formal/final'})
    put(folder/'final/status.json',{'state':'RUNNING'})
    with pytest.raises(RuntimeError,match='CONCURRENT_ACTIVE'):c.concurrent()
@pytest.mark.parametrize('name',['FORMAL_START_CONSUMED.json','STOP.json','STOP_REQUESTED.json','execution.lock'])
def test_repeated_R2_start_or_STOP_rejected(fixture,name):
    root,_,put,_=fixture;put(b.OUT/name,{'SYNTHETIC':True})
    with pytest.raises(RuntimeError,match='ONE_TIME'):c.absent()
def test_preflight_missing_R1_never_reaches_model_or_data(fixture,monkeypatch):
    root,job,put,_=fixture;put(job/'status.json',{'state':'FAILED','PASS':False})
    monkeypatch.setattr(b,'verify',lambda:True)
    def forbidden(*args,**kw):raise AssertionError('SCIENTIFIC_PREFLIGHT_REACHED')
    monkeypatch.setattr(c,'environment',forbidden);monkeypatch.setattr(s,'modules',forbidden);monkeypatch.setattr(s,'final_identity',forbidden);monkeypatch.setattr(b,'verify_data',forbidden)
    with pytest.raises(RuntimeError,match='COMPLETE_PASS_REQUIRED'):c.preflight()
    assert not(b.OUT/'FORMAL_START_CONSUMED.json').exists()
def test_failed_new_attempt_consumed_once_preserves_old(fixture,monkeypatch):
    root,_,_,_=fixture;receipt=g.check();old=g.original_inventory()
    def pf(name=None):c.absent();return {'PASS':True,'R1_identity_audit':receipt}
    monkeypatch.setattr(c,'require_rehearsal',lambda:True)
    monkeypatch.setattr(c,'preflight',pf)
    monkeypatch.setattr(b,'sha',lambda p:'a'*64 if Path(p).name=='package_manifest.json' and Path(p).parent==b.HERE else original_sha(p))
    def fail(*args,**kw):raise RuntimeError('INTENTIONAL_SYNTHETIC_MONITOR_FAILURE')
    monkeypatch.setattr(c,'monitor',fail)
    with pytest.raises(RuntimeError,match='INTENTIONAL_SYNTHETIC'):c.start()
    assert (b.OUT/'STOP.json').is_file() and (b.OUT/'FORMAL_START_CONSUMED.json').is_file()
    assert b.read(b.OUT/'FORMAL_START_CONSUMED.json')['output']=='lab_runs/b5c_paired_r2/formal'
    assert not(b.OUT/'execution.lock').exists() and g.original_inventory()==old
    with pytest.raises(RuntimeError,match='ONE_TIME'):c.start()
original_sha=b.sha
def test_Stop_arriving_during_preflight_prevents_consumption(fixture,monkeypatch):
    _,_,put,_=fixture;receipt=g.check();old=g.original_inventory()
    def pf(name=None):
        c.absent();put(b.OUT/'STOP_REQUESTED.json',{'SYNTHETIC_STOP_DURING_PREFLIGHT':True})
        return {'PASS':True,'R1_identity_audit':receipt}
    monkeypatch.setattr(c,'require_rehearsal',lambda:True)
    monkeypatch.setattr(c,'preflight',pf)
    monkeypatch.setattr(c,'monitor',lambda *a,**k:pytest.fail('MONITOR_STARTED_AFTER_STOP'))
    with pytest.raises(RuntimeError,match='STOP_OR_CONSUMPTION_AFTER_PREFLIGHT'):c.start()
    assert not(b.OUT/'FORMAL_START_CONSUMED.json').exists() and not(b.OUT/'formal').exists()
    assert (b.OUT/'STOP.json').exists() and (b.OUT/'STOP_REQUESTED.json').exists()
    assert g.original_inventory()==old and not(b.OUT/'execution.lock').exists()
def test_direct_worker_cannot_bypass_receipt_before_model(tmp_path):
    code="""from pathlib import Path
import sys,base as b,science
b.ROOT=Path(sys.argv[1]);b.OUT=b.ROOT/'lab_runs/b5c_paired_r2'
def forbidden():raise AssertionError('MODEL_LOADED')
science.modules=forbidden
import worker
sys.argv=['worker.py','--action','evaluate','--job',str(b.OUT/'formal'),'--blas','6','--openmp','1']
worker.main()
"""
    env=os.environ.copy();env['PYTHONPATH']=str(b.HERE)
    r=subprocess.run([sys.executable,'-B','-c',code,str(tmp_path)],env=env,capture_output=True,text=True)
    assert r.returncode!=0 and 'CANONICAL_LOCKED_EXECUTION_REQUIRED' in r.stderr and 'MODEL_LOADED' not in r.stderr
    assert not(tmp_path/'lab_runs/b5c_paired_r2').exists()
def test_science_and_statistics_bytes_are_frozen():
    assert (b.HERE/'science.py').read_bytes()==(b.ROOT/'deployment/b5c_paired_r1_v1/science.py').read_bytes()
    for n in ('paired_stats.py','report.py','publication_plots.py'):assert (b.HERE/n).read_bytes()==(b.ROOT/'deployment/b5c_paired_v1'/n).read_bytes()
    p=b.contract();assert p['N']==1000 and p['score_frames']==list(range(4,10)) and len(p['conditions'])*len(p['snrs'])==35
    assert p['statistics']['replicates']==100000 and p['statistics']['family_size']==35
    assert p['statistics']['tail_probability']==.05/70 and p['selected']=={'K':1,'rank':16,'final_seed':4143049057,'iterations':22,'convergence':'CONVERGED'}
@pytest.mark.parametrize('role',s.ROLES)
def test_six_field_check_exact_and_rejects_each_wrong_seed(role):
    ix={('ID',0,-1,r):k+10 for k,r in enumerate(s.ROLES[:3])}
    ix.update({('ID',0,j,r):100+j*3+k for j in range(10) for k,r in enumerate(s.ROLES[3:])})
    cfg={'fold_hash':'f','pilot_geometry_hash':'g','operator_cache_sha256':'o'}
    proof={'condition':'ID','trajectory_id':0,'group':'EVAL','snr_db':b.contract()['snrs'],'estimators':b.contract()['B2_methods'],'scored_frames':list(range(4,10)),
      'rank':4,'window_frames':5,'fold_hash':'f','pilot_geometry_hash':'g','operator_sha256':'o',
      **{k:True for k in ('method_pairing','CRN_pairing','dense_lr4_shared_coordinates','finite_estimates','valid_coordinate_solve')},
      'channel_seeds':{r:ix['ID',0,-1 if r in s.ROLES[:3] else 0,r] for r in s.ROLES},
      'communication_seeds':[{r:ix['ID',0,-1 if r in s.ROLES[:3] else j,r] for r in s.ROLES} for j in range(10)],
      'inputs':[{'snr_db':v,'frame_id':j} for v in b.contract()['snrs'] for j in range(10)]}
    assert len(s.proof_check(proof,'ID',0,cfg,ix))==70
    proof['channel_seeds'][role]+=1
    with pytest.raises(RuntimeError,match='CHANNEL_SEED_IDENTITY_DRIFT'):s.proof_check(proof,'ID',0,cfg,ix)
def test_scope_rejects_old_attempt_reads_and_writes():
    code="""import control,base as b
control.scope(b.OUT/'synthetic',False)
for q,mode in [(b.ROOT/'lab_runs/b5c_paired/formal/raw/ID/0000.npz','rb'),(b.ROOT/'lab_runs/b5c_paired/STOP.json','wb'),(b.ROOT/'lab_runs/b5c_paired_r1/technical/status.json','wb'),(b.ROOT/'outputs/sp_u12b2b_sealed/raw_evaluation/ID/0000.npz','rb'),(b.ROOT/'outputs/sp_u12b3g_next_sealed/raw_evaluation/ID/0000.npz','rb'),(b.FINAL/'model/means.npy','wb')]:
 try:open(q,mode)
 except RuntimeError:continue
 raise AssertionError('FIREWALL_EXPANDED')
print('SIX_BOUNDARIES_DENIED')
"""
    env=os.environ.copy();env['PYTHONPATH']=str(b.HERE)
    r=subprocess.run([sys.executable,'-B','-c',code],env=env,capture_output=True,text=True)
    assert r.returncode==0,r.stderr;assert 'SIX_BOUNDARIES_DENIED' in r.stdout
def test_Run_requires_explicit_operator_switch(monkeypatch):
    monkeypatch.setattr(sys,'argv',['control.py','Run'])
    with pytest.raises(RuntimeError,match='EXPLICIT_FORMAL'):c.main()
