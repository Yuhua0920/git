"""Meaningful analytic/corruption/permission tests; never fit or real validation."""
from pathlib import Path
import sys,json,subprocess,ast,os
import pytest,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import audit_core as a
from synthetic import fixture
ROW={'name':'K2_r4','K':2,'rank':4,'seed':1443235906}
def made(tmp_path,row=ROW,history=None):
    d=tmp_path/row['name']/'train/model';return d,fixture(d,row,history)
def rewrite(d,m,name,value):
    with(d/(name+'.npy')).open('wb')as f:np.save(f,value,allow_pickle=False)
    m['file_hashes'][name+'.npy']=a.sha(d/(name+'.npy'))
    if name in a.PUBLIC:m['arrays'][name]={'shape':list(value.shape),'dtype':value.dtype.str,'sha256':a.array_sha(value)}
    m['parameter_hash']=a.objsha(m['arrays']);(d/'model_manifest.json').write_text(json.dumps(m),encoding='utf8')
@pytest.mark.parametrize('K,r',[(K,r)for K in(1,2,4)for r in(4,8,16)])
def test_nine_analytic_shapes_and_frozen_all_component_smoke(tmp_path,K,r):
    row={'name':f'K{K}_r{r}','K':K,'rank':r,'seed':1};d,m=made(tmp_path,row);old={p.name:a.sha(p)for p in d.iterdir()}
    v=a.model_check(d,row,m,True);assert v['PASS']and len(v['files'])==7 and v['smoke']['component_calls']==K
    assert v['smoke']['API_calls']==[['float',1.0]]and old=={p.name:a.sha(p)for p in d.iterdir()}
@pytest.mark.parametrize('name',['means','lambdas','psis','amps','covs','inv_covs','training_history'])
def test_each_missing_npy_fails(tmp_path,name):
    d,m=made(tmp_path);(d/(name+'.npy')).unlink()
    with pytest.raises(a.AuditFailure,match='MISSING_NPY'):a.model_check(d,ROW,m,True)
def test_original_bytes_corrupted_even_valid_save_declaration_fails(tmp_path):
    d,m=made(tmp_path)
    with(d/'covs.npy').open('r+b')as f:f.seek(-1,2);x=f.read(1);f.seek(-1,2);f.write(bytes([x[0]^1]))
    with pytest.raises(a.AuditFailure,match='MODEL_FILE_SHA_DRIFT'):a.model_check(d,ROW,m,True)
def test_payload_hash_is_independent_of_npy_file_hash(tmp_path):
    d,m=made(tmp_path);m['arrays']['means']['sha256']='0'*64;m['parameter_hash']=a.objsha(m['arrays']);(d/'model_manifest.json').write_text(json.dumps(m),encoding='utf8')
    with pytest.raises(a.AuditFailure,match='MODEL_PAYLOAD_SHA_DRIFT'):a.model_check(d,ROW,m,True)
@pytest.mark.parametrize('fault',['weight','isotropy','floor','covariance','inverse_nonfinite','mean_nonfinite'])
def test_independent_numeric_checks_reject_hash_consistent_bad_models(tmp_path,fault):
    d,m=made(tmp_path)
    name={'weight':'amps','isotropy':'psis','floor':'psis','covariance':'covs','inverse_nonfinite':'inv_covs','mean_nonfinite':'means'}[fault]
    v=np.load(d/(name+'.npy'),allow_pickle=False)
    if fault=='weight':v[0]=-1.;v[1]=2.
    elif fault=='isotropy':v[0,1]=.6
    elif fault=='floor':v[:]=0.
    elif fault=='covariance':v[0,0,0]+=.1
    else:v.flat[0]=np.nan
    rewrite(d,m,name,v)
    with pytest.raises(RuntimeError,match='ILLEGAL_MODEL_PARAMETERS'):a.model_check(d,ROW,m,True)
def test_history_first_allowed_update_is_seven():
    v=a.first_stop([100,90,80,70,60,60.00001,60.00002]);assert v['first_stop_update']==7
def test_continuation_beyond_first_stop_fails():
    with pytest.raises(a.AuditFailure,match='FIRST_AUTHOR_STOP'):a.first_stop([100,90,80,70,60,50,50.00001,50.00002])
def test_finite_decreasing_maxiter_is_legal_not_monotonic_requirement():
    v=a.first_stop(np.arange(200,100,-1,dtype=float));assert v['convergence']=='MAX_ITER_REACHED'and len(v['decrease_events'])==99
def test_exact_npy_history_jsonl_is_checked(tmp_path):
    d,m=made(tmp_path);p=d.parent/'iteration_history.jsonl';p.write_text(p.read_text(encoding='utf8').replace('50.00001','50.00002'),encoding='utf8')
    with pytest.raises(a.AuditFailure,match='HISTORY_NPY_JSONL'):a.model_check(d,ROW,m,True)
@pytest.mark.parametrize('path', ["../x", "C:/x", "folder\\file", "/outside"])
def test_relpath_traversal_rejected(tmp_path,path):
    with pytest.raises(a.AuditFailure):a.safe(tmp_path,path)
def test_readonly_mmap_and_c_order_reject_fortran(tmp_path):
    d,m=made(tmp_path);v=np.asfortranarray(np.load(d/'means.npy',allow_pickle=False))
    with(d/'means.npy').open('wb')as f:np.save(f,v,allow_pickle=False)
    m['file_hashes']['means.npy']=a.sha(d/'means.npy');(d/'model_manifest.json').write_text(json.dumps(m),encoding='utf8')
    with pytest.raises(a.AuditFailure,match='NPY_LAYOUT'):a.model_check(d,ROW,m,True)
def test_accepted_metric_evidence_does_not_recompute_validation():
    v=a.evidence_binding(a.inputs());assert v['new_metric_calculations']==v['new_real_validation_inference']==0
def test_missing_physical_models_cannot_pass(tmp_path):
    row=a.inputs()['candidate_order'][0];d=tmp_path/'missing_lab/candidates'/row['name']/'train/model'
    with pytest.raises(FileNotFoundError):a.model_check(d,row,a.inputs()['models'][row['name']],False)
def test_boundary_expected_lab_absence_is_not_fake_seal_pass(tmp_path):
    q={'state':'QUEUE_COMPLETE','current_stage':'STOP_AFTER_NINE','completed_candidates':['x'],'remaining_candidates':[],'STOP_reason':None}
    d=tmp_path/'lab_runs/r7_sequential';d.mkdir(parents=True);(d/'queue.json').write_text(json.dumps(q),encoding='utf8');(d/'completion.json').write_text(json.dumps({'PASS':True,'selection_performed':False,'final_fit_performed':False}),encoding='utf8')
    v=a.old_is_idle(tmp_path,{'candidate_order':[{'name':'x'}]});assert v['B2_B3_lab']=='NOT_PRESENT_BY_DESIGN_NOT_PHYSICALLY_VERIFIED'
    (tmp_path/'outputs/sp_u12b2b_sealed').mkdir(parents=True)
    with pytest.raises(a.AuditFailure,match='UNEXPECTED_SEALED_TARGET'):a.old_is_idle(tmp_path,{'candidate_order':[{'name':'x'}]})
def test_new_audit_scope_denies_legacy_write_preparation_decode_and_fit(tmp_path):
    # Isolated child: Python audit hooks/profile cannot be removed safely in pytest process.
    code='''import sys,json,pathlib,numpy
sys.path.insert(0,sys.argv[1]);import audit_core as a
job=pathlib.Path(sys.argv[2]);cfg=a.inputs();a.install_guard(a.ROOT,cfg,job)
checks=[]
for action in (lambda:(a.ROOT/'lab_runs/r6_formal/SHOULD_NOT_EXIST.json').write_text('x',encoding='utf8'),lambda:__import__('numpy').load(a.ROOT/'outputs/sp_u12b5b/selection_train.npy',allow_pickle=False)):
 try:action();raise AssertionError('permission bypass')
 except a.AuditFailure:checks.append(True)
def fit():raise AssertionError('fit reached')
fit.__code__=fit.__code__.replace(co_filename=str(a.ROOT/'src/forbidden_fit.py'))
try:fit();raise AssertionError('fit bypass')
except a.AuditFailure:checks.append(True)
assert checks==[True,True,True]
print('GUARD_PASS')
'''
    r=subprocess.run([sys.executable,'-B','-X','utf8','-c',code,str(a.HERE),str(tmp_path)],capture_output=True,text=True,encoding='utf8');assert r.returncode==0,r.stdout+r.stderr;assert 'GUARD_PASS'in r.stdout
def test_no_scientific_fit_validation_or_selection_calls_in_new_tool():
    banned={'fit','fit_reference','automatic_select','complete_selection','train','validate','startfinal','startcandidate','validatecandidate'}
    for name in('audit_core.py','audit_cli.py','audit_worker.py','audit_monitor.py','synthetic.py'):
        tree=ast.parse((a.HERE/name).read_text(encoding='utf8'))
        for n in ast.walk(tree):
            if isinstance(n,ast.Call):
                f=n.func;label=f.id if isinstance(f,ast.Name)else f.attr if isinstance(f,ast.Attribute)else''
                assert label.lower()not in banned,(name,label)
def test_failed_run_export_cannot_export_any_model(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'OUT',tmp_path);job=tmp_path/'failed';job.mkdir();a.write(job/'status.json',{'state':'FAILED','PASS':False});a.write(job/'report.json',{'verdict':'FROZEN_SELECTION_NOT_RELEASED'})
    r=a.export_job(job);assert r['PASS']and r['models_exported']==0
    import zipfile
    with zipfile.ZipFile(r['path'])as z:assert not any(n.endswith('.npy')for n in z.namelist())
@pytest.mark.parametrize('all_models',[False,True])
def test_original_45_or_63_npy_export_bytes(tmp_path,monkeypatch,all_models):
    root=tmp_path/'analytic_lab';out=root/'lab_runs/r7_evidence_audit';job=out/'success_fixture';job.mkdir(parents=True)
    cfg={'candidate_order':a.inputs()['candidate_order'],'models':{}}
    for row in cfg['candidate_order']:
        cfg['models'][row['name']]=fixture(root/'lab_runs/r6_formal/candidates'/row['name']/'train/model',row)
    monkeypatch.setattr(a,'ROOT',root);monkeypatch.setattr(a,'OUT',out);monkeypatch.setattr(a,'inputs',lambda:cfg)
    a.write(job/'status.json',{'state':'COMPLETE','PASS':True,'synthetic_fixture_only':True});a.write(job/'report.json',{'verdict':'FROZEN_SELECTION_EVIDENCE_READY_REQUIRES_SEPARATE_APPROVAL','synthetic_fixture_only':True})
    result=a.export_job(job,all_models);assert result['models_exported']==(63 if all_models else 45)
    import zipfile,hashlib
    with zipfile.ZipFile(result['path'])as z:
        names=[n for n in z.namelist()if n.endswith('.npy')];assert len(names)==(63 if all_models else 45)
        for n in names:
            _,candidate,file=n.split('/');assert hashlib.sha256(z.read(n)).hexdigest()==cfg['models'][candidate]['file_hashes'][file]
def test_failed_model_is_not_reported_as_pass_and_no_auto_retry():
    source=(a.HERE/'audit_worker.py').read_text(encoding='utf8');assert 'FROZEN_SELECTION_NOT_RELEASED'in source
    assert set(a.GATES)=={'ARCHIVE_INTEGRITY','MODEL_ARTIFACTS_AND_VALIDITY','CANONICAL_VALIDATION_AND_METRIC','FROZEN_PROTOCOL_AND_SEEDS','RESOURCE_AND_STOP','POST_RUN_PRESERVATION'}
