"""Synthetic selection records only; no formal validation values are read.

Admission/evidence/idle-process dependencies are substituted in this private
fixture process. Frozen R6 scoring, selection and selection_frozen are NEVER
substituted. Tiny arrays are not physical D=4032 models: the metadata ABI is
exercised separately from scientific model legality, supplied by real R1 audit
in production. No synthetic certificate is accepted by the production entry.
"""
from pathlib import Path
import sys,json,csv,hashlib,shutil,subprocess,copy,types,traceback
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf8')
def build(root,case):
    root=Path(root);root.mkdir(parents=True,exist_ok=False);new=root/'deployment/r8_selection_v1';new.mkdir(parents=True)
    original=json.loads((HERE/'trusted_inputs.json').read_text());deps={}
    # Original source bytes and static contract metadata only. No real model,
    # preparation payload, validation record, result archive or target copied.
    code_bindings=json.loads((HERE/'selection_contract.json').read_text())['code_bindings']
    for n,v in original['dependencies'].items():
        if n in code_bindings or n.endswith('.py')or n.startswith(('deployment/r6_formal_v1/','deployment/r7_sequential_v1/'))and n.endswith('.json')or n in('configs/sp_u12b5b_mfa_selection.yaml','outputs/sp_u12b5b_candidate_seed_registry.json','deployment/mfa_lab_v1/environment_source.json'):
            src=ROOT/n
            if src.is_file():dst=root/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(src.read_bytes());deps[n]=v
    for src in HERE.iterdir():
        if src.is_file()and src.name not in('preserve_personal.py','prepare_personal.py','package_manifest.json'):(new/src.name).write_bytes(src.read_bytes())
    dump(root/'SYNTHETIC_SELECTION_FIXTURE_ONLY.json',{'scope':'SYNTHETIC_ONLY','tiny_array_dimension':24,'R6_metadata_ABI_dimension':4032,'physical_4032_model_legality_attested':False,'human_approval':False})
    archive=root/'synthetic_archive.zip';archive.write_bytes(b'SYNTHETIC NOT REAL R7 ARCHIVE')
    evidence=root/'synthetic_evidence.zip';evidence.write_bytes(b'SYNTHETIC NOT REAL AUDIT EVIDENCE')
    sys.path[:0]=[str(new),str(root),str(root/'deployment/r6_formal_v1')]
    import numpy as np
    import common,model_ops as ops
    # These imports operate on unchanged source copies whose ROOT is the fixture.
    rows=original['candidate_order'];models={}
    for index,row in enumerate(rows):
        d=root/'lab_runs/r6_formal/candidates'/row['name']/'train';model=d/'model';model.mkdir(parents=True)
        K=row['K'];D=24;r=row['rank'];W=np.zeros((K,D,r),dtype=np.complex128);C=np.stack([np.eye(D,dtype=np.complex128)*.5 for _ in range(K)])
        arrays={'means':np.zeros((K,D),dtype=np.complex128),'lambdas':W,'psis':np.full((K,D),.5),'amps':np.full(K,1/K),'covs':C,'inv_covs':C*4}
        history=np.array([100,90,80,70,60,50,50.00001],dtype=np.float64)
        if index==8:history=np.arange(200,100,-1,dtype=np.float64)
        for n,v in arrays.items():np.save(model/(n+'.npy'),v,allow_pickle=False)
        np.save(model/'training_history.npy',history,allow_pickle=False)
        state=types.SimpleNamespace(n_components=K,D=D,M=r,**arrays);legal=ops.parameter_validity(state,full=True)
        meta={n:{'shape':list(v.shape),'dtype':v.dtype.str,'sha256':ops.array_hash(v)}for n,v in arrays.items()}
        m={'format':'R6_ORIGINAL_MFA_NPY_V1','policy':ops.FROZEN_POLICY,'K':K,'rank':r,'D':4032,'N':12000,
           'context':{'candidate':row['name'],'role':'train','seed':row['seed'],'execution_package_sha256':h(root/'deployment/r6_formal_v1/package_manifest.json'),
             'data_sha256':original['data']['outputs/sp_u12b5b/selection_train.npy']['sha256'],'synthetic_metadata_ABI_only':True},
           'arrays':meta,'initialization':copy.deepcopy(meta),'parameter_hash':common.object_sha(meta),'initialization_hash':common.object_sha(meta),
           'file_hashes':{p.name:h(p)for p in model.glob('*.npy')},'iterations':len(history),'convergence':ops.convergence(history.tolist())}
        dump(model/'model_manifest.json',m);models[row['name']]=m
        t={'PASS':True,'model_manifest_sha256':h(model/'model_manifest.json'),'seed':row['seed'],'convergence':m['convergence'],
           'parameter_validity':legal,'serialization_roundtrip':True,'all_components_smoke':{'PASS':True,'synthetic_ABI_placeholder_not_real_inference':True},'target_access':0}
        dump(d/'training_result.json',t);dump(d/'worker_result.json',{'PASS':True,'synthetic_only':True})
        dump(d/'status.json',{'state':'COMPLETE','PASS':True,'scientific_result_sha256':h(d/'training_result.json'),'worker_result_sha256':h(d/'worker_result.json')})
        vd=d.parent/'validate';vd.mkdir();frames=[];trajectories=[];cells=[]
        base=.125*(index+1)if index!=4 else .0625
        if case=='tie':base=.125
        for ci,(origin,snr)in enumerate(__import__('itertools').product(original['candidate_order']and ['Source/CDL-C-30','CDL-B-30','CDL-C-120'],(10,20))):
            E=np.zeros((100,6));T=np.full((100,6),float(2**ci));E[:]=T*base*(1+.125*ci)
            for i in range(100):
                for fi,frame in enumerate(range(4,10)):
                    frames.append({'candidate':row['name'],'K':K,'rank':r,'origin':origin,'snr_db':snr,'trajectory_id':400+i,'frame_id':frame,
                      'error_energy':float(E[i,fi]),'truth_energy':float(T[i,fi]),'pilot_sha256':'a'*64,'truth_sha256':'b'*64})
                trajectories.append({'candidate':row['name'],'K':K,'rank':r,'origin':origin,'snr_db':snr,'trajectory_id':400+i,'error_energy':float(E[i].sum()),'truth_energy':float(T[i].sum()),'linear_ratio':ops.pooled_ratio(E[i],T[i])})
            cells.append({'origin':origin,'snr_db':snr,'numerator':float(E.sum(dtype=np.float64)),'denominator':float(T.sum(dtype=np.float64)),
                          'linear_pooled_NMSE':ops.pooled_ratio(E,T),'trajectories':100,'frames':600})
        for n,values in [('frame_audit.csv',frames),('trajectory_audit.csv',trajectories)]:
            with(vd/n).open('w',encoding='utf8',newline='')as f:w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
        v={'PASS':True,**row,'cells':cells,'macro_selection_score':ops.macro_score([x['linear_pooled_NMSE']for x in cells]),'prediction_count':3600,
           'model_manifest_sha256':t['model_manifest_sha256'],'frame_audit_sha256':h(vd/'frame_audit.csv'),'trajectory_audit_sha256':h(vd/'trajectory_audit.csv'),
           'input_hashes':{n:x['sha256']for n,x in original['data'].items()if 'validation_'in n},'target_access':0}
        dump(vd/'validation_result.json',v);dump(vd/'worker_result.json',{'PASS':True,'synthetic_only':True});dump(vd/'status.json',{'state':'COMPLETE','PASS':True,'scientific_result_sha256':h(vd/'validation_result.json'),'worker_result_sha256':h(vd/'worker_result.json')})
    dump(root/'lab_runs/r7_sequential/queue.json',{'state':'QUEUE_COMPLETE','current_stage':'STOP_AFTER_NINE','completed_candidates':[x['name']for x in rows],'remaining_candidates':[],'STOP_reason':None})
    dump(root/'lab_runs/r7_sequential/completion.json',{'PASS':True,'selection_performed':False,'final_fit_performed':False})
    settings=original|{'dependencies':deps,'live_archive_files':{},'legacy_run_files':{},'models':models,'archive_SHA256':h(archive)}
    def refresh():
        settings['live_archive_files']={p.relative_to(root).as_posix():{'sha256':h(p),'bytes':p.stat().st_size}for n in('lab_runs/r6_formal/candidates','lab_runs/r7_sequential')for p in(root/n).rglob('*')if p.is_file()}
        dump(new/'trusted_inputs.json',settings)
        files={p.relative_to(root).as_posix():{'sha256':h(p),'bytes':p.stat().st_size}for p in new.iterdir()if p.is_file()and p.name!='package_manifest.json'};dump(new/'package_manifest.json',{'files':files,'synthetic_fixture_only':True})
    def update_result(vd,v):
        dump(vd/'validation_result.json',v);s=json.loads((vd/'status.json').read_text());s['scientific_result_sha256']=h(vd/'validation_result.json');dump(vd/'status.json',s)
    first=root/'lab_runs/r6_formal/candidates'/rows[0]['name'];vd=first/'validate'
    if case=='bad_model_state':
        t=json.loads((first/'train/training_result.json').read_text());t['parameter_validity']['PASS']=False;dump(first/'train/training_result.json',t);s=json.loads((first/'train/status.json').read_text());s['scientific_result_sha256']=h(first/'train/training_result.json');dump(first/'train/status.json',s)
    elif case=='wrong_seed':
        t=json.loads((first/'train/training_result.json').read_text());t['seed']+=1;dump(first/'train/training_result.json',t);s=json.loads((first/'train/status.json').read_text());s['scientific_result_sha256']=h(first/'train/training_result.json');dump(first/'train/status.json',s)
    elif case in('duplicate_identity','missing_cell','wrong_cell_metric'):
        v=json.loads((vd/'validation_result.json').read_text())
        if case=='missing_cell':v['cells'].pop()
        elif case=='wrong_cell_metric':v['cells'][0]['numerator']+=1
        else:
            with(vd/'frame_audit.csv').open(encoding='utf8',newline='')as f:values=list(csv.DictReader(f))
            values[-1]['frame_id']='4'
            with(vd/'frame_audit.csv').open('w',encoding='utf8',newline='')as f:w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
            v['frame_audit_sha256']=h(vd/'frame_audit.csv')
        update_result(vd,v)
    refresh()
    if case=='missing_candidate':(first/'train/model/model_manifest.json').unlink()
    elif case=='corrupt_hash':(first/'train/model/means.npy').write_bytes(b'CORRUPTED SYNTHETIC ONLY')
    elif case=='stop':dump(root/'lab_runs/r6_formal/STOP.json',{'SYNTHETIC':True})
    elif case=='concurrent':(root/'lab_runs/r7_sequential/execution.lock').write_text('SYNTHETIC LOCK')
    elif case=='existing_sha':(root/'lab_runs/r6_formal/SELECTION_MANIFEST.sha256').write_text('SYNTHETIC PARTIAL')
    return settings,evidence,archive
def execute(root,case):
    import selection_core as a
    settings,evidence,archive=build(root,case);root=Path(root).resolve()
    # Use the cloned wrapper modules in a fresh subprocess, not module rebinding.
    child=root/'deployment/r8_selection_v1/toy_case.py'
    code=subprocess.run([sys.executable,'-B','-X','utf8',str(child),'--inside',case],capture_output=True,text=True,encoding='utf8')
    (root/'child_stdout.log').write_text(code.stdout,encoding='utf8');(root/'child_stderr.log').write_text(code.stderr,encoding='utf8')
    if code.returncode:print(code.stdout+code.stderr);raise RuntimeError('SYNTHETIC_CHILD_FAILED')
    print(code.stdout,end='')
def inside(case):
    import selection_core as a
    from authorization import check,GRANTS,ACTION
    c,ops,r,_,_=a.modules();c.environment()
    job=a.OUT/'runs/synthetic_case';job.mkdir(parents=True);evidence=a.ROOT/'synthetic_evidence.zip';archive=a.ROOT/'synthetic_archive.zip'
    instruction=b'SYNTHETIC_TEST_ONLY_NO_HUMAN_APPROVAL'
    record={'format':'R8_SELECTION_ONLY_HUMAN_ATTESTATION_V1','status':'SYNTHETIC_TEST_APPROVED','scope':ACTION,'allowed_actions':[ACTION],
      'grants':GRANTS.copy(),'bindings':a.bindings(),'approval_id':'SYNTHETIC_ONLY','approved_by':'SYNTHETIC_ONLY','approved_UTC':'2000-01-01T00:00:00+00:00',
      'mentor_instruction_reference':'SYNTHETIC_ONLY','mentor_instruction_SHA256':hashlib.sha256(instruction).hexdigest(),'no_automatic_retry':True,'maximum_executions':1}
    auth=a.OUT/'authorization/synthetic_approval.json';text=auth.with_suffix('.txt');dump(auth,record);text.write_bytes(instruction)
    original_authorize=a.authorize
    def synthetic_authorize(p,i,digest):
        receipt=check(a.read(p),a.bindings(),Path(i).read_bytes(),synthetic=True)
        return receipt|{'record_SHA256':h(p),'instruction_SHA256':h(i),'synthetic_only':True}
    if case not in('unauthorized','legacy_r7'):a.authorize=synthetic_authorize
    if case=='legacy_r7':dump(auth,{'status':'MENTOR_APPROVED','scope':'R7_NINE_CANDIDATE_TRAIN_VALIDATE_ONLY','allowed_actions':['train','validate'],'selection_authorized':False,'final_training_authorized':False,'target_access_authorized':False})
    # Admission substitutes only: no real audit certificate is fabricated.
    a.proof=lambda p:{'PASS':True,'synthetic_admission_only':True,'sha256':h(p)}
    a.idle_processes=lambda:{'PASS':True,'synthetic_idle_only':True}
    blocked={};result=None;error=None
    if case=='firewall':
        prep=a.ROOT/'outputs/sp_u12b5b/selection_train.npy';prep.parent.mkdir(parents=True,exist_ok=True);prep.write_bytes(b'SYNTHETIC ONLY')
        b2=a.ROOT/'outputs/sp_u12b2b_sealed/sentinel.txt';b2.parent.mkdir(parents=True);b2.write_text('SYNTHETIC ONLY')
        b3=a.ROOT/'outputs/sp_u12b3g_next_sealed/sentinel.txt';b3.parent.mkdir(parents=True);b3.write_text('SYNTHETIC ONLY')
        a.install_guard(job,evidence,archive,(auth,text))
        model=a.R6OUT/'candidates'/a.cfg()['candidate_order'][0]['name']/'train/model/means.npy'
        calls={'preparation_decode':lambda:__import__('numpy').load(prep),'B2_read':lambda:b2.read_bytes(),'B3_read':lambda:b3.read_bytes(),
          'model_write':lambda:model.write_bytes(b'X'),'validation_write':lambda:(model.parents[2]/'validate/frame_audit.csv').write_bytes(b'X'),
          'legacy_complete_selection':r.complete_selection,'final_retrain':lambda:r.start('final','K1_r4'),
          'validation_rerun':lambda:r.start('validate','K1_r4'),'selection_file_unauthorized_write':lambda:(a.R6OUT/'SELECTION_MANIFEST.json').write_text('x')}
        for name,call in calls.items():
            try:call();blocked[name]=False
            except RuntimeError:blocked[name]=True
        assert all(blocked.values()),blocked
    else:
        a.install_guard(job,evidence,archive,(auth,text))
        try:result=a.complete_once(job,evidence,archive,auth,text,h(auth))
        except BaseException as e:error={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
        good=case in('success','tie','repeat')
        if good:
            assert result and result['PASS'],error
            expected='K1_r4'if case=='tie'else'K2_r8';assert result['selected']['name']==expected
            a._select=True
            try:compatible=r.selection_frozen()
            finally:a._select=False
            assert compatible['selected']==result['selected']and compatible['coverage']['scored_frame_predictions']==32400
            assert compatible['candidates'][-1]['PASS'] # MAX_ITER candidate is retained.
            if case=='repeat':
                try:a.complete_once(job,evidence,archive,auth,text,h(auth));raise AssertionError('duplicate accepted')
                except RuntimeError as e:assert 'ALREADY_CONSUMED'in str(e)
        else:
            assert error and result is None,(case,result,error)
            assert not(a.R6OUT/'SELECTION_MANIFEST.json').exists()
    v={'PASS':True,'case':case,'synthetic_only':True,'real_selection_executed':False,'final_model_trained':False,'target_access':0,
       'selected':result['selected']if result else None,'expected_rejection':error,'firewall':blocked,
       'R6_selection_frozen_compatibility':bool(result),'admission_dependencies_substituted_only_in_private_test':True,'frozen_math_functions_substituted':False}
    a.write(job/'toy_result.json',v,True);print(json.dumps(v,ensure_ascii=False),flush=True)
if __name__=='__main__':
    if sys.argv[1]=='--inside':inside(sys.argv[2])
    else:execute(Path(sys.argv[1]).resolve(),sys.argv[2])
