"""Selection-only wrapper around frozen R6 arithmetic and reader-compatible output."""
from pathlib import Path
import sys,os,json,hashlib,csv,itertools,contextlib,datetime,zipfile,time
from functools import lru_cache
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'lab_runs/r8_selection'
R6OUT=ROOT/'lab_runs/r6_formal'
PUBLIC=('means','lambdas','psis','amps','covs','inv_covs','training_history')
GATES=('ARCHIVE_INTEGRITY','MODEL_ARTIFACTS_AND_VALIDITY','CANONICAL_VALIDATION_AND_METRIC','FROZEN_PROTOCOL_AND_SEEDS','RESOURCE_AND_STOP','POST_RUN_PRESERVATION')
_opaque=False;_commit=False;_select=False
def require(ok,reason):
    if not ok:raise RuntimeError(reason)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def safe(root,name):
    require(isinstance(name,str)and ':'not in name and '\\'not in name and '..'not in Path(name).parts,'UNSAFE_RELATIVE_PATH')
    p=(Path(root)/name).resolve();require(p.is_relative_to(Path(root).resolve()),'PATH_TRAVERSAL');return p
def sha(p):
    global _opaque
    previous=_opaque;_opaque=True
    try:
        h=hashlib.sha256()
        with Path(p).open('rb')as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        return h.hexdigest()
    finally:_opaque=previous
def objsha(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(p,v,exclusive=False):
    p=Path(p).resolve();require(p.is_relative_to(OUT.resolve()),'R8_OUTPUT_ONLY');p.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    if exclusive:
        with p.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
    else:
        tmp=p.with_suffix(p.suffix+'.tmp')
        with tmp.open('wb')as f:f.write(data);f.flush();os.fsync(f.fileno())
        # Windows readers can momentarily deny FILE_SHARE_DELETE. Retry only
        # publication of a mutable R8 log/status file, never a scientific job,
        # authorization decision, consumption marker or selection manifest.
        for attempt in range(101):
            try:os.replace(tmp,p);break
            except PermissionError as e:
                if getattr(e,'winerror',None)not in(5,32,33)or attempt==100:raise
                time.sleep(.01)
@lru_cache(maxsize=1)
def cfg():return read(HERE/'trusted_inputs.json')
@lru_cache(maxsize=1)
def contract():return read(HERE/'selection_contract.json')
def verify():
    m=read(HERE/'package_manifest.json')
    for n,v in m['files'].items():
        p=safe(ROOT,n);require(p.is_relative_to(HERE)and p.stat().st_size==v['bytes']and sha(p)==v['sha256'],'R8_TOOL_DRIFT:'+n)
    for n in contract()['code_bindings']:
        require(sha(ROOT/n)==cfg()['dependencies'][n]['sha256'],'FROZEN_CODE_OR_PROTOCOL_DRIFT:'+n)
    return {'PASS':True,'R8_package_manifest_SHA256':sha(HERE/'package_manifest.json'),'real_selection_executed':False}
def bindings():
    v=verify();p=contract()
    return p['frozen_bindings']|{'R8_package_manifest_SHA256':v['R8_package_manifest_SHA256'],
      'R8_selection_contract_SHA256':sha(HERE/'selection_contract.json'),'R8_authorization_schema_SHA256':sha(HERE/'authorization_schema.json')}
def modules():
    verify();from source_only_loader import install
    install(ROOT,cfg()['dependencies']|read(HERE/'package_manifest.json')['files'])
    sys.path[:0]=[str(ROOT),str(ROOT/'deployment/r6_formal_v1'),str(ROOT/'deployment/r7_sequential_v1')]
    import common,model_ops,runner,windows_resources,r7_common
    return common,model_ops,runner,windows_resources,r7_common
def authorize(path,instruction,expected_record_sha,synthetic=False):
    require(path is not None and instruction is not None and expected_record_sha is not None,'R8_SELECTION_NOT_AUTHORIZED')
    p=Path(path).resolve();i=Path(instruction).resolve();base=(OUT/'authorization').resolve()
    require(p.is_relative_to(base)and i.is_relative_to(base),'R8_INDEPENDENT_AUTHORIZATION_DIRECTORY_REQUIRED')
    require(p.suffix=='.json'and i.suffix=='.txt'and p.stat().st_size<=65536 and i.stat().st_size<=262144,'R8_APPROVAL_METADATA_ONLY')
    require(isinstance(expected_record_sha,str)and len(expected_record_sha)==64 and sha(p)==expected_record_sha,'R8_EXTERNAL_APPROVAL_SHA256_MISMATCH')
    from authorization import check
    record=read(p);receipt=check(record,bindings(),i.read_bytes(),synthetic)
    return receipt|{'record_SHA256':sha(p),'instruction_SHA256':sha(i),'record_timestamp_generated_by_program':False}
def proof(zip_path):
    p=Path(zip_path).resolve();require(p.is_relative_to(ROOT.resolve()),'EVIDENCE_MUST_STAY_IN_PROJECT')
    require(sha(p)==contract()['frozen_bindings']['R7AuditR1_evidence_zip_SHA256'],'R7AUDITR1_EVIDENCE_SHA256_MISMATCH')
    with zipfile.ZipFile(p)as z:
        names=z.namelist();require(len(names)==len(set(names)),'AUDIT_ZIP_DUPLICATES')
        m=json.loads(z.read('export_manifest.json'));require(set(names)==set(m['files'])|{'export_manifest.json'},'AUDIT_ZIP_INVENTORY')
        for n,v in m['files'].items():
            h=hashlib.sha256();size=0
            with z.open(n)as f:
                for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b);size+=len(b)
            require(h.hexdigest()==v['sha256']and size==v['bytes'],'AUDIT_ZIP_MEMBER_DRIFT:'+n)
        r=json.loads(z.read('final_report.json'));s=json.loads(z.read('status.json'))
        require(r['verdict']=='FROZEN_SELECTION_EVIDENCE_READY_REQUIRES_SEPARATE_APPROVAL'and set(r['gates'])==set(GATES)and all(x['status']=='PASS'for x in r['gates'].values()),'SIX_AUDIT_GATES_NOT_PASS')
        require(s['state']=='COMPLETE'and s['PASS']and s['mode']=='lab'and s['exit_code']==0 and s['lab_models_verified']==9 and r['audit_monitor']['PASS'],'AUDIT_NOT_REAL_LAB_COMPLETE')
        require(m['files']['tool_snapshot/package_manifest.json']['sha256']==contract()['frozen_bindings']['R7AuditR1_package_manifest_SHA256'],'AUDIT_TOOL_FREEZE_DRIFT')
        require(len(r['models'])==9 and{v['candidate']for v in r['models']}=={v['name']for v in cfg()['candidate_order']},'AUDIT_NINE_MODELS_MISSING')
        for v in r['models']:
            expected=cfg()['models'][v['candidate']]
            require(v['PASS']and not v['synthetic_fixture']and v['parameter_hash']==expected['parameter_hash']and v['validity']['PASS']and v['validity']['full_covariance_check']and v['smoke']['PASS'],'ACTUAL_MODEL_AUDIT_INVALID')
            require(len(v['files'])==7 and{f['filename']:f['file_sha256']for f in v['files']}==expected['file_hashes'],'AUDIT_MODEL_SEVEN_FILE_BINDING')
    return {'PASS':True,'sha256':sha(p),'six_gates':'PASS','actual_model_records':9,'actual_NPY_hash_records':63,'numerical_model_payloads_decoded':False}
def unchanged_file(root,n,v):
    p=safe(root,n);require(p.is_file()and sha(p)==v['sha256']and('bytes'not in v or p.stat().st_size==v['bytes']),'FROZEN_INPUT_OR_RESULT_HASH_DRIFT:'+n)
    return {'sha256':v['sha256'],'bytes':p.stat().st_size}
def absence(owned_r6_lock=False):
    for d in ('r6_formal','r7_sequential','r7_evidence_audit','r7_evidence_audit_r1','r8_selection'):
        for name in ('STOP.json','STOP_REQUESTED.json','STOP_AUDIT_REQUESTED.json'):
            require(not(ROOT/'lab_runs'/d/name).exists(),'STOP_PRESENT_NO_RETRY:'+d+'/'+name)
    paths=['lab_runs/r7_sequential/execution.lock','lab_runs/r7_evidence_audit/audit_execution.lock','lab_runs/r7_evidence_audit_r1/audit_execution.lock']
    if not owned_r6_lock:paths+=['lab_runs/r6_formal/pipeline.lock']
    for n in paths:require(not(ROOT/n).exists(),'CONCURRENT_OR_STALE_LOCK:'+n)
    for n in ('SELECTION_MANIFEST.json','SELECTION_MANIFEST.sha256','final','FINAL_MODEL_MANIFEST.json'):
        require(not(R6OUT/n).exists(),'SELECTION_OR_FINAL_ALREADY_EXISTS_NO_OVERWRITE:'+n)
    for n in ('outputs/sp_u12b2b_sealed','outputs/sp_u12b3g_next_sealed'):
        require(not(ROOT/n).exists(),'TARGET_DATA_PRESENT_STOP_WITHOUT_READ:'+n)
def idle_processes():
    *_,q=modules();identities=[read(ROOT/'lab_runs/r7_sequential/queue.json')['controller_identity']]
    for row in cfg()['candidate_order']:
        for action in ('train','validate'):
            d=R6OUT/'candidates'/row['name']/action;s=read(d/'status.json');found=None
            with(d/'resources.jsonl').open(encoding='utf8')as f:
                for line in f:
                    for x in json.loads(line)['process_tree']['members']:
                        if x['PID']==s['actual_interpreter_PID']:found={'PID':x['PID'],'creation_time_100ns':x['creation_time_100ns']};break
                    if found:break
            require(found is not None,'HISTORICAL_WORKER_IDENTITY_MISSING');identities.append(found)
    require(not any(q.process_identity(x['PID'])==x for x in identities),'SCIENTIFIC_PROCESS_STILL_ALIVE')
    return {'PASS':True,'identities_checked':len(identities)}
def preflight(evidence,archive,owned_r6_lock=False):
    absence(owned_r6_lock);c,ops,r,_,_=modules();settings=cfg()
    require(sha(archive)==settings['archive_SHA256'],'R7_ORIGINAL_ARCHIVE_SHA256_MISMATCH')
    evidence_result=proof(evidence)
    q=read(ROOT/'lab_runs/r7_sequential/queue.json');done=read(ROOT/'lab_runs/r7_sequential/completion.json')
    require(q['state']=='QUEUE_COMPLETE'and q['current_stage']=='STOP_AFTER_NINE'and q['completed_candidates']==[x['name']for x in settings['candidate_order']]and not q['remaining_candidates']and q.get('STOP_reason')is None,'R7_NINE_QUEUE_NOT_COMPLETE')
    require(done['PASS']and done['selection_performed']is False and done['final_fit_performed']is False,'R7_COMPLETION_INVALID')
    hashes={}
    for mapping in (settings['dependencies'],settings['live_archive_files'],settings['legacy_run_files']):
        for n,v in mapping.items():hashes[n]=unchanged_file(ROOT,n,v)
    all_frames=[]
    for row in settings['candidate_order']:
        d=R6OUT/'candidates'/row['name']/'train/model'
        require(read(d/'model_manifest.json')==settings['models'][row['name']],'FROZEN_MODEL_MANIFEST_CHANGED')
        for name,h in settings['models'][row['name']]['file_hashes'].items():
            n=(d/name).relative_to(ROOT).as_posix();hashes[n]=unchanged_file(ROOT,n,{'sha256':h})
        t=r.stage_result(d.parent,'training_result.json')
        require(t['convergence']in('CONVERGED','MAX_ITER_REACHED')and t['parameter_validity']['PASS']and t['parameter_validity']['full_covariance_check']and t['serialization_roundtrip']and t['all_components_smoke']['PASS'],'ILLEGAL_MODEL_STATE')
        v,frames=r.valid_candidate(row)
        require(v['target_access']==0 and t.get('target_access',0)==0 and all(x['candidate']==row['name']for x in frames),'TARGET_OR_CANDIDATE_IDENTITY_INVALID')
        expected_inputs={n:x['sha256']for n,x in settings['data'].items()if 'validation_'in n}
        require(v['input_hashes']==expected_inputs,'VALIDATION_INPUT_HASH_DRIFT')
        all_frames.extend(frames)
    coverage=ops.total_cardinality(all_frames);require(coverage['scored_frame_predictions']==32400,'32400_PREDICTIONS_REQUIRED')
    return {'PASS':True,'evidence':evidence_result,'coverage':coverage,'cell_count':54,'seeds_verified':9,
      'frozen_hashes':hashes,'real_selection_executed':False,'selection_rule_called':False,'target_access':0,
      'model_legality_basis':'prior real full numerical audit plus exact current model bytes; no model decoding or inference here'}
@contextlib.contextmanager
def lease():
    absence();lock=R6OUT/'pipeline.lock'
    with lock.open('x',encoding='utf8')as f:f.write('R8_SELECTION_ONLY '+str(os.getpid()));f.flush();os.fsync(f.fileno())
    try:yield
    finally:lock.unlink(missing_ok=True)
def install_guard(job,evidence,archive,approval_files=()):
    job=Path(job).resolve();settings=cfg();exact={safe(ROOT,n)for m in (settings['dependencies'],settings['live_archive_files'],settings['legacy_run_files'])for n in m}
    models={R6OUT/'candidates'/r['name']/'train/model'/(n+'.npy')for r in settings['candidate_order']for n in PUBLIC}
    exact|={R6OUT/n for n in ('SELECTION_MANIFEST.json','SELECTION_MANIFEST.sha256')}|models|{Path(evidence).resolve(),Path(archive).resolve()}|{Path(x).resolve()for x in approval_files}
    runtime=[Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()];outputs={R6OUT/n for n in ('SELECTION_MANIFEST.json','SELECTION_MANIFEST.sha256')}
    def hook(event,args):
        if event=='open'and isinstance(args[0],(str,bytes,os.PathLike)):
            name=os.fsdecode(args[0])
            if name.upper()in('NUL',r'\\.\NUL'):return
            p=Path(name).resolve();mode,flags=args[1:3];writing=any(x in(mode or '')for x in' wax+'.strip())or isinstance(flags,int)and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if writing:require(p.is_relative_to(OUT.resolve())or p==R6OUT/'pipeline.lock'or _commit and p in outputs,'R8_SELECTION_WRITE_DENIED:'+str(p))
            else:
                require(p in exact or p.is_relative_to(HERE)or p.is_relative_to(OUT)or any(p.is_relative_to(x)for x in runtime),'R8_SELECTION_READ_DENIED:'+str(p))
                if p.suffix.lower()in('.npy','.npz','.pkl','.pickle','.mat','.h5','.hdf5'):
                    c=sys.modules.get('common');require(p in exact and (_opaque or c is not None and c._opaque),'R8_NUMERICAL_MODEL_PREPARATION_TARGET_DECODE_DENIED')
        elif event in('os.remove','os.rename','os.rmdir','os.mkdir','os.utime','os.link'):
            for x in args[:2]if event in('os.rename','os.link')else args[:1]:
                if isinstance(x,(str,bytes,os.PathLike)):
                    p=Path(os.fsdecode(x)).resolve();require(p.is_relative_to(OUT.resolve())or p==R6OUT/'pipeline.lock','R8_HISTORICAL_MUTATION_DENIED')
    sys.addaudithook(hook)
    forbidden={'fit','fit_reference','_initialize','_EM_per_component','train','validate','start','complete_selection','load_model','inference_smoke','automatic_select'}
    def profile(frame,event,arg):
        if event!='call' or frame.f_code.co_name not in forbidden:return
        if not Path(frame.f_code.co_filename).is_absolute()or not Path(frame.f_code.co_filename).is_relative_to(ROOT):return
        name=frame.f_code.co_name
        require(name not in('fit','fit_reference','_initialize','_EM_per_component','train','validate','start','complete_selection','load_model','inference_smoke','automatic_select')or name=='automatic_select'and _select,'R8_FORBIDDEN_ACTION:'+name)
    sys.setprofile(profile)
def publish(job,result,authorization_receipt):
    global _commit
    outputs=[R6OUT/'SELECTION_MANIFEST.json',R6OUT/'SELECTION_MANIFEST.sha256']
    require(all(not p.exists()for p in outputs),'SELECTION_ALREADY_FROZEN_NO_OVERWRITE')
    result=result|{'UTC':utc(),'R8_selection_authorization':authorization_receipt,'R8_bindings':bindings()}
    data=(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode();digest=hashlib.sha256(data).hexdigest()
    _commit=True
    try:
        with outputs[0].open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
        with outputs[1].open('xb')as f:f.write((digest+'\n').encode('ascii'));f.flush();os.fsync(f.fileno())
    finally:_commit=False
    require(sha(outputs[0])==digest and outputs[1].read_text(encoding='ascii').strip()==digest,'SELECTION_FREEZE_INTEGRITY_FAILURE_STOP_NO_RETRY')
    write(job/'selection_commit_receipt.json',{'PASS':True,'selection_manifest_SHA256':digest,'selected':result['selected'],'final_model_trained':False,'target_access':0},True)
    return {'PASS':True,'selection_manifest_SHA256':digest,'selected':result['selected'],'real_selection_executed':True,'final_model_trained':False,'target_access':0}
def complete_once(job,evidence,archive,approval,instruction,approval_sha):
    global _select
    receipt=authorize(approval,instruction,approval_sha)
    require(not(OUT/'SELECTION_EXECUTION_CONSUMED.json').exists(),'R8_SELECTION_ALREADY_CONSUMED_NO_RETRY')
    with lease():
        write(OUT/'SELECTION_EXECUTION_CONSUMED.json',{'record_SHA256':receipt['record_SHA256'],'approval_id':receipt['approval_id'],'job':str(job),'UTC':utc(),'maximum_executions':1},True)
        before=preflight(evidence,archive,True);idle_processes();absence(True)
        c,ops,r,_,_=modules();_select=True
        try:result=r.selection_audit()
        finally:_select=False
        for n,v in before['frozen_hashes'].items():unchanged_file(ROOT,n,v)
        require(sha(archive)==cfg()['archive_SHA256']and proof(evidence)==before['evidence']and sha(approval)==receipt['record_SHA256']and sha(instruction)==receipt['instruction_SHA256'],'AUTHORIZATION_OR_EVIDENCE_CHANGED_DURING_SELECTION')
        require(authorize(approval,instruction,approval_sha)==receipt,'R8_AUTHORIZATION_RECHECK_FAILED')
        absence(True);return publish(job,result,receipt)
