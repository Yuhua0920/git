"""Read-only evidence checks. No fit, validation, selection or scientific runner."""
from pathlib import Path
import os,sys,json,hashlib,math,io,zipfile,contextlib,datetime
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'lab_runs/r7_evidence_audit_r1'
PUBLIC=('means','lambdas','psis','amps','covs','inv_covs');SMALL=('means','lambdas','psis','amps','training_history')
GATES=('ARCHIVE_INTEGRITY','MODEL_ARTIFACTS_AND_VALIDITY','CANONICAL_VALIDATION_AND_METRIC','FROZEN_PROTOCOL_AND_SEEDS','RESOURCE_AND_STOP','POST_RUN_PRESERVATION')
class AuditFailure(RuntimeError):pass
def require(v,msg):
    if not v:raise AuditFailure(msg)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v,exclusive=False):
    p=Path(p).resolve();require(p.is_relative_to(OUT.resolve()),'AUDIT_OUTPUT_ONLY');p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if exclusive:
        with p.open('x',encoding='utf8')as f:f.write(text)
    else:
        tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(text,encoding='utf8');os.replace(tmp,p)
_opaque=False
def sha(p):
    global _opaque
    old=_opaque;_opaque=True
    try:
        h=hashlib.sha256()
        with Path(p).open('rb')as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        return h.hexdigest()
    finally:_opaque=old
def objsha(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def safe(root,name):
    require(isinstance(name,str)and '\\'not in name and ':'not in name,'UNSAFE_RELATIVE_PATH')
    p=(Path(root)/name).resolve();require(p.is_relative_to(Path(root).resolve())and '..'not in Path(name).parts,'PATH_TRAVERSAL');return p
def inputs():return read(HERE/'trusted_inputs.json')
def verify_tool():
    v=read(HERE/'package_manifest.json')
    for n,x in v['files'].items():
        p=safe(ROOT,n);require(p.is_relative_to(HERE)and p.stat().st_size==x['bytes']and sha(p)==x['sha256'],'TOOL_DRIFT:'+n)
    return {'PASS':True,'package_manifest_SHA256':sha(HERE/'package_manifest.json'),'files':len(v['files'])}
def dependencies(root,cfg,progress=None):
    rows=[]
    for n,x in cfg['dependencies'].items():
        p=safe(root,n);require(p.is_file(),'MISSING_DEPENDENCY:'+n);h=sha(p)
        require(h==x['sha256']and('bytes'not in x or p.stat().st_size==x['bytes']),'POST_RUN_DEPENDENCY_DRIFT:'+n)
        rows.append({'path':n,'bytes':p.stat().st_size,'sha256':h})
        if progress:progress('dependency:'+n)
    import numpy as np
    headers={}
    for n,x in cfg['data'].items():
        global _opaque
        _opaque=True
        try:
            with safe(root,n).open('rb')as f:
                version=np.lib.format.read_magic(f);require(version in ((1,0),(2,0)),'NPY_VERSION_INVALID')
                shape,order,dtype=(np.lib.format.read_array_header_1_0(f)if version==(1,0)else np.lib.format.read_array_header_2_0(f));offset=f.tell()
        finally:_opaque=False
        require(not dtype.hasobject and list(shape)==x['shape']and dtype.str==x['dtype']and order==x['fortran_order'],'PREPARATION_HEADER_DRIFT:'+n)
        require(safe(root,n).stat().st_size==offset+math.prod(shape)*dtype.itemsize,'PREPARATION_SIZE_DRIFT:'+n)
        headers[n]={'shape':list(shape),'dtype':dtype.str,'fortran_order':order,'header_bytes':offset,'payload_decoded':False}
    return rows,headers
def archive_leaves(root,cfg,progress=None):
    rows=[]
    for n,x in cfg['live_archive_files'].items():
        p=safe(root,n);require(p.is_file(),'MISSING_LIVE_ARCHIVE_FILE:'+n);h=sha(p)
        require(h==x['sha256']and p.stat().st_size==x['bytes'],'LIVE_ARCHIVE_DRIFT:'+n)
        rows.append({'path':n,'bytes':p.stat().st_size,'sha256':h})
        if progress:progress('archive_leaf:'+n)
    require(len(rows)==353,'ARCHIVE_LEAF_COUNT');return rows
def legacy_history(root,cfg,progress=None):
    rows=[]
    for n,v in cfg['legacy_run_files'].items():
        p=safe(root,n);require(p.is_file(),'MISSING_ACTUAL_R5_R6_HISTORY:'+n);h=sha(p)
        require(h==v['sha256']and p.stat().st_size==v['bytes'],'R5_R6_HISTORICAL_DRIFT:'+n)
        rows.append({'path':n,'bytes':p.stat().st_size,'sha256':h})
        if progress:progress('legacy:'+n)
    return rows
def source_population(root,cfg):
    expected={n for n in cfg['dependencies']if n.startswith(('src/','reference/','configs/','deployment/r5_128gb_v1/','deployment/r6_formal_v1/','deployment/r7_sequential_v1/'))}
    actual=set()
    for name in ('src','reference','configs','deployment/r5_128gb_v1','deployment/r6_formal_v1','deployment/r7_sequential_v1'):
        for p in(Path(root)/name).rglob('*'):
            if p.is_file()and '__pycache__'not in p.parts and '.pytest_cache'not in p.parts:actual.add(p.relative_to(root).as_posix())
    require(actual==expected,'FROZEN_SOURCE_POPULATION_MISMATCH:'+str({'missing':sorted(expected-actual),'additional':sorted(actual-expected)}))
    return {'PASS':True,'files':len(actual),'unexpected_source_files':0}
def evidence_binding(cfg):
    for n,h in cfg['accepted_evidence_SHA256'].items():require(sha(HERE/n)==h,'ACCEPTED_EVIDENCE_DRIFT')
    a=read(HERE/'SP_U12B5_R7_audit_evidence.json');s=read(HERE/'SP_U12B5_R7_supplementary_evidence.json')
    require(a['archive']['sha256']==cfg['archive_SHA256']and not a['failed_checks']and not s['failed_checks'],'PRIOR_AUDIT_NOT_ACCEPTABLE')
    return {'PASS':True,'source':'INHERITED_ACCEPTED_INDEPENDENT_AUDIT','new_metric_calculations':0,'new_real_validation_inference':0,
      'accepted_evidence_SHA256':cfg['accepted_evidence_SHA256'],'archive_SHA256':cfg['archive_SHA256'],'live_bytes_must_match':353}
def old_is_idle(root,cfg):
    root=Path(root);q=read(root/'lab_runs/r7_sequential/queue.json');c=read(root/'lab_runs/r7_sequential/completion.json')
    require(q['state']=='QUEUE_COMPLETE'and q['current_stage']=='STOP_AFTER_NINE'and q['completed_candidates']==[r['name']for r in cfg['candidate_order']]and not q['remaining_candidates'],'QUEUE_NOT_COMPLETE')
    require(q.get('STOP_reason')is None and c['PASS']and c['selection_performed']is False and c['final_fit_performed']is False,'QUEUE_FAILED_OR_SCOPE_EXCEEDED')
    for n in ('lab_runs/r6_formal/pipeline.lock','lab_runs/r7_sequential/execution.lock','lab_runs/r6_formal/STOP.json','lab_runs/r6_formal/STOP_REQUESTED.json','lab_runs/r7_sequential/STOP.json','lab_runs/r7_sequential/STOP_REQUESTED.json',
      'lab_runs/r6_formal/SELECTION_MANIFEST.json','lab_runs/r6_formal/FINAL_MODEL_MANIFEST.json','lab_runs/r6_formal/final'):
        require(not(root/n).exists(),'SCIENTIFIC_LOCK_STOP_OR_UNAUTHORIZED_ARTIFACT:'+n)
    # Expected absence is an explicit boundary, never a laboratory seal PASS.
    for n in ('outputs/sp_u12b2b_sealed','outputs/sp_u12b3g_next_sealed'):
        require(not(root/n).exists(),'UNEXPECTED_SEALED_TARGET_DATA_ON_LAB; STOP_NO_READ')
    return {'PASS':True,'B2_B3_lab':'NOT_PRESENT_BY_DESIGN_NOT_PHYSICALLY_VERIFIED','manuscript_lab':'NOT_MIGRATED_BY_DESIGN',
      'historical_target_access':'NOT_FORENSICALLY_PROVABLE_FROM_DECLARATIONS','selection_manifest':'ABSENT','final_model':'ABSENT'}
def original_modules():
    from source_only_loader import install
    verify_tool()
    install(ROOT, inputs()['dependencies'] | read(HERE/'package_manifest.json')['files'])
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'deployment/r6_formal_v1'))
    import model_ops,common,windows_resources
    return model_ops,common,windows_resources
def first_stop(history):
    import numpy as np
    h=np.asarray(history,dtype=np.float64);require(h.ndim==1 and 7<=len(h)<=100 and np.isfinite(h).all(),'HISTORY_INVALID')
    eligible=[];declines=[]
    for index in range(1,len(h)):
        if h[index]<h[index-1]:declines.append({'update':index+1,'previous':float(h[index-1]),'current':float(h[index])})
        if index>5:
            with np.errstate(divide='ignore',invalid='ignore'):v=abs((h[index]-h[index-1])/h[index])
            if v<1e-4:eligible.append(index+1)
    first=eligible[0]if eligible else None
    require(first is None and len(h)==100 or first==len(h),'FIRST_AUTHOR_STOP_POSITION_MISMATCH')
    return {'first_eligible_update':7,'first_stop_update':first,'iterations':len(h),'convergence':'MAX_ITER_REACHED'if len(h)==100 else'CONVERGED',
      'decrease_events':declines,'interpretation':'stopping criterion only; no monotonic/global-optimum/across-seed claim'}
def array_sha(a):
    import numpy as np
    require(a.flags.c_contiguous,'ARRAY_NOT_C_ORDER');h=hashlib.sha256();h.update(memoryview(a).cast('B'));return h.hexdigest()
def model_check(directory,row,expected,synthetic=False,progress=None):
    """Read original NPYs; frozen validity/smoke with no scientific threshold changes."""
    import numpy as np
    directory=Path(directory);require(read(directory/'model_manifest.json')==expected,'MODEL_MANIFEST_CHANGED')
    require(expected['K']==row['K']and expected['rank']==row['rank']and expected['D']==(24 if synthetic else 4032)and expected['N']==(12 if synthetic else 12000),'MODEL_CONTEXT_DIMENSION')
    context=expected['context'];require(context['candidate']==row['name']and context['seed']==row['seed']and context['role']=='train'and context['fresh_fit']is True and context['candidate_checkpoint_used']is False,'MODEL_FIT_CONTEXT_INVALID')
    require(set(expected['file_hashes'])=={n+'.npy'for n in PUBLIC+('training_history',)},'SEVEN_NPY_REQUIRED')
    ops,_,_=original_modules();require(expected['policy']==ops.FROZEN_POLICY,'POLICY_DRIFT')
    require(objsha(expected['arrays'])==expected['parameter_hash']and objsha(expected['initialization'])==expected['initialization_hash'],'COMBINED_MODEL_HASH_INVALID')
    m=ops.make_reference_model(row['K'],row['rank']);m.D=expected['D'];m.N=expected['N'];records=[]
    for name in PUBLIC+('training_history',):
        p=directory/(name+'.npy');require(p.is_file(),'MISSING_NPY:'+str(p));h=sha(p);require(h==expected['file_hashes'][p.name],'MODEL_FILE_SHA_DRIFT:'+p.name)
        a=np.load(p,mmap_mode='r',allow_pickle=False);require(not a.dtype.hasobject and a.flags.c_contiguous,'NPY_LAYOUT_INVALID:'+name)
        info={'shape':list(a.shape),'dtype':a.dtype.str,'sha256':array_sha(a)}
        if name in PUBLIC:require(info==expected['arrays'][name],'MODEL_PAYLOAD_SHA_DRIFT:'+name);setattr(m,name,a)
        else:
            require(a.dtype.str=='<f8'and a.ndim==1 and len(a)==expected['iterations'],'HISTORY_DTYPE_OR_SHAPE');m.L_all=a.tolist()
        # NPY header + actual payload yields independently computed full size.
        with p.open('rb')as f:
            v=np.lib.format.read_magic(f)
            _,order,dt=(np.lib.format.read_array_header_1_0(f)if v==(1,0)else np.lib.format.read_array_header_2_0(f));offset=f.tell()
        require(not order,'NPY_LAYOUT_INVALID_FORTRAN_HEADER:'+name)
        require(p.stat().st_size==offset+a.nbytes,'MODEL_NPY_SIZE_INVALID')
        records.append({'candidate':row['name'],'filename':p.name,'bytes':p.stat().st_size,'file_sha256':h,'array_sha256':info['sha256'],'shape':info['shape'],'dtype':info['dtype'],'C_order':True})
        if progress:progress(row['name']+':'+name)
    history=first_stop(m.L_all);require(history['convergence']==expected['convergence'],'CONVERGENCE_LABEL_DRIFT')
    raw=[json.loads(line)for line in(directory.parent/'iteration_history.jsonl').read_text(encoding='utf8').splitlines()if line]
    require(len(raw)==len(m.L_all)and all(v['iteration']==i+1 and float(v['author_log_likelihood'])==m.L_all[i]for i,v in enumerate(raw)),'HISTORY_NPY_JSONL_MISMATCH')
    original=ops.parameter_hashes(m);valid=ops.parameter_validity(m,full=True)
    # Small parameters and history round-trip entirely in memory, with original bytes untouched.
    for n in SMALL:
        a=np.asarray(m.L_all,dtype=np.float64)if n=='training_history'else getattr(m,n)
        buf=io.BytesIO();np.save(buf,a,allow_pickle=False);buf.seek(0);b=np.load(buf,allow_pickle=False)
        require(b.dtype==a.dtype and b.shape==a.shape and array_sha(b)==array_sha(a),'MEMORY_ROUNDTRIP_INVALID')
    smoke=ops.inference_smoke(m)
    require(ops.parameter_hashes(m)==original,'SYNTHETIC_SMOKE_MUTATED_SAVED_PARAMETERS')
    for x in records:require(sha(directory/x['filename'])==x['file_sha256'],'MODEL_BYTES_CHANGED_DURING_AUDIT')
    return {'candidate':row['name'],'PASS':True,'K':row['K'],'rank':row['rank'],'seed':row['seed'],'parameter_hash':expected['parameter_hash'],
      'validity':valid,'history':history,'smoke':smoke,'memory_roundtrip_small_arrays':True,'actual_cov_and_inverse_file_read':True,
      'inverse_scope':'finite + shape/dtype/hash of saved inverse; frozen checker defines no additional inverse-residual threshold',
      'files':records,'synthetic_fixture':synthetic}
def resource_history(root,cfg):
    rows=[];last=None;limits=cfg['resources']['limits']
    for row in cfg['candidate_order']:
        total=0
        for action in ('train','validate'):
            base=Path(root)/'lab_runs/r6_formal/candidates'/row['name']/action;s=read(base/'status.json');w=read(base/'worker_result.json')
            require(s['PASS']and w['PASS']and s['state']=='COMPLETE'and s['stop_reason']is None and s['exit_code']==0,'HISTORICAL_STAGE_FAILED')
            require(s['blas_threads']==6 and s['openmp_threads']==1 and s['limits']==limits,'RESOURCE_CONTRACT_DRIFT')
            require(s['actual_interpreter_PID']==w['actual_PID']and s['monitor_not_assumed_venv_launcher_PID']and s['all_descendants_kernel_contained'],'ACTUAL_PID_PROOF_MISSING')
            start=datetime.datetime.fromisoformat(s['started_UTC']);end=datetime.datetime.fromisoformat(s['completed_UTC']);require(start<=end and(last is None or start>=last),'STAGE_OVERLAP');last=end
            require(s['wall_seconds']<=86400 and(end-start).total_seconds()<=86400,'HISTORICAL_STAGE_TIME_LIMIT');total+=s['wall_seconds']
            rss=commit=count=0;ram=available=None;previous=0.;gap=0.
            with(base/'resources.jsonl').open(encoding='utf8')as f:
                for line in f:
                    x=json.loads(line);t=x['process_tree'];u=x['system'];count+=1
                    rss=max(rss,t['RSS_bytes'],max((p['peak_RSS_bytes']for p in t['members']),default=0))
                    commit=max(commit,t['private_commit_bytes'],max((p['native_peak_commit_bytes']for p in t['members']),default=0))
                    ram=u['available_physical_bytes']if ram is None else min(ram,u['available_physical_bytes'])
                    available=u['commit_available_bytes']if available is None else min(available,u['commit_available_bytes'])
                    gap=max(gap,x['elapsed_seconds']-previous);previous=x['elapsed_seconds']
            n=w['native_final_memory'];rss=max(rss,n['peak_working_set_bytes']);commit=max(commit,n['peak_commit_bytes'])
            require(count==s['resource_samples']and rss==s['peak_RSS_bytes']and commit==s['peak_private_commit_bytes']and ram==s['minimum_available_physical_bytes']and available==s['minimum_remaining_commit_bytes'],'RESOURCE_PEAK_OR_COUNT_MISMATCH')
            require(rss<=limits['RSS_bytes']and commit<=limits['private_commit_bytes']and ram>=limits['min_available_physical_bytes']and available>=limits['min_remaining_commit_bytes'],'HISTORICAL_RESOURCE_LIMIT_EXCEEDED')
            require(s['sample_target_seconds']==.1 and s['automatic_retry']is False,'SAMPLING_OR_RETRY_DRIFT')
            rows.append({'candidate':row['name'],'stage':action,'samples':count,'peak_RSS':rss,'peak_private_commit':commit,'wall_seconds':s['wall_seconds'],'sample_max_observed_gap':gap,'actual_PID':w['actual_PID'],'JobObject_name':s['JobObject_name'],'PASS':True})
        require(total<=86400,'COMBINED_CANDIDATE_TIME_EXCEEDED')
    return rows
def install_guard(root,cfg,job):
    """Only new audit output writes; only exact authorized project input reads."""
    root=Path(root).resolve();job=Path(job).resolve()
    authorized={safe(root,n)for n in cfg['dependencies']}|{safe(root,n)for n in cfg['live_archive_files']}|{safe(root,n)for n in cfg.get('legacy_run_files',{})}
    models={root/'lab_runs/r6_formal/candidates'/r['name']/'train/model'/(n+'.npy')for r in cfg['candidate_order']for n in PUBLIC+('training_history',)}
    def hook(event,args):
        if event=='open'and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2]
            writing=any(c in(mode or '')for c in 'wax+')or isinstance(flags,int)and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if writing:require(p.is_relative_to(job),'READONLY_AUDIT_WRITE_DENIED:'+str(p))
            elif p.is_relative_to(root)and not p.is_relative_to(job):
                require(p.is_relative_to(HERE)or p in authorized or p in models or p.is_relative_to(root/'.venv-lab'),'READONLY_AUDIT_READ_DENIED:'+str(p))
                if p.suffix.lower()in('.npy','.npz','.pkl','.h5','.mat')and p not in models:require(_opaque,'NO_PREPARATION_OR_TARGET_NUMERICAL_DECODE')
        elif event in('os.remove','os.rename','os.rmdir','os.mkdir'):
            for x in args[:2]if event=='os.rename'else args[:1]:
                if isinstance(x,(str,bytes,os.PathLike)):require(Path(os.fsdecode(x)).resolve().is_relative_to(job),'READONLY_AUDIT_MUTATION_DENIED')
    sys.addaudithook(hook)
    def profile(frame,event,arg):
        if event=='call'and frame.f_code.co_name in('fit','fit_reference','_initialize','_EM_per_component','automatic_select','complete_selection','validate','train','selection_audit','start'):
            if Path(frame.f_code.co_filename).is_relative_to(root):raise AuditFailure('FORBIDDEN_SCIENTIFIC_CALL:'+frame.f_code.co_name)
    sys.setprofile(profile)
def csv_write(p,rows,fields=None):
    import csv
    rows=list(rows);fields=fields or(list(rows[0])if rows else['empty'])
    with Path(p).open('x',encoding='utf-8-sig',newline='')as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False)if isinstance(v,(dict,list))else v for k,v in row.items()})
def export_job(job,all_models=False):
    job=Path(job).resolve();require(job.is_relative_to(OUT)and(job/'status.json').is_file(),'UNKNOWN_AUDIT_JOB')
    state=read(job/'status.json');require(state['state']in('COMPLETE','FAILED','STOP')and not(job/'export_report.json').exists(),'AUDIT_RUNNING_OR_EXPORT_ALREADY_EXISTS')
    report=read(job/'final_report.json')if(job/'final_report.json').exists()else read(job/'report.json')if(job/'report.json').exists()else{};success=state.get('PASS')is True and report.get('verdict')=='FROZEN_SELECTION_EVIDENCE_READY_REQUIRES_SEPARATE_APPROVAL'
    files={p.relative_to(job).as_posix():p for p in job.rglob('*')if p.is_file()and 'temporary'not in p.parts and p.suffix!='.zip'}
    if success:
        for row in inputs()['candidate_order']:
            base=ROOT/'lab_runs/r6_formal/candidates'/row['name']/'train/model'
            for n in (PUBLIC+('training_history',)if all_models else SMALL):
                p=base/(n+'.npy');require(sha(p)==inputs()['models'][row['name']]['file_hashes'][p.name],'MODEL_DRIFT_BEFORE_EXPORT')
                files['original_models/'+row['name']+'/'+p.name]=p
    manifest={n:{'bytes':p.stat().st_size,'sha256':sha(p)}for n,p in files.items()};dest=job/'evidence_export.zip'
    with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED)as z:
        for n,p in files.items():z.write(p,n)
        z.writestr('export_manifest.json',json.dumps({'files':manifest,'original_model_arrays':63 if all_models and success else 45 if success else 0,'large_matrices_retained_on_lab':not all_models},indent=2))
    with zipfile.ZipFile(dest)as z:
        require(z.testzip()is None and len(z.namelist())==len(manifest)+1==len(set(z.namelist())),'EXPORT_CRC_OR_INVENTORY')
        for n,v in manifest.items():
            h=hashlib.sha256();count=0
            with z.open(n)as f:
                for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b);count+=len(b)
            require(count==v['bytes']and h.hexdigest()==v['sha256'],'EXPORT_BYTE_DRIFT')
    h=sha(dest);Path(str(dest)+'.sha256').write_text(h+'  '+dest.name+'\n',encoding='ascii')
    result={'PASS':True,'path':str(dest),'sha256':h,'bytes':dest.stat().st_size,'files':len(manifest)+1,'models_exported':63 if all_models and success else 45 if success else 0,
      'remote_limit':'18 large matrix NPY bytes remain on lab by default; remote reviewer verifies hashes and original host full-matrix logs, not direct matrix payloads'}
    write(job/'export_report.json',result,True);return result
