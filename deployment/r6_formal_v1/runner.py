"""Manual PHASE A/B/C. No all-grid launch, retry or target-stage entry."""
from pathlib import Path
import os,sys,argparse,traceback,csv,itertools,zipfile,time,contextlib
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from common import ROOT,OUT,LOCAL,read,write,sha,object_sha,require,utc,scope,integrity,approval,science,registry,candidate,npy_header,environment,protocol

def input_check():
    contract=read(HERE/'input_contract.json');cfg=science();rows=registry();results={}
    require(rows==cfg['candidate_order']==contract['candidate_order'],'CANDIDATE_REGISTRY_DRIFT')
    require(cfg['canonical_origins']==contract['canonical_origins'] and cfg['noise_variances']==contract['noise_variances'],'ORIGIN_OR_NOISE_DRIFT')
    require(cfg['pilot_geometry']==contract['pilot_geometry'],'PILOT_GEOMETRY_METADATA_DRIFT')
    for key,ids in [('training',list(range(400))),('validation',list(range(400,500))),('full_retraining',list(range(500)))]:
        require(cfg['trajectory_ids'][key]==ids,'TRAJECTORY_SPLIT_DRIFT')
    split=read(ROOT/'outputs/sp_u12b5b_preparation_split_manifest.json')
    for origin in cfg['canonical_origins']:
        require(split['origin_partitions'][origin]=={'train_ids':list(range(400)),'validation_ids':list(range(400,500))},'PREPARATION_SPLIT_DRIFT')
    noise=read(ROOT/'outputs/sp_u12b5b/validation_noise_registry.json')
    noise_ids=[(v['origin'],v['trajectory_id'],v['frame_id']) for v in noise]
    require(len(noise_ids)==1800 and len(set(noise_ids))==1800 and set(noise_ids)==set(itertools.product(cfg['canonical_origins'],range(400,500),range(4,10))),'NOISE_REGISTRY_IDENTITY_DRIFT')
    require(all(isinstance(v['seed'],int) and 0<=v['seed']<2**32 and len(v['unit_owner_divided_noise_sha256'])==64 for v in noise),'NOISE_SEED_OR_HASH_INVALID')
    proofs=read(ROOT/'outputs/sp_u12b5b/preparation_channel_proofs.json')
    require(len(proofs)==1500,'PREPARATION_PROOF_COUNT')
    require(len({(v['origin'],v['trajectory_id']) for v in proofs})==1500 and {(v['origin'],v['trajectory_id']) for v in proofs}==set(itertools.product(cfg['canonical_origins'],range(500))),'PREPARATION_PROOF_IDENTITIES')
    from src.baselines.mfa_reference_adapter import ProjectPilotInterface
    interface=ProjectPilotInterface()
    require(interface.A.shape==(64,4032) and interface.data_positions.shape==(3968,), 'LIVE_PILOT_GEOMETRY_MISMATCH')
    geometry={'selected_full':interface.selected_full.tolist(),'data_full':interface.data_full.tolist(),'joint_full':interface.joint_full.tolist(),
              'pilot_positions':interface.pilot_positions.tolist(),'data_positions':interface.data_positions.tolist(),
              'owner_symbols_real':interface.owner_symbols.real.tolist(),'owner_symbols_imag':interface.owner_symbols.imag.tolist()}
    require(object_sha(geometry)==contract['pilot_order_sha256'],'PILOT_ORDER_DRIFT')
    for name,expected in contract['data'].items():
        path=ROOT/name;require(path.is_file(),'MISSING_FROZEN_INPUT:'+name)
        require(sha(path)==expected['sha256'],'INPUT_SHA256_MISMATCH:'+name)
        header=npy_header(path);require(all(header[k]==expected[k] for k in ('shape','dtype','fortran_order')),'INPUT_HEADER_MISMATCH:'+name)
        results[name]=header
    return {'PASS':True,'headers':results,'train_rows':12000,'final_rows':15000,'val_predictions_per_candidate':3600,
            'val_predictions_total':32400,'noise_base_identities':1800,'pilot_order_sha256':object_sha(geometry),'numerical_arrays_read':False}

def stage_result(job,name):
    require(read(job/'status.json')['state']=='COMPLETE' and read(job/'status.json')['PASS'],'PREREQUISITE_JOB_NOT_COMPLETE')
    status=read(job/'status.json'); r=read(job/name)
    require(r['PASS'] and r.get('target_access',0)==0 and sha(job/'worker_result.json')==status['worker_result_sha256'],'PREREQUISITE_RESULT_HASH_MISMATCH')
    require(sha(job/name)==status['scientific_result_sha256'],'SCIENTIFIC_RESULT_HASH_MISMATCH')
    return r

def valid_candidate(row,include_validation=True):
    from model_ops import check_model_files,candidate_cardinality,macro_score,pooled_ratio
    trainjob=OUT/'candidates'/row['name']/'train';t=stage_result(trainjob,'training_result.json');v=check_model_files(trainjob/'model')
    require(sha(trainjob/'model/model_manifest.json')==t['model_manifest_sha256'],'TRAINED_MODEL_MANIFEST_DRIFT')
    require(v['K']==row['K'] and v['rank']==row['rank'] and v['N']==12000 and v['D']==4032 and v['context']['seed']==row['seed'] and t['seed']==row['seed'],'TRAINED_MODEL_SEED_OR_DIMENSION_DRIFT')
    require(v['context']['execution_package_sha256']==sha(HERE/'package_manifest.json') and v['context']['data_sha256']==read(HERE/'input_contract.json')['data']['outputs/sp_u12b5b/selection_train.npy']['sha256'],'TRAINED_MODEL_PROVENANCE_DRIFT')
    if not include_validation:return t
    valjob=OUT/'candidates'/row['name']/'validate';r=stage_result(valjob,'validation_result.json')
    require(all(r[k]==row[k] for k in row) and r['model_manifest_sha256']==t['model_manifest_sha256'],'VALIDATION_MODEL_OR_SEED_DRIFT')
    require(sha(valjob/'frame_audit.csv')==r['frame_audit_sha256'] and sha(valjob/'trajectory_audit.csv')==r['trajectory_audit_sha256'],'VALIDATION_AUDIT_HASH_DRIFT')
    with (valjob/'frame_audit.csv').open(encoding='utf8',newline='') as f: frames=list(csv.DictReader(f))
    require(candidate_cardinality(frames,row['K'],row['rank'])==r['prediction_count']==3600,'VALIDATION_COUNT_DRIFT')
    expected=set(itertools.product(science()['canonical_origins'],(10,20)))
    require(len(r['cells'])==6 and {(v['origin'],v['snr_db']) for v in r['cells']}==expected,'VALIDATION_CELL_INVENTORY_INVALID')
    import numpy as np
    for cell in r['cells']:
        records=[v for v in frames if v['origin']==cell['origin'] and int(v['snr_db'])==cell['snr_db']]
        E=np.asarray([float(v['error_energy']) for v in records],dtype=np.float64).reshape(100,6)
        T=np.asarray([float(v['truth_energy']) for v in records],dtype=np.float64).reshape(100,6)
        require(float(E.sum(dtype=np.float64))==cell['numerator'] and float(T.sum(dtype=np.float64))==cell['denominator'] and pooled_ratio(E,T)==cell['linear_pooled_NMSE'],'POOLED_CELL_RECONSTRUCTION_DRIFT')
    require(macro_score([v['linear_pooled_NMSE'] for v in r['cells']])==r['macro_selection_score'],'SELECTION_SCORE_DRIFT')
    return r,frames

def selection_audit():
    from model_ops import automatic_select,total_cardinality
    rows=[];frames=[]
    for row in registry():
        result,frame=valid_candidate(row);rows.append(result);frames.extend(frame)
    coverage=total_cardinality(frames)
    require(coverage['scored_frame_predictions']==32400,'TOTAL_VALIDATION_CARDINALITY')
    chosen=automatic_select(rows)
    return {'PASS':True,'selected':{k:chosen[k] for k in ('name','K','rank','seed','final_seed','macro_selection_score')},
            'candidates':rows,'coverage':coverage,'execution_package_sha256':sha(HERE/'package_manifest.json'),
            'input_contract_sha256':sha(HERE/'input_contract.json'),'target_access':0,
            'criterion':'six linear pooled NMSE macro average; exact floating tie then smaller K/rank'}

def selection_frozen():
    path=OUT/'SELECTION_MANIFEST.json';require(path.is_file(),'SELECTION_NOT_FROZEN')
    require(sha(path)==(OUT/'SELECTION_MANIFEST.sha256').read_text(encoding='ascii').strip(),'SELECTION_MANIFEST_HASH_DRIFT')
    v=read(path);check=selection_audit();require(all(v[k]==x for k,x in check.items()),'SELECTION_PREREQUISITES_DRIFT');return v

@contextlib.contextmanager
def lock():
    path=OUT/'pipeline.lock';OUT.mkdir(parents=True,exist_ok=True)
    require(not path.exists(),'SINGLE_JOB_LOCK_PRESENT: query status; no automatic stale-lock deletion')
    with path.open('x',encoding='utf8') as f:f.write(str(os.getpid()))
    try:yield
    finally:path.unlink(missing_ok=True)

def start(action,name):
    # Approval first, before numpy/model imports or scientific data reads.
    approval();require(action in ('train','validate','final'),'UNKNOWN_STAGE')
    with lock():
        integrity();input_check();row=candidate(name)
        require((OUT/'preflight.json').is_file() and read(OUT/'preflight.json')['PASS'] and
                read(OUT/'preflight.json')['integrity']['package_manifest_SHA256']==sha(HERE/'package_manifest.json'),'LAB_PREFLIGHT_RECEIPT_REQUIRED')
        require((LOCAL/'toy_test_report.json').is_file() and read(LOCAL/'toy_test_report.json')['PASS'] and
                read(LOCAL/'toy_test_report.json')['tested_package_manifest_sha256']==sha(HERE/'package_manifest.json'),'LAB_R6_TOY_FIDELITY_RECEIPT_REQUIRED')
        env=environment()
        from windows_resources import capacity,hardware
        cap=capacity(hardware());require(cap['PASS'],'LAB_HOST_CAPACITY_GATE:'+str(cap['reasons']))
        if action=='train':
            index=registry().index(row)
            for previous in registry()[:index]:valid_candidate(previous)
            job=OUT/'candidates'/name/'train'
        elif action=='validate':
            valid_candidate(row,False);job=OUT/'candidates'/name/'validate'
        else:
            selected=selection_frozen()['selected'];require(name==selected['name'],'FINAL_STRUCTURE_MISMATCH');job=OUT/'final'
        require(not job.exists(),'EXISTING_JOB_OR_CHECKPOINT_NO_RETRY')
        job.mkdir(parents=True)
        phase='C' if action=='final' else 'A'
        write(OUT/('PHASE_'+phase+'_STATUS.json'),{'state':'RUNNING','action':action,'candidate':name,'job':job.relative_to(ROOT).as_posix(),'UTC':utc()})
        write(job/'execution_context.json',{'action':action,'candidate':row,'package_manifest_sha256':sha(HERE/'package_manifest.json'),
              'resource_proposal_sha256':sha(HERE/'resource_amendment.json'),'approval':approval(),'environment':env,'UTC':utc()})
        from supervisor import run_worker
        timeout=protocol()['worker_seconds'][action]
        if action=='validate':timeout-=read(OUT/'candidates'/name/'train/status.json')['wall_seconds'];require(timeout>0,'CANDIDATE_COMBINED_24H_EXHAUSTED')
        state,result=run_worker(action,job,blas=6,openmp=1,extra=['--candidate',name],timeout=timeout)
        filename='validation_result.json' if action=='validate' else 'training_result.json'
        state['worker_result_sha256']=sha(job/'worker_result.json');state['scientific_result_sha256']=sha(job/filename)
        write(job/'status.json',state)
        if action=='final':
            selected=selection_frozen()['selected']
            manifest={'PASS':True,'final_model_manifest_sha256':result['model_manifest_sha256'],
                      'selected':selected,'final_seed':row['final_seed'],'full_data_sha256':read(HERE/'input_contract.json')['data']['outputs/sp_u12b5b/full_retraining.npy']['sha256'],
                      'training_result_sha256':sha(job/'training_result.json'),'SELECTION_MANIFEST_SHA256':sha(OUT/'SELECTION_MANIFEST.json'),
                      'package_manifest_sha256':sha(HERE/'package_manifest.json'),'inference_config':{'joint_dimension':4032,'data_dimension':3968,'pilot_dimension':64,
                      'geometry_order_sha256':read(HERE/'input_contract.json')['pilot_order_sha256'],'all_components_float':1.0,
                      'validation_noise_variances':science()['noise_variances'],'operating_noise_rule':'known positive sigma2 times I64; original author noise-dependent posterior is instantiated per call',
                      'online_inputs':'received owner pilots, known pilot symbols/geometry, known sigma2; no target label/covariance/CSI'},
                      'target_access':0,'frozen_before_B2_unsealing':True,'fresh_fit':True,'UTC':utc()}
            write(OUT/'FINAL_MODEL_MANIFEST.json',manifest,exclusive=True)
            (OUT/'FINAL_MODEL_MANIFEST.sha256').write_text(sha(OUT/'FINAL_MODEL_MANIFEST.json')+'\n',encoding='ascii')
        completed_validation=sum((OUT/'candidates'/r['name']/'validate/status.json').exists() and
                 read(OUT/'candidates'/r['name']/'validate/status.json')['state']=='COMPLETE' for r in registry())
        write(OUT/('PHASE_'+phase+'_STATUS.json'),{'state':'COMPLETE' if action=='final' or completed_validation==9 else 'AWAITING_MANUAL_NEXT_STAGE',
              'last_action':action,'last_candidate':name,'complete_candidate_validations':completed_validation,'UTC':utc()})
        return state

def complete_selection():
    approval()
    with lock():
        integrity();result=selection_audit();result['UTC']=utc()
        require(not (OUT/'SELECTION_MANIFEST.json').exists(),'SELECTION_ALREADY_FROZEN_NO_OVERWRITE')
        write(OUT/'SELECTION_MANIFEST.json',result,exclusive=True)
        (OUT/'SELECTION_MANIFEST.sha256').write_text(sha(OUT/'SELECTION_MANIFEST.json')+'\n',encoding='ascii')
        write(OUT/'PHASE_B_STATUS.json',{'state':'COMPLETE','selection_manifest_sha256':sha(OUT/'SELECTION_MANIFEST.json'),'UTC':utc()})
        return {'PASS':True,'selected':result['selected'],'manifest_sha256':sha(OUT/'SELECTION_MANIFEST.json')}

def status():
    jobs=[]
    for p in sorted((OUT/'candidates').rglob('status.json'))+sorted((OUT/'final').glob('status.json')):
        jobs.append({'path':p.relative_to(ROOT).as_posix(),**read(p)})
    return {'resource_proposal_status':protocol()['status'],'approval_present':(OUT/'MENTOR_APPROVAL.json').exists(),
            'STOP':read(OUT/'STOP.json') if (OUT/'STOP.json').exists() else None,'lock_present':(OUT/'pipeline.lock').exists(),
            'jobs':jobs,'phases':{p:read(OUT/('PHASE_'+p+'_STATUS.json')) if (OUT/('PHASE_'+p+'_STATUS.json')).exists() else {'state':'NOT_STARTED'} for p in ('A','B','C')},
            'selection_manifest_present':(OUT/'SELECTION_MANIFEST.json').exists(),'final_manifest_present':(OUT/'FINAL_MODEL_MANIFEST.json').exists()}

def stop():
    write(OUT/'STOP_REQUESTED.json',{'UTC':utc(),'reason':'USER_STOP_REQUESTED'})
    killed=False
    if (OUT/'active_worker.json').exists():
        from windows_resources import stop_named_job
        killed=stop_named_job(read(OUT/'active_worker.json')['JobObject_name'])
    write(OUT/'STOP.json',{'UTC':utc(),'reason':'USER_STOP_REQUESTED','no_retry':True})
    return {'PASS':True,'kernel_job_terminated':killed,'resume_requires_new_mentor_task':True}

def export_results(include_final=False):
    dest=OUT/'exports';dest.mkdir(parents=True,exist_ok=True);name='SP_U12B5_R6_Results_'+time.strftime('%Y%m%d_%H%M%S')+'.zip';path=dest/name
    require(not path.exists(),'EXPORT_EXISTS')
    require(not (OUT/'pipeline.lock').exists(),'EXPORT_WAIT_FOR_FORMAL_JOB_COMPLETION')
    if include_final:
        require((OUT/'FINAL_MODEL_MANIFEST.json').is_file(),'FINAL_MODEL_NOT_FROZEN')
        require(sha(OUT/'FINAL_MODEL_MANIFEST.json')==(OUT/'FINAL_MODEL_MANIFEST.sha256').read_text(encoding='ascii').strip(),'FINAL_FREEZE_HASH_DRIFT')
        from model_ops import check_model_files
        v=check_model_files(OUT/'final/model');require(sha(OUT/'final/model/model_manifest.json')==read(OUT/'FINAL_MODEL_MANIFEST.json')['final_model_manifest_sha256'],'FINAL_MODEL_HASH_DRIFT')
    def technical_receipt(p):
        if not p.is_relative_to(LOCAL):return True
        rel=p.relative_to(LOCAL)
        if len(rel.parts)==1:return p.name in ('toy_test_report.json','background_probe_result.json','background_probe_latest.json') or p.name.startswith('pytest_') and p.suffix=='.log'
        return rel.parts[0] in ('toy_models','monitor') and p.suffix.lower() in ('.json','.jsonl','.log')
    files=[p for p in OUT.rglob('*') if p.is_file() and not p.is_relative_to(dest) and technical_receipt(p)
           and 'temporary' not in p.parts and (p.suffix.lower()not in ('.npy','.npz') or include_final and p.is_relative_to(OUT/'final/model'))]
    manifest={p.relative_to(OUT).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,'r6_formal/'+p.relative_to(OUT).as_posix())
        import json
        z.writestr('result_manifest.json',json.dumps({'files':manifest,'package_manifest_sha256':sha(HERE/'package_manifest.json'),
                   'includes_final_model':include_final,'candidate_models_retained_on_lab':True},indent=2))
    with zipfile.ZipFile(path)as z:
        import hashlib
        for member_name,row in manifest.items():
            h=hashlib.sha256()
            with z.open('r6_formal/'+member_name)as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
            require(h.hexdigest()==row['sha256'],'EXPORT_INPUT_CHANGED_DURING_CAPTURE')
    (dest/(name+'.sha256')).write_text(sha(path)+'  '+name+'\n',encoding='ascii')
    return {'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,'files':len(files)+1}

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('verify','preflight','train','validate','select','final','status','stop','export'))
    p.add_argument('--candidate');p.add_argument('--include-final-model',action='store_true');p.add_argument('--personal-metadata-only',action='store_true');a=p.parse_args()
    scope()
    try:
        if a.action=='verify':result=integrity(not a.personal_metadata_only)
        elif a.action=='preflight':
            result={'integrity':integrity(not a.personal_metadata_only),'inputs':input_check(),'environment':environment()}
            if not a.personal_metadata_only:
                from windows_resources import hardware,capacity
                result['capacity']=capacity(hardware());require(result['capacity']['PASS'],'PREFLIGHT_CAPACITY_STOP')
            result['PASS']=True;write(LOCAL/'metadata_preflight.json' if a.personal_metadata_only else OUT/'preflight.json',result)
        elif a.action in ('train','validate'):result=start(a.action,a.candidate)
        elif a.action=='select':result=complete_selection()
        elif a.action=='final':result=start('final',selection_frozen()['selected']['name']) if approval() else None
        elif a.action=='status':result=status()
        elif a.action=='stop':result=stop()
        else:result=export_results(a.include_final_model)
        import json
        print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),flush=True)
    except BaseException as error:
        # Missing approval is a refusal, not a scientific attempt. All other
        # failures preserve evidence and stop the whole pipeline.
        formal=a.action in ('train','validate','select','final')
        log={'UTC':utc(),'action':a.action,'type':type(error).__name__,'reason':str(error),'traceback':traceback.format_exc(), 'formal_science_started':(OUT/'candidates').exists() or (OUT/'final').exists()}
        if formal and 'MENTOR_APPROVAL_PENDING' not in str(error):
            write(OUT/'STOP.json',log)
            phase='B' if a.action=='select' else 'C' if a.action=='final' else 'A'
            write(OUT/('PHASE_'+phase+'_STATUS.json'),{'state':'STOP','reason':str(error),'UTC':utc()})
        else:write(LOCAL/('refusal_'+a.action+'.json'),log)
        raise

if __name__=='__main__':main()
