"""R6 personal preparation only: opaque preservation and verified technical R5 review."""
from pathlib import Path
import sys,json,hashlib,zipfile,time,subprocess,concurrent.futures,argparse
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=ROOT/'lab_runs/r6_formal';LOCAL=OUT/'preparation_personal'
R5_ZIP_SHA='d5472254f66f8f6afa4dbac883858f5f6273e2b9d94017e3ce5b96cc2f13da45'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def preserve(mode):
    LOCAL.mkdir(parents=True,exist_ok=True);before=LOCAL/'historical_before.json'
    assert mode!='before' or not before.exists(),'BEFORE_ALREADY_EXISTS'
    folders=('src','scripts','tests','configs','docs','paper','outputs','reference','experiments','deployment','lab_runs')
    paths=sorted(p for name in folders for p in(ROOT/name).rglob('*')if p.is_file()
        and not p.is_relative_to(HERE)and not p.is_relative_to(OUT)
        and not any(v in('__pycache__','.pytest_cache','pytest_tmp')or v.startswith('.venv')for v in p.parts))
    paths+=sorted(p for p in ROOT.iterdir()if p.is_file());hashes={};tick=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:
        for p,h in zip(paths,pool.map(sha,paths)):
            hashes[p.relative_to(ROOT).as_posix()]=h
            if time.monotonic()-tick>25:print(json.dumps({'stage':'R6_opaque_preservation','files':len(hashes)}),flush=True);tick=time.monotonic()
    sys.path.insert(0,str(ROOT));from scripts import sp_u12b4a_preserve as legacy
    seals=legacy.seals(hashes)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    status=subprocess.check_output(['git','status','--porcelain=v1','-uall'],cwd=ROOT,text=True,encoding='utf8')
    (LOCAL/f'git_status_{mode}.txt').write_text(status,encoding='utf8')
    if mode=='before':write(before,{'hashes':hashes,'B2_B3':seals,'Git_HEAD':head})
    else:
        old=read(before);changed=[n for n,h in old['hashes'].items()if hashes.get(n)!=h];added=[n for n in hashes if n not in old['hashes']]
        diff=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True,encoding='utf8')
        result={'PASS':not changed and not added and seals==old['B2_B3']and head==old['Git_HEAD']and diff.returncode==0,
            'protected_files':len(old['hashes']),'changed':changed,'unauthorized_added':added,'B2_B3':seals,'Git_HEAD_unchanged':head==old['Git_HEAD'],
            'Git_diff_check':diff.returncode,'Git_commit':False,'manuscript_drift':sum(n.startswith('paper/')for n in changed),
            'production_drift':sum(n.startswith(('src/','scripts/'))for n in changed),'reference_drift':sum(n.startswith('reference/')for n in changed),
            'R5_drift':sum(n.startswith(('deployment/r5_','lab_runs/r5_','SP_U12B5_R5_'))for n in changed),
            'scientific_values_decoded':0}
        write(LOCAL/'historical_preservation.json',result);assert result['PASS'],result
    print(json.dumps({'PASS':True,'stage':mode,'protected_files':len(hashes)}),flush=True)
def review_r5():
    archive=ROOT/'SP_U12B5_R5_Results_20261008_193708.zip';assert sha(archive)==R5_ZIP_SHA,'R5_RETURN_ZIP_SHA_MISMATCH'
    with zipfile.ZipFile(archive)as z:
        names=z.namelist();assert len(names)==len(set(names))
        assert all(not Path(n).is_absolute()and'..'not in Path(n).parts for n in names)
        assert all(Path(n).suffix.lower()not in('.npy','.npz','.pkl','.pickle')for n in names)
        mf=json.loads(z.read('result_manifest.json'));verified=0
        for n,v in mf['files'].items():
            h=hashlib.sha256()
            with z.open('r5_128gb/'+n)as f:
                for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
            assert h.hexdigest()==v['sha256'],'R5_LEAF_SHA_MISMATCH:'+n;verified+=1
        def member(n):return json.loads(z.read('r5_128gb/'+n))
        stage={n:member(n+'_stage.json')for n in('preflight','cpu','smoke12000','smoke15000')}
        results={}
        for n,s in stage.items():
            assert s['state']=='COMPLETE'and s['PASS']and s['package_manifest_SHA256']==mf['package_manifest_SHA256']
            path=s['job'].split('lab_runs/r5_128gb/',1)[1]+'/stage_result.json'
            assert hashlib.sha256(z.read('r5_128gb/'+path)).hexdigest()==s['result_sha256'];results[n]=member(path)
        report=member('MFA_128GB_HOST_QUALIFICATION_REPORT.json')
        proposal=member('PROPOSED_FORMAL_B5B_EXECUTION_CONFIG.json')
        assert report['PASS']and report['cross_host_fidelity_PASS']and report['actual_laboratory_tests_performed']
        assert proposal['BLAS_threads']==6 and proposal['OpenMP_threads']==1
        assert report['formal_candidates_trained']==report['validation_scientific_score']==report['target_scientific_data_reads']==0
        assert sha(ROOT/'deployment/r5_128gb_v1/package_manifest.json')==mf['package_manifest_SHA256']
        value={'PASS':True,'archive_sha256':R5_ZIP_SHA,'result_manifest_SHA256':hashlib.sha256(z.read('result_manifest.json')).hexdigest(),
            'R5_package_manifest_SHA256':mf['package_manifest_SHA256'],'technical_leaf_files_verified':verified,
            'stages':stage,'qualification_report':report,'proposed_execution_config':proposal,
            'scope':'actual returned technical evidence; no scientific outcomes','formal_training_started':False}
        write(HERE/'R5_EVIDENCE.json',value);write(LOCAL/'r5_review.json',value)
        print(json.dumps({k:v for k,v in value.items()if k not in('stages','qualification_report','proposed_execution_config')},indent=2))
        print(json.dumps({'N12000':results['smoke12000']['resources'],'N15000':results['smoke15000']['resources']},indent=2))
def metadata():
    for n in('configs/sp_u12b5b_mfa_selection.yaml','outputs/sp_u12b5b_candidate_seed_registry.json',
              'outputs/sp_u12b5b/input_manifest.json','outputs/sp_u12b5b_preparation_split_manifest.json','outputs/sp_u12b5b/validation_noise_registry.json'):
        v=read(ROOT/n);print(n)
        if n.endswith('.yaml'):
            print(json.dumps({k:x for k,x in v.items()if k!='trajectory_ids'},indent=2))
            assert v['trajectory_ids']=={'training':list(range(400)),'validation':list(range(400,500)),'full_retraining':list(range(500))}
        elif n.endswith('split_manifest.json'):
            for part in v['origin_partitions'].values():
                assert part['train_ids']==list(range(400))and part['validation_ids']==list(range(400,500))
            print(json.dumps({k:x for k,x in v.items()if k!='origin_partitions'},indent=2))
            print('all three origin_partitions train0..399/validation400..499 verified')
        elif n.endswith('validation_noise_registry.json'):
            assert isinstance(v,list)
            print(json.dumps({'type':'list','rows':len(v),'sample':v[:2]},indent=2))
        else:print(json.dumps(v,indent=2))
def contracts():
    base=ROOT/'outputs/mfa_lab_migration_v1/package/Project_MFA_Lab/migration_manifest.json'
    b=read(base);cfg=read(ROOT/'configs/sp_u12b5b_mfa_selection.yaml')
    seeds=read(ROOT/'outputs/sp_u12b5b_candidate_seed_registry.json')
    inputs=read(ROOT/'outputs/sp_u12b5b/input_manifest.json')
    assert seeds['candidates']==cfg['candidate_order'] and seeds['seeds_reused_exactly'] and not seeds['seeds_regenerated']
    for row in seeds['candidates']:
        for key,role in [('seed','candidate'),('final_seed','full_retraining')]:
            token=f"SP_U12B5A_MFA_AUTHOR2023_V1|{role}|K={row['K']}|L={row['rank']}|fit=0"
            assert row[key]==int.from_bytes(hashlib.sha256(token.encode()).digest()[:4],'big')
    shapes={'selection_train.npy':[12000,4032],'full_retraining.npy':[15000,4032],
            'validation_truth.npy':[3,100,6,3968],'validation_pilots.npy':[3,2,100,6,64]}
    data={}
    for name,shape in shapes.items():
        rel='outputs/sp_u12b5b/'+name
        assert inputs['files'][name]==b['files'][rel]['sha256']
        data[rel]={'sha256':inputs['files'][name],'shape':shape,'dtype':'<c16','fortran_order':False,'source':'original migration manifest + frozen input manifest'}
    frozen={n:v['sha256']for n,v in b['files'].items()if n not in('necessary_files.csv','migration_provenance.json') and not n.startswith(('docs/','tests/','scripts/','deployment/mfa_lab_v1/wheelhouse/'))}
    # The original runner remains preserved but is never imported by R6.
    for n in ('scripts/sp_u12b5b_prepare.py','scripts/sp_u12b5b_model.py','scripts/sp_u12b5b_worker.py','scripts/run_sp_u12b5b.py',
              'experiments/mfa_low_memory_fidelity_r4/canonical_validation_contract.py',
              'deployment/r5_128gb_v1/windows_resources.py','deployment/r5_128gb_v1/supervisor.py','deployment/r5_128gb_v1/package_manifest.json',
              'deployment/r5_128gb_v1/protocol.json'):
        frozen[n]=sha(ROOT/n)
    assert all(not n.startswith(('outputs/sp_u12b2','outputs/sp_u12b3'))for n in frozen)
    metadata=[n for n in frozen if n not in data and Path(n).suffix.lower()not in('.py','.ps1','.npy','.npz','.whl')]
    contract={'format':'R6_SCIENTIFIC_INPUT_CONTRACT_V1','scientific_protocol_status':'FROZEN',
        'base_manifest_sha256':sha(base),'frozen_hashes':frozen,'metadata_allowlist':metadata,'data':data,
        'canonical_origins':cfg['canonical_origins'],'candidate_order':seeds['candidates'],'pilot_geometry':cfg['pilot_geometry'],
        'train_ids':[0,399],'validation_ids':[400,499],'final_ids':[0,499],
        'train_frames':list(range(10)),'validation_frames':list(range(4,10)),'final_frames':list(range(10)),
        'array_order':'origin order, ascending trajectory ID, ascending frame; joint vector in C-order',
        'noise_variances':cfg['noise_variances'],'validation_predictions_per_candidate':3600,'validation_predictions_total':32400,
        'metric':cfg['selection_metric'],'tie':cfg['tie_breaker'],'policy':cfg['policy'],
        'policy_precedence':'mentor_amendment overrides historical policy.maxiter_exhaustion; MAX_ITER_REACHED is legal only with finite lawful parameters and valid inference/serialization',
        'initializations_per_fit':1,'restarts':0,'final_fit':'fresh original initialization using selected row final_seed; not a candidate checkpoint',
        'target_data_authorized':False,'preparation_arrays_already_owner_divided':True,
        'target_data_denied':['B2','B3','target trajectories','unapproved scientific inputs'],
        'header_check':'header-only read until formal mentor approval; numerical arrays only in role-scoped formal child',
        'canonical_cardinality_source':'experiments/mfa_low_memory_fidelity_r4/canonical_validation_contract.py'}
    sys.path.insert(0,str(ROOT))
    from src.baselines.mfa_reference_adapter import ProjectPilotInterface
    i=ProjectPilotInterface()
    geometry={'selected_full':i.selected_full.tolist(),'data_full':i.data_full.tolist(),'joint_full':i.joint_full.tolist(),
              'pilot_positions':i.pilot_positions.tolist(),'data_positions':i.data_positions.tolist(),
              'owner_symbols_real':i.owner_symbols.real.tolist(),'owner_symbols_imag':i.owner_symbols.imag.tolist()}
    contract['pilot_order_sha256']=hashlib.sha256(json.dumps(geometry,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    with zipfile.ZipFile(ROOT/'SP_U12B5_R5_Results_20261008_193708.zip')as z:
        contract['lab_prerequisite_hashes']={'lab_runs/r5_128gb/'+name:hashlib.sha256(z.read('r5_128gb/'+name)).hexdigest()
            for name in ('MFA_128GB_HOST_QUALIFICATION_REPORT.json','PROPOSED_FORMAL_B5B_EXECUTION_CONFIG.json')}
    contract['lab_prerequisite_hashes'].update({n:b['files'][n]['sha256']for n in('necessary_files.csv','migration_provenance.json')})
    write(HERE/'input_contract.json',contract)
    e=read(HERE/'R5_EVIDENCE.json')
    p={'format':'R6_RESOURCE_AMENDMENT_V1','status':'MENTOR_APPROVAL_PENDING','historical_contract_overwritten':False,
       'BLAS_threads':6,'OpenMP_threads':1,'candidate_concurrency':1,'sample_seconds':.1,
       'limits':{'RSS_bytes':48*2**30,'private_commit_bytes':64*2**30,'min_available_physical_bytes':32*2**30,'min_remaining_commit_bytes':32*2**30},
       'launch_minimum':read(ROOT/'deployment/r5_128gb_v1/protocol.json')['launch_minimum'],
       'worker_seconds':{'train':86400,'validate':86400,'final':86400,'unit-probe':10},
       'candidate_training_plus_validation_seconds':86400,
       'reference_R5_archive_sha256':e['archive_sha256'],'reference_R5_result_manifest_SHA256':e['result_manifest_SHA256'],
       'evidence':'R5_EVIDENCE.json; actual two-update N12000 RSS15.874/private16.242 GiB and N15000 RSS19.323/private19.698 GiB; verified BLAS6/OMP1 fidelity and CPU selection',
       'justification':{'48GiB_RSS':'above measured 19.323GiB peak with host headroom; finite safety envelope, no 100-update guarantee',
          '64GiB_private_commit':'above measured 19.698GiB peak; retain R5 protected kernel JobObject threshold',
          '32GiB_floors':'retain R5 system safety envelope on measured127.92GiB host',
          '80GiB_launch':'retain R5 launch requirement before any heavy allocation',
          '6_BLAS_1_OMP':'R5 CPU selection and cross-host fidelity PASS; native KMeans nested BLAS cap remains author/sklearn behavior',
          '24h':'retain original B5B budget; per-job plus stricter combined candidate training+validation budget',
          'concurrency1':'retain original single-model schedule; avoids overlapping original dense covariance/transient allocations',
          '100ms':'retain R5 sampled monitor; sampling gaps recorded, native private commit cap is kernel enforced; RSS/floor STOP sampled'},
       'no_retry':True,'math_hyperparameters_changed':False,'R5_only_two_synthetic_updates':True,
       'cost_status':'R5_EXTRAPOLATED_NOT_FULL_FIT_MEASURED','cost_hours':{'nine_candidates_central':1.42,'nine_candidates_conservative':13.11,
          'final_central':[.06,.48],'final_conservative':[.54,4.39],'validation_central':.78,'future_inference':[2.09,8.87]},
       'cost_may_change_experiment':False,'enable_only_after_next_mentor_approval':True}
    write(HERE/'resource_amendment.json',p)
    assert p['limits']==read(ROOT/'deployment/r5_128gb_v1/protocol.json')['limits']
    write(HERE/'scientific_protocol_audit.json',{'PASS':True,'scientific_config_sha256':sha(ROOT/'configs/sp_u12b5b_mfa_selection.yaml'),
        'seed_registry_sha256':sha(ROOT/'outputs/sp_u12b5b_candidate_seed_registry.json'),
        'all_candidate_rows':seeds['candidates'],'seed_roles_verified_against_B5A':18,'seed_regeneration':False,
        'train_trajectory_count':1200,'train_N':12000,'val_trajectory_count':300,'val_predictions_per_candidate':3600,'val_predictions_total':32400,
        'final_trajectory_count':1500,'final_N':15000,'D':4032,'dtype':'complex128','trajectory_split_preserved':True,
        'metric_preserved':True,'exact_tie_preserved':True,'array_hashes_from_original_migration':True,
        'historical_contract_unchanged':True,'mentor_maxiter_amendment_precedence_explicit':True,
        'formal_stage_instances_created':0,'formal_scores_generated':0,'formal_selected_structure':'NONE','final_model':'NONE'})
    # Byte-identical R5 containment API, new namespace only.
    if not (HERE/'windows_resources.py').exists():
        (HERE/'windows_resources.py').write_bytes((ROOT/'deployment/r5_128gb_v1/windows_resources.py').read_bytes())
    source=(ROOT/'deployment/r5_128gb_v1/supervisor.py').read_text(encoding='utf8')
    source=source.replace("assert action=='unit-probe','PERSONAL_NUMERICAL_BENCHMARK_FORBIDDEN'", "assert action=='unit-probe','PERSONAL_FORMAL_EXECUTION_FORBIDDEN'")
    source=source.replace("os.environ.get('R5_PERSONAL_SELFTEST')", "os.environ.get('R6_PERSONAL_SELFTEST')")
    source=source.replace('SP_U12B5_R5_','SP_U12B5_R6_')
    source=source.replace("state['CPU_utilization_recorded_in']", "state['CPU_utilization_recorded_in']")
    if not (HERE/'supervisor.py').exists():(HERE/'supervisor.py').write_text(source,encoding='utf8')
    write(HERE/'monitor_reuse.json',{'Windows_JobObject_source':'deployment/r5_128gb_v1/windows_resources.py',
      'Windows_JobObject_sha256':sha(ROOT/'deployment/r5_128gb_v1/windows_resources.py'),'copied_bytes_identical':True,
      'supervisor_source':'deployment/r5_128gb_v1/supervisor.py','supervisor_source_sha256':sha(ROOT/'deployment/r5_128gb_v1/supervisor.py'),
      'changes':['R6 namespace and personal-test flag/error label; local synthetic active/STOP markers isolated in personal job directory; all kernel and sampling plumbing retained'],
      'host_inventory':'R3 portable/host_inventory.py inventory-only; no R3/R4 MFA implementation imported'})
    print(json.dumps({'PASS':True,'frozen_files':len(frozen),'data_files':4,'candidate_seeds_verified':9,'proposal_status':p['status']}))
def freeze():
    files={p.relative_to(ROOT).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(HERE.rglob('*'))
           if p.is_file() and p.name not in('prepare.py','deployment_check.py','package_manifest.json') and '__pycache__'not in p.parts}
    assert all(not n.endswith(('.npy','.npz','.whl','.zip'))for n in files)
    write(HERE/'package_manifest.json',{'format':'R6_INCREMENTAL_PACKAGE_V1','files':files,
       'resource_amendment_status':'MENTOR_APPROVAL_PENDING','contains_preparation_data':False,
       'contains_B2_B3_scientific_outcomes':False,'contains_personal_environment':False,
       'frozen_prior_inputs':'input_contract.json; original lab migration_manifest.json','new_namespace':'deployment/r6_formal_v1'})
    print(json.dumps({'PASS':True,'payload_files':len(files),'package_manifest_sha256':sha(HERE/'package_manifest.json')}))
def package():
    manifest=read(HERE/'package_manifest.json');names=list(manifest['files'])+['deployment/r6_formal_v1/package_manifest.json']
    for name,row in manifest['files'].items():assert sha(ROOT/name)==row['sha256']
    directory=LOCAL/'package';directory.mkdir(parents=True,exist_ok=True)
    dest=directory/'SP_U12B5_R6_Formal_Execution_Package.zip'
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9)as z:
        for name in names:z.write(ROOT/name,name)
    with zipfile.ZipFile(dest)as z:
        assert z.namelist()==names
        for name in names:assert hashlib.sha256(z.read(name)).hexdigest()==sha(ROOT/name)
    h=sha(dest);(directory/(dest.name+'.sha256')).write_text(h+'  '+dest.name+'\n',encoding='ascii')
    import csv
    with(directory/'SP_U12B5_R6_File_Manifest.csv').open('w',encoding='utf-8-sig',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);writer.writeheader()
        for name in names:writer.writerow({'path':name,'bytes':(ROOT/name).stat().st_size,'sha256':sha(ROOT/name)})
    (directory/'Install-R6.ps1').write_bytes((HERE/'Install-R6.ps1').read_bytes())
    report={'PASS':True,'path':str(dest),'sha256':h,'bytes':dest.stat().st_size,'files':len(names),'uncompressed_bytes':sum((ROOT/n).stat().st_size for n in names),
        'manifest_sha256':sha(HERE/'package_manifest.json'),'target_root':'D:/Research/Project_MFA_Lab','conflict_policy':'check all paths before extract; any existing destination STOP',
        'preparation_arrays_in_zip':0,'environment_or_keys_in_zip':0,'B2_B3_outputs_in_zip':0}
    write(LOCAL/'package_report.json',report);print(json.dumps(report,indent=2))
def finish():
    import re
    t=read(LOCAL/'toy_test_report.json');q=read(LOCAL/'deployment_qa.json');h=read(LOCAL/'historical_preservation.json');p=read(LOCAL/'package_report.json')
    manifest=read(HERE/'package_manifest.json');msha=sha(HERE/'package_manifest.json')
    assert t['PASS']and q['PASS']and h['PASS']and t['tested_package_manifest_sha256']==q['package_manifest_sha256']==p['manifest_sha256']==msha
    assert q['package_sha256']==p['sha256']==sha(p['path'])
    assert all(sha(ROOT/n)==v['sha256']for n,v in manifest['files'].items())
    assert not any((OUT/n).exists()for n in('candidates','final','SELECTION_MANIFEST.json','FINAL_MODEL_MANIFEST.json','MENTOR_APPROVAL.json'))
    cfg=read(ROOT/'configs/sp_u12b5b_mfa_selection.yaml');contract=read(HERE/'input_contract.json');resource=read(HERE/'resource_amendment.json')
    assert cfg['resource_contract']['BLAS_threads']==1 and cfg['resource_contract']['max_process_bytes']==6*2**30 and resource['status']=='MENTOR_APPROVAL_PENDING'
    models=[read(LOCAL/'toy_models'/r['name']/'toy_report.json')for r in contract['candidate_order']]
    assert len(models)==9 and all(v['PASS']and v['N']==96 and v['D']==32 and v['bitwise_author_fit_parity']for v in models)
    result={'verdict':'MFA_FORMAL_EXECUTION_PACKAGE_READY','R5_review':read(LOCAL/'r5_review.json')['PASS'],
        'R5_result_manifest_SHA256':read(HERE/'R5_EVIDENCE.json')['result_manifest_SHA256'],'resource_amendment_status':resource['status'],
        'resource_amendment_SHA256':sha(HERE/'resource_amendment.json'),'input_contract_SHA256':sha(HERE/'input_contract.json'),
        'package':p,'pytest_passed':int(re.search(r'(\d+) passed',t['pytest_stdout']).group(1)),'pytest_failed':0,
        'toy_models':models,'deployment_QA_PASS':True,'historical_preservation':h,
        'formal_candidates_trained':0,'real_validation_scores_generated':0,'selected_structure':'NONE','final_models_trained':0,
        'MENTOR_APPROVAL_record_created':False,'target_data_unsealed':False,'original_author_source_drift':0,'adapter_drift':0,
        'resource_amendment_applied_to_historical_contract':False,'production_code_drift':0,'manuscript_drift':0,'Git_commit':False,
        'Windows_commands':'deployment/r6_formal_v1/README_CN.md; technical actions only before mentor approval',
        'remaining_gates':['next mentor approval bound to final package/resource/input SHA','lab R6 Verify/Preflight/ToyTests/MonitorToy technical results','no full-fit success guarantee from R5 two updates'],
        'STOP':True}
    write(LOCAL/'R6_DELIVERY_REPORT.json',result)
    notes=('R6 preparation only. Original reference/adapter/history unchanged.\n'
           'Technical failures retained in pytest and deployment QA attempt logs: float64 fixture reduction mismatch; fault injection before PID handshake; migration-only metadata location; Windows PowerShell module inheritance; Windows NUL device exception; export sidecar variable shadow.\n'
           'Repairs changed deployment/testing only; no real scientific input decoded or scored. Final pytest and final ZIP relocation QA are bound to the delivered package manifest.\n'
           'No formal candidate, score, selection, final fit, target unseal or Git commit. STOP awaiting mentor.\n')
    (LOCAL/'change_log.txt').write_text(notes,encoding='utf8')
    print(json.dumps({'verdict':result['verdict'],'pytest_passed':result['pytest_passed'],'protected_files':h['protected_files'],
        'package_manifest_SHA256':msha,'resource_amendment_SHA256':result['resource_amendment_SHA256'],
        'input_contract_SHA256':result['input_contract_SHA256'],'formal_candidates_trained':0,'real_validation_scores':0,'final_models':0},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('before','after','review-r5','metadata','contracts','freeze','package','finish'));a=p.parse_args()
    if a.action in('before','after'):preserve(a.action)
    elif a.action=='review-r5':review_r5()
    elif a.action=='contracts':contracts()
    elif a.action=='freeze':freeze()
    elif a.action=='package':package()
    elif a.action=='finish':finish()
    else:metadata()
