"""Technical identity forensics only: never deserialize scientific arrays."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, subprocess, sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/sp_u12b5c_r1'
OLD = ROOT/'deployment/b5c_paired_v1'
ROLES = ('sionna_seed','torch_seed','numpy_seed','bits_seed_tx0','bits_seed_tx1','noise_seed')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def need(v,m):
    if not v: raise RuntimeError(m)
def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f: json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def snapshot():
    contract=read(OLD/'contract.json')
    files={ROOT/n for n in contract['source_dependencies']}
    folders=[ROOT/'paper',OLD,ROOT/'lab_runs/b5c_paired']
    folders += [p for p in (ROOT/'outputs').iterdir() if p.is_dir() and p.name.startswith(('sp_u12b2','sp_u12b3')) and 'sealed' in p.name]
    for folder in folders:
        files.update(p for p in folder.rglob('*') if p.is_file())
    files.update(p for p in (ROOT/'outputs/sp_u12b5c_preparation').glob('*') if p.is_file())
    files.update(ROOT/n for n in ('FINAL_MODEL_MANIFEST.json','FINAL_MODEL_MANIFEST.sha256','SELECTION_MANIFEST.json','SELECTION_MANIFEST.sha256','configs/sp_u12b2b_sealed.json','outputs/sp_u12b2a_seed_registry.json','outputs/sp_u12b2b_sealed_manifest.json','outputs/sp_u12b3h_manifest.json','outputs/sp_u12b3h_preservation.json','outputs/sp_u12b3h_history_preservation.json','scripts/run_sp_u12b2a.py','scripts/run_sp_u12b2b.py','scripts/sp_u12b2b_common.py'))
    values={}
    for i,p in enumerate(sorted(files)):
        values[p.relative_to(ROOT).as_posix()]={'sha256':sha(p),'bytes':p.stat().st_size}
        if i%10000==0: print('Opaque preservation files hashed:',i,flush=True)
    return {'files':values,'git_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'tracked_diff_sha256':hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=ROOT)).hexdigest(),
        'staged_diff_sha256':hashlib.sha256(subprocess.check_output(['git','diff','--cached','--binary'],cwd=ROOT)).hexdigest(),
        'scientific_arrays_deserialized':0,'lab_failure_records_received':False}
def before():
    v=snapshot();dump(OUT/'preservation_before.json',v)
    print(json.dumps({'protected_files':len(v['files']),'bytes':sum(x['bytes'] for x in v['files'].values())}))
def after():
    old=read(OUT/'preservation_before.json');now=snapshot()
    changed=[n for n,v in old['files'].items() if now['files'].get(n)!=v]
    extra=sorted(set(now['files'])-set(old['files']))
    ok=not changed and not extra and all(now[k]==old[k] for k in ('git_HEAD','tracked_diff_sha256','staged_diff_sha256'))
    result={'PASS':ok,'protected_files':len(old['files']),'changed':changed,'unexpected_added_protected_files':extra,
        'scientific_arrays_deserialized':0,'B2_B3_preservation':'all local sealed directories opaque-byte SHA256 comparison',
        'manuscript_modified':False if not any(n.startswith('paper/') for n in changed) else True,
        'original_B5C_namespace_modified':False if not any(n.startswith('deployment/b5c_paired_v1/') for n in changed) else True,
        'original_personal_B5C_logs_modified':False if not any(n.startswith('lab_runs/b5c_paired/') for n in changed) else True,
        'actual_lab_failed_attempt_verified_on_personal':False,'git_commit_created':False}
    dump(OUT/'preservation_after.json',result);need(ok,'PRESERVATION_DRIFT');print(json.dumps(result))
def metadata_audit(root, data_root, revised_science=None):
    """Authenticate fixed JSON paths before parsing; reject numeric scientific fields."""
    old=root/'deployment/b5c_paired_v1'
    contract=read(old/'contract.json');inventory=read(old/'data_manifest.json')['files']
    count=0
    def authenticated(n):
        nonlocal count
        p=data_root/n;v=inventory[n]
        need(p.suffix=='.json','ONLY_TECHNICAL_JSON_MAY_BE_PARSED')
        need(p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],'FROZEN_METADATA_HASH_DRIFT:'+n)
        obj=read(p);count+=1
        need(not (set(obj)&{'nmse','ber','pooled_nmse_db','frame_error','trajectory_error','frame_truth','trajectory_truth','bit_error','bit_errors','method_coordinates','coordinates','estimate_db','simul_lower','simul_upper'}),'SCIENTIFIC_FIELD_FORBIDDEN')
        return obj
    cfg=authenticated('configs/sp_u12b2b_sealed.json')
    # Config is an identity/protocol declaration, not scientific outcomes.
    reg=authenticated('outputs/sp_u12b2a_seed_registry.json')
    ix={}
    for r in reg['rows']:
        if r['group']!='EVAL':continue
        key=(r['condition'],r['trajectory_id'],r['frame_id'],r['role'])
        need(key not in ix and type(r['seed']) is int,'SEED_REGISTRY_IDENTITY_OR_TYPE_DRIFT');ix[key]=r['seed']
    need(len(ix)==165000,'SEED_REGISTRY_CARDINALITY')
    need(cfg['conditions']==contract['conditions'] and cfg['snr_db']==contract['snrs'],'CONDITION_SNR_MAPPING_DRIFT')
    # Compile precisely the authenticated source function, bypassing importlib's
    # attempted cache reads (-B disables cache writes, not cache reads).
    import ast, types
    facade=types.SimpleNamespace(require=need,contract=lambda:contract)
    def source_checker(path):
        manifest=read(path.parent/'package_manifest.json')['files']
        n=path.relative_to(root).as_posix();data=path.read_bytes()
        need(hashlib.sha256(data).hexdigest()==manifest[n]['sha256'],'PROOF_CHECK_SOURCE_HASH_DRIFT')
        tree=ast.parse(data,filename=str(path))
        functions=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='proof_check']
        need(len(functions)==1,'UNIQUE_PROOF_CHECK_REQUIRED')
        namespace={'b':facade,'ROLES':ROLES}
        exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),namespace)
        return types.SimpleNamespace(proof_check=namespace['proof_check'])
    original=source_checker(old/'science.py')
    corrected=None
    if revised_science:
        corrected=source_checker(Path(revised_science))
    first=None;old_failures=0;new_passes=0;cells=[]
    hdigest=hashlib.sha256();ndigest=hashlib.sha256();pdigest=hashlib.sha256()
    for condition in contract['conditions']:
        for i in range(contract['N']):
            prefix='outputs/sp_u12b2b_sealed/'
            proof=authenticated(prefix+f'raw_evaluation/{condition}/{i:04d}.json')
            hi=authenticated(prefix+f'cache/{condition}/H_{i:04d}.json')
            ci=authenticated(prefix+f'cache/{condition}/comm_{i:04d}.json')
            actual=proof['channel_seeds'];old_expected={r:ix[condition,i,-1,r] for r in ROLES[:3]}
            expected={r:ix[condition,i,-1 if r in ROLES[:3] else 0,r] for r in ROLES}
            need(type(actual) is dict and set(actual)==set(ROLES) and all(type(v) is int for v in actual.values()),'FROZEN_SIX_FIELD_INTEGER_SCHEMA_DRIFT')
            need(actual==expected==hi['channel_seeds'],'TRUE_SCIENTIFIC_SEED_MISMATCH')
            frames=[{r:ix[condition,i,-1 if r in ROLES[:3] else j,r] for r in ROLES} for j in range(10)]
            need(proof['communication_seeds']==frames==ci['frame_seeds'],'COMMUNICATION_SEED_MISMATCH')
            need((hi['condition'],hi['trajectory_id'],hi['group'])==(condition,i,'EVAL') and (ci['condition'],ci['trajectory_id'])==(condition,i),'CACHE_TRAJECTORY_MAPPING_DRIFT')
            need(proof['H_sha256']==hi['H_sha256'],'H_ARRAY_HASH_LINK_DRIFT')
            need(proof['H_artifact_sha256']==hi['artifacts'][0]['sha256']==inventory[prefix+f'cache/{condition}/H_{i:04d}.npz']['sha256'],'H_FILE_HASH_LINK_DRIFT')
            need(proof['communication_sha256']==ci['artifacts'][0]['sha256']==inventory[prefix+f'cache/{condition}/comm_{i:04d}.npz']['sha256'],'COMM_FILE_HASH_LINK_DRIFT')
            for info in (hi,ci,proof):
                for rec in info['artifacts']: need(rec['sha256']==inventory[rec['path']]['sha256'] and rec['bytes']==inventory[rec['path']]['bytes'],'FROZEN_ARTIFACT_COMMIT_LINK_DRIFT')
            need(actual!=old_expected,'ORIGINAL_FAILURE_NOT_REPRODUCIBLE')
            try:original.proof_check(proof,condition,i,cfg,ix)
            except RuntimeError as e:need(str(e)=='CHANNEL_SEED_IDENTITY_DRIFT','UNEXPECTED_ORIGINAL_FAILURE');old_failures+=1
            else:raise RuntimeError('ORIGINAL_COMPARISON_UNEXPECTEDLY_PASSED')
            if first is None:
                first={'condition':condition,'canonical_condition':contract['conditions'][condition],'trajectory_id':i,'snr_scope':contract['snrs'],
                    'first_planned_cell_snr_db':contract['snrs'][0],'comparison_occurs_before_SNR_loop':True,
                    'actual':actual,'original_expected':old_expected,'correct_six_field_expected':expected,
                    'actual_types':{k:type(v).__name__ for k,v in actual.items()},'expected_types':{k:type(v).__name__ for k,v in old_expected.items()},
                    'actual_field_order':list(actual),'original_expected_field_order':list(old_expected),
                    'additional_fields':sorted(set(actual)-set(old_expected)),
                    'shared_field_value_mismatches':{k:[actual[k],old_expected[k]] for k in old_expected if actual[k]!=old_expected[k]},
                    'proof_sha256':sha(data_root/(prefix+f'raw_evaluation/{condition}/{i:04d}.json')),
                    'H_receipt_sha256':sha(data_root/(prefix+f'cache/{condition}/H_{i:04d}.json'))}
            ins={(v['snr_db'],v['frame_id']):v for v in proof['inputs']}
            need(len(proof['inputs'])==len(ins)==70 and set(ins)=={(s,j) for s in contract['snrs'] for j in range(10)},'70_INPUT_IDENTITY_GRID_DRIFT')
            for (snr,j),v in ins.items():
                need(set(v)=={'snr_db','frame_id','truth_sha256','unit_noise_sha256','pilot_sha256','Y_sha256'},'INPUT_IDENTITY_SCHEMA_DRIFT')
                need(v['unit_noise_sha256']==ci['unit_noise_sha256'][j],'NOISE_ARRAY_HASH_LINK_DRIFT')
                need(v['truth_sha256']==ins[contract['snrs'][0],j]['truth_sha256'],'CROSS_SNR_TRUTH_HASH_DRIFT')
                need(all(type(v[k]) is str and len(v[k])==64 and set(v[k])<=set('0123456789abcdef') for k in ('truth_sha256','unit_noise_sha256','pilot_sha256','Y_sha256')),'INVALID_SHA256_IDENTITY')
                pdigest.update(json.dumps([condition,i,snr,j,v],sort_keys=True).encode())
            hdigest.update(json.dumps([condition,i,hi['H_sha256']],sort_keys=True).encode())
            ndigest.update(json.dumps([condition,i,ci['unit_noise_sha256']],sort_keys=True).encode())
            if corrected:
                need(corrected.proof_check(proof,condition,i,cfg,ix)==ins,'CORRECTED_PROOF_CHECK_MISMATCH');new_passes+=1
        for snr in contract['snrs']:cells.append({'condition':condition,'snr_db':snr,'trajectories':1000,'scored_frames':6000})
    return {'PASS':True,'root_cause':'B5C expected three fields; frozen B2 consumed(j=0) serializes six fields',
        'first_mismatch':first,'real_scientific_seed_mismatches':0,'authenticated_metadata_files':count,'EVAL_registry_entries':len(ix),
        'raw_trajectories_checked':5000,'channel_seed_scalar_values_checked':30000,'communication_seed_scalar_values_checked':300000,
        'original_failure_reproductions':old_failures,'corrected_proof_passes':new_passes,'cells':cells,
        'full_input_identities_checked':350000,'scored_input_identities':210000,
        'H_identity_digest':hdigest.hexdigest(),'noise_identity_digest':ndigest.hexdigest(),'pilot_truth_Y_identity_digest':pdigest.hexdigest(),
        'original_H_noise_pilot_checks_unchanged':True,'formal_scientific_arrays_deserialized':0,'formal_predictions':0,
        'actual_lab_failed_attempt_inspected':False,'scope':'Authenticated PERSONAL B2 metadata; no laboratory model or scientific outcome decoding'}
def audit():
    revised=ROOT/'deployment/b5c_paired_r1_v1/science.py'
    result=metadata_audit(ROOT,ROOT,revised if revised.exists() else None)
    destination=OUT/('identity_after_fix.json' if revised.exists() else 'identity_original.json')
    if destination.exists():destination=OUT/'identity_authenticated_source_loading.json'
    dump(destination,result)
    print(json.dumps({k:v for k,v in result.items() if k!='cells'},ensure_ascii=False,indent=2))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=('before','audit','after'));x=a.parse_args();{'before':before,'audit':audit,'after':after}[x.action]()
