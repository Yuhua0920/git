"""Personal-only R7 preservation/package builder. Never shipped to laboratory."""
from pathlib import Path
import sys,json,time,hashlib,subprocess,concurrent.futures,zipfile,csv,argparse
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=ROOT/'lab_runs/r7_sequential/preparation_personal'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def preserve(mode):
    before=OUT/'historical_before.json';assert mode!='before' or not before.exists()
    files=sorted(p for d in ('src','scripts','tests','configs','docs','paper','outputs','reference','experiments','deployment','lab_runs') for p in (ROOT/d).rglob('*') if p.is_file()
        and not p.is_relative_to(HERE) and not p.is_relative_to(ROOT/'lab_runs/r7_sequential')
        and not any(v in ('__pycache__','.pytest_cache','pytest_tmp') or v.startswith('.venv') for v in p.parts))
    files+=sorted(p for p in ROOT.iterdir() if p.is_file());hashes={};last=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for p,h in zip(files,pool.map(sha,files)):
            hashes[p.relative_to(ROOT).as_posix()]=h
            if time.monotonic()-last>25:print(json.dumps({'stage':'R7_opaque_preservation','files':len(hashes)}),flush=True);last=time.monotonic()
    sys.path.insert(0,str(ROOT));from scripts.sp_u12b4a_preserve import seals
    seal=seals(hashes);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    status=subprocess.check_output(['git','status','--porcelain=v1','-uall'],cwd=ROOT,text=True,encoding='utf8')
    OUT.mkdir(parents=True,exist_ok=True);(OUT/f'git_status_{mode}.txt').write_text(status,encoding='utf8')
    if mode=='before':write(before,{'hashes':hashes,'seals':seal,'Git_HEAD':head})
    else:
        old=read(before);changed=[n for n,h in old['hashes'].items() if hashes.get(n)!=h];added=[n for n in hashes if n not in old['hashes']]
        diff=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True,encoding='utf8')
        result={'PASS':not changed and not added and seal==old['seals'] and head==old['Git_HEAD'] and diff.returncode==0,
                'protected_files':len(old['hashes']),'changed':changed,'unauthorized_added':added,'B2_B3':seal,'Git_HEAD':head,'Git_HEAD_unchanged':head==old['Git_HEAD'],
                'Git_diff_check':diff.returncode,'Git_commit':False,'scientific_values_decoded':0,
                'manuscript_drift':sum(n.startswith('paper/') for n in changed),'production_drift':sum(n.startswith(('src/','scripts/')) for n in changed),
                'R5_R6_drift':sum(n.startswith(('deployment/r5','deployment/r6','lab_runs/r5','lab_runs/r6')) for n in changed)}
        write(OUT/'historical_preservation.json',result);assert result['PASS'],result
    print(json.dumps({'PASS':True,'stage':mode,'protected_files':len(hashes),'B2_files':seal['B2']['files'],'B3_files':seal['B3']['files']}),flush=True)
def contracts():
    r6=ROOT/'deployment/r6_formal_v1'; original=read(r6/'input_contract.json');resources=read(r6/'resource_amendment.json')
    rows=read(ROOT/'outputs/sp_u12b5b_candidate_seed_registry.json')['candidates']
    assert rows==original['candidate_order'];assert [r['name'] for r in rows]==['K1_r4','K1_r8','K1_r16','K2_r4','K2_r8','K2_r16','K4_r4','K4_r8','K4_r16']
    required={n:sha(ROOT/n) for n in read(r6/'package_manifest.json')['files']}
    required['deployment/r6_formal_v1/package_manifest.json']=sha(r6/'package_manifest.json')
    value={'format':'R7_QUEUE_CONTRACT_V1','status':'MENTOR_APPROVAL_PENDING','candidate_order':rows,'resource_contract':resources,
           'R6_dependencies':required,'allowed_stages':['train','validate'],'after_nine':'QUEUE_COMPLETE_9_OF_9_THEN_STOP',
           'queue_total_timeout_seconds':None,'per_job_timeout_seconds':86400,'R6_stricter_combined_candidate_timeout_seconds':86400,
           'selection_automated':False,'final_fit_automated':False,'target_access':False,'automatic_retry':False,
           'original_author_worker':'deployment/r6_formal_v1/worker.py','timing_bridge_changes_math':False,
           'R6_global_lock_held_for_entire_queue':True,'candidate_outputs':'lab_runs/r6_formal/candidates/<candidate>/<train|validate>',
           'science':{k:original[k] for k in ('canonical_origins','data','metric','noise_variances','pilot_order_sha256','validation_predictions_per_candidate','validation_predictions_total')},
           'authorization_mechanism':'Exact import of independently supplied hash-bound human record and instruction; no generated approval or signature',
           'compatibility_verdict':'R6_HASH_BOUND_PROSPECTIVE_ATTESTATION_SUPPORTED_WITHOUT_MODIFICATION'}
    write(HERE/'queue_contract.json',value)
    write(OUT/'scientific_protocol_audit.json',{'PASS':True,'candidate_order':rows,'seeds_regenerated':False,'R6_dependencies':required,
          'policy':original['policy'],'policy_precedence':original['policy_precedence'],'science':value['science'],
          'formal_training_count':0,'validation_scores':0,'target_values_read':0,'resource_authorization':'MENTOR_APPROVAL_PENDING'})
def package():
    paths=sorted(p for p in HERE.iterdir() if p.is_file() and p.name not in ('prepare_personal.py','package_manifest.json','deployment_check.py'))
    files={p.relative_to(ROOT).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in paths}
    write(HERE/'package_manifest.json',{'format':'R7_INCREMENTAL_PACKAGE_V1','files':files,'approval_generated':False})
    paths.append(HERE/'package_manifest.json');dest=OUT/'package';dest.mkdir(parents=True,exist_ok=True)
    path=dest/'SP_U12B5_R7_Sequential_Execution_Package.zip'
    assert not path.exists(),'PACKAGE_ALREADY_FROZEN'
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,p.relative_to(ROOT).as_posix())
    h=sha(path);Path(str(path)+'.sha256').write_text(h+'  '+path.name+'\n',encoding='ascii')
    allfiles={p.relative_to(ROOT).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in paths}
    with(dest/'SP_U12B5_R7_File_Manifest.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','sha256','bytes']);w.writeheader();w.writerows({'path':n,**v} for n,v in allfiles.items())
    (dest/'Install-R7.ps1').write_bytes((HERE/'Install-R7.ps1').read_bytes())
    with zipfile.ZipFile(path) as z:
        assert len(z.namelist())==len(allfiles)==len(set(z.namelist()))
        for n,v in allfiles.items():assert hashlib.sha256(z.read(n)).hexdigest()==v['sha256']
    report={'PASS':True,'path':str(path),'sha256':h,'bytes':path.stat().st_size,'files':len(paths),'uncompressed_bytes':sum(v['bytes'] for v in allfiles.values()),
            'manifest_sha256':sha(HERE/'package_manifest.json'),'queue_contract_sha256':sha(HERE/'queue_contract.json'),'file_manifest':str(dest/'SP_U12B5_R7_File_Manifest.csv'),'files_manifest':allfiles,
            'formal_training_started':False,'approval_record_in_package':False}
    write(OUT/'package_report.json',report);print(json.dumps({k:v for k,v in report.items() if k!='files_manifest'},indent=2),flush=True)
if __name__=='__main__':
    action=sys.argv[1]
    if action in ('before','after'):preserve(action)
    elif action=='contracts':contracts()
    else:package()
