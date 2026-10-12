"""Real PowerShell relocation QA; LAB failure/model claims are never simulated."""
from pathlib import Path
import json, os, shutil, subprocess, sys, uuid
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/sp_u12b5c_r1'
sys.path.insert(0,str(ROOT/'scripts'));from prepare_sp_u12b5c_r1 import sha,read,dump,need
def run(python):
    qa=OUT/('windows_qa_'+uuid.uuid4().hex[:8]);qa.mkdir();root=qa/'实验室搬迁 Root';root.mkdir();incoming=qa/'Incoming';incoming.mkdir()
    old=ROOT/'deployment/b5c_paired_v1';records=read(old/'package_manifest.json')['files']
    records['deployment/b5c_paired_v1/package_manifest.json']={'sha256':sha(old/'package_manifest.json')}
    for n in records:
        q=root/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,q)
    original=root/'lab_runs/b5c_paired';(original/'formal').mkdir(parents=True)
    for n in ('STOP.json','FORMAL_START_CONSUMED.json','formal/exception.json','formal/stderr.log','formal/status.json'):
        (original/n).write_text('SYNTHETIC_FAILURE_PRESERVATION_FIXTURE_ONLY\n',encoding='utf8')
    cache=root/'deployment/b5c_paired_v1/__pycache__';cache.mkdir()
    (cache/'science.cpython-313.pyc').write_bytes(b'UNAUTHENTICATED_SYNTHETIC_CACHE_MUST_NEVER_BE_READ')
    protected={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file()}
    # Only exact technical JSON is transferred into the QA stand-in. No NPZ/NPY.
    data=root/'lab_inputs/b5c_b2_v1';inventory=read(old/'data_manifest.json')['files'];metadata=[]
    for n in inventory:
        if n in ('configs/sp_u12b2b_sealed.json','outputs/sp_u12b2a_seed_registry.json') or n.startswith('outputs/sp_u12b2b_sealed/') and n.endswith('.json'):
            q=data/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,q);metadata.append(n)
    need(len(metadata)==15002,'QA_EXACT_METADATA_COUNT');need(not list(data.rglob('*.npz')) and not list(data.rglob('*.npy')),'QA_NUMERIC_DATA_FORBIDDEN')
    z=OUT/'SP_U12B5C_R1_Seed_Identity_Fix.zip';installer=incoming/'Install-B5C-R1.ps1';shutil.copy2(OUT/'Install-B5C-R1.ps1',installer)
    ps=['powershell','-NoProfile','-ExecutionPolicy','Bypass'];reports=[]
    def call(label,command,ok=True):
        r=subprocess.run(command,cwd=root,capture_output=True,encoding='utf8',errors='replace')
        (qa/(label+'.stdout.log')).write_text(r.stdout,encoding='utf8');(qa/(label+'.stderr.log')).write_text(r.stderr,encoding='utf8')
        passed=(r.returncode==0) if ok else (r.returncode!=0)
        reports.append({'test':label,'PASS':passed,'exit_code':r.returncode});need(passed,'WINDOWS_QA_FAILED:'+label)
        for n,h in protected.items():need(sha(root/n)==h,'OLD_FILES_CHANGED:'+n)
        print(label,'PASS',flush=True);return r
    install=ps+['-File',str(installer),'-ProjectRoot',str(root),'-ZipPath',str(z),'-ExpectedSHA256',sha(z),'-Python',python]
    call('wrong_zip_hash_refused',install[:-3]+['0'*64,'-Python',python],False)
    call('add_only_install',install)
    call('duplicate_install_refused',install,False)
    prefix=ps+['-File',str(root/'deployment/b5c_paired_r1_v1/R1.ps1'),'-Python',python]
    for action in ('Verify','ToyTests','IdentityAudit','Status','Export'):call(action,prefix+['-Action',action])
    controller=root/'deployment/b5c_paired_r1_v1/control.py';worker=root/'deployment/b5c_paired_r1_v1/worker.py'
    call('controller_formal_refused',[python,'-B',str(controller),'Run','--formal'],False)
    call('worker_formal_refused',[python,'-B',str(worker),'--action','evaluate','--job',str(root/'lab_runs/b5c_paired_r1/formal'),'--blas','6','--openmp','1'],False)
    need(not(root/'lab_runs/b5c_paired_r1/FORMAL_START_CONSUMED.json').exists() and not(root/'lab_runs/b5c_paired_r1/formal').exists(),'QA_FORMAL_RUN_OCCURRED')
    jobs=list((root/'lab_runs/b5c_paired_r1/technical').glob('identity_*/identity_report.json'));need(len(jobs)==1,'IDENTITY_QA_REPORT_REQUIRED')
    identity=read(jobs[0]);need(identity['corrected_proof_passes']==5000 and identity['formal_scientific_arrays_deserialized']==0,'IDENTITY_QA_NOT_PASS')
    need(all(sha(data/n)==inventory[n]['sha256'] for n in metadata),'QA_METADATA_CHANGED')
    result={'PASS':True,'tests':reports,'qa_root':str(root),'technical_metadata_files':len(metadata),'corrected_identity_checks':5000,
        'protected_original_files':len(protected),'original_files_changed':[],'formal_predictions':0,'formal_model_accessed':False,
        'numeric_B2_B3_data_transferred':False,'actual_LAB_failed_attempt_verified':False,'unauthenticated_pyc_present_untouched_and_ignored':True,
        'scope':'Actual Windows PowerShell deployment in relocated QA root; synthetic failure markers; exact authenticated PERSONAL technical identity copies only'}
    dump(OUT/'windows_qa.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--python',default=sys.executable);run(p.parse_args().python)
