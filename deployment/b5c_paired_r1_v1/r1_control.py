"""R1 technical checks only. Formal restart requires a later mentor decision."""
from pathlib import Path
import argparse, json, os, subprocess, sys, traceback, uuid, zipfile
sys.dont_write_bytecode=True
import base as b
ORIGINAL_MANIFEST='556e96ddc788454d9060ad850ae982cba706879a318acbbbbc98a345670519b5'
def verify():
    result=b.verify()
    b.require(b.sha(b.ROOT/'deployment/b5c_paired_v1/package_manifest.json')==ORIGINAL_MANIFEST,'ORIGINAL_B5C_FREEZE_DRIFT')
    result.update(formal_restart_enabled=False,original_namespace_unchanged=True)
    return result
def protected():
    result={}
    for folder in (b.ROOT/'deployment/b5c_paired_v1',b.ROOT/'lab_runs/b5c_paired'):
        for q in folder.rglob('*'):
            if q.is_file():result[q.relative_to(b.ROOT).as_posix()]={'sha256':b.sha(q),'bytes':q.stat().st_size}
    return result
def metadata_scope():
    """Narrow, temporary-phase read policy; never admit numeric result arrays."""
    active={'enabled':True}
    exact={b.safe(b.ROOT,n) for n in b.read(b.HERE/'package_manifest.json')['files']}
    exact.add(b.HERE/'package_manifest.json')
    names={'configs/sp_u12b2b_sealed.json','outputs/sp_u12b2a_seed_registry.json'}
    for condition in b.contract()['conditions']:
        for i in range(1000):
            names.update('outputs/sp_u12b2b_sealed/'+name for name in (f'raw_evaluation/{condition}/{i:04d}.json',f'cache/{condition}/H_{i:04d}.json',f'cache/{condition}/comm_{i:04d}.json'))
    exact.update(b.safe(b.DATA,n) for n in names)
    runtimes=(Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve())
    def hook(event,args):
        if not active['enabled']:return
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
            writing=isinstance(mode,str) and any(c in mode for c in 'wax+') or isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if writing:b.require(p.is_relative_to(b.OUT.resolve()),'R1_HISTORICAL_WRITE_DENIED');return
            b.require(p.suffix.lower() not in ('.npz','.npy'),'R1_NUMERIC_ARRAY_READ_DENIED')
            b.require(p in exact or any(p.is_relative_to(q) for q in runtimes),'R1_METADATA_READ_DENIED:'+str(p))
        elif event in ('os.remove','os.rename','os.rmdir','os.mkdir','os.utime','os.link'):
            for p in args[:2] if event in ('os.rename','os.link') else args[:1]:
                if isinstance(p,(str,bytes,os.PathLike)):b.require(Path(os.fsdecode(p)).resolve().is_relative_to(b.OUT.resolve()),'R1_HISTORICAL_WRITE_DENIED')
        elif event in ('subprocess.Popen','os.system','os.exec'):raise RuntimeError('R1_METADATA_CHILD_EXECUTION_DENIED')
    sys.addaudithook(hook);return active
def identity_audit():
    verify();old=b.ROOT/'lab_runs/b5c_paired'
    b.require((old/'STOP.json').exists() and (old/'FORMAL_START_CONSUMED.json').exists(),'ORIGINAL_FAILED_STOP_AND_CONSUMPTION_REQUIRED')
    b.require(not(old/'execution.lock').exists(),'OLD_ATTEMPT_STILL_RUNNING')
    job=b.OUT/'technical'/('identity_'+uuid.uuid4().hex[:12]);job.mkdir(parents=True,exist_ok=False)
    prior=protected();b.write(job/'original_preservation_before.json',prior,True)
    b.write(job/'status.json',{'state':'RUNNING','formal_predictions':0},True)
    try:
        from identity_audit import metadata_audit
        guard=metadata_scope()
        try:report=metadata_audit(b.ROOT,b.DATA,b.HERE/'science.py')
        finally:guard['enabled']=False
        now=protected();b.require(now==prior,'ORIGINAL_FAILURE_OR_CODE_CHANGED')
        report.update(actual_lab_failed_attempt_inspected=False,lab_failed_files_opaque_hashes_preserved=True,
            scope='Authenticated LAB transferred identity metadata only; failed artifacts preserved by byte hashes; no outcome decoding')
        b.write(job/'original_preservation_after.json',{'PASS':True,'files':len(prior),'changed':[]},True)
        b.write(job/'identity_report.json',report,True)
        b.write(job/'status.json',{'PASS':True,'state':'COMPLETE','formal_predictions':0,'corrected_proof_passes':report['corrected_proof_passes']})
        return {'PASS':True,'job':str(job),'metadata_trajectories':5000,'formal_predictions':0,'failed_attempt_unchanged':True}
    except BaseException as e:
        b.write(job/'exception.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()},True)
        b.write(job/'status.json',{'PASS':False,'state':'FAILED','automatic_retry':False,'formal_predictions':0})
        raise
def toys():
    verify()
    job=b.OUT/'technical'/('pytest_'+uuid.uuid4().hex[:12]);job.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['TEMP']=env['TMP']=str(job);env['PYTHONPATH']=str(b.HERE)
    command=[sys.executable,'-B','-m','pytest',str(b.HERE/'test_b5c.py'),str(b.HERE/'test_seed_fix.py'),'-q','-p','no:cacheprovider','--basetemp',str(job/'tmp'),'--junitxml',str(job/'pytest.xml')]
    with (job/'pytest.log').open('x',encoding='utf8') as f:r=subprocess.run(command,cwd=b.ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    result={'PASS':r.returncode==0,'state':'COMPLETE' if r.returncode==0 else 'FAILED','exit_code':r.returncode,'job':str(job),'synthetic_only':True,'formal_predictions':0}
    b.write(job/'status.json',result,True);b.require(result['PASS'],'R1_SYNTHETIC_TEST_FAILED');return result
def status():
    paths=list((b.OUT/'technical').glob('*/status.json'))
    return b.read(max(paths,key=lambda p:p.stat().st_mtime_ns)) if paths else {'state':'NOT_STARTED','formal_restart_enabled':False}
def export():
    paths=[p for p in (b.OUT/'technical').rglob('*') if p.is_file() and 'tmp' not in p.parts]
    b.require(bool(paths),'NO_TECHNICAL_RECORDS');b.require(not any(b.read(p)['state']=='RUNNING' for p in (b.OUT/'technical').glob('*/status.json')),'WAIT_FOR_COMPLETION')
    dest=b.OUT/('R1_Technical_Export_'+uuid.uuid4().hex[:12]+'.zip');dest.parent.mkdir(parents=True,exist_ok=True)
    hashes={p.relative_to(b.OUT).as_posix():b.sha(p) for p in paths}
    with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,p.relative_to(b.OUT).as_posix())
        z.writestr('EXPORT_SHA256.json',json.dumps(hashes,sort_keys=True,indent=2))
    dest.with_suffix('.zip.sha256').write_text(b.sha(dest)+'\n',encoding='ascii')
    return {'path':str(dest),'sha256':b.sha(dest),'bytes':dest.stat().st_size,'formal_predictions':0}
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=('Verify','ToyTests','IdentityAudit','Status','Export'));x=a.parse_args()
    print(json.dumps({'Verify':verify,'ToyTests':toys,'IdentityAudit':identity_audit,'Status':status,'Export':export}[x.action](),ensure_ascii=False,indent=2))
