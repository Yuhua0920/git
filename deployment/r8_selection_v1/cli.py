"""Selection-only actions. No approval creator, training, validation or final action."""
from pathlib import Path
import sys,os,argparse,uuid,time,json,subprocess,zipfile,traceback
sys.dont_write_bytecode=True
import selection_core as a
ACTIONS=('verify','toy-tests','authorization-bindings','selection-preflight','complete-selection-once','monitor-toy','status','export','stop')
def job(kind):
    p=a.OUT/'runs'/(kind+'_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]);p.mkdir(parents=True,exist_ok=False);return p
def export(folder):
    s=a.read(folder/'status.json');a.require(s['state']in('COMPLETE','FAILED','STOP'),'WAIT_FOR_TERMINAL_STATE')
    dest=folder/'R8_evidence_export.zip';a.require(not dest.exists(),'EXPORT_EXISTS_NO_OVERWRITE')
    files={p.relative_to(folder).as_posix():p for p in folder.rglob('*')if p.is_file()and not any(x in p.parts for x in('toy_project','temporary','pytest_tmp'))and p.suffix not in('.zip','.npy','.npz')}
    if s['mode']=='complete'and s['PASS']:
        r=a.read(folder/'selection_commit_receipt.json');a.require(a.sha(a.R6OUT/'SELECTION_MANIFEST.json')==r['selection_manifest_SHA256'],'SELECTION_EXPORT_HASH_DRIFT')
        for n in('SELECTION_MANIFEST.json','SELECTION_MANIFEST.sha256'):files['selection/'+n]=a.R6OUT/n
    manifest={n:{'sha256':a.sha(p),'bytes':p.stat().st_size}for n,p in files.items()}
    with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED)as z:
        for n,p in files.items():z.write(p,n)
        z.writestr('export_manifest.json',json.dumps({'files':manifest,'contains_models':False,'contains_target_outcomes':False},indent=2))
    with zipfile.ZipFile(dest)as z:
        a.require(z.testzip()is None,'EXPORT_CRC_FAILED')
        for n,v in manifest.items():a.require(__import__('hashlib').sha256(z.read(n)).hexdigest()==v['sha256'],'EXPORT_INPUT_CHANGED')
    h=a.sha(dest);Path(str(dest)+'.sha256').write_text(h+'  '+dest.name+'\n',encoding='ascii');return {'PASS':True,'path':str(dest),'sha256':h,'files':len(manifest)+1}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=ACTIONS);parser.add_argument('--evidence',type=Path);parser.add_argument('--archive',type=Path);parser.add_argument('--approval',type=Path);parser.add_argument('--instruction',type=Path);parser.add_argument('--approval-sha');parser.add_argument('--job',type=Path);args=parser.parse_args();a.verify();a.OUT.mkdir(parents=True,exist_ok=True);folder=None
    try:
        if args.action=='verify':result=a.verify()
        elif args.action=='authorization-bindings':
            result={'PASS':True,'bindings':a.bindings(),'approval_created':False,'real_selection_executed':False,'final_model_trained':False,'target_access':0}
            if args.evidence:result['evidence']=a.proof(args.evidence)
            if args.archive:a.require(a.sha(args.archive)==a.cfg()['archive_SHA256'],'ORIGINAL_R7_ARCHIVE_HASH_MISMATCH')
        elif args.action=='toy-tests':
            folder=job('pytest')
            with(folder/'pytest.log').open('x',encoding='utf8')as f:
                code=subprocess.call([sys.executable,'-B','-X','utf8','-m','pytest','-q',str(a.HERE/'test_selection.py'),'-p','no:cacheprovider','--basetemp',str(folder/'pytest_tmp')],stdout=f,stderr=subprocess.STDOUT,cwd=a.ROOT)
            result={'PASS':code==0,'exit_code':code,'synthetic_only':True,'real_selection_executed':False,'final_model_trained':False,'target_access':0,'R8_package_manifest_SHA256':a.sha(a.HERE/'package_manifest.json')}
            a.write(folder/'status.json',result|{'state':'COMPLETE'if code==0 else'FAILED','mode':'pytest'});a.require(code==0,'TOY_TESTS_FAILED_NO_RETRY');a.write(a.OUT/'latest_tests.json',{'job':str(folder)})
        elif args.action in('selection-preflight','complete-selection-once','monitor-toy'):
            mode={'selection-preflight':'preflight','complete-selection-once':'complete','monitor-toy':'toy'}[args.action]
            if mode!='toy':a.require(args.evidence and args.archive,'EXPLICIT_FROZEN_EVIDENCE_AND_ARCHIVE_PATHS_REQUIRED')
            if mode=='complete':a.authorize(args.approval,args.instruction,args.approval_sha)
            folder=job(mode);lock=a.OUT/'execution.lock'
            with lock.open('x',encoding='utf8')as f:f.write(str(os.getpid()))
            try:
                a.require(not(a.OUT/'STOP_REQUESTED.json').exists(),'R8_STOP_PRESENT_NO_RETRY')
                request={'mode':mode,'evidence':str(args.evidence.resolve())if args.evidence else None,'archive':str(args.archive.resolve())if args.archive else None,
                         'approval':str(args.approval.resolve())if args.approval else None,'instruction':str(args.instruction.resolve())if args.instruction else None,'approval_sha':args.approval_sha}
                a.write(folder/'request.json',request,True);a.write(a.OUT/('latest_toy.json'if mode=='toy'else'latest.json'),{'job':str(folder)})
                import monitor
                result=monitor.run(folder,request);a.require(result['PASS'],'R8_FAILED_OR_STOP_NO_AUTOMATIC_RETRY')
            finally:lock.unlink(missing_ok=True)
        else:
            if args.job:folder=args.job.resolve()
            else:
                latest=next((p for p in(a.OUT/'latest.json',a.OUT/'latest_toy.json',a.OUT/'latest_tests.json')if p.exists()),None)
                if not latest:print(json.dumps({'state':'NOT_STARTED','real_selection_executed':False,'final_model_trained':False,'target_access':0}));return
                folder=Path(a.read(latest)['job'])
            a.require(folder.is_relative_to(a.OUT),'STATUS_EXPORT_JOB_SCOPE')
            if args.action=='status':result=a.read(folder/'status.json')if(folder/'status.json').exists()else{'state':'PREPARING','job':str(folder)}
            elif args.action=='stop':
                a.require(a.read(folder/'status.json')['state']=='RUNNING','NO_RUNNING_R8_JOB')
                a.write(a.OUT/'STOP_REQUESTED.json',{'UTC':a.utc(),'reason':'USER_R8_STOP','automatic_retry':False},True);result={'STOP_requested':True,'historical_scientific_STOP_files_modified':False}
            else:result=export(folder)
        print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),flush=True)
    except BaseException as e:
        if folder:a.write(folder/'cli_exception.json',{'message':str(e),'traceback':traceback.format_exc()},True)
        raise
if __name__=='__main__':main()
