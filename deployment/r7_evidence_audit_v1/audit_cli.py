"""Only audit/technical actions are exposed. No scientific pipeline entry imports."""
from pathlib import Path
import sys,os,argparse,time,uuid,json,subprocess,traceback
sys.dont_write_bytecode=True
import audit_core as a
def new_job(kind):
    p=a.OUT/'runs'/(kind+'_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]);p.mkdir(parents=True,exist_ok=False);return p
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('verify','toy-tests','monitor-toy','audit','status','stop-audit','export'))
    p.add_argument('--job',type=Path);p.add_argument('--include-all-models',action='store_true');args=p.parse_args();a.verify_tool();a.OUT.mkdir(parents=True,exist_ok=True)
    job=None
    try:
        if args.action=='verify':result=a.verify_tool()|{'lab_physical_audit':'NOT_EXECUTED','training_authorized':False,'selection_authorized':False}
        elif args.action=='toy-tests':
            job=new_job('pytest');before={p.name:a.sha(p)for p in a.HERE.iterdir()if p.is_file()};tmp=job/'pytest_tmp'
            with(job/'pytest.log').open('x',encoding='utf8')as f:
                code=subprocess.call([sys.executable,'-B','-X','utf8','-m','pytest','-q',str(a.HERE/'test_audit.py'),'-p','no:cacheprovider','--basetemp',str(tmp)],cwd=a.ROOT,stdout=f,stderr=subprocess.STDOUT)
            result={'PASS':code==0,'exit_code':code,'log':str(job/'pytest.log'),'lab_model_count':0,'synthetic_only':True,
              'source_unchanged':before=={p.name:a.sha(p)for p in a.HERE.iterdir()if p.is_file()}};a.write(job/'test_report.json',result,True)
            a.require(result['PASS']and result['source_unchanged'],'SYNTHETIC_TESTS_FAILED');a.write(a.OUT/'latest_tests.json',{'job':str(job)})
        elif args.action in('audit','monitor-toy'):
            mode='lab'if args.action=='audit'else'toy';job=new_job(mode);lock=a.OUT/'audit_execution.lock'
            with lock.open('x',encoding='utf8')as f:f.write(str(os.getpid()))
            try:
                a.write(a.OUT/('latest_lab.json'if mode=='lab'else'latest_toy.json'),{'job':str(job),'mode':mode})
                # Snapshot exact new tool bytes; protected dependencies are snapshotted by the read-only child.
                for src in a.HERE.iterdir():
                    if src.is_file()and src.name not in('prepare_personal.py','personal_qa.py'):
                        dst=job/'tool_snapshot'/src.name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(src.read_bytes())
                import audit_monitor
                result=audit_monitor.run(job,mode)
                if mode=='lab':
                    final=a.read(job/'report.json')if(job/'report.json').exists()else{'gates':{n:{'status':'FAIL','scope':'NOT_EXECUTED'}for n in a.GATES}}
                    final['audit_monitor']=result
                    if not result['PASS']:
                        final['gates']['RESOURCE_AND_STOP']={'status':'FAIL','scope':'new audit monitor FAILED/STOP: '+str(result.get('stop_reason'))}
                        final['verdict']='FROZEN_SELECTION_NOT_RELEASED'
                    else:a.require(all(x['status']=='PASS'for x in final['gates'].values()),'WORKER_GATE_INCONSISTENCY')
                    a.write(job/'final_report.json',final,True)
                    (job/'final_report.md').write_text('# R7 final read-only evidence report\n\n'+json.dumps(final,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
                a.require(result['PASS'],'AUDIT_FAILED_OR_STOPPED; NO_RETRY')
            finally:lock.unlink(missing_ok=True)
        else:
            if args.job:job=args.job.resolve()
            elif(a.OUT/'latest_lab.json').exists():job=Path(a.read(a.OUT/'latest_lab.json')['job'])
            elif(a.OUT/'latest_toy.json').exists():job=Path(a.read(a.OUT/'latest_toy.json')['job'])
            else:result={'state':'NOT_STARTED','lab_physical_audit':'NOT_EXECUTED'};print(json.dumps(result));return
            a.require(job.is_relative_to(a.OUT.resolve()),'JOB_OUTSIDE_AUDIT_NAMESPACE')
            if args.action=='status':result=a.read(job/'status.json')if(job/'status.json').exists()else{'state':'PREPARING','job':str(job)}
            elif args.action=='stop-audit':
                state=a.read(job/'status.json');a.require(state['state']=='RUNNING','AUDIT_NOT_RUNNING')
                a.write(job/'STOP_AUDIT_REQUESTED.json',{'UTC':a.utc(),'audit_only':True},True);result={'STOP_requested':True,'scientific_pipeline_modified':False}
            else:result=a.export_job(job,args.include_all_models)
        print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),flush=True)
    except BaseException as e:
        if job:
            a.write(job/'cli_exception.json',{'message':str(e),'traceback':traceback.format_exc(),'UTC':a.utc()},True)
            if not(job/'status.json').exists():a.write(job/'status.json',{'state':'FAILED','PASS':False,'stop_reason':str(e),'lab_models_verified':0},True)
            if args.action=='audit'and not(job/'report.json').exists():
                r={'verdict':'FROZEN_SELECTION_NOT_RELEASED','error':str(e),'lab_models_verified':0,'gates':{n:{'status':'FAIL','scope':'NOT_EXECUTED_PRECONDITION_REFUSAL'}for n in a.GATES}}
                a.write(job/'report.json',r,True)
        raise
if __name__=='__main__':main()
