"""Audit child; performs no fitting or real-observation inference."""
from pathlib import Path
import os,sys,time,traceback,argparse,platform,importlib.metadata,shutil
sys.dont_write_bytecode=True
import audit_core as a
def report_text(v):
    lines=['# R7 read-only evidence audit','',f"Host: {v['host']}",f"Root: {v['project_root']}",f"Actual interpreter: {v['interpreter']}",f"Verdict: {v['verdict']}",'',
      'Selected structure: NONE. Final model: NONE. No training, real validation, metric recalculation or selection.']
    lines+=['',* [f"- {k}: {x['status']} — {x.get('scope','')}"for k,x in v['gates'].items()]]
    lines+=['','B2/B3 are absent from lab by design; no laboratory seal verification is asserted.',
      'Historical access absence cannot be proved by declaration fields; the audit establishes current artifacts, logs and byte preservation.',
      'Stored inv_covs are read and checked for shape/dtype/finite/hash under the frozen checker; no new inverse-consistency eligibility threshold is added.',
      'Default export contains 45 original small NPY files. Eighteen original dense matrices remain on lab with full-file/payload hashes and frozen full-matrix checks.']
    if v.get('error'):lines+=['','Failure: '+v['error']]
    return '\n'.join(lines)+'\n'
def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--mode',choices=('lab','toy'),required=True);args=p.parse_args();job=args.job.resolve()
    a.require(job.is_relative_to(a.OUT.resolve()),'WORKER_JOB_SCOPE');cfg=a.inputs();a.verify_tool()
    a.write(job/'actual_pid.json',{'PID':os.getpid(),'executable':sys.executable,'host':os.environ.get('COMPUTERNAME'),'UTC':a.utc()},True)
    start=time.monotonic()
    while not(job/'monitor_ready.json').exists():a.require(time.monotonic()-start<60,'HANDSHAKE_TIMEOUT');time.sleep(.02)
    for n in('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[n]='6'
    os.environ['OMP_NUM_THREADS']='1'
    ops,common,wr=a.original_modules()
    v={'host':os.environ.get('COMPUTERNAME'),'project_root':str(a.ROOT),'interpreter':sys.executable,'actual_PID':os.getpid(),'UTC':a.utc(),
      'gates':{k:{'status':'FAIL','scope':'NOT_EXECUTED'}for k in a.GATES},'verdict':'FROZEN_SELECTION_NOT_RELEASED',
      'new_training':0,'new_real_validation':0,'new_metric_calculations':0,'selection_executed':False,'final_training':False,'models':[]}
    def progress(text):a.write(job/'phase.json',{'phase':text,'UTC':a.utc()})
    try:
        env=common.environment();a.write(job/'environment.json',env,True)
        with common.threads(6,1)as pools:
            a.write(job/'actual_threads.json',pools,True)
            if args.mode=='toy':
                sys.path.insert(0,str(a.ROOT));from synthetic import fixture
                for row in cfg['candidate_order']:
                    d=job/'synthetic_fixtures'/row['name']/'train/model';expected=fixture(d,row)
                    v['models'].append(a.model_check(d,row,expected,True,progress))
                time.sleep(1)
                result={'PASS':True,'synthetic_only':True,'fixture_models':9,'fixture_NPY_files':63,'lab_physical_models_verified':0,'actual_PID':os.getpid()}
            else:
                a.install_guard(a.ROOT,cfg,job)
                a.old_is_idle(a.ROOT,cfg)
                rows,headers=a.dependencies(a.ROOT,cfg,progress);v['source_population']=a.source_population(a.ROOT,cfg);a.csv_write(job/'post_run_dependency_hashes.csv',rows);a.write(job/'preparation_headers.json',headers,True)
                legacy=a.legacy_history(a.ROOT,cfg,progress);a.csv_write(job/'R5_R6_historical_hashes.csv',legacy)
                # Original small source/config/contract bytes, never model reconstruction or approval creation.
                snapshots={}
                for x in rows:
                    src=a.ROOT/x['path']
                    if src.suffix.lower()in('.py','.ps1','.json','.yaml','.yml','.toml','.md','.txt')and src.stat().st_size<=1024*1024:
                        dst=job/'frozen_dependency_snapshot'/x['path'];dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(src.read_bytes());snapshots[x['path']]=x['sha256']
                a.write(job/'dependency_snapshot_manifest.json',snapshots,True)
                v['gates']['FROZEN_PROTOCOL_AND_SEEDS']={'status':'PASS','scope':'278 original migration/R5/R6/R7 dependencies; frozen seeds/protocol/data hashes and headers'}
                live=a.archive_leaves(a.ROOT,cfg,progress);a.csv_write(job/'live_archive_hashes.csv',live)
                v['gates']['ARCHIVE_INTEGRITY']={'status':'PASS','scope':'accepted fixed ZIP SHA/CRC353 leaves; all353 actual lab leaves match original returned bytes'}
                v['validation']=a.evidence_binding(cfg);v['gates']['CANONICAL_VALIDATION_AND_METRIC']={'status':'PASS','scope':cfg['validation_scope']}
                v['resources']=a.resource_history(a.ROOT,cfg);a.csv_write(job/'historical_resource_check.csv',v['resources'])
                v['gates']['RESOURCE_AND_STOP']={'status':'PASS','scope':'18 original stage resource streams/status/native peaks; serial24h and STOP_AFTER_NINE; no retry'}
                for row in cfg['candidate_order']:
                    expected=cfg['models'][row['name']];a.require(expected['context']['data_sha256']==cfg['data']['outputs/sp_u12b5b/selection_train.npy']['sha256'],'MODEL_TRAIN_DATA_BINDING')
                    d=a.ROOT/'lab_runs/r6_formal/candidates'/row['name']/'train/model'
                    v['models'].append(a.model_check(d,row,expected,False,progress));a.write(job/(row['name']+'_parameter_audit.json'),v['models'][-1],True)
                a.csv_write(job/'model63_hashes.csv',[x for model in v['models']for x in model['files']])
                a.csv_write(job/'nine_candidate_audit.csv',[{k:m[k]for k in('candidate','PASS','K','rank','seed','parameter_hash')}|{'iterations':m['history']['iterations'],'convergence':m['history']['convergence'],'likelihood_decreases':len(m['history']['decrease_events'])}for m in v['models']])
                v['gates']['MODEL_ARTIFACTS_AND_VALIDITY']={'status':'PASS','scope':'63 actual saved NPY; frozen parameter checker; exact history; zero-observation all-K smoke; no parameter mutations'}
                after,after_headers=a.dependencies(a.ROOT,cfg,progress);after_live=a.archive_leaves(a.ROOT,cfg,progress)
                after_legacy=a.legacy_history(a.ROOT,cfg,progress)
                a.require(after==rows and after_live==live and after_headers==headers and after_legacy==legacy,'POST_AUDIT_DRIFT');a.source_population(a.ROOT,cfg);v['boundary']=a.old_is_idle(a.ROOT,cfg)
                v['gates']['POST_RUN_PRESERVATION']={'status':'PASS','scope':'manifest-covered actual migrated dependencies and returned run history; before/after exact hashes; no laboratory manuscript/B2/B3 attestation'}
                v['verdict']='FROZEN_SELECTION_EVIDENCE_READY_REQUIRES_SEPARATE_APPROVAL'
                result={'PASS':True,'actual_PID':os.getpid(),'report_sha256':None,'lab_physical_models_verified':9,'lab_physical_NPY_verified':63,'selection_executed':False,'final_model':'NONE'}
        v['native_final_memory']=wr.frozen_probe().process_memory();result['native_final_memory']=v['native_final_memory']
        if args.mode=='toy':v['verdict']='SYNTHETIC_TEST_ONLY_NO_LAB_ATTESTATION';v['lab_physical_models_verified']=0
        a.write(job/'report.json',v,True);(job/'report.md').write_text(report_text(v),encoding='utf8');result['report_sha256']=a.sha(job/'report.json');a.write(job/'worker_result.json',result,True)
    except BaseException as e:
        v['error']=str(e);a.write(job/'report.json',v,True);(job/'report.md').write_text(report_text(v),encoding='utf8')
        a.write(job/'exception.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()},True);raise
if __name__=='__main__':main()
