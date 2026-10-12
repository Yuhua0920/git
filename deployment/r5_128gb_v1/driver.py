"""Manually launched stages; gates are verified again before every worker."""
from pathlib import Path
import os,sys,time,json,argparse,traceback,uuid,contextlib,msvcrt,zipfile,csv
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from common import ROOT,OUT,HERE,protocol,read,write,sha,integrity,utc,scope,scientific_counters
from windows_resources import hardware,capacity,stop_named_job
from supervisor import run_worker,optional
from analysis import select_threads,inference_cost,training_budget

@contextlib.contextmanager
def exclusive():
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'execution.lock'
    with path.open('a+b')as f:
        if path.stat().st_size==0:f.write(b'0');f.flush()
        f.seek(0)
        try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise RuntimeError('ANOTHER_R5_STAGE_IS_RUNNING_STOP')
        try:yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)

def gate(stage):
    saved=read(OUT/f'{stage}_stage.json')
    assert saved['state']=='COMPLETE'and saved['PASS'],'PREVIOUS_STAGE_NOT_COMPLETE_PASS'
    assert saved['package_manifest_SHA256']==sha(HERE/'package_manifest.json'),'PREREQUISITE_CODE_FREEZE_DIFFERENT'
    job=ROOT/saved['job'];assert job.is_relative_to(OUT)
    result=read(job/'stage_result.json');assert sha(job/'stage_result.json')==saved['result_sha256'],'STAGE_RESULT_DRIFT'
    return result

def start_stage(stage,job):
    check=integrity();scope();assert not(OUT/'STOP_REQUESTED.json').exists(),'PERSISTENT_STOP_REQUEST_REQUIRES_MENTOR_REVIEW'
    cfg=protocol();deadline=time.monotonic()+cfg['stage_seconds'][stage]
    job=job.resolve();assert job.is_relative_to(OUT);job.mkdir(parents=True,exist_ok=True)
    with exclusive():
        path=OUT/f'{stage}_stage.json';assert not path.exists(),'STAGE_ALREADY_ATTEMPTED_NO_AUTOMATIC_OR_UNREVIEWED_RETRY'
        state={'state':'RUNNING','PASS':False,'stage':stage,'job':job.relative_to(ROOT).as_posix(),'started_UTC':utc(),
               'package_manifest_SHA256':check['package_manifest_SHA256'],'protocol_SHA256':check['protocol_SHA256'],**scientific_counters()}
        write(path,state);write(job/'status.json',state);write(OUT/'latest.json',{'job':state['job'],'stage':stage})
        write(job/'integrity_before.json',check)
        try:
            cap=capacity(hardware());write(job/'hardware_and_capacity.json',cap)
            assert cap['PASS'],'LAB_CAPACITY_STOP:'+str(cap['reasons'])
            def launch(action,name,blas=1,omp=1,extra=None):
                remaining=deadline-time.monotonic();assert remaining>0,'STAGE_TECHNICAL_TIME_BUDGET_EXCEEDED'
                budget=min(cfg['worker_seconds'].get(action,1800),remaining)
                return run_worker(action,job/name,blas,omp,extra,timeout=budget)
            if stage=='preflight':
                resources,result=launch('preflight','fidelity')
                result={'PASS':True,'hardware_and_capacity':cap,'fidelity':result,'resources':resources}
            elif stage=='cpu':
                gate('preflight');scratch=OUT/'_scratch/benchmark_anchor';scratch.mkdir(parents=True,exist_ok=False)
                launch('anchor','single_thread_anchor',1,1,['--anchor',str(scratch)])
                records=[];candidates=cfg['thread_candidates'];order=[]
                for block in range(cfg['benchmark_blocks']):
                    rotated=candidates[block:]+candidates[:block];order.append(rotated)
                    for t in rotated:
                        resources,record=launch('benchmark',f'block{block}_threads{t}',t,t,['--anchor',str(scratch)])
                        row={'block':block,'threads':t,'blas':t,'openmp':t,'kernels':{v['kernel']:v for v in record['rows']},
                             'native_KMeans':record['native_KMeans'],'resources':resources,'input_sha256':record['input_sha256']}
                        records.append(row);write(job/'cpu_records.json',records)
                assert len({r['input_sha256']for r in records})==1,'BENCHMARK_INPUT_DRIFT'
                recommendation=select_threads(records);write(job/'thread_recommendation.json',recommendation)
                blas=recommendation['RECOMMENDED_BLAS_THREADS'];omp=recommendation['RECOMMENDED_OPENMP_THREADS']
                launch('pair-fidelity','recommended_pair_fidelity',blas,omp)
                _,pair=launch('benchmark','recommended_pair_native_benchmark',blas,omp,['--anchor',str(scratch)])
                inference=[]
                for t in candidates:
                    for K in cfg['inference']['K']:
                        resources,report=launch('inference',f'inference_threads{t}_K{K}',t,omp,['--K',str(K)])
                        inference.append({'blas':t,'openmp':omp,'K':K,'report':report,'resources':resources})
                        write(job/'inference_records.json',inference)
                costs=inference_cost(inference,blas);write(job/'inference_cost_scenarios.json',costs)
                flat=[]
                for row in records:
                    for name,v in row['kernels'].items():flat.append({'block':row['block'],'threads':row['threads'],'kernel':name,'wall_seconds':v['wall_seconds'],'numeric_parity_PASS':v['PASS']})
                with(job/'cpu_timing_raw.csv').open('w',encoding='utf-8-sig',newline='')as f:
                    writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
                result={'PASS':True,'recommendation':recommendation,'records':records,'inference_records':inference,
                    'inference_costs':costs,'balanced_order':order,'recommended_pair_fidelity_PASS':True,'recommended_pair_native_benchmark':pair}
            else:
                gate('preflight');cpu=gate('cpu')
                if stage=='smoke15000':gate('smoke12000')
                recommendation=cpu['recommendation'];blas=recommendation['RECOMMENDED_BLAS_THREADS'];omp=recommendation['RECOMMENDED_OPENMP_THREADS']
                resources,report=launch(stage,'author_full_scale',blas,omp)
                result={'PASS':True,'report':report,'resources':resources,'blas':blas,'openmp':omp}
                if stage=='smoke15000':
                    first=gate('smoke12000');costs=training_budget(first['report'],report,recommendation,cpu['records'])
                    write(job/'training_cost_scenarios.json',costs)
                    qualified={'title':'MFA_128GB_HOST_QUALIFICATION_REPORT','PASS':True,'implementation':'frozen 2023 author code + original adapter',
                        'hardware':cap,'cross_host_fidelity_PASS':True,'CPU':recommendation,'N12000':first,'N15000':result,
                        'training_cost_scenarios':costs,'inference_cost_scenarios':cpu['inference_costs'],
                        'actual_laboratory_tests_performed':True,'scope':'two full-batch synthetic updates, not converged formal fits',**scientific_counters()}
                    proposal={'title':'PROPOSED_FORMAL_B5B_EXECUTION_CONFIG','status':'MENTOR_REVIEW_REQUIRED_NOT_APPLIED',
                        'implementation':'B5A_AUTHOR_REFERENCE_AND_ORIGINAL_ADAPTER','BLAS_threads':blas,'OpenMP_threads':omp,'concurrency':1,
                        'historical_formal_threads':1,'historical_formal_RSS_bytes':6442450944,'historical_per_job_seconds':86400,
                        'formal_resource_contract_changed':False,'formal_resource_amendment_required_before_author_full_scale_fit':True,
                        'qualified_synthetic_limits_only':cfg['limits'],'policy':'unchanged FROZEN_POLICY; K/r selection NONE',
                        'formal_24h_safety_certified':False,'budget_scenarios':costs,**scientific_counters()}
                    write(OUT/'MFA_128GB_HOST_QUALIFICATION_REPORT.json',qualified)
                    write(OUT/'PROPOSED_FORMAL_B5B_EXECUTION_CONFIG.json',proposal)
            result.update(**scientific_counters());write(job/'stage_result.json',result)
            final_check=integrity();write(job/'integrity_after.json',final_check)
            state.update(state='COMPLETE',PASS=True,completed_UTC=utc(),result_sha256=sha(job/'stage_result.json'))
            write(path,state);write(job/'status.json',state)
        except Exception:
            active=optional(OUT/'active_worker.json');worker_status=None
            if active:
                active_path=ROOT/active['job']
                if active_path.is_relative_to(job):worker_status=optional(active_path/'status.json')
            state.update(state='STOP'if worker_status and worker_status['state']=='STOP'else'FAILED',PASS=False,completed_UTC=utc())
            write(path,state);write(job/'status.json',state);write(job/'exception.json',{'traceback':traceback.format_exc(),**scientific_counters()});raise
    return state

def export_results():
    OUT.mkdir(parents=True,exist_ok=True)
    scope()
    with exclusive():
        stamp=time.strftime('%Y%m%d_%H%M%S');folder=OUT/'exports';folder.mkdir(exist_ok=True)
        archive=folder/f'SP_U12B5_R5_Results_{stamp}.zip';assert not archive.exists()
        files=[]
        for p in OUT.rglob('*'):
            if p.is_file()and not any(v in('_scratch','exports','preparation_personal','temporary','pytest_tmp','__pycache__')for v in p.relative_to(OUT).parts):
                assert not p.is_symlink(),'RESULT_EXPORT_SYMLINK_FORBIDDEN'
                if p.suffix.lower()in('.json','.jsonl','.csv','.log','.txt','.md')and p.name!='execution.lock':files.append(p)
        manifest={p.relative_to(OUT).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size}for p in files}
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6)as z:
            for p in files:z.write(p,'r5_128gb/'+p.relative_to(OUT).as_posix())
            z.writestr('result_manifest.json',json.dumps({'files':manifest,'scope':'TECHNICAL_ONLY_NO_ARRAYS_OR_SCIENTIFIC_INPUTS',
                'package_manifest_SHA256':sha(HERE/'package_manifest.json'),**scientific_counters()},indent=2))
        h=sha(archive);archive.with_suffix('.zip.sha256').write_text(h+'  '+archive.name+'\n',encoding='ascii')
        with zipfile.ZipFile(archive)as z:
            for name,v in manifest.items():
                import hashlib
                assert hashlib.sha256(z.read('r5_128gb/'+name)).hexdigest()==v['sha256']
        report={'archive':str(archive),'sha256':h,'files':len(files)+1,'bytes':archive.stat().st_size,'large_arrays_included':0}
        write(folder/'latest_export.json',report);return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('verify','preflight','cpu','smoke12000','smoke15000','status','stop','export'))
    parser.add_argument('--job');args=parser.parse_args()
    if args.action=='status':
        latest=optional(OUT/'latest_launch.json')
        if latest:print(json.dumps(optional(ROOT/latest['job']/'status.json')or latest))
        for stage in('preflight','cpu','smoke12000','smoke15000'):print(json.dumps(optional(OUT/f'{stage}_stage.json')or{'stage':stage,'state':'NOT_STARTED'}))
        active=optional(OUT/'active_worker.json')
        if active:print(json.dumps(optional(ROOT/active['job']/'status.json')or active))
        return
    if args.action=='stop':
        write(OUT/'STOP_REQUESTED.json',{'time':utc(),'reason':'USER_MANUAL_STOP'})
        active=optional(OUT/'active_worker.json');killed=stop_named_job(active['JobObject_name'])if active else False
        print(json.dumps({'state':'STOP','kernel_job_terminated':killed,'automatic_retry':False}));return
    if args.action=='export':print(json.dumps(export_results(),indent=2));return
    if args.action=='verify':print(json.dumps(integrity(),indent=2));return
    assert args.job,'MANUAL_POWERSHELL_LAUNCH_WITH_JOB_REQUIRED'
    print(json.dumps(start_stage(args.action,Path(args.job)),indent=2))

if __name__=='__main__':
    try:main()
    except Exception:
        if '--job'in sys.argv:
            p=Path(sys.argv[sys.argv.index('--job')+1]).resolve()
            if p.is_relative_to(OUT):
                old=optional(p/'status.json')or{};old.update(state='FAILED',PASS=False,completed_UTC=utc())
                # Preserve an already recorded resource/manual STOP classification.
                recorded=optional(p/'status.json')
                if not recorded or recorded.get('state')not in('STOP','FAILED'):write(p/'status.json',old)
                write(p/'driver_exception.json',{'traceback':traceback.format_exc(),**scientific_counters()})
        traceback.print_exc();sys.exit(1)
