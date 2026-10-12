"""One independent LAB recovery, never original inference or status rewriting."""
from pathlib import Path
import argparse,importlib.metadata,json,os,sys,time,traceback,uuid,zipfile
import re
sys.dont_write_bytecode=True
import recovery_base as r
def environment():
    got={n:importlib.metadata.version(n) for n in ('numpy','scipy','threadpoolctl')}
    r.require(sys.version_info[:3]==(3,13,15) and got=={'numpy':'2.5.3','scipy':'1.18.1','threadpoolctl':'3.7.0'},'FROZEN_STATISTICS_ENVIRONMENT_REQUIRED')
    return {'Python':sys.version,'packages':got}
def source_scope(b,snapshot,freeze):
    exact={r.ROOT/n for n in r.bindings()['frozen_files']}
    exact|={r.ROOT/n for n in r.read(r.HERE/'package_manifest.json')['files']}
    exact|={r.HERE/'package_manifest.json'} # Include the manifest itself explicitly.
    exact|={r.OLD/n for n in snapshot}
    exact|={b.DATA/n for n in b.data_inventory()}
    exact|={r.ROOT/'lab_runs/r6_formal'/n for n in ('FINAL_MODEL_MANIFEST.json','FINAL_MODEL_MANIFEST.sha256')}
    exact|={b.FINAL/'training_result.json',b.FINAL/'model/model_manifest.json'}
    exact|={b.FINAL/'model'/n for n in b.contract()['model_file_names']}
    q=r.ROOT/freeze['preflight']['R1_identity_audit']['job']
    r.require(q.parent==r.ROOT/'lab_runs/b5c_paired_r1/technical' and re.fullmatch(r'identity_[0-9a-f]{12}',q.name),'R1_RECEIPT_SCOPE')
    names={'status.json','identity_report.json','original_preservation_before.json','original_preservation_after.json'}
    r.require(set(freeze['preflight']['R1_identity_audit']['receipt_hashes'])==names,'R1_RECEIPT_FIELD_SCOPE')
    exact|={q/n for n in names}
    return exact
def forbid_inference(s):
    def guard(frame,event,arg):
        if event=='call' and frame.f_code.co_filename==s.__file__ and frame.f_code.co_name in ('infer','evaluate','load_final','modules'):
            raise RuntimeError('RECOVERY_MFA_INFERENCE_FORBIDDEN')
    sys.setprofile(guard)
def recover():
    r.plain(r.OUT,r.ROOT);r.plain(r.OLD,r.ROOT);r.plain(r.SOURCE,r.ROOT)
    r.verify();env=environment();r.require(r.SOURCE.is_dir(),'REAL_RESULTS_PRESENT_ONLY_ON_LAB')
    r.require(not(r.OUT/'latest.json').exists(),'RECOVERY_ALREADY_STARTED_NO_AUTOMATIC_RETRY')
    b,s,stats,report,h=r.frozen_modules()
    inspection=h.inspect()
    try:
        r.require(inspection.report['PASS'],'HISTORICAL_ACTIVITY_OR_UNKNOWN:'+json.dumps(inspection.report['blockers']))
        inspection.closed_status(r.SOURCE/'status.json','b5c_paired_r2')
        if (r.OLD/'execution.lock').exists():inspection.classify_lock(r.OLD/'execution.lock','b5c_paired_r2')
        r.OUT.mkdir(parents=True,exist_ok=True)
        with (r.OUT/'execution.lock').open('x',encoding='ascii') as f:f.write(str(os.getpid()))
        job=r.OUT/('attempt_'+uuid.uuid4().hex[:12]);job.mkdir(exist_ok=False)
        r.write(r.OUT/'latest.json',{'job':job.relative_to(r.ROOT).as_posix(),'PID':os.getpid(),'UTC':r.utc()})
        r.status(job,'SNAPSHOTTING',PASS=False,original_attempt_state='FAILED',new_inference_calls=0)
        begin=time.monotonic();snapshot=None
        try:
            snapshot=r.snapshot(r.OLD);r.write(job/'original_snapshot.json',snapshot)
            r.write(job/'snapshot_receipt.json',{'sha256':r.sha(job/'original_snapshot.json'),'files':len(snapshot),'old_attempt_state':'FAILED','UTC':r.utc()})
            freeze=r.read(r.SOURCE/'execution_freeze.json')
            # Authenticate and initialize only pure replay/statistical modules.
            # No author model initialization, training or inference is imported.
            from src.cdl_benchmark import observation_geometry,frames
            from src.link import assemble_pilot_data_frame,apply_frequency_mimo_channel,estimate_ls_linear_channel
            from threadpoolctl import threadpool_limits
            r.firewall(source_scope(b,snapshot,freeze));forbid_inference(s)
            def cancel():r.require(not(r.OUT/'STOP_REQUESTED.json').exists(),'RECOVERY_USER_STOP_REQUESTED')
            r.status(job,'VERIFYING',PASS=False,performance_released=False,new_inference_calls=0)
            state=__import__('recovery_core').failed_end_gate(b,s,h)
            r.verify();b.verify_data();p=b.contract();cfg=b.read(b.DATA/'configs/sp_u12b2b_sealed.json');index=s.seed_index()
            gate=__import__('recovery_core').trajectory_gate(r.SOURCE,p,s,index,cfg,cancel)
            r.unchanged(snapshot);gate.update(schema='INDEPENDENT_RECOVERY_GATE',original_attempt_state='FAILED',resource_evidence_verified=True,
                model_and_source_hashes_verified=True,snapshot_sha256=r.sha(job/'original_snapshot.json'),UTC=r.utc())
            r.write(job/'integrity_gate.json',gate)
            # Aggregate release is reachable only after the complete independent Gate.
            with threadpool_limits(limits=6,user_api='blas'),threadpool_limits(limits=1,user_api='openmp'):
                __import__('recovery_core').analyze(r.SOURCE,job,p,s,stats,report,gate,state,cancel)
            r.unchanged(snapshot);r.verify()
            r.write(job/'original_preservation_after.json',{'PASS':True,'unchanged_files':len(snapshot),'snapshot_sha256':gate['snapshot_sha256'],'old_state':'FAILED'})
            r.status(job,'COMPLETE',PASS=True,original_attempt_state='FAILED',new_inference_calls=0,elapsed_seconds=time.monotonic()-begin,
                analysis_manifest_sha256=r.sha(job/'analysis/analysis_manifest.json'),figures='RenderFigures after Export, using the authenticated original publication renderer')
            return status()
        except BaseException as error:
            r.write(job/'exception.json',{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(),'new_inference_calls':0})
            # Never publish aggregate metrics on a failed Gate. No missing data fabricated.
            r.status(job,'STOP',PASS=False,original_attempt_state='FAILED',new_inference_calls=0,exception='exception.json',automatic_retry=False)
            raise
        finally:(r.OUT/'execution.lock').unlink()
    finally:inspection.close()
def latest():
    value=r.read(r.OUT/'latest.json');p=r.plain(r.ROOT/value['job'],r.OUT)
    r.require(p.parent==r.OUT and p.name.startswith('attempt_'),'CANONICAL_RECOVERY_JOB');return p
def status():
    if not(r.OUT/'latest.json').exists():return {'state':'NOT_STARTED','real_results_verified':False,'new_inference_calls':0}
    job=latest();return {'job':str(job),**r.read(job/'status.json')}
def stop():
    r.plain(r.OUT,r.ROOT);r.OUT.mkdir(parents=True,exist_ok=True);r.write(r.OUT/'STOP_REQUESTED.json',{'UTC':r.utc(),'automatic_retry':False});return {'STOP_requested':True}
def check_analysis(job):
    state=r.read(job/'status.json');path=job/'analysis/analysis_manifest.json'
    r.require(state['PASS'] is True and r.sha(path)==state['analysis_manifest_sha256'],'ANALYSIS_MANIFEST_CHANGED_AFTER_COMPLETE')
    value=r.read(path)
    for name,digest in value['files'].items():
        r.require(Path(name).name==name and r.sha(job/'analysis'/name)==digest,'ANALYSIS_FILE_CHANGED:'+name)
    return value
def export():
    r.plain(r.OUT,r.ROOT)
    r.require(not(r.OUT/'execution.lock').exists(),'WAIT_FOR_RECOVERY_EXIT_BEFORE_EXPORT')
    job=latest();r.require(r.read(job/'status.json')['state'] in ('STOP','COMPLETE'),'RECOVERY_NOT_TERMINAL')
    if r.read(job/'status.json')['state']=='COMPLETE':check_analysis(job)
    path=r.OUT/('Recovery_Export_'+uuid.uuid4().hex[:12]+'.zip')
    files=[]
    for p in job.rglob('*'):
        r.plain(p,r.OUT)
        if p.is_file():files.append(p)
    hashes={p.relative_to(job).as_posix():r.sha(p) for p in files}
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for p in files:z.write(p,p.relative_to(job).as_posix())
        z.writestr('EXPORT_FILE_HASHES.json',json.dumps(hashes,indent=2));z.writestr('SOURCE_READONLY_FAILED_STATUS.json',json.dumps(r.read(r.SOURCE/'status.json'),indent=2))
    path.with_suffix('.zip.sha256').write_text(r.sha(path)+'\n',encoding='ascii')
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':r.sha(path),'files':len(files)+2,'original_attempt_state':'FAILED'}
def render_figures():
    # Separate post-Gate rendering process; no model/data access and no statistics changes.
    r.plain(r.OUT,r.ROOT);r.verify();job=latest();r.require(r.read(job/'status.json')['PASS'] is True,'RECOVERY_STATISTICS_NOT_COMPLETE')
    manifest=check_analysis(job)
    csv=job/'analysis/nmse_comparison_35_cells.csv';r.require(r.sha(csv)==manifest['files'][csv.name],'CSV_CHANGED_BEFORE_RENDER')
    source=r.ROOT/'deployment/b5c_paired_v1/publication_plots.py'
    import importlib.util
    spec=importlib.util.spec_from_file_location('_original_publication_renderer',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    os.environ['MPLCONFIGDIR']=str(job/'figure_cache')
    module.render(csv,job/'figures')
    return {'PASS':True,'figures':str(job/'figures'),'omitted_cells':0,'human_visual_QA':'REQUIRED'}
def main():
    a=argparse.ArgumentParser();a.add_argument('action',choices=('Verify','Recover','Status','Stop','Export','RenderFigures'));x=a.parse_args()
    answer={'Verify':r.verify,'Recover':recover,'Status':status,'Stop':stop,'Export':export,'RenderFigures':render_figures}[x.action]()
    print(json.dumps(answer,indent=2,allow_nan=False))
if __name__=='__main__':
    try:main()
    except BaseException as e:print(json.dumps({'state':'STOP','message':str(e),'automatic_retry':False}));raise
