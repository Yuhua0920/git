"""One isolated R6 job. Real arrays remain gated by mentor approval."""
from pathlib import Path
import os,sys,time,argparse,traceback,json,csv
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from common import ROOT,OUT,LOCAL,read,write,require,utc,scope,environment,threads,science,candidate,sha,numeric_inputs,integrity,approval

def train(job,row,final=False):
    import numpy as np
    from model_ops import make_reference_model,fit_reference,Observe,parameter_validity,save_model,load_model,inference_smoke,parameter_hashes
    role='final' if final else 'train'; name='full_retraining.npy' if final else 'selection_train.npy'
    seed=row['final_seed'] if final else row['seed']; data_path=ROOT/'outputs/sp_u12b5b'/name
    with numeric_inputs(role):
        # Memmap only changes data transport: frozen adapter np.asarray sees the
        # same complete C-order complex128 array; no batching/optimized EM.
        x=np.load(data_path,mmap_mode='r',allow_pickle=False)
        require(x.shape==((15000 if final else 12000),4032) and x.dtype==np.complex128 and x.flags.c_contiguous,'FORMAL_INPUT_SHAPE_DTYPE_ORDER')
        require(np.isfinite(x).all(),'TRAINING_DATA_NONFINITE')
        m=make_reference_model(row['K'],row['rank'])
        with Observe(job) as observer: fit_reference(m,x,seed)
        require(observer.initial is not None and observer.steps==len(m.L_all),'INITIALIZATION_OR_ITERATION_AUDIT_MISSING')
        legal=parameter_validity(m)
        context={'candidate':row['name'],'role':role,'seed':seed,'data_sha256':sha(data_path),
                 'fresh_fit':True,'candidate_checkpoint_used':False,'author_and_adapter_hashes':read(HERE/'input_contract.json')['frozen_hashes'],
                 'execution_package_sha256':sha(HERE/'package_manifest.json'),'environment':read(job/'environment.json')}
        if final: context['SELECTION_MANIFEST_SHA256']=sha(OUT/'SELECTION_MANIFEST.json')
        manifest=save_model(m,job/'model',context,observer.initial)
        q,saved=load_model(job/'model'); parameter_validity(q)
        require(parameter_hashes(q)==parameter_hashes(m),'ROUNDTRIP_PARAMETER_DRIFT')
        # Free original fit arrays before original conditional inference setup.
        del m,x
        smoke=inference_smoke(q)
        result={'PASS':True,'model_manifest_sha256':sha(job/'model/model_manifest.json'),'parameter_hash':saved['parameter_hash'],
                'initialization_hash':saved['initialization_hash'],'iterations':saved['iterations'],'convergence':saved['convergence'],
                'seed':seed,'parameter_validity':legal,'serialization_roundtrip':True,'all_components_smoke':smoke,
                'target_access':0,'role':role,'candidate':row['name']}
        write(job/'training_result.json',result,exclusive=True)
        return result

def validate(job,row):
    import numpy as np
    from model_ops import load_model,parameter_validity,MFAReferenceAdapter,ProjectPilotInterface,pooled_ratio,macro_score,candidate_cardinality,array_hash
    trainjob=OUT/'candidates'/row['name']/'train'
    with numeric_inputs('validate'):
        m,manifest=load_model(trainjob/'model'); parameter_validity(m)
        require(manifest['context']['candidate']==row['name'] and manifest['context']['seed']==row['seed'] and manifest['N']==12000 and manifest['D']==4032,'CANDIDATE_MODEL_IDENTITY_MISMATCH')
        pilots=np.load(ROOT/'outputs/sp_u12b5b/validation_pilots.npy',mmap_mode='r',allow_pickle=False)
        truth=np.load(ROOT/'outputs/sp_u12b5b/validation_truth.npy',mmap_mode='r',allow_pickle=False)
        interface=ProjectPilotInterface(); adapter=MFAReferenceAdapter(m,interface.A); cfg=science(); cells=[]; identities=[]
        fields=['candidate','K','rank','origin','snr_db','trajectory_id','frame_id','error_energy','truth_energy','pilot_sha256','truth_sha256']
        with (job/'frame_audit.csv').open('x',encoding='utf8',newline='') as f,(job/'trajectory_audit.csv').open('x',encoding='utf8',newline='') as tf:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            tw=csv.DictWriter(tf,fieldnames=fields[:6]+['error_energy','truth_energy','linear_ratio']);tw.writeheader()
            for o,origin in enumerate(cfg['canonical_origins']):
                # Preserve the old worker's summation axes and float64 order.
                T=np.sum(abs(np.asarray(truth[o]))**2,axis=2,dtype=np.float64)
                for si,snr in enumerate((10,20)):
                    E=np.empty((100,6),dtype=np.float64); max_resp_error=0.
                    for i in range(100):
                        y=np.asarray(pilots[o,si,i]); target=np.asarray(truth[o,i]); require(np.isfinite(y).all() and np.isfinite(target).all(),'VALIDATION_INPUT_NONFINITE')
                        estimated=adapter.estimate(y,cfg['noise_variances'][str(snr)]*np.eye(64,dtype=np.complex128))
                        responsibility=m.predict_proba(y)
                        require(np.isfinite(estimated).all() and np.isfinite(responsibility).all() and np.all(responsibility>=0),'INFERENCE_NONFINITE_OR_INVALID')
                        err=float(np.max(abs(responsibility.sum(axis=1)-1),initial=0));require(err<=1e-10,'RESPONSIBILITY_NORMALIZATION_FAILED');max_resp_error=max(max_resp_error,err)
                        E[i]=np.sum(abs(estimated[:,interface.data_positions]-target)**2,axis=1,dtype=np.float64)
                        for fi,frame in enumerate(range(4,10)):
                            rec={'candidate':row['name'],'K':row['K'],'rank':row['rank'],'origin':origin,'snr_db':snr,
                                 'trajectory_id':400+i,'frame_id':frame,'error_energy':float(E[i,fi]),'truth_energy':float(T[i,fi]),
                                 'pilot_sha256':array_hash(y[fi]),'truth_sha256':array_hash(target[fi])}
                            writer.writerow(rec);identities.append(rec)
                        tw.writerow({**{k:rec[k] for k in fields[:6]},'error_energy':float(E[i].sum(dtype=np.float64)),
                                     'truth_energy':float(T[i].sum(dtype=np.float64)),'linear_ratio':pooled_ratio(E[i],T[i])})
                        write(job/'phase.json',{'phase':'preparation_validation','origin':origin,'snr_db':snr,'trajectory_id':400+i,'predictions_complete':len(identities)})
                    cells.append({'origin':origin,'snr_db':snr,'numerator':float(E.sum(dtype=np.float64)),'denominator':float(T.sum(dtype=np.float64)),
                                  'linear_pooled_NMSE':pooled_ratio(E,T),'trajectories':100,'frames':600,'responsibility_normalization_max_error':max_resp_error})
        count=candidate_cardinality(identities,row['K'],row['rank'],cfg)
        result={'PASS':True,**row,'cells':cells,'macro_selection_score':macro_score([v['linear_pooled_NMSE'] for v in cells]),
                'prediction_count':count,'model_manifest_sha256':sha(trainjob/'model/model_manifest.json'),
                'frame_audit_sha256':sha(job/'frame_audit.csv'),'trajectory_audit_sha256':sha(job/'trajectory_audit.csv'),
                'input_hashes':{n:v['sha256'] for n,v in read(HERE/'input_contract.json')['data'].items() if 'validation_' in n},
                'target_access':0,'metric':'six pooled linear NMSE, then equal macro mean; no resource penalty'}
        write(job/'validation_result.json',result,exclusive=True);return result

def probe(job,mode):
    # Stdlib-only bounded resource/exception fixtures, never a scientific fit.
    require(job.is_relative_to(LOCAL),'TOY_JOB_OUTSIDE_PERSONAL_NAMESPACE')
    if mode=='exception': raise RuntimeError('INTENTIONAL_TOY_WORKER_EXCEPTION')
    if mode=='allocate': data=bytearray(64*2**20); data[:]=b'1'*len(data)
    if mode=='tree':
        import subprocess
        child=subprocess.Popen([sys.executable,'-B','-c','import time; x=bytearray(32*2**20); time.sleep(4)'])
        write(job/'probe_child.json',{'PID':child.pid})
    time.sleep(3 if mode!='success' else .5)
    return {'PASS':True,'synthetic_only':True}

def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=('train','validate','final','unit-probe'),required=True)
    p.add_argument('--job',type=Path,required=True);p.add_argument('--candidate');p.add_argument('--blas',type=int,required=True);p.add_argument('--openmp',type=int,required=True)
    p.add_argument('--probe-mode',default='success');args=p.parse_args();job=args.job.resolve()
    require(job.is_relative_to(OUT),'JOB_PATH_OUTSIDE_R6')
    scope();write(job/'actual_pid.json',{'PID':os.getpid(),'interpreter':sys.executable,'UTC':utc()})
    start=time.monotonic()
    while not (job/'monitor_ready.json').exists():
        require(time.monotonic()-start<60,'MONITOR_HANDSHAKE_TIMEOUT');time.sleep(.02)
    try:
        if args.action=='unit-probe': result=probe(job,args.probe_mode)
        else:
            for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]=str(args.blas)
            os.environ['OMP_NUM_THREADS']=str(args.openmp)
            approval();integrity(); row=candidate(args.candidate)
            expected=OUT/'final' if args.action=='final' else OUT/'candidates'/row['name']/args.action
            require(job==expected,'FORMAL_JOB_CANONICAL_PATH_REQUIRED')
            context=read(job/'execution_context.json')
            require(context['action']==args.action and context['candidate']==row and
                    context['package_manifest_sha256']==sha(HERE/'package_manifest.json') and
                    context['resource_proposal_sha256']==sha(HERE/'resource_amendment.json') and
                    context['approval']==approval(),'FORMAL_SUPERVISOR_EXECUTION_CONTEXT_REQUIRED')
            require(args.blas==6 and args.openmp==1,'FROZEN_FORMAL_THREADS_REQUIRED')
            from runner import valid_candidate,selection_frozen
            if args.action=='final':require(selection_frozen()['selected']['name']==row['name'],'FINAL_SELECTION_MISMATCH')
            elif args.action=='validate':valid_candidate(row,False)
            else:
                from common import registry
                for previous in registry()[:registry().index(row)]:valid_candidate(previous)
            write(job/'environment.json',environment());
            with threads(args.blas,args.openmp) as pools:
                write(job/'actual_threads.json',pools)
                if args.action in ('train','final'):result=train(job,row,args.action=='final')
                else:result=validate(job,row)
        from windows_resources import frozen_probe
        result.update(native_final_memory=frozen_probe().process_memory(),actual_PID=os.getpid())
        write(job/'worker_result.json',result)
    except BaseException as error:
        write(job/'exception.json',{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()});raise

if __name__=='__main__':main()
