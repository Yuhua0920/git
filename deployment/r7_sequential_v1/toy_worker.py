"""Bounded original-author synthetic fixture, never accesses preparation arrays."""
from pathlib import Path
import sys,os,time,argparse,traceback,json
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import r7_common as q

def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=('unit-probe',),required=True);p.add_argument('--job',type=Path,required=True)
    p.add_argument('--blas',type=int,required=True);p.add_argument('--openmp',type=int,required=True);p.add_argument('--probe-mode',default='success');p.add_argument('--candidate');p.add_argument('--fault',default='none');a=p.parse_args();job=a.job.resolve()
    q.require(job.is_relative_to(q.OUT/'technical') and os.environ.get('R7_SYNTHETIC_TEST')=='1','TOY_ONLY_NAMESPACE_REQUIRED')
    q.write(job/'actual_pid.json',{'PID':os.getpid(),'interpreter':sys.executable,'UTC':q.utc()})
    until=time.monotonic()+60
    while not (job/'monitor_ready.json').exists():q.require(time.monotonic()<until,'MONITOR_HANDSHAKE_TIMEOUT');time.sleep(.01)
    # Real data are disallowed even if a malformed toy tries to load them.
    def firewall(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            if os.fsdecode(args[0]).lower()==os.devnull.lower():return
            path=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
            writing=isinstance(mode,str) and any(v in mode for v in 'wax+') or isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            if writing:q.require(path.is_relative_to(q.OUT/'technical'),'TOY_HISTORICAL_WRITE_DENIED')
            elif path.suffix.lower() in ('.npy','.npz','.pkl','.pickle','.h5','.mat'):q.require(path.is_relative_to(q.OUT/'technical'),'TOY_REAL_DATA_READ_DENIED')
    sys.addaudithook(firewall)
    try:
        c,r,s,w=q.r6_imports()
        if a.probe_mode=='exception':raise RuntimeError('INTENTIONAL_THIRD_CANDIDATE_FAULT')
        if a.probe_mode=='allocate':x=bytearray(64*2**20);time.sleep(2)
        if a.probe_mode=='timeout':time.sleep(3)
        if a.probe_mode in ('train','validate'):
            import numpy as np
            from model_ops import parameter_validity,parameter_hashes,make_reference_model,fit_reference,convergence,MFAReferenceAdapter,pooled_ratio,macro_score
            row=next(v for v in q.contract()['candidate_order'] if v['name']==a.candidate)
            rng=np.random.default_rng(928716);x=(rng.normal(size=(96,32))+1j*rng.normal(size=(96,32)))/np.sqrt(2);x[:48]+=1.1
            with c.threads(a.blas,a.openmp):
                if a.probe_mode=='train':
                    m=make_reference_model(row['K'],row['rank']);fit_reference(m,x,row['seed']);legal=parameter_validity(m)
                    folder=job/'model';folder.mkdir(exist_ok=False)
                    for n in ('means','lambdas','psis','amps','covs','inv_covs'):
                        with (folder/(n+'.npy')).open('xb') as f:np.save(f,getattr(m,n),allow_pickle=False)
                    np.save(folder/'history.npy',np.asarray(m.L_all),allow_pickle=False)
                    hashes={p.name:q.digest(p) for p in folder.glob('*.npy')}
                    result={'PASS':True,'synthetic_only':True,'candidate':row,'seed':row['seed'],'N':96,'D':32,'file_hashes':hashes,
                            'legal':legal,'iterations':len(m.L_all),'convergence':convergence(m.L_all),'parameters':parameter_hashes(m)}
                else:
                    parent=job.parent/'train';manifest=q.read(parent/'toy_result.json');m=make_reference_model(row['K'],row['rank']);m.D=32;m.N=96
                    for n in ('means','lambdas','psis','amps','covs','inv_covs'):
                        q.require(q.digest(parent/'model'/ (n+'.npy'))==manifest['file_hashes'][n+'.npy'],'TOY_MODEL_HASH_DRIFT')
                        setattr(m,n,np.load(parent/'model'/(n+'.npy'),allow_pickle=False))
                    parameter_validity(m);A=np.eye(32,dtype=complex)[:4];adapter=MFAReferenceAdapter(m,A);frames=[];cells=[]
                    # 3600 toy identities exercise exact production coverage;
                    # toy channel dimension/statistics are deliberately unrelated.
                    for origin in q.contract()['science']['canonical_origins']:
                        for snr,noise in ((10,.37),(20,.037)):
                            target=(rng.normal(size=(600,32))+1j*rng.normal(size=(600,32)))/np.sqrt(2)
                            y=target[:,:4]+np.sqrt(noise/2)*(rng.normal(size=(600,4))+1j*rng.normal(size=(600,4)))
                            estimate=adapter.estimate(y,noise*np.eye(4));E=np.sum(abs(estimate[:,4:]-target[:,4:])**2,axis=1);T=np.sum(abs(target[:,4:])**2,axis=1)
                            q.require(np.isfinite(estimate).all(),'TOY_INFERENCE_NONFINITE');cells.append(pooled_ratio(E,T))
                            for i in range(100):
                                for f in range(4,10):frames.append({'K':row['K'],'rank':row['rank'],'origin':origin,'snr_db':snr,'trajectory_id':400+i,'frame_id':f})
                    if a.fault=='missing_frame':frames.pop()
                    if a.fault=='duplicate_frame':frames[-1]=frames[0]
                    result={'PASS':True,'synthetic_only':True,'candidate':row,'seed':row['seed'],'frames':frames,'cells':cells,'toy_macro_score':macro_score(cells)}
            q.write(job/'toy_result.json',result,exclusive=True)
        else:result={'PASS':True,'synthetic_only':True};time.sleep(.3)
        result.update(native_final_memory=w.frozen_probe().process_memory(),actual_PID=os.getpid())
        q.write(job/'worker_result.json',result,exclusive=True)
    except BaseException as e:
        q.write(job/'exception.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()},exclusive=True);raise

if __name__=='__main__':main()
