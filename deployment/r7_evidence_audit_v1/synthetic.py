"""Analytic tiny fixtures only: no fit, seeds, channels or validation metrics."""
from pathlib import Path
import json, numpy as np
import audit_core as a
def fixture(directory,row,history=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);K=row['K'];D=24;r=row['rank']
    W=(np.arange(K*D*r).reshape(K,D,r)%17-8).astype(np.complex128)*.001
    means=np.zeros((K,D),dtype=np.complex128);psi=np.ones((K,D),dtype=np.float64)*.5;pi=np.ones(K,dtype=np.float64)/K
    C=np.stack([x@x.conj().T+.5*np.eye(D,dtype=np.complex128)for x in W]);inv=np.stack([np.linalg.inv(x)for x in C])
    arrays=dict(means=means,lambdas=W,psis=psi,amps=pi,covs=C,inv_covs=inv)
    history=np.array(history if history is not None else[100,90,80,70,60,50,50.00001],dtype=np.float64)
    meta={n:{'shape':list(v.shape),'dtype':v.dtype.str,'sha256':a.array_sha(v)}for n,v in arrays.items()}
    for n,v in{**arrays,'training_history':history}.items():
        with(directory/(n+'.npy')).open('xb')as f:np.save(f,v,allow_pickle=False)
    from src.baselines.mfa_reference_adapter import FROZEN_POLICY
    value={'format':'R6_ORIGINAL_MFA_NPY_V1','K':K,'rank':r,'D':D,'N':12,'policy':FROZEN_POLICY,
      'context':{'candidate':row['name'],'role':'train','seed':row['seed'],'fresh_fit':True,'candidate_checkpoint_used':False,'synthetic_fixture':True},
      'arrays':meta,'initialization':json.loads(json.dumps(meta)),'parameter_hash':a.objsha(meta),'initialization_hash':a.objsha(meta),
      'iterations':len(history),'convergence':'MAX_ITER_REACHED'if len(history)==100 else'CONVERGED',
      'file_hashes':{p.name:a.sha(p)for p in directory.glob('*.npy')}}
    (directory/'model_manifest.json').write_text(json.dumps(value,indent=2),encoding='utf8')
    (directory.parent/'iteration_history.jsonl').write_text(''.join(json.dumps({'iteration':i+1,'author_log_likelihood':float(v)})+'\n'for i,v in enumerate(history)),encoding='utf8')
    return value
