"""Bounded analytic D4032 inference timing, not a trained/scientific model."""
import time,json
import numpy as np
import base as b
from science import modules,infer
def main():
    c,_,_,wr=modules()
    from src.baselines.mfa_reference_adapter import make_reference_model,MFAReferenceAdapter,ProjectPilotInterface
    m=make_reference_model(1,16);m.D=4032;m.N=1
    m.means=np.zeros((1,4032),np.complex128);m.lambdas=np.zeros((1,4032,16),np.complex128)
    m.psis=np.ones((1,4032));m.amps=np.ones(1);m.covs=np.eye(4032,dtype=np.complex128)[None,:,:];m.inv_covs=np.eye(4032,dtype=np.complex128)[None,:,:]
    a=MFAReferenceAdapter(m,ProjectPilotInterface().A);samples=[]
    with c.threads(6,1):
        for _ in range(2):
            _,timing=infer(a,np.full((6,64),.25+.5j,np.complex128),.9941134822523973);samples.append(timing)
    v={'synthetic_only':True,'model_fit_calls':0,'formal_inputs_read':0,'formal_predictions':0,'dimension':4032,'rows_per_call':6,'BLAS':6,'OpenMP':1,'timings':samples,
        'median_seconds_per_call':float(np.median([x['adapter_wall_seconds'] for x in samples])),
        '35000_call_linear_extrapolation_hours':float(np.median([x['adapter_wall_seconds'] for x in samples])*35000/3600),
        'native_memory':wr.frozen_probe().process_memory(),'warning':'Personal CPU, analytic covariance and two calls only; excludes data I/O, integrity and bootstrap; not a LAB performance claim or fair speedup.'}
    b.write(b.OUT/'synthetic/cost_probe.json',v,True);print(json.dumps(v,indent=2))
if __name__=='__main__':main()
