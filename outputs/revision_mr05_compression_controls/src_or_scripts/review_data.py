"""Independent sufficient-data review, no project scientific imports or new RNG."""
from pathlib import Path
import hashlib,json,csv,zipfile,sys
import numpy as np

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def review(root):
    receipt=json.loads((root/'reports/development_receipt.json').read_text())
    conditions=('ID','Profile','Mobility','Composed','Boundary');snrs=(10,20,30)
    methods=('SOURCE','POOLED_NATURAL','DENSE_TW5','CORR_ORIGIN_R4','CORR_I_R4','CORR_ORIGIN_M2_R4','CORR_POOL_M2_R4','WHOLE_POOL_M2_R23','WHOLE_POOL_M2_R24')
    E=np.empty((5,100,3,9));Q=np.empty((5,100));AA=np.zeros_like(E);BB=np.zeros_like(E);seen=set();max_error=0.
    for row in receipt['receipts']:
        key=(row['condition'],row['trajectory_id']);assert key not in seen;seen.add(key)
        p=root/row['path'];assert digest(p)==row['sha256'];assert digest(p.with_suffix('.json'))==row['json_sha256']
        with zipfile.ZipFile(p) as z:assert z.testzip() is None
        with np.load(p,allow_pickle=False) as z:
            e=z['frame_error'];q=z['frame_truth'];a=z['A'];b=z['B']
            assert tuple(z['method_names'])==methods and tuple(z['snr_db'])==snrs and tuple(z['frame_ids'])==tuple(range(4,10))
            assert e.shape==a.shape==b.shape==(3,9,6);assert q.shape==(6,)
            assert all(np.isfinite(v).all() for v in (e,q,a,b,z['coordinates'])) and np.all(e>=0) and np.all(q>0) and np.all(a>=0)
            residual=e-e[:,2:3]-a-b;scale=np.abs(a)+np.abs(b)+e+e[:,2:3]
            assert np.all(np.abs(residual)<=1e-10+1e-10*scale)
            max_error=max(max_error,float(np.max(np.abs(residual)/(1+scale))))
            ci=conditions.index(key[0]);i=key[1];E[ci,i]=e.sum(-1);Q[ci,i]=q.sum();AA[ci,i]=a.sum(-1);BB[ci,i]=b.sum(-1)
    assert seen=={(c,i) for c in conditions for i in range(100)}
    point=10*np.log10(E.sum(1)/Q.sum(1)[:,None,None]);table=list(csv.DictReader((root/'tables/nmse.csv').open()))
    assert len(table)==135
    for r in table:
        ci=conditions.index(r['condition']);si=snrs.index(int(r['snr_db']));mi=methods.index(r['method'])
        assert abs(point[ci,si,mi]-float(r['nmse_db']))<1e-10
    indices=dict(np.load(root/'data/bootstrap_indices.npz',allow_pickle=False));draws=[]
    for ci,c in enumerate(conditions):
        ix=indices[c];assert ix.shape==(2000,100) and ix.dtype==np.int64 and ix.min()>=0 and ix.max()<100
        draws.append(10*np.log10(E[ci][ix].sum(1)/Q[ci][ix].sum(1)[:,None,None]))
    contrasts=list(csv.DictReader((root/'tables/contrasts.csv').open()));assert len(contrasts)==285
    CI_error=0.
    for r in contrasts:
        ci=conditions.index(r['condition']);si=snrs.index(int(r['snr_db']));i,j=map(methods.index,(r['first'],r['second']))
        sample=draws[ci][:,si,i]-draws[ci][:,si,j]
        lower,upper=np.quantile(sample,[.025,.975],method='linear')
        CI_error=max(CI_error,abs(lower-float(r['CI_lower'])),abs(upper-float(r['CI_upper'])))
        assert abs(lower-float(r['CI_lower']))<1e-10 and abs(upper-float(r['CI_upper']))<1e-10
        assert abs(point[ci,si,i]-point[ci,si,j]-float(r['estimate_db']))<1e-10
    answer=dict(PASS=True,trajectories=len(seen),scored_positions_per_method=9000,nmse_rows=135,contrasts=285,
        max_signed_identity_scaled_error=max_error,max_independent_CI_absolute_error=CI_error,new_random_draws=0,review='read-only sufficient-data recomputation; original hashes untouched')
    print(json.dumps(answer,indent=2));return answer
if __name__=='__main__':review(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1])
