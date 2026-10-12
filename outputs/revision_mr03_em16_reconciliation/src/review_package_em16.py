"""Recompute only persisted MR03B energies/paired intervals from a light ZIP."""
from pathlib import Path
import argparse,csv,json,hashlib,zipfile,os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();out=args.out
    cfg=json.loads((out/'config.json').read_text(encoding='utf8'));methods=cfg['methods']
    with (out/'tables/capacity_comparison.csv').open(encoding='utf8',newline='') as f:nm={(r['dataset'],r['condition'],int(r['snr_db']),r['method']):r for r in csv.DictReader(f)}
    with (out/'tables/paired_contrasts.csv').open(encoding='utf8',newline='') as f:co=list(csv.DictReader(f))
    checked=0;maximum=0
    def close(actual,target):
        nonlocal checked,maximum
        e=abs(float(actual)-float(target));maximum=max(maximum,e);checked+=1
        if e>1e-10+1e-9*max(abs(float(actual)),abs(float(target))):raise ValueError('PACKAGE_STATISTICAL_MISMATCH')
    for ds in cfg['datasets']:
        receipt=json.loads((out/'reports'/f'{ds}_receipt.json').read_text(encoding='utf8'))
        analysis=json.loads((out/'reports/analysis.json').read_text(encoding='utf8'))
        draws_path=out/'review_reference'/f'bootstrap_{ds}.npz'
        if sha(draws_path)!=analysis['original_draws_SHA256'][ds]:raise ValueError('DRAW_HASH')
        with np.load(draws_path,allow_pickle=False) as z:draws=dict(z)
        for c in cfg['conditions']:
            es=[];qs=[]
            for rec in receipt['receipts']:
                if rec['condition']!=c:continue
                path=out/rec['raw_path']
                if sha(path)!=rec['raw_sha256'] or sha(path.with_suffix('.json'))!=rec['JSON_sha256']:raise ValueError('RAW_HASH')
                with zipfile.ZipFile(path) as z:
                    if z.testzip() is not None:raise ValueError('CRC')
                with np.load(path,allow_pickle=False) as z:es.append(z['frame_error']);qs.append(z['frame_truth'])
            e=np.array(es).sum(axis=-1);q=np.array(qs).sum(axis=-1)
            if e.shape!=(100,3,12) or q.shape!=(100,) or not np.isfinite(e).all() or not np.isfinite(q).all() or np.any(e<0) or np.any(q<=0):raise ValueError('ENERGY_GRID')
            point=10*np.log10(e.sum(axis=0)/q.sum());boot=10*np.log10(e[draws[c]].sum(axis=1)/q[draws[c]].sum(axis=1)[:,None,None]);bounds=np.quantile(boot,[.025,.975],axis=0)
            for si,s in enumerate(cfg['snrs']):
                for mi,m in enumerate(methods):
                    r=nm[ds,c,s,m];close(point[si,mi],r['nmse_db']);close(bounds[0,si,mi],r['CI_lower']);close(bounds[1,si,mi],r['CI_upper'])
                for r in co:
                    if (r['dataset'],r['condition'],int(r['snr_db']))!=(ds,c,s):continue
                    a,b=methods.index(r['first']),methods.index(r['second']);ci=np.quantile(boot[:,si,a]-boot[:,si,b],[.025,.975]);close(point[si,a]-point[si,b],r['estimate_db']);close(ci[0],r['CI_lower']);close(ci[1],r['CI_upper'])
    print(json.dumps(dict(PASS=True,compared_numbers=checked,max_absolute_difference=maximum,new_random_draws=0,new_inference=0,new_training=0,original_data_writes=0)))
if __name__=='__main__':main()
