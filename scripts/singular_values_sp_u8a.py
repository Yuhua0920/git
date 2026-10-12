"""Retain both singular values for every RE of the seven fixed-ID estimates."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import sp_u8a_common as c
import numpy as np

def main():
    c.install_guard();assert c.read(c.OUT/'conditioning_summary.json')['PASS']
    g=c.observation_geometry();rows=[]
    target=c.OUT/'fixed_singular_values';target.mkdir(exist_ok=True)
    previous={(int(r['trajectory_id']),int(r['snr_db']),r['method']):r for r in c.csv_read(c.OUT/'fixed_ID_diagnostics.csv')}
    for i in c.IDS:
        for s in c.SNRS:
            for m in c.METHODS:
                values=[]
                for j in range(4,10):
                    with np.load(c.OUT/f'fixed/{i:04d}_{s}_{j}_{m}.npz',allow_pickle=False) as z:
                        h=z['H_est']
                    sv=np.linalg.svd(h[:,:,g.layout.data_mask[0]].transpose(2,0,1),compute_uv=False)
                    assert np.all(np.isfinite(sv)) and np.all(sv[:,1]>0)
                    values.append(sv)
                values=np.stack(values)
                assert float(np.median(values[:,:,1]))==float(previous[i,s,m]['sigma_min_median'])
                np.savez(target/f'{i:04d}_{s}_{m}.npz',notice=c.NOTICE,frames=np.arange(4,10),columns=np.array(['sigma_max','sigma_min']),values=values)
                r=dict(trajectory_id=i,snr_db=s,method=m,REs=5952)
                for k,name in enumerate(('sigma_max','sigma_min')):
                    for q,label in ((.05,'P05'),(.5,'median'),(.95,'P95'),(1,'max')):r[name+'_'+label]=float(np.quantile(values[:,:,k],q))
                rows.append(r)
    c.csv_write(c.OUT/'fixed_ID_singular_values.csv',rows)
    c.write(c.OUT/'fixed_singular_value_gate.json',dict(PASS=True,rows=len(rows),archives=70,RE_matrices=416640,
        both_singular_values_retained=True,previous_sigma_min_exact_identity=True,new_estimates=False))
    print(c.NOTICE,flush=True);print('Fixed-ID singular-value retention PASS: 70 summaries / 416640 RE matrices',flush=True)

if __name__=='__main__':main()
