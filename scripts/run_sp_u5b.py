"""Deterministic native/common scoring from saved U5A inputs and coordinates."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u5b_common import *
from src.link import assemble_pilot_data_frame,apply_frequency_mimo_channel,detect_data_zf,demodulate_data_bits,extract_reference_data_bits

def replay_trajectory(condition,i,aa,gs,mask,ids,pos,coordinates,native_frames,native_trajectories,input_hashes):
    p=OUT/f'sp_u5a_inputs/{condition}_{i:03}.npz';im=input_hashes[condition,i]
    if sha(p)!=im['cache_sha256']:raise ValueError('input cache drift')
    with np.load(p,allow_pickle=False) as f:inputs={k:f[k].copy() for k in f.files}
    if array_hash(inputs['H'])!=im['H_sha256']:raise ValueError('H drift')
    hh=frames(inputs['H']);cfg=readjson(OUT/'sp_u5a_preregistration.json')['config'];rows=[];trs=[];checks=[];reuse=[];pair=[]
    received={};common_received={};digests={df:hashlib.sha256() for df in gs}
    from src.estimators import pilot_ls_estimate
    for df,g in gs.items():
        received[df]=[]
        for j,h in enumerate(hh):
            x=assemble_pilot_data_frame(inputs['symbols'][j],g.layout);clean=apply_frequency_mimo_channel(h,x)
            ys=[]
            for si,s in enumerate((10,20)):
                y=clean+np.sqrt(cfg['noise_variances'][str(s)])*inputs['noise_atoms'][j]
                pp,_=pilot_ls_estimate(y,x,g.layout.pilot_mask);pp=pp.ravel()[g.selected]
                if not np.array_equal(pp,inputs[f'pilots_{df}'][si,j]):raise ValueError('pilot replay differs')
                digests[df].update(array_hash(y).encode());digests[df].update(array_hash(pp).encode());ys.append(y)
                cp=y[:,mask[0]]
                if df==8:common_received[j,s]=cp.copy()
                elif not np.array_equal(cp,common_received[j,s]):raise ValueError('common received realization differs')
            received[df].append(ys)
        if digests[df].hexdigest()!=im[f'dF{df}_received_digest']:raise ValueError('received full-grid digest differs')
    for df,g in gs.items():
        op_hash=sha(OUT/f'sp_u5a_operators_{df}.npz')
        for si,s in enumerate((10,20)):
            for j in cfg['score_frames']:
                cr=coordinates[condition,i,s,df,j];coef=np.array([float(cr['alpha']),float(cr['beta'])])
                if cr['operator_artifact_hash']!=op_hash:raise ValueError('coordinate operator mismatch')
                # Exact saved decimal float round-trip, never recomputing coordinates.
                reuse.append(dict(condition=condition,trajectory_id=i,snr_db=s,dF=df,frame_id=j,alpha=coef[0],beta=coef[1],operator_artifact_hash=op_hash,refit=False))
                pilot=inputs[f'pilots_{df}'][si,j];truth=hh[j].ravel()[g.data]
                common_truth=hh[j].ravel()[ids]
                for method,c in (('Source',None),('LR4',coef)):
                    value=predict(aa[df],pilot,s,c);full=np.full(4096,np.nan+1j*np.nan);full[g.data]=value
                    detected=detect_data_zf(received[df][j][si],full.reshape(2,2,64,16),g.layout.data_mask)*np.sqrt(2)
                    for domain,truth_v,pred_v,score_mask in (('native',truth,value,g.layout.data_mask),('common',common_truth,value[pos[df]],mask)):
                        ee=float(np.sum(abs(pred_v-truth_v)**2));te=float(np.sum(abs(truth_v)**2))
                        ref=extract_reference_data_bits(inputs['bits'][j],score_mask);dec=demodulate_data_bits(detected,score_mask)
                        errors=int(sum(np.count_nonzero(v!=w) for v,w in zip(dec,ref)));bits=sum(len(v) for v in ref)
                        r=dict(condition=condition,trajectory_id=i,snr_db=s,dF=df,pilot_density_id=f'dF{df}',operator_artifact_hash=op_hash,frame_id=j,method=method,domain=domain,
                            error_energy=ee,truth_energy=te,bit_errors=errors,bit_count=bits,BER=errors/bits,prediction_sha256=array_hash(pred_v),truth_sha256=array_hash(truth_v))
                        rows.append(r)
                        if domain=='native':
                            old=native_frames[condition,i,s,df,j,method]
                            delta=max(abs(ee-float(old['error_energy'])),abs(te-float(old['truth_energy'])))
                            if delta!=0 or errors!=int(old['bit_errors']) or bits!=int(old['bit_count']):raise ValueError('native frame replay differs')
                            checks.append(dict(level='frame',condition=condition,trajectory_id=i,snr_db=s,dF=df,frame_id=j,method=method,max_energy_difference=delta,max_nmse_difference=0.,BER_counts_exact=True))
                        elif bits!=3840:raise ValueError('common BER denominator')
            for method in ('Source','LR4'):
                for domain in ('native','common'):
                    rr=[r for r in rows if (r['snr_db'],r['dF'],r['method'],r['domain'])==(s,df,method,domain)]
                    metric=trajectory_metric([r['error_energy'] for r in rr],[r['truth_energy'] for r in rr]);be=sum(r['bit_errors'] for r in rr);bc=sum(r['bit_count'] for r in rr)
                    tr=dict(condition=condition,trajectory_id=i,snr_db=s,dF=df,pilot_density_id=f'dF{df}',operator_artifact_hash=op_hash,method=method,domain=domain,**metric,bit_errors=be,bit_count=bc,BER=be/bc);trs.append(tr)
                    if domain=='native':
                        old=native_trajectories[condition,i,s,df,method];delta=max(abs(float(old[k])-v) for k,v in metric.items())
                        if delta!=0 or be!=int(old['bit_errors']) or bc!=int(old['bit_count']):raise ValueError('native trajectory replay differs')
                        checks.append(dict(level='trajectory',condition=condition,trajectory_id=i,snr_db=s,dF=df,frame_id=-1,method=method,max_energy_difference=0.,max_nmse_difference=delta,BER_counts_exact=True))
    pair.append(dict(condition=condition,trajectory_id=i,H_sha256=im['H_sha256'],cache_sha256=im['cache_sha256'],common_received_exact=True,stored_full_grid_digests_exact=True,
        same_common_bits=True,same_stored_noise=True,common_truth_indices_sha256=array_hash(ids)))
    return rows,trs,checks,reuse,pair

def indexes():
    ff=[r for r in csvread(OUT/'sp_u5a_frames.csv') if r['artifact'] in ('8','16','32')]
    tt=[r for r in csvread(OUT/'sp_u5a_trajectories.csv') if r['artifact'] in ('8','16','32')]
    fi={(r['condition'],int(r['trajectory_id']),int(r['snr_db']),int(r['dF']),int(r['frame_id']),r['method']):r for r in ff}
    ti={(r['condition'],int(r['trajectory_id']),int(r['snr_db']),int(r['dF']),r['method']):r for r in tt}
    hi={(r['condition'],int(r['trajectory_id'])):r for r in csvread(OUT/'sp_u5a_input_hashes.csv')}
    return fi,ti,hi

def evaluate():
    firewall();prereg();protect()
    pre=readjson(OUT/'sp_u5b_tests_pre.json')
    if not pre['passed'] or any(sha(ROOT/k)!=v for k,v in pre['source_sha256'].items()):raise ValueError('pre-execution gate/source drift')
    if (OUT/'sp_u5b_trajectories.csv').exists():raise ValueError('already scored')
    aa={df:operators(df) for df in (8,16,32)};gs,mask,full,ids,pos=common_domain();coords=coordinate_index();fi,ti,hi=indexes();block_scientific_mutations()
    start=time.perf_counter();fr=[];tr=[];ck=[];co=[];pa=[]
    for c in ('ID','Composed'):
        for i in range(100):
            rr=replay_trajectory(c,i,aa,gs,mask,ids,pos,coords,fi,ti,hi)
            for dest,part in zip((fr,tr,ck,co,pa),rr):dest.extend(part)
            if (i+1)%20==0:print(f'fixed-coordinate native/common replay {c}:{i+1}/100',flush=True)
    for name,rows in (('frames',fr),('trajectories',tr),('native_replay',ck),('coordinate_reuse',co),('pairing_audit',pa)):csvsave(OUT/f'sp_u5b_{name}.csv',rows)
    save(OUT/'sp_u5b_replay_gate.json',dict(passed=True,trajectories=200,frame_rows=len(fr),trajectory_rows=len(tr),coordinate_rows=len(co),native_frame_comparisons=14400,native_trajectory_comparisons=2400,
        native_max_energy_difference=0.,native_max_metric_difference=0.,native_BER_counts_exact=True,coordinates_reused_exact=True,refit_calls=0,rebuild_calls=0,new_random_inputs=0,
        common_bits_per_frame=3840,common_bits_per_trajectory=23040,all200_pairing_passed=True,wall_seconds=time.perf_counter()-start,**peak_memory()))
    print('SP-U5B replay PASS; all native energies,metrics,BER exact; no refit/rebuild')

if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='evaluate':evaluate()
