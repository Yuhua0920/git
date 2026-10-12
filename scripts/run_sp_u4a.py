"""Recover clean training only; rebuild all budgets; stop on historical500 mismatch."""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u4a_common import *
from src.cdl_benchmark import OfficialCDL, frames, SYSTEM
from src.cdl_mismatch import ShiftCDL
from src.link import (generate_master_qpsk,assemble_pilot_data_frame,apply_frequency_mimo_channel,
    detect_data_zf,demodulate_data_bits,extract_reference_data_bits)
from src.estimators import pilot_ls_estimate


def recover():
    pre=prereg(); firewall()
    path=OUT/'sp_u4a_clean.npy'
    if path.exists(): raise ValueError('clean corpus exists')
    source_ref={int(r['trajectory_id']):r['replayed_h_sha256'] for r in csvread(OUT/'t15a_fix_replay_h_hashes.csv') if r['split']=='train_dev'}
    other_ref={(r['shift'],int(r['trajectory_id'])):r['H_sha256'] for r in csvread(OUT/'t15c_cdl_channel_hashes.csv') if r['split']=='train_dev'}
    x=np.lib.format.open_memmap(path,mode='w+',dtype=np.complex128,shape=(3,500,10,4096))
    rows=[]; start=time.perf_counter(); devices={}
    for ei,e in enumerate(pre['config']['environments']):
        name={'P':'CDL-B-30','M':'CDL-C-120'}.get(e)
        model=ShiftCDL(name) if name else OfficialCDL(); devices[e]=model.device
        for i in range(500):
            h=model.generate(i,split='train_dev') if name else model.generate(i)
            actual=array_hash(h);expected=other_ref[name,i] if name else source_ref[i]
            if actual!=expected: raise ValueError(f'STOP historical clean H mismatch {e}/{i}')
            x[ei,i]=frames(h).reshape(10,4096)
            rows.append(dict(environment=e,trajectory_id=i,H_sha256=actual,historical_exact=True))
            if (i+1)%100==0: print(f'clean {e}: {i+1}/500 historical H hashes exact',flush=True)
    x.flush(); del x
    csvsave(OUT/'sp_u4a_clean_hashes.csv',rows)
    save(OUT/'sp_u4a_clean_gate.json',dict(passed=True,trajectories=1500,frames=15000,generation_seconds=time.perf_counter()-start,
        generation_devices=devices,all_H_hashes_exact=True,corpus_sha256=sha(path),**peak_memory(),old_evaluation_read=False))


def prepare(only=None):
    pre=prereg(); cfg=pre['config']; firewall()
    if only is None:
        import subprocess
        for n in cfg['budgets']:
            subprocess.check_call([sys.executable,__file__,'prepare','--budget',str(n)],cwd=ROOT)
        rows=[]
        for n in cfg['budgets']:
            m=readjson(OUT/f'sp_u4a_construction_{n}.json')
            rows.append({k:m[k] for k in ('budget','trajectories_per_environment','frames_per_environment','stats_seconds','operator_seconds','compression_seconds','peak_working_set_bytes','peak_commit_bytes','artifact_storage_bytes','statistics_sha256','operators_sha256')})
        csvsave(OUT/'sp_u4a_table1_costs.csv',rows)
        return
    gate=readjson(OUT/'sp_u4a_clean_gate.json')
    if not gate['passed'] or gate['corpus_sha256']!=sha(OUT/'sp_u4a_clean.npy'): raise ValueError('corpus gate')
    x=np.load(OUT/'sp_u4a_clean.npy',mmap_mode='r'); costs=[]; histrows=[]; tolerance=cfg['historical_relative_roundoff_gate']
    old=historical_arrays() if only==500 else {}
    for n in [only]:
        if n not in cfg['budgets']:raise ValueError('budget not frozen grid')
        tick=time.perf_counter(); stats={}; warnings_seen=[]
        with warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter('always')
            for ei,e in enumerate(cfg['environments']): stats[e]=build_statistics(x[ei],pre['permutations'][e][:n])
            seconds=time.perf_counter()-tick
            a,timing=build_operators(stats,cfg)
            warnings_seen=[str(w.message) for w in warns]
        arr={}
        for e,v in stats.items():
            for k in ('mean_p','mean_d','rpp','rdp'): arr[f'{e}_{k}']=getattr(v,k)
            arr[f'{e}_ids']=np.array(v.train_trajectory_ids,np.int64); arr[f'{e}_samples']=np.array(v.samples)
        sp=OUT/f'sp_u4a_statistics_{n}.npz'; op=OUT/f'sp_u4a_operators_{n}.npz'
        np.savez_compressed(sp,**arr);np.savez_compressed(op,**a)
        meta=dict(budget=n,rank=4,all_object_budget=n,trajectories_per_environment=n,frames_per_environment=10*n,
            stats_seconds=seconds,**timing,**peak_memory(),statistics_sha256=sha(sp),operators_sha256=sha(op),
            artifact_storage_bytes=sp.stat().st_size+op.stat().st_size,warnings=warnings_seen,
            array_sha256={k:array_hash(v) for k,v in a.items()},clean_power={e:float(np.mean(abs(x[ei,pre['permutations'][e][:n]])**2)) for ei,e in enumerate(cfg['environments'])})
        save(OUT/f'sp_u4a_construction_{n}.json',meta)
        costs.append({k:meta[k] for k in ('budget','trajectories_per_environment','frames_per_environment','stats_seconds','operator_seconds','compression_seconds','peak_working_set_bytes','peak_commit_bytes','artifact_storage_bytes','statistics_sha256','operators_sha256')})
        print(f'N={n} rebuilt all means/covariances/operators/factors/CV; {seconds+timing["operator_seconds"]+timing["compression_seconds"]:.3f}s',flush=True)
        if n==500:
            for e,filename in HIST_STATS.items():
                with np.load(OUT/filename,allow_pickle=False) as f:
                    if sorted(f['train_trajectory_ids'].tolist())!=list(range(500)) or int(f['samples'])!=5000: raise ValueError('historical not full500')
                    for k in ('mean_p','mean_d','rpp','rdp'):
                        v=getattr(stats[e],k); ref=f[k]
                        histrows.append(dict(object=f'{e}.{k}',relative_error=rel_error(v,ref),maximum_absolute_error=float(np.max(abs(v-ref)))))
            comparisons={k:(a[k],old[k]) for k in ('source_mean_p','source_mean_d')}
            for s in (10,20):
                comparisons[f'Wsource_{s}']=(a[f'Wsource_{s}'],old[f'Wsource_{s}'])
                for e in ('P','M'):
                    comparisons[f'W{e}_{s}']=(a[f'W{e}_{s}'],old[f'W{e}_{s}'])
                    for stem in ('DeltaW','DeltaK'): comparisons[f'{stem}{e}_{s}']=(a[f'{stem}{e}_{s}'],old[f'{stem}{e}_{s}'])
                    comparisons[f'B{e}4_{s}']=(a[f'A_{e}_{s}']@a[f'Q_{e}_{s}'],old[f'A_{e}_{s}']@old[f'Q_{e}_{s}'])
                for e in cfg['environments']: comparisons[f'K{e}_{s}']=(a[f'K{e}_{s}'],old[f'K{e}_{s}'])
            for k,(v,ref) in comparisons.items(): histrows.append(dict(object=k,relative_error=rel_error(v,ref),maximum_absolute_error=float(np.max(abs(v-ref)))))
    if only!=500:return
    csvsave(OUT/'sp_u4a_historical500_errors.csv',histrows)
    passed=all(r['relative_error']<=tolerance for r in histrows)
    save(OUT/'sp_u4a_historical500_gate.json',dict(passed=passed,historical_full500_per_environment=True,relative_tolerance=tolerance,
        maximum_relative_error=max(r['relative_error'] for r in histrows),objects=histrows,effective_products_compared=True,
        performance_permitted=passed,case='provenance PASS' if passed else 'Case D STOP',raw_factor_equality_required=False))
    if not passed: raise ValueError('Case D STOP: N500 differs beyond fixed roundoff gate; no development evaluation')


class DevelopmentCDL(OfficialCDL):
    def __init__(self,condition):
        super().__init__();self.condition=condition
        from sionna.phy.channel.tr38901 import CDL
        from src.cdl_benchmark import ARRAY
        c=config()['conditions'][condition];speed=c['speed_kmh']/3.6
        self.model=CDL(model=c['profile'],delay_spread=100e-9,carrier_frequency=3.5e9,ut_array=self.array,bs_array=self.array,
            direction='downlink',ut_orientation=ARRAY['ut_orientation_radians'],bs_orientation=ARRAY['bs_orientation_radians'],
            ut_velocity=[speed,0.,0.],min_speed=speed,max_speed=speed,normalize_delays=True,precision='double',device=self.device,spec_version='19.2')

    def generate(self,trajectory):
        import torch
        from sionna.phy import config as sconfig
        from sionna.phy.channel import cir_to_ofdm_channel
        from src.cdl_benchmark import SYMBOL_SECONDS
        s=development_seeds(self.condition,trajectory);sconfig.seed=s['sionna_seed'];torch.manual_seed(s['torch_seed'])
        self.numpy_rng=np.random.default_rng(s['numpy_seed'])
        with torch.no_grad():
            a,tau=self.model(batch_size=1,num_time_steps=160,sampling_frequency=1/SYMBOL_SECONDS)
            hf=cir_to_ofdm_channel(self.frequencies,a,tau,normalize=False)
        return hf[0,0,:,0,:,:,:].permute(0,1,3,2).cpu().numpy().astype(np.complex128,copy=True)


def evaluate():
    pre=prereg();cfg=pre['config'];firewall()
    if not readjson(OUT/'sp_u4a_historical500_gate.json')['passed']: raise ValueError('historical gate FAIL')
    tested=readjson(OUT/'sp_u4a_tests_pre.json')
    if not tested['passed']: raise ValueError('scientific pytest gate FAIL')
    if any(sha(ROOT/p)!=h for p,h in tested['source_sha256'].items()):raise ValueError('tested execution source drift')
    if SYSTEM['sample_rate_hz']!=SYSTEM['nfft']*SYSTEM['delta_f_hz']: raise ValueError('sampling config invalid')
    if (OUT/'sp_u4a_trajectories.csv').exists():raise ValueError('development output protected')
    arrays={}
    for n in cfg['budgets']:
        meta=readjson(OUT/f'sp_u4a_construction_{n}.json');path=OUT/f'sp_u4a_operators_{n}.npz'
        if sha(path)!=meta['operators_sha256']:raise ValueError('operator drift')
        with np.load(path,allow_pickle=False) as f:arrays[n]={k:f[k].copy() for k in f.files}
    arrays['Original500']=historical_arrays()
    g=observation_geometry();scores=cfg['score_frames'];tr=[];wr=[];pr=[];sr=[];fr=[];det=[];start=time.perf_counter()
    cache=OUT/'sp_u4a_development_cache';cache.mkdir()
    oldhashes={r['H_sha256'] for r in csvread(OUT/'sp_u4a_clean_hashes.csv')}
    cv={(n,s):cv_model(a,s) for n,a in arrays.items() for s in (10,20)}
    factor={(n,s,e):RankFactors(a[f'A_{e}_{s}'],a[f'Q_{e}_{s}']) for n,a in arrays.items() for s in (10,20) for e in ('P','M')}
    for condition in cfg['conditions']:
        model=DevelopmentCDL(condition)
        for i in range(100):
            h=model.generate(i);hhash=array_hash(h)
            if hhash in oldhashes:raise ValueError('evaluation/train H leakage')
            truths=frames(h);pilots=np.empty((2,10,64),np.complex128);energy=[];pairhash=hashlib.sha256();frameinputs=[];signal=[]
            for j,truth in enumerate(truths):
                seed=development_seeds(condition,i,j);sr.append(dict(namespace=cfg['namespace'],condition=condition,trajectory_id=i,frame_id=j,**seed))
                bits,symbols=generate_master_qpsk(bits_seeds=(seed['bits_seed_tx0'],seed['bits_seed_tx1']))
                x=assemble_pilot_data_frame(symbols,g.layout);clean=apply_frequency_mimo_channel(truth,x)
                rng=np.random.default_rng(seed['noise_seed']);atom=(rng.standard_normal(clean.shape)+1j*rng.standard_normal(clean.shape))/np.sqrt(2)
                energy.append(float(np.sum(abs(truth.ravel()[g.data])**2)));signal.append(float(np.mean(abs(clean[:,g.layout.data_mask[0]])**2)))
                ys=[]
                for si,s in enumerate((10,20)):
                    y=clean+np.sqrt(cfg['noise_variances'][str(s)])*atom;ys.append(y)
                    sparse,_=pilot_ls_estimate(y,x,g.layout.pilot_mask);pilots[si,j]=sparse.ravel()[g.selected]
                    pairhash.update(array_hash(y).encode());pairhash.update(array_hash(pilots[si,j]).encode())
                for value in (truth,bits,symbols,atom):pairhash.update(array_hash(value).encode())
                frameinputs.append((bits,ys))
            np.savez_compressed(cache/f'{condition}_{i:03}.npz',H=h,pilots=pilots,truth_energy=np.array(energy))
            pr.append(dict(condition=condition,trajectory_id=i,H_sha256=hhash,paired_inputs_sha256=pairhash.hexdigest(),budgets_share_identical_inputs=True,
                score_truth_energy=float(sum(energy[4:])),score_useful_receive_power=float(np.mean(signal[4:]))))
            for si,s in enumerate((10,20)):
                for n,a in arrays.items():
                    error={'Source':[],'LR4':[]};coordinates=[]
                    for j in scores:
                        c,diag=cv[n,s].fit(pilots[si,j-4:j+1],modules=('P','M'))
                        base=source_predict(a,pilots[si,j],s);z=pilots[si,j]-a['source_mean_p']
                        pred=base+c[0]*factor[n,s,'P'].apply(z)+c[1]*factor[n,s,'M'].apply(z)
                        for method,value in (('Source',base),('LR4',pred)):
                            ee=float(np.sum(abs(value-truths[j].ravel()[g.data])**2));error[method].append(ee)
                            fr.append(dict(condition=condition,trajectory_id=i,snr_db=s,budget=str(n),frame_id=j,method=method,error_energy=ee,truth_energy=energy[j]))
                            # One fixed receiver smoke, all budgets/methods, before claiming unchanged detector.
                            if i==0 and j==4:
                                full=np.full(4096,np.nan+1j*np.nan);full[g.data]=value
                                detected=detect_data_zf(frameinputs[j][1][si],full.reshape(2,2,64,16),g.layout.data_mask)*np.sqrt(2)
                                decoded=demodulate_data_bits(detected,g.layout.data_mask);reference=extract_reference_data_bits(frameinputs[j][0],g.layout.data_mask)
                                det.append(dict(condition=condition,snr_db=s,budget=str(n),method=method,bit_errors=int(sum(np.count_nonzero(a0!=b0) for a0,b0 in zip(decoded,reference))),bit_count=sum(len(b0) for b0 in reference),smoke_only=True))
                        wr.append(dict(condition=condition,trajectory_id=i,snr_db=s,budget=str(n),frame_id=j,alpha=float(c[0]),beta=float(c[1]),condition_number=diag['condition_number'],normal_equation_residual=diag['normal_equation_relative_residual']))
                        coordinates.append(c)
                    for method,errors in error.items():tr.append(dict(condition=condition,trajectory_id=i,snr_db=s,budget=str(n),method=method,**trajectory_metric(errors,energy[4:])))
            if (i+1)%20==0:print(f'{condition} paired Source/LR4 four budgets + original500: {i+1}/100',flush=True)
    for name,rows in (('trajectories',tr),('windows',wr),('pair_hashes',pr),('seed_manifest',sr),('frames',fr),('detector_smoke',det)):csvsave(OUT/f'sp_u4a_{name}.csv',rows)
    save(OUT/'sp_u4a_development_gate.json',dict(passed=True,trajectories=200,trajectory_metric_rows=len(tr),window_rows=len(wr),frame_rows=len(fr),
        score_frames=scores,paired_budgets=True,common_standardized_noise_across_snr=True,channel_normalized=False,
        historical500_gate_sha256=sha(OUT/'sp_u4a_historical500_gate.json'),dataset=cfg['dataset'],namespace=cfg['namespace'],
        wall_seconds=time.perf_counter()-start,**peak_memory(),formal_outputs_read=False,detector='frozen ZF; 40 first-trajectory smoke calls only; no BER scientific claim'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('recover','prepare','evaluate'));p.add_argument('--budget',type=int);args=p.parse_args()
    if args.stage=='prepare':prepare(args.budget)
    else:globals()[args.stage]()
