"""SP-U4B prepare12 independent subsets, replay existing inputs, evaluate fixed arms."""
import argparse
import sys
import time
import warnings
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u4b_common import *
from src.cdl_benchmark import frames,SYSTEM
from src.link import generate_master_qpsk,assemble_pilot_data_frame,apply_frequency_mimo_channel
from src.estimators import pilot_ls_estimate


def prepare_one(r,n):
    p=prereg();firewall()
    if r not in ('R1','R2','R3','R4') or n not in (50,100,250):raise ValueError('unregistered preparation')
    x=np.load(OUT/'sp_u4a_clean.npy',mmap_mode='r');start=time.perf_counter();stats={};descriptors=[]
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        for ei,e in enumerate(('Source','P','M')):
            ids=p['permutations'][r][e][:n];stats[e]=build_statistics(x[ei],ids);v=stats[e]
            descriptors.append(dict(replicate=r,budget=n,environment=e,clean_power=float(np.mean(abs(x[ei,ids])**2)),
                pilot_covariance_trace_per_dimension=float(np.trace(v.rpp).real/64),mean_norm=float(np.linalg.norm(np.r_[v.mean_p,v.mean_d]))))
        a,diag=build_operators(stats,p['config']);warnings_seen=[str(w.message) for w in ws]
    arr={}
    for e,v in stats.items():
        for k in ('mean_p','mean_d','rpp','rdp'):arr[f'{e}_{k}']=getattr(v,k)
        arr[f'{e}_ids']=np.array(v.train_trajectory_ids,np.int64);arr[f'{e}_samples']=np.array(v.samples)
    np.savez_compressed(artifact(r,n,'statistics'),**arr);np.savez_compressed(artifact(r,n),**a)
    save(OUT/f'sp_u4b_construction_{r}_{n}.json',dict(replicate=r,budget=n,object_identity=[r,n],rank=4,wall_seconds=time.perf_counter()-start,
        statistics_sha256=sha(artifact(r,n,'statistics')),operators_sha256=sha(artifact(r,n)),warnings=warnings_seen,descriptors=descriptors,
        construction_diagnostics=diag,N500_used_to_construct=False,other_replicate_used_to_construct=False))
    print(f'{r} N{n}: complete statistics/operators/rank4/CV construction saved',flush=True)


def prepare():
    p=prereg()
    if sha(OUT/'sp_u4a_clean.npy')!=readjson(OUT/'sp_u4a_clean_gate.json')['corpus_sha256']:raise ValueError('clean corpus changed')
    for r in ('R1','R2','R3','R4'):
        for n in (50,100,250):subprocess.check_call([sys.executable,__file__,'prepare_one','--replicate',r,'--budget',str(n)],cwd=ROOT)
    save(OUT/'sp_u4b_construction_gate.json',dict(passed=True,new_preparations=12,new_npz_artifacts=24,N500_rebuilt=False,
        common500_operators_sha256=sha(artifact('R0',500)),common500_statistics_sha256=sha(artifact('R0',500,'statistics'))))


def replay_inputs():
    p=prereg();firewall();cfg=p['config'];g=observation_geometry()
    seeds={(r['condition'],int(r['trajectory_id']),int(r['frame_id'])):r for r in csvread(OUT/'sp_u4a_seed_manifest.csv')}
    evidence={(r['condition'],int(r['trajectory_id'])):r for r in csvread(OUT/'sp_u4a_pair_hashes.csv')};rows=[]
    for condition in cfg['conditions']:
        for i in range(100):
            with np.load(OUT/'sp_u4a_development_cache'/f'{condition}_{i:03}.npz') as f:h=f['H'];expected_pilots=f['pilots']
            # Restore frozen native export physical layout [rx,tx,t,k] backing.
            # NPY persistence retains values but cannot retain these mixed strides.
            h=h.transpose(0,1,3,2).copy(order='C').transpose(0,1,3,2)
            if array_hash(h)!=evidence[condition,i]['H_sha256']:raise ValueError('cached H drift')
            digest=hashlib.sha256()
            for j,truth in enumerate(frames(h)):
                s=seeds[condition,i,j];bits,symbols=generate_master_qpsk(bits_seeds=(int(s['bits_seed_tx0']),int(s['bits_seed_tx1'])))
                x=assemble_pilot_data_frame(symbols,g.layout);clean=apply_frequency_mimo_channel(truth,x)
                rng=np.random.default_rng(int(s['noise_seed']));atom=(rng.standard_normal(clean.shape)+1j*rng.standard_normal(clean.shape))/np.sqrt(2)
                for si,snr in enumerate((10,20)):
                    y=clean+np.sqrt(cfg['noise_variances'][str(snr)])*atom;sparse,_=pilot_ls_estimate(y,x,g.layout.pilot_mask);pilots=sparse.ravel()[g.selected]
                    if not np.array_equal(pilots,expected_pilots[si,j]):raise ValueError(f'cached pilots not exact:{condition}/{i}/{j}/{snr}')
                    digest.update(array_hash(y).encode());digest.update(array_hash(pilots).encode())
                for v in (truth,bits,symbols,atom):digest.update(array_hash(v).encode())
            if digest.hexdigest()!=evidence[condition,i]['paired_inputs_sha256']:raise ValueError('complete communication replay digest mismatch')
            rows.append(dict(condition=condition,trajectory_id=i,H_sha256=array_hash(h),paired_inputs_sha256=digest.hexdigest(),pilot_exact=True,
                native_H_strides=list(h.strides)))
    save(OUT/'sp_u4b_input_replay_gate.json',dict(passed=True,trajectories=200,all_H_and_communication_digests_exact=True,all4000_frame_snr_pilots_exact=True,
        new_channel_generations=0,layout_recovered_from_frozen_export=True,independent_validation=False))
    csvsave(OUT/'sp_u4b_input_replay.csv',rows);print('All200 U4A H/communication digests and4000 pilot vectors EXACT; new H generation0',flush=True)


def evaluate():
    p=prereg();firewall();cfg=p['config']
    for filename in ('sp_u4b_construction_gate.json','sp_u4b_input_replay_gate.json','sp_u4b_tests_pre.json'):
        if not readjson(OUT/filename)['passed']:raise ValueError('prerequisite failed:'+filename)
    tested=readjson(OUT/'sp_u4b_tests_pre.json')
    if any(sha(ROOT/k)!=h for k,h in tested['source_sha256'].items()):raise ValueError('execution source drift')
    if SYSTEM['sample_rate_hz']!=SYSTEM['nfft']*SYSTEM['delta_f_hz']:raise ValueError('sampling config invalid')
    if (OUT/'sp_u4b_trajectories.csv').exists():raise ValueError('results already exist')
    start=time.perf_counter();tr=[];wr=[];fr=[]
    for r in csvread(OUT/'sp_u4a_trajectories.csv'):
        if r['budget']=='Original500':continue
        tr.append(dict(replicate='Common500' if r['budget']=='500' else 'R0',**r))
    for r in csvread(OUT/'sp_u4a_windows.csv'):
        if r['budget']=='Original500':continue
        wr.append(dict(replicate='Common500' if r['budget']=='500' else 'R0',**r))
    models={}
    for rep in ('R1','R2','R3','R4'):
        for n in (50,100,250):
            meta=readjson(OUT/f'sp_u4b_construction_{rep}_{n}.json')
            if sha(artifact(rep,n))!=meta['operators_sha256']:raise ValueError('prepared operator changed')
            a=load_arrays(rep,n)
            for s in (10,20):models[rep,n,s]=(a,cv_model(a,s),{e:RankFactors(a[f'A_{e}_{s}'],a[f'Q_{e}_{s}']) for e in ('P','M')})
    g=observation_geometry()
    for condition in cfg['conditions']:
        for i in range(100):
            with np.load(OUT/'sp_u4a_development_cache'/f'{condition}_{i:03}.npz') as f:h=f['H'];pilots=f['pilots'];energy=f['truth_energy']
            truths=frames(h)
            for (rep,n,s),(a,cv,factors) in models.items():
                si=0 if s==10 else 1;errors={'Source':[],'LR4':[]}
                for j in range(4,10):
                    c,d=cv.fit(pilots[si,j-4:j+1],modules=('P','M'));z=pilots[si,j]-a['source_mean_p'];base=source_predict(a,pilots[si,j],s)
                    pred=base+c[0]*factors['P'].apply(z)+c[1]*factors['M'].apply(z)
                    for method,value in (('Source',base),('LR4',pred)):
                        ee=float(np.sum(abs(value-truths[j].ravel()[g.data])**2));errors[method].append(ee)
                        fr.append(dict(replicate=rep,condition=condition,trajectory_id=i,snr_db=s,budget=n,frame_id=j,method=method,error_energy=ee,truth_energy=float(energy[j])))
                    wr.append(dict(replicate=rep,condition=condition,trajectory_id=i,snr_db=s,budget=str(n),frame_id=j,alpha=float(c[0]),beta=float(c[1]),
                        condition_number=d['condition_number'],normal_equation_residual=d['normal_equation_relative_residual']))
                for method,err in errors.items():tr.append(dict(replicate=rep,condition=condition,trajectory_id=i,snr_db=s,budget=str(n),method=method,**trajectory_metric(err,energy[4:])))
            if (i+1)%25==0:print(f'{condition}: fixed U4A input {i+1}/100,12 preparations evaluated',flush=True)
    csvsave(OUT/'sp_u4b_trajectories.csv',tr);csvsave(OUT/'sp_u4b_windows.csv',wr);csvsave(OUT/'sp_u4b_new_frames.csv',fr)
    save(OUT/'sp_u4b_evaluation_gate.json',dict(passed=True,new_channel_generations=0,R0_raw_reused=True,N500_common_raw_reused=True,
        trajectory_rows=len(tr),window_rows=len(wr),new_frame_rows=len(fr),wall_seconds=time.perf_counter()-start,
        scientific_unit='5 independently permuted offline subset realizations on same finite corpus;100 reused evaluation trajectories per condition',formal_inference=False))


if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('stage',choices=('prepare','prepare_one','replay_inputs','evaluate'));pa.add_argument('--replicate');pa.add_argument('--budget',type=int);args=pa.parse_args()
    if args.stage=='prepare_one':prepare_one(args.replicate,args.budget)
    else:globals()[args.stage]()
