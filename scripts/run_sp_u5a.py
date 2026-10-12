"""Self-consistent preparations, mandatory historical gate, paired true-pilot scan."""
import sys,time,warnings,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u5a_common import *
from src.link import generate_master_qpsk,assemble_pilot_data_frame,apply_frequency_mimo_channel,detect_data_zf,demodulate_data_bits,extract_reference_data_bits
from src.estimators import pilot_ls_estimate

class PilotDevelopmentCDL(OfficialCDL):
    def __init__(self,condition):
        super().__init__();self.condition=condition
        from sionna.phy.channel.tr38901 import CDL
        from src.cdl_benchmark import ARRAY
        c=cfg()['conditions'][condition];v=c['speed_kmh']/3.6
        self.model=CDL(model=c['profile'],delay_spread=100e-9,carrier_frequency=3.5e9,ut_array=self.array,bs_array=self.array,direction='downlink',
            ut_orientation=ARRAY['ut_orientation_radians'],bs_orientation=ARRAY['bs_orientation_radians'],ut_velocity=[v,0.,0.],
            min_speed=v,max_speed=v,normalize_delays=True,precision='double',device=self.device,spec_version='19.2')
    def generate(self,i):
        import torch
        from sionna.phy import config as sc
        from sionna.phy.channel import cir_to_ofdm_channel
        from src.cdl_benchmark import SYMBOL_SECONDS
        ss=seed_map(self.condition,i);sc.seed=ss['sionna_seed'];torch.manual_seed(ss['torch_seed']);self.numpy_rng=np.random.default_rng(ss['numpy_seed'])
        with torch.no_grad():a,tau=self.model(batch_size=1,num_time_steps=160,sampling_frequency=1/SYMBOL_SECONDS);hf=cir_to_ofdm_channel(self.frequencies,a,tau,normalize=False)
        # A new namespace has one declared canonical layout, preserved by cache reload.
        return hf[0,0,:,0,:,:,:].permute(0,1,3,2).cpu().numpy().astype(np.complex128).copy(order='C')

def prepare(df=None):
    prereg();firewall()
    if df is None:
        for d in (16,8,32):subprocess.run([sys.executable,__file__,'prepare',str(d)],cwd=ROOT,check=True)
        return
    g=geometry(df);gate=readjson(OUT/'sp_u4a_clean_gate.json')
    if not gate['passed'] or sha(OUT/'sp_u4a_clean.npy')!=gate['corpus_sha256']:raise ValueError('frozen clean corpus changed')
    if (OUT/f'sp_u5a_operators_{df}.npz').exists():raise ValueError('preparation already exists')
    x=np.load(OUT/'sp_u4a_clean.npy',mmap_mode='r');tick=time.perf_counter()
    with warnings.catch_warnings(record=True) as ww:
        warnings.simplefilter('always');stats={e:build_stats(x[ei],g) for ei,e in enumerate(cfg()['environments'])};stats_seconds=time.perf_counter()-tick
        a,diags=build_ops(stats,g)
    st={f'{e}_{k}':getattr(v,k) for e,v in stats.items() for k in ('mean_p','mean_d','rpp','rdp')}
    for e,v in stats.items():st[f'{e}_ids']=np.array(v.train_trajectory_ids);st[f'{e}_samples']=np.array(v.samples)
    sp=OUT/f'sp_u5a_statistics_{df}.npz';op=OUT/f'sp_u5a_operators_{df}.npz';np.savez_compressed(sp,**st);np.savez_compressed(op,**a)
    save(OUT/f'sp_u5a_construction_{df}.json',dict(passed=True,dF=df,trajectories_per_environment=500,frames_per_environment=5000,
        pilot_dimension=len(g.selected),target_dimension=int(g.data.sum()),statistics_seconds=stats_seconds,wall_seconds=time.perf_counter()-tick,
        statistics_sha256=sha(sp),operators_sha256=sha(op),warnings=[str(w.message) for w in ww],diagnostics=diags,
        shapes={k:list(v.shape) for k,v in a.items()},geometry_sha256=sha(OUT/f'sp_u5a_geometry_{df}.npz'),fold_sha256=folds_for(g)[1]['sha256'],**peak_memory()))
    print(f'dF{df} complete500 preparation; pilot{len(g.selected)},target{g.data.sum()}',flush=True)

def inputs():
    prereg();firewall();cache=OUT/'sp_u5a_inputs';cache.mkdir(exist_ok=True)
    if (OUT/'sp_u5a_input_gate.json').exists():raise ValueError('inputs already frozen')
    trainhash={r['H_sha256'] for r in csvread(OUT/'sp_u4a_clean_hashes.csv')};rows=[];devices={};seen=set();start=time.perf_counter()
    for c in cfg()['conditions']:
        model=PilotDevelopmentCDL(c);devices[c]=model.device
        for i in range(100):
            h=model.generate(i);hh=array_hash(h)
            if hh in trainhash or hh in seen:raise ValueError('H leakage/duplicate')
            seen.add(hh);truths=frames(h);bb=[];sy=[];nn=[];pp={df:[] for df in cfg()['densities']};digests={df:hashlib.sha256() for df in cfg()['densities']}
            for j,truth in enumerate(truths):
                ss=seed_map(c,i,j);bits,symbols=generate_master_qpsk(bits_seeds=(ss['bits_seed_tx0'],ss['bits_seed_tx1']));rng=np.random.default_rng(ss['noise_seed'])
                atom=(rng.standard_normal((2,64,16))+1j*rng.standard_normal((2,64,16)))/np.sqrt(2)
                bb.append(bits);sy.append(symbols);nn.append(atom)
                for df in cfg()['densities']:
                    g=geometry(df);x=assemble_pilot_data_frame(symbols,g.layout);clean=apply_frequency_mimo_channel(truth,x);pair=[]
                    if not np.allclose(np.sum(abs(x)**2,axis=0),1,rtol=0,atol=1e-12):raise ValueError('active RE power')
                    for s in (10,20):
                        y=clean+np.sqrt(cfg()['noise_variances'][str(s)])*atom;p,_=pilot_ls_estimate(y,x,g.layout.pilot_mask);p=p.ravel()[g.selected];pair.append(p)
                        digests[df].update(array_hash(y).encode());digests[df].update(array_hash(p).encode())
                    pp[df].append(pair)
            payload=dict(H=h,bits=np.array(bb),symbols=np.array(sy),noise_atoms=np.array(nn),**{f'pilots_{df}':np.array(v).transpose(1,0,2) for df,v in pp.items()})
            path=cache/f'{c}_{i:03}.npz'
            if path.exists():raise ValueError('input cache collision')
            np.savez_compressed(path,**payload)
            rows.append(dict(condition=c,trajectory_id=i,H_sha256=hh,bits_sha256=array_hash(payload['bits']),symbols_sha256=array_hash(payload['symbols']),
                full_noise_sha256=array_hash(payload['noise_atoms']),cache_sha256=sha(path),dF8_received_digest=digests[8].hexdigest(),
                dF16_received_digest=digests[16].hexdigest(),dF32_received_digest=digests[32].hexdigest(),all_densities_share_H_bits_symbols_noise=True))
            if (i+1)%20==0:print(f'new paired inputs {c}:{i+1}/100',flush=True)
    csvsave(OUT/'sp_u5a_input_hashes.csv',rows);save(OUT/'sp_u5a_input_gate.json',dict(passed=True,trajectories=200,distinct_H=200,
        generation_devices=devices,continuous_states_per_H=160,dataset=cfg()['dataset'],namespace=cfg()['namespace'],channel_normalized=False,
        full_grid_bits=True,full_grid_noise=True,common_noise_across_SNR=True,wall_seconds=time.perf_counter()-start,U4_evaluation_read=False,
        source_sha256={n:sha(ROOT/n) for n in ('scripts/sp_u5a_common.py','scripts/run_sp_u5a.py','configs/sp_u5a_protocol.json')}))

def regression():
    prereg();firewall();new=arrays(16);old=historical_arrays();rows=[];replay=[];tol=cfg()['historical_relative_roundoff_gate']
    def compare(name,a,b):rows.append(dict(object=name,relative_error=rel_error(a,b),maximum_absolute_error=float(np.max(abs(a-b)))))
    with np.load(OUT/'sp_u5a_statistics_16.npz') as st:
        for e,path in HIST_STATS.items():
            with np.load(OUT/path) as hist:
                for k in ('mean_p','mean_d','rpp','rdp'):compare(f'{e}.{k}',st[f'{e}_{k}'],hist[k])
    for k in ('source_mean_p','source_mean_d'):compare(k,new[k],old[k])
    for s in (10,20):
        for k in [f'Wsource_{s}',*[f'W{e}_{s}' for e in ('P','M')],*[f'K{e}_{s}' for e in ('Source','P','M')],
            *[f'{stem}{e}_{s}' for stem in ('DeltaW','DeltaK') for e in ('P','M')]]:compare(k,new[k],old[k])
        for e in ('P','M'):compare(f'B{e}4_{s}',new[f'A_{e}_{s}']@new[f'Q_{e}_{s}'],old[f'A_{e}_{s}']@old[f'Q_{e}_{s}'])
    from src.spatial_information import observation_geometry
    g=geometry(16);histg=observation_geometry()
    masks={k:np.array_equal(getattr(g.layout,k),getattr(histg.layout,k)) for k in ('pilot_mask','data_mask','mute_mask','pilot_owner')}
    masks['selected']=np.array_equal(g.selected,histg.selected);masks['data']=np.array_equal(g.data,histg.data)
    histfolds,_=frozen_folds(histg.selected);fs,_=folds_for(g);masks['folds']=all(np.array_equal(a,b) for ff,hh in zip(fs,histfolds) for a,b in zip(ff,hh))
    if not all(masks.values()) or not all(r['relative_error']<=tol for r in rows):
        save(OUT/'sp_u5a_regression_gate.json',dict(passed=False,objects=rows,masks=masks,stop=True));raise ValueError('STOP historical construction mismatch')
    if not readjson(OUT/'sp_u5a_input_gate.json')['passed']:raise ValueError('missing paired inputs')
    for c in cfg()['conditions']:
        for i in (0,1):
            with np.load(OUT/f'sp_u5a_inputs/{c}_{i:03}.npz') as f:
                for si,s in enumerate((10,20)):
                    pilots=f['pilots_16'][si];histcv=PilotCV(histfolds,old['source_mean_p'],tuple(old[f'KSource_{s}']),tuple(old[f'DeltaKP_{s}']),tuple(old[f'DeltaKM_{s}']))
                    for j in cfg()['score_frames']:
                        cn,dn=cv_model(new,g,s).fit(pilots[j-4:j+1],modules=('P','M'));co,do=histcv.fit(pilots[j-4:j+1],modules=('P','M'))
                        vals=dict(coordinate_max_abs=float(max(abs(cn-co))),gram_max_abs=float(np.max(abs(np.array(dn['gram'])-do['gram']))))
                        for m,cc,oc in [('Source',None,None),('LR4',cn,co)]:
                            a=predict(new,pilots[j],s,cc);b=predict(old,pilots[j],s,oc)
                            replay.append(dict(condition=c,trajectory_id=i,snr_db=s,frame_id=j,method=m,prediction_relative_error=rel_error(a,b),prediction_max_abs=float(max(abs(a-b))),**vals))
    csvsave(OUT/'sp_u5a_regression_objects.csv',rows);csvsave(OUT/'sp_u5a_regression_replay.csv',replay)
    passed=all(r['prediction_relative_error']<=tol and r['prediction_max_abs']<=cfg()['historical_replay_absolute_gate'] and r['coordinate_max_abs']<=cfg()['historical_replay_absolute_gate'] for r in replay)
    save(OUT/'sp_u5a_regression_gate.json',dict(passed=passed,tolerance_relative=tol,tolerance_absolute=cfg()['historical_replay_absolute_gate'],objects=rows,masks=masks,
        maximum_relative_error=max(r['relative_error'] for r in rows),replay_rows=len(replay),replay_max_prediction_relative=max(r['prediction_relative_error'] for r in replay),
        replay_max_prediction_abs=max(r['prediction_max_abs'] for r in replay),replay_max_coordinate_abs=max(r['coordinate_max_abs'] for r in replay),scientific_results_permitted=passed))
    if not passed:raise ValueError('STOP actual historical replay mismatch')
    print('historical dF16 full construction and actual4-trajectory replay PASS',flush=True)

def evaluate():
    p=prereg();firewall();pre=readjson(OUT/'sp_u5a_tests_pre.json')
    if not pre['passed'] or not readjson(OUT/'sp_u5a_regression_gate.json')['passed']:raise ValueError('scientific gate denied')
    if any(sha(ROOT/k)!=v for k,v in pre['source_sha256'].items()):raise ValueError('execution drift')
    if (OUT/'sp_u5a_trajectories.csv').exists():raise ValueError('already evaluated')
    aa={str(df):arrays(df) for df in cfg()['densities']};aa['Historical16']=historical_arrays();geos={str(df):geometry(df) for df in cfg()['densities']};geos['Historical16']=geometry(16)
    hashes={str(df):sha(OUT/f'sp_u5a_operators_{df}.npz') for df in cfg()['densities']};hashes['Historical16']=hashlib.sha256(''.join(sha(OUT/n) for n in ('t16e_frozen_operators.npz','t16f_frozen_candidates.npz','t16g_frozen_module_factors.npz')).encode()).hexdigest()
    cvs={(key,s):cv_model(a,geos[key],s) for key,a in aa.items() for s in (10,20)}
    factors={(key,s,e):RankFactors(a[f'A_{e}_{s}'],a[f'Q_{e}_{s}']) for key,a in aa.items() for s in (10,20) for e in ('P','M')}
    tr=[];fr=[];wr=[];fails=[];start=time.perf_counter();inputrows=csvread(OUT/'sp_u5a_input_hashes.csv')
    for c in cfg()['conditions']:
        for i in range(100):
            path=OUT/f'sp_u5a_inputs/{c}_{i:03}.npz';meta=next(r for r in inputrows if r['condition']==c and int(r['trajectory_id'])==i)
            if sha(path)!=meta['cache_sha256']:raise ValueError('input cache drift')
            with np.load(path) as f:inputs={k:f[k].copy() for k in f.files}
            truths=frames(inputs['H']);received={}
            for key,g in geos.items():
                if key=='Historical16':received[key]=received['16'];continue
                ys=[]
                for j,truth in enumerate(truths):
                    x=assemble_pilot_data_frame(inputs['symbols'][j],g.layout);clean=apply_frequency_mimo_channel(truth,x)
                    ys.append([clean+np.sqrt(cfg()['noise_variances'][str(s)])*inputs['noise_atoms'][j] for s in (10,20)])
                received[key]=ys
            for key,a in aa.items():
                g=geos[key];pilots=inputs[f'pilots_{g.df}'];artifact=hashes[key]
                for si,s in enumerate((10,20)):
                    errs={'Source':[],'LR4':[]};energies=[];berr={'Source':0,'LR4':0};bitcount=0
                    for j in cfg()['score_frames']:
                        try:cfit,diag=cvs[key,s].fit(pilots[si,j-4:j+1],modules=('P','M'))
                        except np.linalg.LinAlgError:
                            fails.append(dict(condition=c,trajectory_id=i,snr_db=s,dF=g.df,artifact=key,frame_id=j,reason='singular Gram; no fallback'))
                            csvsave(OUT/'sp_u5a_rank_failures.csv',fails);save(OUT/'sp_u5a_evaluation_gate.json',dict(passed=False,stop=True,rank_failures=fails));raise
                        z=pilots[si,j]-a['source_mean_p'];base=a['source_mean_d']+z@a[f'Wsource_{s}'].T
                        adapted=base+cfit[0]*factors[key,s,'P'].apply(z)+cfit[1]*factors[key,s,'M'].apply(z)
                        energy=float(np.sum(abs(truths[j].ravel()[g.data])**2));energies.append(energy)
                        ref=extract_reference_data_bits(inputs['bits'][j],g.layout.data_mask);bc=sum(len(v) for v in ref);bitcount+=bc
                        common=dict(condition=c,trajectory_id=i,snr_db=s,dF=g.df,pilot_density_id=f'dF{g.df}',artifact=key,operator_artifact_hash=artifact,frame_id=j)
                        for method,value in [('Source',base),('LR4',adapted)]:
                            ee=float(np.sum(abs(value-truths[j].ravel()[g.data])**2));errs[method].append(ee)
                            full=np.full(4096,np.nan+1j*np.nan);full[g.data]=value
                            detected=detect_data_zf(received[key][j][si],full.reshape(2,2,64,16),g.layout.data_mask)*np.sqrt(2)
                            dec=demodulate_data_bits(detected,g.layout.data_mask);be=int(sum(np.count_nonzero(v!=w) for v,w in zip(dec,ref)));berr[method]+=be
                            fr.append(dict(**common,method=method,error_energy=ee,truth_energy=energy,bit_errors=be,bit_count=bc,BER=be/bc))
                        wr.append(dict(**common,alpha=float(cfit[0]),beta=float(cfit[1]),coordinate_norm=float(np.linalg.norm(cfit)),gram_rank=diag['gram_rank'],condition_number=diag['condition_number'],
                            normal_equation_residual=diag['normal_equation_relative_residual'],gram00=diag['gram'][0][0],gram01=diag['gram'][0][1],gram11=diag['gram'][1][1],rhs0=diag['rhs'][0],rhs1=diag['rhs'][1],window_start=j-4,window_end=j,fallback=False))
                    for method in ('Source','LR4'):tr.append(dict(condition=c,trajectory_id=i,snr_db=s,dF=g.df,pilot_density_id=f'dF{g.df}',artifact=key,operator_artifact_hash=artifact,
                        method=method,**trajectory_metric(errs[method],energies),bit_errors=berr[method],bit_count=bitcount,BER=berr[method]/bitcount))
            if (i+1)%20==0:print(f'true pilot NMSE/full BER {c}:{i+1}/100',flush=True)
    for name,r in [('trajectories',tr),('frames',fr),('windows',wr)]:csvsave(OUT/f'sp_u5a_{name}.csv',r)
    save(OUT/'sp_u5a_evaluation_gate.json',dict(passed=True,new_trajectories=200,densities_paired=True,U4_evaluation_used=False,trajectory_rows=len(tr),frame_rows=len(fr),window_rows=len(wr),
        rank_failures=0,fallback_count=0,wall_seconds=time.perf_counter()-start,dataset=cfg()['dataset'],formal=False,**peak_memory()))

if __name__=='__main__':
    stage=sys.argv[1]
    if stage=='prepare':prepare(int(sys.argv[2]) if len(sys.argv)>2 else None)
    else:globals()[stage]()
