"""Strict A/B -> C -> D receiver diagnostic ladder on USED R1 data."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import sp_u8a_common as c
import numpy as np

def prepare():
    c.OUT.mkdir(exist_ok=True)
    c.install_guard()
    if (c.OUT/'fixtures.json').exists():raise RuntimeError('completed A/B gate immutable')
    if (c.OUT/'STOP.json').exists():
        stopped=c.read(c.OUT/'STOP.json')
        if stopped['phase']!='prepare':raise RuntimeError('cannot resume a scientific stop')
        (c.OUT/'STOP.json').replace(c.OUT/'preparation_firewall_failure.json')
    if not (c.OUT/'before.json').exists():c.write(c.OUT/'before.json',c.snapshot())
    frozen=c.read(c.R1/'freeze.json')
    deployable=set(c.read(c.frozen.PROTOCOL)['deployable']['objects'])
    for p,h in frozen['hashes'].items():
        if p.startswith('outputs/') and p not in deployable and not p.startswith('outputs/sp_u7b_r1/'):continue
        if c.sha(c.ROOT/p)!=h:raise RuntimeError('R1 source freeze drift '+p)
    c.write(c.OUT/'scope.json',dict(task='SP-U8A',data='USED SP-U7B-R1 only',primary_case='A_STRONG_TRANSFER FROZEN',
        primary_metric='NMSE',secondary_metric='BER',OOD_branch='FROZEN',fixed_IDs=c.IDS,frames=list(range(4,10)),
        methods=c.METHODS,Perfect_label='POST-HOC DIAGNOSTIC ONLY',rank=4,
        coordinate_policy='read saved coordinates; no fitting',
        replay_policy='reconstruct original predictions with frozen functions and frozen artifacts; no new channel-estimation run',
        deployable_artifacts='unchanged R1 frozen operators/factors, not historical performance data',
        receiver='frozen exact square solve, no new regularization',confidence_claims=False,
        thresholds=[10,100],new_H_bits_noise_trajectories=0))
    print(c.NOTICE,flush=True)
    c.write(c.OUT/'fixtures.json',c.fixtures())
    print('A/B fixtures PASS; preservation baseline complete',flush=True)

def replay():
    c.install_guard()
    assert c.read(c.OUT/'fixtures.json')['PASS']
    assert c.read(c.OUT/'pytest_pre.json')['PASS']
    if (c.OUT/'replay_gate.json').exists():raise RuntimeError('completed replay immutable')
    print(c.NOTICE,flush=True)
    c.write(c.OUT/'execution_freeze.json',dict(hashes={p:c.sha(c.ROOT/p) for p in
        ('scripts/sp_u8a_common.py','scripts/run_sp_u8a.py','scripts/test_sp_u8a.py','tests/test_sp_u8a.py')},
        detector_sha256=c.sha(c.ROOT/'src/detectors.py'),link_sha256=c.sha(c.ROOT/'src/link.py')))
    a,f,matched=c.load_methods();g=c.observation_geometry()
    rows=[];traj=[];inputchecks=0;outputchecks=0;counts=np.zeros((2,5),np.int64);frozen_counts=np.zeros((2,4),np.int64)
    nb=np.zeros(2,np.int64);max_delta=0
    for i in range(1000):
        hh,comm,raw,proof=c.ingredients(i)
        expected_inputs={(v['snr_db'],v['frame_id']):v for v in proof['inputs']}
        expected_outputs={(v['snr_db'],v['frame_id'],v['method']):v for v in proof['outputs']}
        per=np.zeros((2,5),np.int64)
        for j,(bits,symbols,unit) in enumerate(comm):
            h=hh[j];x=c.assemble_pilot_data_frame(symbols,g.layout);clean=c.apply_frequency_mimo_channel(h,x)
            ref=c.extract_reference_data_bits(bits,g.layout.data_mask)
            for si,s in enumerate(c.SNRS):
                noise=np.sqrt(c.VARIANCES[s])*unit;y=clean+noise;y.flags.writeable=False
                sparse,linear=c.estimate_ls_linear_channel(y,g.layout)
                pilot=sparse.ravel()[g.selected];pilot.flags.writeable=False
                actual=dict(H_sha256=c.array_hash(h),bits_sha256=c.array_hash(bits),symbols_sha256=c.array_hash(symbols),
                    unit_noise_sha256=c.array_hash(unit),Y_sha256=c.array_hash(y),pilot_sha256=c.array_hash(pilot))
                expected=expected_inputs[s,j]
                if any(actual[k]!=expected[k] for k in actual):raise RuntimeError(f'input replay failed {i}/{s}/{j}')
                inputchecks+=len(actual)
                if j<4:continue
                ji=j-4;co=raw['coordinates'][si*6+ji]
                assert co[0]==s and co[1]==j
                pred,grids=c.replay_prediction(pilot,linear,sparse,s,co[2:4],a,f,matched)
                grids={'Perfect':h,**grids}
                bits_per=sum(v.size for v in ref)
                if bits_per!=int(raw['frame_total_bits'][si,ji]) or bits_per!=3968:raise RuntimeError('bit-count identity')
                for mi,m in enumerate(c.METHODS):
                    if m!='Perfect':
                        if c.array_hash(pred[m])!=expected_outputs[s,j,m]['prediction_sha256']:raise RuntimeError(f'prediction replay failed {i}/{s}/{j}/{m}')
                        outputchecks+=1
                    errors,_=c.count_bits(y,grids[m],g.layout.data_mask,ref)
                    per[si,mi]+=errors
                    delta=0
                    if m!='Perfect':
                        original=int(raw['frame_bit_errors'][si,c.frozen.METHODS.index(m),ji]);delta=abs(errors-original)
                        if delta:raise RuntimeError(f'BER frame replay failed {i}/{s}/{j}/{m}')
                    max_delta=max(max_delta,delta)
                    rows.append(dict(trajectory_id=i,snr_db=s,frame_id=j,method=m,bit_errors=errors,total_bits=bits_per,
                        BER=errors/bits_per,replay_bit_error_delta=delta,label='POST-HOC DIAGNOSTIC ONLY' if m=='Perfect' else 'Frozen comparator replay'))
                    if i in c.IDS:
                        fixed=c.OUT/'fixed';fixed.mkdir(exist_ok=True)
                        np.savez(fixed/f'{i:04d}_{s}_{j}_{m}.npz',notice=c.NOTICE,H=h,H_est=grids[m],Y=y,X=x,N=noise)
                nb[si]+=bits_per
        for si,s in enumerate(c.SNRS):
            assert raw['total_bits'][si]==23808
            for mi,m in enumerate(c.METHODS):
                traj.append(dict(trajectory_id=i,snr_db=s,method=m,bit_errors=int(per[si,mi]),total_bits=23808,BER=per[si,mi]/23808))
            frozen_counts[si]+=raw['bit_errors'][si,[c.frozen.METHODS.index(m) for m in c.METHODS[1:]]]
        counts+=per
        if (i+1)%25==0:print(f'C replay {i+1}/1000: exact inputs/predictions/frame counts PASS',flush=True)
    assert np.array_equal(counts[:,1:],frozen_counts) and np.all(nb==23808000)
    c.csv_write(c.OUT/'frames.csv',rows);c.csv_write(c.OUT/'trajectories.csv',traj)
    c.write(c.OUT/'replay_gate.json',dict(PASS=True,trajectories=1000,frames=6000,SNR_frame_pairs=12000,
        input_hash_comparisons=inputchecks,prediction_hash_comparisons=outputchecks,
        exact_frame_count_replays=48000,max_bit_count_error=max_delta,max_aggregate_BER_error=0.,
        aggregate_counts=counts.tolist(),frozen_counts=frozen_counts.tolist(),total_bits=nb.tolist(),
        no_regeneration=c.COUNTERS.copy(),data_mask_sha256=c.array_hash(g.layout.data_mask),
        selected_sha256=c.array_hash(g.selected),data_channel_mask_sha256=c.array_hash(g.data),
        saved_alpha_beta_rows_reused=12000,Perfect_H_identity=True,primary_rebootstrap=False))
    print('C PASS: exact full-data replay; D authorized',flush=True)

def diagnose():
    c.install_guard();assert c.read(c.OUT/'replay_gate.json')['PASS']
    print(c.NOTICE,flush=True)
    g=c.observation_geometry();allvalues=[];condition=[];frame_summary=[];nonfinite=0;singular=0;fixed=[]
    (c.OUT/'conditioning').mkdir(exist_ok=True)
    for i in range(1000):
        hh,_,_,_=c.ingredients(i);v=[]
        for j in range(4,10):
            mx,mn,kap,log,enh,_=c.conditioning(hh[j],g.layout.data_mask)
            one=np.stack([mx,mn,kap,log,enh],axis=-1);v.append(one)
            frame_summary.append(dict(trajectory_id=i,frame_id=j,sigma_min_median=float(np.median(mn)),
                sigma_min_P95=float(np.quantile(mn,.95)),log_condition_median=float(np.median(log)),
                log_condition_P95=float(np.quantile(log,.95)),noise_enhancement_median=float(np.median(enh)),
                noise_enhancement_P95=float(np.quantile(enh,.95))))
        v=np.stack(v);flat=v.reshape(-1,5);allvalues.append(flat)
        nonfinite+=int(np.count_nonzero(~np.isfinite(flat)));singular+=int(np.count_nonzero(flat[:,1]==0))
        np.savez(c.OUT/f'conditioning/{i:04d}.npz',notice=c.NOTICE,columns=np.array(['sigma_max','sigma_min','condition','log10_condition','noise_enhancement']),frames=np.arange(4,10),values=v)
        row=dict(trajectory_id=i,REs=flat.shape[0],condition_gt10=float(np.mean(flat[:,2]>10)),condition_gt100=float(np.mean(flat[:,2]>100)))
        for k,name in enumerate(('sigma_max','sigma_min','condition','log10_condition','noise_enhancement')):
            for q,label in ((.05,'P05'),(.5,'median'),(.95,'P95'),(1,'max')):row[name+'_'+label]=float(np.quantile(flat[:,k],q))
        condition.append(row)
        if (i+1)%100==0:print(f'D conditioning {i+1}/1000',flush=True)
    # User-fixed trajectories only: actual symbol error decomposition, including cross term.
    max_relative=0.;max_symbol_residual=0.
    for i in c.IDS:
        for s in c.SNRS:
            sums={m:dict(mismatch=0.,noise=0.,cross=0.,error=0.,energy=0.,eq=[],sigma=[],ser=0,bits=0) for m in c.METHODS}
            for j in range(4,10):
                for m in c.METHODS:
                    with np.load(c.OUT/f'fixed/{i:04d}_{s}_{j}_{m}.npz',allow_pickle=False) as z:
                        h,he,y,x,n=(z[k] for k in ('H','H_est','Y','X','N'))
                    mx,mn,kap,log,enh,eq=c.conditioning(he,g.layout.data_mask)
                    ht=h[:,:,g.layout.data_mask[0]].transpose(2,0,1)
                    xx=x[:,g.layout.data_mask[0]].T;nn=n[:,g.layout.data_mask[0]].T
                    mismatch=np.einsum('nij,nj->ni',eq@ht-np.eye(2),xx)
                    noisepart=np.einsum('nij,nj->ni',eq,nn)
                    detected=c.detect_data_zf(y,he,g.layout.data_mask)[:,g.layout.data_mask[0]].T
                    error=detected-xx
                    residual=float(np.max(abs(error-mismatch-noisepart)))
                    max_symbol_residual=max(max_symbol_residual,residual)
                    energies=[float(np.sum(abs(mismatch)**2)),float(np.sum(abs(noisepart)**2)),
                        float(2*np.vdot(mismatch,noisepart).real),float(np.sum(abs(error)**2))]
                    relative=abs(sum(energies[:3])-energies[3])/max(energies[3],1e-300)
                    max_relative=max(max_relative,relative)
                    assert relative<1e-10
                    row=sums[m]
                    for key,value in zip(('mismatch','noise','cross','error'),energies):row[key]+=value
                    row['energy']+=float(np.sum(abs(xx)**2));row['eq'].extend(enh.tolist());row['sigma'].extend(mn.tolist())
                    errors=((detected.real<0)!=(xx.real<0)) | ((detected.imag<0)!=(xx.imag<0))
                    row['ser']+=int(np.count_nonzero(errors));row['bits']+=int(np.count_nonzero((detected.real<0)!=(xx.real<0))+np.count_nonzero((detected.imag<0)!=(xx.imag<0)))
            for m,row in sums.items():
                fixed.append(dict(trajectory_id=i,snr_db=s,method=m,BER=row['bits']/23808,SER=row['ser']/11904,
                    sigma_min_median=float(np.median(row['sigma'])),sigma_min_P95=float(np.quantile(row['sigma'],.95)),
                    equalizer_squared_Frobenius_median=float(np.median(row['eq'])),equalizer_squared_Frobenius_P95=float(np.quantile(row['eq'],.95)),
                    mismatch_energy=row['mismatch'],noise_energy=row['noise'],cross_term=row['cross'],symbol_error_energy=row['error'],
                    mismatch_per_symbol=row['mismatch']/11904,noise_per_symbol=row['noise']/11904,cross_per_symbol=row['cross']/11904,
                    symbol_MSE=row['error']/11904,transmitted_energy=row['energy']))
    allvalues=np.concatenate(allvalues)
    quantiles={name:{label:float(np.quantile(allvalues[:,k],q)) for q,label in ((0,'min'),(.05,'P05'),(.5,'median'),(.95,'P95'),(.99,'P99'),(1,'max'))}
        for k,name in enumerate(('sigma_max','sigma_min','condition','log10_condition','noise_enhancement'))}
    c.csv_write(c.OUT/'channel_trajectories.csv',condition);c.csv_write(c.OUT/'channel_frames.csv',frame_summary);c.csv_write(c.OUT/'fixed_ID_diagnostics.csv',fixed)
    c.write(c.OUT/'conditioning_summary.json',dict(PASS=True,REs=len(allvalues),nonfinite_values=nonfinite,singular_REs=singular,
        quantiles=quantiles,condition_gt10=float(np.mean(allvalues[:,2]>10)),condition_gt100=float(np.mean(allvalues[:,2]>100)),
        formula='noise enhancement = ||solve(H,I)||_F^2; same square solve, no pinv or clipping',
        mismatch_identity_relative_energy_residual_max=max_relative,mismatch_identity_symbol_residual_max=max_symbol_residual,
        fixed_IDs=c.IDS,counters=c.COUNTERS.copy()))
    print('D conditioning PASS',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('prepare','replay','diagnose','verify'));args=p.parse_args()
    try:
        if args.phase=='prepare':prepare()
        elif args.phase=='replay':replay()
        elif args.phase=='diagnose':diagnose()
        else:
            c.install_guard();c.write(c.OUT/'preservation.json',c.verify_unchanged());print('preservation PASS',flush=True)
    except BaseException as exc:
        if c.OUT.exists():c.write(c.OUT/'STOP.json',dict(phase=args.phase,error=repr(exc),receiver_case='C' if args.phase=='replay' else 'UNDECIDED',
            action='STOP and report; no receiver change or rescue'))
        raise
