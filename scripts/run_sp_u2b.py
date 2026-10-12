"""New CDL development diagnostic; never invoke historical experiment runners."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import sp_u2b_diagnostic as d
d.install_guard()
import numpy as np
from src.pilot_cv import causal_windows, deployment
from src.lowrank_modules import factored_affine_predict
from src.cdl_benchmark import OfficialCDL, ARRAY, SYMBOL_SECONDS, frames, SYSTEM
from src.spatial_information import observation_geometry

OUT = d.OUT
PROTOCOL = ROOT/'configs/sp_u2b_protocol.json'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def protect():
    if git('rev-parse', 'HEAD') != d.HEAD or git('diff', '--name-only') or git('diff', '--cached', '--name-only'):
        raise RuntimeError('frozen HEAD/tree differs')


def snapshot():
    protect()
    # Exclude all saved outcomes, including hashes. Protect source, paper and prior task documents.
    paths = set(git('ls-files').splitlines()) | set(git('ls-files','--others','--exclude-standard').splitlines())
    protected = {p:d.sha(ROOT/p) for p in sorted(paths)
                 if p.startswith(('src/','scripts/','tests/','configs/','docs/','paper/'))
                 and 'sp_u2b' not in p and '__pycache__' not in p and (ROOT/p).is_file()}
    sources = {str(p.relative_to(ROOT)).replace('\\','/'):d.sha(p)
               for folder in ('scripts','tests','configs') for p in (ROOT/folder).glob('*sp_u2b*') if p.is_file()}
    config = d.load(PROTOCOL)
    c = config['condition']
    assert c['sample_rate_hz'] == c['nfft']*c['subcarrier_spacing_hz'] == SYSTEM['sample_rate_hz']
    arrays, cvs, factors, proof = d.frozen_method()
    manifest = []
    unique = []
    for stage, count in (('smoke',20),('development',100)):
        for i in range(count):
            for j in range(10):
                seed = d.seeds(stage,i,j)
                manifest.append(dict(namespace='SP_U2B_UPGRADE_DEV_v1_'+stage, split='upgrade-development',
                                     stage=stage, trajectory_id=i, frame_id=j, start_symbol=16*j,
                                     channel_seed_scope='trajectory', communication_seed_scope='frame', **seed))
                unique.extend(seed[k] for k in seed if j == 0 or k not in ('sionna_seed','torch_seed','numpy_seed'))
    assert len(set(unique)) == len(unique) == 3960
    d.csvsave(OUT/'sp_u2b_seed_manifest.csv', manifest)
    d.save(OUT/'sp_u2b_preregistration.json', dict(task='SP-U2B', head=d.HEAD,
           frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), hypotheses_and_protocol=config,
           protocol_sha256=d.sha(PROTOCOL), protected_sha256=protected, execution_source_sha256=sources,
           frozen_method=proof, manifest_sha256=d.sha(OUT/'sp_u2b_seed_manifest.csv'),
           seed_unique_role_values=len(unique), historical_seed_manifest_read=False,
           historical_numeric_seed_disjointness_claimed=False, new_namespace=True,
           old_outcomes_read_or_hashed=False, formal_outcome_access_blocked=True,
           source_reference_power_frozen=True, generator_calls=0, access=d.ACCESS))
    print('Preregistered before any CDL generation; protected',len(protected),'files; unique consumed role seeds',len(unique),flush=True)


def verify():
    protect()
    pre=d.load(OUT/'sp_u2b_preregistration.json')
    for p,h in {**pre['protected_sha256'],**pre['execution_source_sha256']}.items():
        if d.sha(ROOT/p)!=h:
            raise RuntimeError('protected source drift: '+p)
    assert d.sha(PROTOCOL)==pre['protocol_sha256']
    assert d.sha(OUT/'sp_u2b_seed_manifest.csv')==pre['manifest_sha256']
    return pre


class UpgradeCDL(OfficialCDL):
    def __init__(self):
        super().__init__()
        from sionna.phy.channel.tr38901 import CDL
        self.model=CDL(model='B',delay_spread=100e-9,carrier_frequency=3.5e9,
                       ut_array=self.array,bs_array=self.array,direction='downlink',
                       ut_orientation=ARRAY['ut_orientation_radians'],bs_orientation=ARRAY['bs_orientation_radians'],
                       ut_velocity=[120/3.6,0.,0.],min_speed=120/3.6,max_speed=120/3.6,
                       normalize_delays=True,precision='double',device=self.device,spec_version='19.2')

    def generate_upgrade(self, stage, trajectory):
        import torch
        from sionna.phy import config
        from sionna.phy.channel import cir_to_ofdm_channel
        seed=d.seeds(stage,trajectory)
        config.seed=seed['sionna_seed']
        torch.manual_seed(seed['torch_seed'])
        self.numpy_rng=np.random.default_rng(seed['numpy_seed'])
        with torch.no_grad():
            a,tau=self.model(batch_size=1,num_time_steps=160,sampling_frequency=1/SYMBOL_SECONDS)
            hf=cir_to_ofdm_channel(self.frequencies,a,tau,normalize=False)
        h=hf[0,0,:,0,:,:,:].permute(0,1,3,2).cpu().numpy().astype(np.complex128,copy=True)
        assert h.shape==(2,2,64,160) and np.isfinite(h).all()
        h.flags.writeable=False
        return h


def diagnose(pilots, h, arrays, cvs, factors, stage, trajectory):
    g=observation_geometry()
    rows, design_cache=[],{}
    truth_frames=frames(h)
    for si,s in enumerate(d.SNRS,1):
        cv=cvs[s]
        for j,window in causal_windows(pilots[si],5):
            clean_window=pilots[0,j-4:j+1]
            c0,diag0=cv.fit(clean_window,modules=('P','M'))
            c,diag=cv.fit(window,modules=('P','M'))
            G0,z0=d.explicit_design(cv,clean_window)
            G,z=d.explicit_design(cv,window)
            geo0,geo=d.geometry(G0),d.geometry(G)
            errors=[d.close(G0.T@G0,diag0['gram']),d.close(G0.T@z0,diag0['rhs']),
                    d.close(G.T@G,diag['gram']),d.close(G.T@z,diag['rhs']),
                    d.close(np.linalg.solve(G.T@G,G.T@z),c)]
            joint=d.joint_diagnostics(G0,z0,G,z,c0,c)
            pilot=window[-1]
            u=pilot-arrays['source_mean_p']
            v=pilots[0,j]-arrays['source_mean_p']
            n=pilot-pilots[0,j]
            true=truth_frames[j].ravel()[g.data]
            ws=arrays[f'Wsource_{s}']
            args=(pilot,arrays['source_mean_p'],arrays['source_mean_d'],ws)
            fp,fm=factors['P',s],factors['M',s]
            X=np.column_stack([fp.apply(u),fm.apply(u)])
            base=arrays['source_mean_d']+u@ws.T
            oracle,M,oracle_error=d.channel_oracle(true,base,X)
            # Production calls before truth-assisted diagnostic oracle is used for evaluation.
            pred=factored_affine_predict(*args,c,fp,fm)
            pred0=factored_affine_predict(*args,c0,fp,fm)
            predoracle=factored_affine_predict(*args,oracle,fp,fm)
            dc=c-c0
            measured=pred-pred0
            propagated=X@dc
            energy=float(np.vdot(measured,measured).real)
            quadratic=float(dc@M@dc)
            prop_error=d.close(measured,propagated)
            pure_relative=float(np.linalg.norm(measured-propagated)/max(np.linalg.norm(measured),np.finfo(float).tiny))
            d.close(energy,quadratic)
            nd=len(true)
            mse=lambda p:float(np.vdot(p-true,p-true).real/nd)
            R,R0,Ro=mse(pred),mse(pred0),mse(predoracle)
            penalties=(R-R0,R0-Ro,R-Ro)
            d.close(sum(penalties[:2]),penalties[2])
            # Oracle output quadratic identity, with the tiny numerical normal-equation cross residual retained.
            oracle_quad=float((c-oracle)@M@(c-oracle)/nd)
            d.close(penalties[2],oracle_quad)
            noise_cross=float(2*np.vdot(pred0-true,propagated).real/nd)
            d.close(penalties[0],quadratic/nd+noise_cross)
            dp,dm=arrays[f'DeltaWP_{s}'],arrays[f'DeltaWM_{s}']
            dense=deployment(*args,dp,dm,c,('P','M'))
            compression_energy=float(np.vdot(pred-dense,pred-dense).real/nd)
            comp_penalty=R-mse(dense)
            # Explicit pure-noise four-vector bookkeeping, using current rank4 oracle as declared reference.
            rep=arrays['source_mean_d']+v@ws.T+oracle[0]*(dp@v)+oracle[1]*(dm@v)-true
            coord=(c[0]-oracle[0])*(dp@v)+(c[1]-oracle[1])*(dm@v)
            comp=c[0]*(fp.apply(v)-dp@v)+c[1]*(fm.apply(v)-dm@v)
            noise=ws@n+c[0]*fp.apply(n)+c[1]*fm.apply(n)
            vectors=[rep,coord,comp,noise]
            decomp_error=d.close(sum(vectors),pred-true)
            pure_energy=[float(np.vdot(a,a).real/nd) for a in vectors]
            cross=[float(2*np.vdot(vectors[a],vectors[b]).real/nd) for a in range(4) for b in range(a+1,4)]
            d.close(sum(pure_energy)+sum(cross),R)
            # A clean-input oracle is a secondary representation-floor diagnostic, not a deployed method.
            Xclean=np.column_stack([fp.apply(v),fm.apply(v)])
            baseclean=arrays['source_mean_d']+v@ws.T
            cleanoracle,_,_=d.channel_oracle(true,baseclean,Xclean)
            cleanfloor=mse(baseclean+Xclean@cleanoracle)
            # Dense/noisy oracle probes attainable compression loss within each actual operator class.
            Xdense=np.column_stack([dp@u,dm@u])
            denseoracle,_,_=d.channel_oracle(true,base,Xdense)
            densefloor=mse(base+Xdense@denseoracle)
            row=dict(stage=stage,trajectory_id=trajectory,frame_id=j,window_start=j-4,snr_db=s,
                     clean_flag=False,design_shape='640x2',truth_energy=float(np.vdot(true,true).real),ndata=nd)
            row.update({'clean_'+k:v for k,v in geo0.items()})
            row.update({'noisy_'+k:v for k,v in geo.items()})
            row.update(joint)
            for label,value in (('cleanCV',c0),('noisyCV',c),('channelOracle',oracle),
                                ('delta_noise',dc),('delta_surrogate',c0-oracle),('delta_total',c-oracle)):
                row[label+'_alpha'],row[label+'_beta']=map(float,value)
                row[label+'_norm']=float(np.linalg.norm(value))
            row.update(M_PP=float(M[0,0]),M_PM=float(M[0,1]),M_MM=float(M[1,1]),
                       output_lambda_min=float(np.linalg.eigvalsh(M)[0]),output_lambda_max=float(np.linalg.eigvalsh(M)[-1]),
                       coord_output_energy=energy,coord_output_quadratic=quadratic,coord_output_mse=quadratic/nd,
                       propagation_identity_error=prop_error,propagation_pure_relative_error=pure_relative,
                       normal_equation_identity_error_max=max(errors),oracle_LS_identity_error=oracle_error,
                       MSE_noisyCV=R,MSE_cleanCV=R0,MSE_channelOracle=Ro,
                       delta_R_noise=penalties[0],delta_R_surrogate=penalties[1],delta_R_total=penalties[2],
                       noise_penalty_cross_term=noise_cross,total_oracle_quadratic_mse=oracle_quad,
                       MSE_dense_same_coordinate=mse(dense),compression_output_mse=compression_energy,
                       signed_compression_MSE_penalty=comp_penalty,
                       MSE_clean_input_rank4_oracle=cleanfloor,MSE_noisy_input_dense_oracle=densefloor,
                       rank4_vs_dense_oracle_MSE=Ro-densefloor,
                       decomposition_identity_error=decomp_error,
                       representation_vector_mse=pure_energy[0],coordinate_vector_mse=pure_energy[1],
                       compression_vector_mse=pure_energy[2],observation_noise_vector_mse=pure_energy[3],
                       cross_rep_coord=cross[0],cross_rep_comp=cross[1],cross_rep_noise=cross[2],
                       cross_coord_comp=cross[3],cross_coord_noise=cross[4],cross_comp_noise=cross[5],
                       conditioning_improved=bool(geo['kappa_G']<geo0['kappa_G']),
                       minimum_singular_increased=bool(geo['sigma_min']>geo0['sigma_min']),
                       source_clean_MSE=mse(baseclean),source_noisy_MSE=mse(base),
                       current_clean_pilot_norm=float(np.linalg.norm(v)),
                       clean_channel_mean_power=float(np.mean(abs(true)**2)),
                       prediction_sha256=d.array_hash(pred),cleanCV_prediction_sha256=d.array_hash(pred0))
            numeric=[v for v in row.values() if isinstance(v,(int,float)) and not isinstance(v,bool)]
            assert np.isfinite(numeric).all(), 'nonfinite diagnostic; preserve failure, no dropping'
            assert geo0['rank_real']==geo['rank_real']==2, 'singular production solve; no fallback'
            assert not joint['bound_failure']
            rows.append(row)
            for label,value in (('G0',G0),('z0',z0),('G',G),('z',z),('c0',c0),('c',c),('oracle',oracle)):
                design_cache[f'{label}_{s}_{j}']=value
    return rows,design_cache


def run(stage):
    if Path(sys.prefix).resolve()!=(ROOT/'.venv-cdl').resolve():
        raise RuntimeError('use frozen .venv-cdl')
    pre=verify()
    if stage=='development':
        smoke=d.load(OUT/'sp_u2b_smoke_gate.json')
        assert smoke['all_checks_passed'] and smoke['trajectories']==20
    count=20 if stage=='smoke' else 100
    result_path=OUT/f'sp_u2b_{stage}_gate.json'
    if result_path.exists():
        raise RuntimeError('existing completed run protected')
    tests=d.load(OUT/'sp_u2b_tests_pass.json')
    assert tests['exit_code']==0 and tests['execution_source_sha256']==pre['execution_source_sha256']
    arrays,cvs,factors,proof=d.frozen_method()
    assert json.loads(json.dumps(proof))==pre['frozen_method']
    start=time.perf_counter()
    adapter=UpgradeCDL()
    provenance=adapter.provenance()
    from sionna.phy.channel.tr38901 import models
    provenance['source_hashes']={k:v for k,v in provenance['source_hashes'].items() if not k.endswith('CDL-C.json')}
    bpath=Path(models.parameter_file('CDL-B.json','19.2'))
    provenance['source_hashes'][str(bpath)]=d.sha(bpath)
    provenance.update(profile='CDL-B',speed_kmh=120,nominal_DS_seconds=1e-7)
    import sionna, torch, scipy, os
    environment=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,
                     torch=torch.__version__,sionna=getattr(sionna,'__version__','2.2.0'),device=adapter.device,
                     torch_threads=torch.get_num_threads(),
                     thread_environment={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')})
    allrows,pairs,power=[],[],[]
    cache_root=OUT/f'sp_u2b_{stage}_cache'
    cache_root.mkdir(exist_ok=False)
    replay_calls=0
    for i in range(count):
        h=adapter.generate_upgrade(stage,i)
        pilots,paired,powa=d.pair_channel(h,stage,i)
        if i==0:
            repeat=adapter.generate_upgrade(stage,i)
            replay_calls+=1
            assert np.array_equal(h,repeat), 'channel seed replay not exact'
            rep_pilots,rep_hashes,_=d.pair_channel(repeat,stage,i)
            assert np.array_equal(pilots,rep_pilots) and paired==rep_hashes
        rows,design=diagnose(pilots,h,arrays,cvs,factors,stage,i)
        path=cache_root/f'trajectory_{i:03d}.npz'
        assert not path.exists()
        np.savez_compressed(path,H=h,pilots=pilots,**design)
        d.csvsave(cache_root/f'trajectory_{i:03d}_windows.csv',rows)
        allrows.extend(rows)
        pairs.extend(paired)
        power.append(dict(stage=stage,trajectory_id=i,H_sha256=d.array_hash(h),cache_sha256=d.sha(path),**powa))
        print(f'{stage}: {i+1}/{count} trajectories; {len(allrows)} paired SNR-window rows; elapsed {time.perf_counter()-start:.1f}s',flush=True)
    d.csvsave(OUT/f'sp_u2b_{stage}_windows.csv',allrows)
    d.csvsave(OUT/f'sp_u2b_{stage}_pair_hashes.csv',pairs)
    verify()
    result=dict(task='SP-U2B',stage=stage,all_checks_passed=True,trajectories=count,frames=count*10,
                windows_per_snr=count*6,total_paired_snr_windows=count*12,
                unique_channel_generations=count,channel_replay_calls=replay_calls,
                smoke_not_scientific_evidence=(stage=='smoke'),statistical_unit='trajectory',
                rank_failures=0,NaN_count=0,bound_failures=0,pairing_verified=True,
                rows_path=str((OUT/f'sp_u2b_{stage}_windows.csv').relative_to(ROOT)),
                power_and_channel_hashes=power,generator=provenance,environment=environment,
                protocol_sha256=pre['protocol_sha256'],execution_source_sha256=pre['execution_source_sha256'],
                frozen_method=proof,wall_seconds=time.perf_counter()-start,access=d.ACCESS,
                production_changed=False,paper_changed=False,committed=False,formal_outcomes_accessed=False)
    d.save(result_path,result)
    print(stage,'complete; all numerical gates PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=('prepare','smoke','development','verify'))
    a=p.parse_args()
    try:
        if a.mode=='prepare': snapshot()
        elif a.mode=='verify': verify(); print('protected integrity PASS')
        else: run(a.mode)
    except Exception as error:
        stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        d.save(OUT/f'sp_u2b_failure_{a.mode}_{stamp}.json',dict(mode=a.mode,error=repr(error),traceback=traceback.format_exc(),access=d.ACCESS))
        raise
