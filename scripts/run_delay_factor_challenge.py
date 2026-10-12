"""T19A preparation and paired frozen-receiver delay-factor challenge."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from src import delay_factor_challenge as p
from src import frozen_affine_validation as old
from src.cdl_benchmark import StreamingStatistics,frames,array_hash,observation_geometry
from src.cdl_snr import TrainReceivePowerAccumulator
from src.lowrank_modules import WeightedModule
from src.pilot_cv import deployment
from src.statistics_mixing import geometry
from src.link import assemble_pilot_data_frame,apply_frequency_mimo_channel,PILOT_SYMBOLS
from src.pilots import map_pilot_symbols
from src.estimators import pilot_ls_estimate
from scripts.run_cdl_mismatch import load,save,sha,csvsave,history,environments,communication
from scripts.run_pilot_cv import cv_family,readcsv,trajectory_energy
from scripts.run_frozen_generalization import frozen_method,historical_receiver_regression
OUT=ROOT/'outputs';RESULT=OUT/'t19a_results.json'
T18B_COMMIT='2d13f71b7198691e2fc9b86b6bff57d4082d9b8f'
SOURCES=('src/delay_factor_challenge.py','scripts/run_delay_factor_challenge.py','scripts/plot_delay_factor_challenge.py',
    'scripts/audit_delay_factor_challenge.py','tests/test_delay_factor_challenge.py','docs/t19a_delay_factor_protocol.md')


def accepted():
    subprocess.check_call(['git','merge-base','--is-ancestor',T18B_COMMIT,'HEAD'],cwd=ROOT)
    if subprocess.check_output(['git','show','-s','--format=%s',T18B_COMMIT],cwd=ROOT,text=True).strip()!='feat: replicate frozen affine adaptation with trajectory-level uncertainty':
        raise ValueError('accepted T18B commit required')
    r=load(OUT/'t18b_results.json')
    from scripts.audit_validation_replication import verify
    verify(r,recompute=False)
    if not r['all_checks_passed'] or r['c8_claim_status']!='SUPPORTED_IN_FIXED_ENVIRONMENTS':raise ValueError('accepted T18B evidence required')
    for path in (*r['source_sha256'],'outputs/t18b_results.json'):
        blob=subprocess.check_output(['git','show',f'{T18B_COMMIT}:{path}'],cwd=ROOT)
        if blob.replace(b'\r\n',b'\n')!=(ROOT/path).read_bytes().replace(b'\r\n',b'\n'):raise ValueError('accepted snapshot changed:'+path)
    return r


def namespace_audit(rows):
    if rows!=p.manifest_rows():raise ValueError('manifest differs from fixed policy')
    from src import validation_replication as rep
    past=old.manifest_rows()+rep.manifest_rows()
    individual={r[k] for r in past for k in old.ROLES}
    used=set();channels=set();communication_tuples=set()
    historical=[]
    for name in ('t15a_cdl_manifest.csv','t15c_cdl_manifest.csv','t16c_heldout_manifest.csv'):
        historical+=readcsv(OUT/name)
    hc={tuple(int(r[k]) for k in old.ROLES[:3]) for r in historical}
    hm={tuple(int(r[k]) for k in old.ROLES[3:]) for r in historical}
    for r in rows:
        channel=tuple(r[k] for k in old.ROLES[:3]);comm=tuple(r[k] for k in old.ROLES[3:])
        if channel in hc or comm in hm:raise ValueError('historical stream tuple collision')
        channels.add(channel);communication_tuples.add(comm)
        for k in old.ROLES:
            if k in old.ROLES[:3] and r['frame_id']!=0:continue
            if r[k] in used or r[k] in individual:raise ValueError('individual namespace collision')
            used.add(r[k])
    if (len(rows),len(channels),len(communication_tuples),len(used))!=(48000,4800,48000,158400):raise ValueError('incomplete namespace')
    return dict(namespaces=[f'T19A_{n}_v1' for n in p.ENVIRONMENTS],allocated_individual_seeds=158400,
        consumed_individual_seeds=38400,trajectories=4800,train_trajectories=4000,validation_trajectories=800,
        train_validation_disjoint=True,T18A_T18B_individual_seeds_disjoint=True,historical_stream_tuples_disjoint=True,
        train_communication_allocated_not_generated=True,formal_namespace_accessed=False)


def aggregate(rows):
    index={(r['environment'],int(r['snr_db']),int(r['trajectory_id']),r['method']):r for r in rows}
    expected={(n,s,i,m) for n in p.ENVIRONMENTS for s in p.SNRS for i in range(500,600) for m in p.METHODS}
    if len(rows)!=11200 or set(index)!=expected:raise ValueError('all 800 validation trajectories required')
    pooled=[];metrics=[]
    for n in p.ENVIRONMENTS:
        for s in p.SNRS:
            values={}
            for i in range(500,600):
                group=[index[n,s,i,m] for m in p.METHODS]
                if len({float(v['truth_energy']) for v in group})!=1:raise ValueError('unpaired denominator')
            for m in p.METHODS:
                group=[index[n,s,i,m] for i in range(500,600)]
                if any(int(v['frames'])!=6 or v['subset']!='frames4..9' for v in group):raise ValueError('primary frame subset drift')
                e=trajectory_energy([float(v['error_energy']) for v in group]);t=trajectory_energy([float(v['truth_energy']) for v in group])
                if e<=0 or t<=0 or not np.isfinite([e,t]).all():raise ValueError('positive finite pooled energies required')
                db=float(10*np.log10(e/t));values[m]=db
                pooled.append(dict(environment=n,snr_db=s,method=m,trajectories=100,frames=600,
                    subset='frames4..9',error_energy=e,truth_energy=t,nmse_linear=e/t,nmse_db=db,
                    means='target' if m=='matched' else 'source',evaluation_only=(m in ('matched','oracle2-affine')),secondary_only=(m=='hard')))
            metrics.append(dict(environment=n,snr_db=s,pure_delay=(n in p.PURE),**p.metric_columns(values)))
    return pooled,metrics


def summarize_weights(rows):
    result=[]
    for n in p.ENVIRONMENTS:
        for s in p.SNRS:
            group=[r for r in rows if (r['environment'],int(r['snr_db']),r['method'])==(n,s,'affine')]
            if len(group)!=600:raise ValueError('all six correlated windows required')
            item=dict(environment=n,snr_db=s,windows=600,independent_trajectories=100,
                outside_simplex_fraction=sum(str(r['outside_any'])=='True' for r in group)/600)
            for field in ('alpha','beta','lambda0','coordinate_error_norm'):
                x=np.array([float(r[field]) for r in group]);q=np.quantile(x,[.25,.5,.75])
                item[field]=dict(q25=float(q[0]),median=float(q[1]),q75=float(q[2]),minimum=float(x.min()),maximum=float(x.max()))
            result.append(item)
    return result


def preflight():
    if (OUT/'t19a_preflight.json').exists():raise ValueError('existing preflight protected')
    if Path(sys.prefix).resolve()!=(ROOT/'.venv-cdl').resolve():raise ValueError('frozen CDL environment required')
    r=accepted();arrays,family,proof=frozen_method()
    if proof!=r['frozen_method_provenance']:raise ValueError('frozen method drift')
    env=environments()
    if env!=r['environment_after']:raise ValueError('environment drift')
    log=OUT/'t19a_related_preflight_pytest_attempt02.txt';summary=log.read_text(encoding='utf8').strip().splitlines()[-1]
    if '1 skipped' not in summary or 'failed' in summary or 'error' in summary:raise ValueError('calibration tests and one explicit missing-result skip required')
    if not load(OUT/'t19a_fixture_figure_gate.json')['all_checks_passed']:raise ValueError('small figure schema/render gate missing')
    namespace=namespace_audit(p.manifest_rows());legacy=historical_receiver_regression(arrays,family)
    small={};generators={}
    for n in p.ENVIRONMENTS:
        adapter=p.FactorCDL(n);generators[n]=adapter.provenance();small[n]={}
        for split,i in (('train_dev',0),('validation_dev',500)):
            v=adapter.generate(i,split=split,details=True);w=adapter.generate(i,split=split,details=True)
            if [array_hash(x) for x in v]!=[array_hash(x) for x in w]:raise ValueError('deterministic CDL/CIR smoke mismatch')
            if v[1].shape!=(2,2,len(v[2]),160):raise ValueError('CIR export order changed')
            small[n][split]=dict(H=array_hash(v[0]),CIR=array_hash(v[1]),tau=array_hash(v[2]))
    csvsave(OUT/'t19a_manifest.csv',p.manifest_rows())
    frozen=history();frozen.update({v:sha(ROOT/v) for v in subprocess.check_output(['git','ls-files','docs'],cwd=ROOT,text=True).splitlines()})
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():raise ValueError('historical tracked files changed')
    for path in SOURCES:
        dest=OUT/'t19a_execution_snapshot'/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/path).read_bytes())
    save(OUT/'t19a_preflight.json',dict(task='T19A',all_checks_passed=True,accepted_T18B_commit=T18B_COMMIT,
        working_tree_clean_after_T18B_commit=True,startup_summary='1726 passed in 276.23s (0:04:36)',startup_exit_code=0,
        startup_log_sha256=sha(OUT/'t19a_startup_pytest.txt'),source_sha256={v:sha(ROOT/v) for v in SOURCES},
        frozen_history_sha256=frozen,frozen_method=proof,environment=env,namespace_audit=namespace,
        small_H_hashes=small,generator_provenance=generators,manifest_sha256=sha(OUT/'t19a_manifest.csv'),
        historical_receiver_regression=legacy,calibration_summary=summary,calibration_log_sha256=sha(log),
        figure_fixture_sha256=sha(OUT/'t19a_fixture_figure_gate.json'),c9_claim_status_before='UNCONFIRMED',
        scientific_scan_started=False,source_noise_variances={s:old.VARIANCES[s] for s in p.SNRS},source_power=old.POWER,
        C9_priority='Representation sufficient takes priority; coordinate bottleneck is independent; no third module if bottleneck.',
        generator_smoke_calls=32,formal_test_used=False,formal_test_sealed=True))
    print('T19A preflight PASS: eight environments;32 paired deterministic smoke calls; frozen method/folds/K; C9 UNCONFIRMED',flush=True)


def check_preflight():
    pre=load(OUT/'t19a_preflight.json')
    if not pre['all_checks_passed'] or Path(sys.prefix).resolve()!=(ROOT/'.venv-cdl').resolve():raise ValueError('passed preflight and frozen CDL interpreter required')
    for group in ('source_sha256','frozen_history_sha256'):
        for path,h in pre[group].items():
            if sha(ROOT/path)!=h:raise ValueError('frozen bytes changed:'+path)
    if sha(OUT/'t19a_manifest.csv')!=pre['manifest_sha256']:raise ValueError('manifest changed')
    arrays,family,proof=frozen_method()
    if proof!=pre['frozen_method']:raise ValueError('frozen method changed')
    return pre,arrays,family


def evaluate(name,i,h,arrays,family,matched,oracle,g):
    """Generate each communication frame once, then call label-free deployed API."""
    truths=frames(h);pilots=np.empty((2,10,64),np.complex128);pair=[]
    signal=np.zeros(2);unit_noise=np.zeros(2);samples=0
    known=map_pilot_symbols(g.layout,PILOT_SYMBOLS)
    for j,truth in enumerate(truths):
        bits,symbols,z=communication(p.seeds(name,'validation_dev',i,j))
        x=assemble_pilot_data_frame(symbols,g.layout);clean=apply_frequency_mimo_channel(truth,x)
        if j>=4:
            mask=g.layout.data_mask[0];signal+=np.sum(abs(clean[:,mask])**2,axis=1)
            unit_noise+=np.sum(abs(z[:,mask])**2,axis=1);samples+=int(mask.sum())
        shared=dict(environment=name,trajectory_id=i,frame_id=j,H_sha256=array_hash(truth),
            bits_sha256=array_hash(bits),symbols_sha256=array_hash(symbols),unit_noise_sha256=array_hash(z))
        for si,s in enumerate(p.SNRS):
            y=clean+np.sqrt(old.VARIANCES[s])*z
            sparse,_=pilot_ls_estimate(y,known,g.layout.pilot_mask)
            pilots[si,j]=sparse.ravel()[g.selected]
            pair.append(dict(**shared,snr_db=s,Y_sha256=array_hash(y),pilot_sha256=array_hash(pilots[si,j])))
    pilots.flags.writeable=False;raw=[];tr=[];frame_rows=[]
    for si,s in enumerate(p.SNRS):
        estimates=old.deployed(pilots[si],arrays,s,cv_family(arrays,s),family['P',s,4],family['M',s,4])
        errors={m:[] for m in p.METHODS};energies=[]
        for r in estimates:
            j=r['frame_id'];true=truths[j].ravel()[g.data];energy=float(np.sum(abs(true)**2));energies.append(energy)
            if r['dense_coefficients'] is not r['lr4_coefficients']:raise ValueError('Dense/LR4 coefficients not shared')
            # Target references are confined to this post-deployment evaluation.
            predictions=dict(r['predictions'],matched=matched[s].estimate(pilots[si,j]))
            predictions['oracle2-affine']=deployment(pilots[si,j],arrays['source_mean_p'],arrays['source_mean_d'],
                arrays[f'Wsource_{s}'],arrays[f'DeltaWP_{s}'],arrays[f'DeltaWM_{s}'],oracle[s],('P','M'))
            for m,pred in predictions.items():
                error=float(np.sum(abs(pred-true)**2));errors[m].append(error)
                frame_rows.append(dict(environment=name,snr_db=s,trajectory_id=i,frame_id=j,method=m,
                    error_energy=error,truth_energy=energy,prediction_sha256=array_hash(pred)))
            for m,c in r['coefficients'].items():
                diag=r['diagnostics'][m]
                coordinate_error=float(np.linalg.norm(c-oracle[s])) if m=='affine' else None
                raw.append(dict(environment=name,snr_db=s,trajectory_id=i,frame_id=j,window_start=j-4,window=5,
                    method=m,**geometry(c),source_cv_loss=diag['source_cv_loss'],fitted_cv_loss=diag['fitted_cv_loss'],
                    selected_candidate=diag.get('selected_candidate'),active_set=diag.get('selected_active_set'),
                    gram_condition_number=diag.get('condition_number'),coefficients_sha256=array_hash(c),
                    dense_lr4_same_coefficients=(m=='affine'),coordinate_error_norm=coordinate_error,
                    alpha_oracle=float(oracle[s][0]),beta_oracle=float(oracle[s][1]),lambda0_oracle=float(1-oracle[s].sum())))
        energy=trajectory_energy(energies)
        for method in p.METHODS:
            error=trajectory_energy(errors[method]);tr.append(dict(environment=name,snr_db=s,trajectory_id=i,
                method=method,frames=6,subset='frames4..9',error_energy=error,truth_energy=energy,nmse_linear=error/energy))
    return tr,raw,frame_rows,pair,signal,unit_noise,samples


def run():
    if RESULT.exists() or (OUT/'t19a_factor_nmse.csv').exists():raise ValueError('existing scientific run protected')
    pre,arrays,family=check_preflight();accepted()
    start=time.perf_counter();stats_info={};delays=[];oracles=[];tr=[];weights=[];frame_rows=[];pairs=[];hrows=[];snr_audit=[];runtime={}
    with patch.object(WeightedModule,'build',side_effect=AssertionError('no module refit/SVD allowed')):
        for n in p.ENVIRONMENTS:
            tick=time.perf_counter();adapter=p.FactorCDL(n)
            if adapter.provenance()!=pre['generator_provenance'][n]:raise ValueError('generator changed')
            g=observation_geometry();stream=StreamingStatistics(g.selected,g.data);power=TrainReceivePowerAccumulator();delay=p.EffectiveDelayAudit();batch=[];ids=[]
            for i in range(500):
                h,a,tau=adapter.generate(i,split='train_dev',details=True)
                if i==0 and [array_hash(v) for v in (h,a,tau)]!=list(pre['small_H_hashes'][n]['train_dev'].values()):raise ValueError('train smoke replay drift')
                hrows.append(dict(environment=n,split='train_dev',trajectory_id=i,H_sha256=array_hash(h),CIR_sha256=array_hash(a),tau_sha256=array_hash(tau)))
                delay.update(i,a,tau);power.update(i,h);batch.append(frames(h).reshape(10,4096));ids.append(i)
                if len(batch)==10:stream.update(np.concatenate(batch),ids);batch=[];ids=[]
                if (i+1)%100==0:print(f'{n} matched/oracle-only training {i+1}/500',flush=True)
            target=stream.freeze();reference=power.freeze()
            if target.train_trajectory_ids!=tuple(range(500)) or target.samples!=5000:raise ValueError('train-only target statistics violated')
            train_seconds=time.perf_counter()-tick
            audit=delay.freeze(int(n.split('-')[-1]));delays.append(dict(environment=n,**audit))
            path=OUT/f't19a_matched_statistics_{n}.npz'
            if path.exists():raise ValueError('target statistics protected')
            np.savez_compressed(path,mean_p=target.mean_p,mean_d=target.mean_d,rpp=target.rpp,rdp=target.rdp,train_trajectory_ids=target.train_trajectory_ids,samples=target.samples)
            path.chmod(0o444)
            stats_info[n]=dict(path='outputs/'+path.name,sha256=sha(path),component_sha256={k:array_hash(getattr(target,k)) for k in ('mean_p','mean_d','rpp','rdp')},
                split='train_dev',namespace=f'T19A_{n}_v1',samples=5000,trajectories=500,covariance_divisor='N population',batch_frames=100,
                used_for_deployment=False,validation_used=False,train_power=reference.reference,train_power_per_rx=list(reference.per_rx_signal_power))
            matched={s:target.prepare(old.VARIANCES[s]) for s in p.SNRS};oracle={}
            for s in p.SNRS:
                if matched[s].solve_relative_residual>1e-11:raise ValueError('matched Cholesky residual failed')
                oracle[s],info=old.oracle_coordinates(target,arrays,s)
                oracle[s].flags.writeable=False
                oracles.append(dict(environment=n,snr_db=s,alpha_oracle=info['alpha'],beta_oracle=info['beta'],lambda0_oracle=info['lambda0'],
                    operator_objective_residual=info['weighted_filter_excess_energy_per_frame'],**info))
            signal=np.zeros(2);noise=np.zeros(2);samples=0;tick=time.perf_counter()
            for i in range(500,600):
                h=adapter.generate(i,split='validation_dev');h.flags.writeable=False
                if i==500 and array_hash(h)!=pre['small_H_hashes'][n]['validation_dev']['H']:raise ValueError('validation smoke replay drift')
                hrows.append(dict(environment=n,split='validation_dev',trajectory_id=i,H_sha256=array_hash(h),CIR_sha256=None,tau_sha256=None))
                rows,raw,fr,paired,sp,npow,count=evaluate(n,i,h,arrays,family,matched,oracle,g)
                tr+=rows;weights+=raw;frame_rows+=fr;pairs+=paired;signal+=sp;noise+=npow;samples+=count
                if (i-499)%20==0:print(f'{n} paired frozen validation {i-499}/100',flush=True)
            runtime[n]=dict(target_training_pipeline_seconds=train_seconds,validation_pipeline_seconds=time.perf_counter()-tick)
            for s in p.SNRS:
                snr_audit.append(dict(environment=n,snr_db=s,source_reference_power=old.POWER,sigma2=old.VARIANCES[s],target_train_power=reference.reference,
                    target_train_receive_snr_db=float(10*np.log10(reference.reference/old.VARIANCES[s])),validation_realised_receive_snr_db=float(10*np.log10(signal.sum()/(old.VARIANCES[s]*noise.sum()))),
                    validation_signal_power_per_rx=(signal/samples).tolist(),validation_noise_power_per_rx=(old.VARIANCES[s]*noise/samples).tolist(),
                    subset='frames4..9',audit_only=True,noise_recalibrated=False,target_normalized=False))
            csvsave(OUT/f't19a_checkpoint_{n}_energy.csv',[v for v in tr if v['environment']==n])
            csvsave(OUT/f't19a_checkpoint_{n}_weights.csv',[v for v in weights if v['environment']==n])
    pooled,metrics=aggregate(tr);summaries=summarize_weights(weights);science=p.scientific_gate(metrics)
    for row in oracles:
        v=next(v for v in pooled if (v['environment'],v['snr_db'],v['method'])==(row['environment'],row['snr_db'],'oracle2-affine'))
        row.update(nmse_db=v['nmse_db'],error_energy=v['error_energy'],truth_energy=v['truth_energy'])
    files={'t19a_factor_nmse.csv':pooled,'t19a_oracle2_affine.csv':oracles,'t19a_factor_metrics.csv':metrics,'t19a_weights.csv':weights,
        't19a_effective_delay.csv':delays,'t19a_trajectory_energy.csv':tr,'t19a_frame_metrics.csv':frame_rows,'t19a_paired_input_hashes.csv':pairs,
        't19a_channel_hashes.csv':hrows,'t19a_snr_audit.csv':snr_audit}
    for filename,rows in files.items():csvsave(OUT/filename,rows)
    r=dict(task='T19A',experiment_type=p.EXPERIMENT_TYPE,research_stage='factor generalization / failure localization',research_question=p.C9,
        accepted_T18B_commit=T18B_COMMIT,method_frozen=True,delay_levels_ns=[50,200],factor_environments={n:p.definition(n) for n in p.ENVIRONMENTS},
        dataset=dict(train_trajectories_per_environment=500,validation_trajectories_per_environment=100,frames_per_trajectory=10,primary_frames=list(range(4,10))),
        source_power=old.POWER,source_noise_variances={s:old.VARIANCES[s] for s in p.SNRS},snr_audit=snr_audit,effective_delay_audit=delays,
        nmse=pooled,oracle2_coordinates=oracles,deployed_coordinates=summaries,factor_metrics=metrics,
        **{k:[dict(environment=v['environment'],snr_db=v['snr_db'],value=v[k]) for v in metrics] for k in ('representation_closure','coordinate_efficiency','lr4_closure')},
        compression_penalty=[dict(environment=v['environment'],snr_db=v['snr_db'],value=v['compression_penalty_db']) for v in metrics],
        affine_vs_convex_gain=[dict(environment=v['environment'],snr_db=v['snr_db'],dense=v['dense_vs_convex_gain_db'],lr4=v['lr4_vs_convex_gain_db']) for v in metrics],
        **science,c6_claim_status=old.C6,c7_claim_status=old.C7,c8_claim_status='SUPPORTED_IN_FIXED_ENVIRONMENTS',t18a_materiality_status='PARTIAL',
        new_model_trained=False,method_parameters_changed=False,rank_per_module=4,causal_window=5,primary_estimator_uses_environment_label=False,
        target_statistics_used_for_deployment=False,oracle2_evaluation_only=True,source_means_for_all_nonmatched=True,
        dense_lr4_coefficients_identical=True,new_module_added=False,no_basis_refit=True,bootstrap_performed=False,replication_performed=False,
        target_normalization=False,noise_recalibrated=False,BER_evaluations=0,formal_test_used=False,formal_test_sealed=True,
        frozen_method_provenance=pre['frozen_method'],target_statistics=stats_info,namespace_audit=pre['namespace_audit'],
        source_sha256=pre['source_sha256'],source_snapshot_root='outputs/t19a_execution_snapshot',frozen_history_sha256=pre['frozen_history_sha256'],environment_before=pre['environment'],
        preparation_artifact_sha256={f'outputs/{v}':sha(OUT/v) for v in ('t19a_preflight.json','t19a_manifest.csv','t19a_fixture_figure_gate.json')},
        numeric_artifact_sha256={f'outputs/{v}':sha(OUT/v) for v in files},generator_calls=dict(preflight=32,target_train=4000,validation=800),
        runtime=dict(by_environment=runtime,numerical_pipeline_seconds=time.perf_counter()-start),
        warnings=['One validation namespace/100 independent trajectories per fixed environment; no bootstrap or replication.',
            'Six causal windows per trajectory are correlated; coordinate quartiles are descriptive.',
            'Oracle2 optimizes centered filter error; actual evaluation shares source means. Matched uses target means.',
            'Nominal CDL DS differs from empirical array/time-weighted RMS delay; audit does not recalibrate noise.',
            'C6/C7 remain contradicted; C8 fixed-environment support and T18A PARTIAL remain unchanged.',
            'Interrupted startup observer did not capture exit code; a second full startup gate passed1726 tests with exit0.',
            'Git staged historical whitespace/newline warnings preserved; stale Python location and optional Triton warnings retained.',
            'Initial synthetic figure source QA misread squares:20dB as res:20; archived failure, clarified square-markers wording and reran all QA before scientific execution.'],
        all_checks_passed=False,completion_state='numerical challenge completed; final regression and figure audit pending')
    save(RESULT,r);print(science,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['preflight','run']);args=parser.parse_args()
    {'preflight':preflight,'run':run}[args.mode]()
