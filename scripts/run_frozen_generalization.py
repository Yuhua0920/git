"""T18A six predeclared new namespaces; frozen receiver, train-only references."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from src import frozen_affine_validation as protocol
from src.cdl_benchmark import StreamingStatistics, frames, array_hash, observation_geometry
from src.cdl_snr import TrainReceivePowerAccumulator
from src.pilot_cv import frozen_folds, deployment
from src.statistics_mixing import geometry
from src.link import assemble_pilot_data_frame, apply_frequency_mimo_channel, PILOT_SYMBOLS
from src.pilots import map_pilot_symbols
from src.estimators import pilot_ls_estimate
from scripts.run_cdl_mismatch import load, save, sha, csvsave, history, environments, source_statistics, communication
from scripts.run_statistics_selection import frozen_candidates
from scripts.run_lowrank_modules import frozen_factors
from scripts.run_pilot_cv import cv_family, readcsv, trajectory_energy
OUT=ROOT/'outputs'; RESULT=OUT/'t18a_results.json'
T17B_COMMIT='ad29e0b97cc2f44610977868540d449bbbc850dc'
SOURCES=('src/frozen_affine_validation.py','scripts/run_frozen_generalization.py',
    'scripts/plot_frozen_generalization.py','scripts/audit_frozen_generalization.py',
    'tests/test_frozen_generalization.py','docs/t18a_generalization_protocol.md')


def accepted():
    subprocess.check_call(['git','merge-base','--is-ancestor',T17B_COMMIT,'HEAD'],cwd=ROOT)
    if subprocess.check_output(['git','show','-s','--format=%s',T17B_COMMIT],cwd=ROOT,text=True).strip()!='feat: identify cancellation-dominated exact affine interaction':
        raise ValueError('T17B accepted commit required')
    old=load(OUT/'t17b_results.json')
    if not old['all_checks_passed'] or old['completion_state']!='completed' or old['c6_claim_status']!=protocol.C6 or old['c7_claim_status']!=protocol.C7:
        raise ValueError('T17B completion and permanent negative claims required')
    for group in ('frozen_history_sha256','numeric_artifact_sha256','preparation_artifact_sha256'):
        for p,h in old[group].items():
            if sha(ROOT/p)!=h:raise ValueError('historical bytes changed:'+p)
    # Compare accepted text snapshots after declared Git newline conversion.
    for p in (*old['source_sha256'],'outputs/t17b_results.json'):
        blob=subprocess.check_output(['git','show',f'{T17B_COMMIT}:{p}'],cwd=ROOT)
        if blob.replace(b'\r\n',b'\n')!=(ROOT/p).read_bytes().replace(b'\r\n',b'\n'):
            raise ValueError('accepted historical snapshot drift:'+p)
    return old


def frozen_method():
    arrays,candidate=frozen_candidates();family,factors=frozen_factors()
    e=load(OUT/'t16e_pilot_cv_preflight.json')
    _,fold=frozen_folds(observation_geometry().selected)
    if fold!=e['pilot_fold_definition'] or fold['sha256']!='16594a306157e0b6ac114100ddebbba262f8e4205a98c8f48726e5519e55ab18':
        raise ValueError('frozen fold definition changed')
    d=load(OUT/'t16d_modular_results.json')
    if {int(k):v for k,v in d['source_noise_variances'].items()}!=protocol.VARIANCES:
        raise ValueError('production noise scaling drift')
    if any(protocol.POWER/10**(s/10)!=protocol.VARIANCES[s] for s in protocol.SNRS):
        raise ValueError('source SNR reference identity drift')
    info=dict(operator_artifact=e['operator_artifact'],candidate_artifact=candidate['candidate_artifact'],
        factor_artifact=factors['factor_artifact'],fold=fold,
        source_and_module_array_sha256={k:array_hash(v) for k,v in arrays.items()},
        rank4_array_sha256={f'{m}_{s}_{part}':array_hash(getattr(family[m,s,4],part))
                           for m in ('P','M') for s in protocol.SNRS for part in ('A','Q')},
        ranks={'P':4,'M':4},means='source',no_svd=True,no_module_refit=True,
        coefficient_solver='original T16E real two-coordinate normal equations',
        cross_pilot_operators='frozen exact T16E',causal_window=5)
    return arrays,family,info


def namespace_audit(rows):
    channel_keys=protocol.ROLES[:3];comm_keys=protocol.ROLES[3:]
    channels={tuple(r[k] for k in channel_keys) for r in rows}
    comm={tuple(r[k] for k in comm_keys) for r in rows}
    if len(rows)!=36000 or len(channels)!=3600 or len(comm)!=36000:
        raise ValueError('new random stream tuple collision')
    past=[]
    for filename in ('t15a_cdl_manifest.csv','t15c_cdl_manifest.csv','t16c_heldout_manifest.csv'):
        path=OUT/filename
        if path.exists():past.extend(readcsv(path))
    old_channel={tuple(int(r[k]) for k in channel_keys) for r in past}
    old_comm={tuple(int(r[k]) for k in comm_keys) for r in past}
    if channels&old_channel or comm&old_comm:raise ValueError('historical namespace collision')
    return dict(namespaces=[f'T18A_{n}_v1' for n in protocol.ENVIRONMENTS],
        trajectory_seed_tuples=3600,frame_communication_seed_tuples=36000,
        consumed_validation_communication_seed_tuples=6000,train_validation_disjoint=True,
        historical_channel_and_communication_tuples_disjoint=True,
        train_communication_allocated_but_not_generated=True,formal_namespace_accessed=False)


def historical_receiver_regression(arrays,family):
    """C75/B120 first frozen trajectory: all 3 SNRs and 6 causal frames exact.

    Inputs loaded from old caches; no generator, target fitting or tuning.
    Complete remaining historical regression runs in startup/final pytest.
    """
    old=load(OUT/'t16f_results.json')
    dense=readcsv(OUT/'t16f_frame_metrics.csv');compressed=readcsv(OUT/'t16g_frame_metrics.csv')
    weights=readcsv(OUT/'t16f_weight_geometry.csv');proof=[]
    for name in ('CDL-C-75','CDL-B-120'):
        info=old['historical_replay_gate']['cache'][f'{name}/500']
        if sha(ROOT/info['path'])!=info['sha256']:raise ValueError('old pilot cache changed')
        with np.load(ROOT/info['path'],allow_pickle=False) as f:pilots=f['pilots']
        from scripts.run_modular_correction import cached_H
        hinfo=load(OUT/'t16d_modular_results.json')['validation_replay_gate']['cache'][name]
        if sha(ROOT/hinfo['path'])!=hinfo['file_sha256']:raise ValueError('old H cache changed')
        truth=frames(cached_H(hinfo,0));g=observation_geometry()
        for si,s in enumerate(protocol.SNRS):
            inferred=protocol.deployed(pilots[si],arrays,s,cv_family(arrays,s),family['P',s,4],family['M',s,4])
            for r in inferred:
                j=r['frame_id']
                prior=next(v for v in weights if (v['shift'],int(v['snr_db']),int(v['trajectory_id']),int(v['frame_id']),int(v['window']),v['method'])==(name,s,500,j,5,'affine'))
                if not np.array_equal(r['coefficients']['affine'],[float(prior['alpha']),float(prior['beta'])]):raise ValueError('old affine coefficients drift')
                for method,old_method in (('source','source'),('hard','hard'),('convex','convex'),('dense-affine','affine'),('lr4-affine','lr-affine')):
                    reference=compressed if method=='lr4-affine' else dense
                    v=next(v for v in reference if (v['shift'],int(v['snr_db']),int(v['trajectory_id']),int(v['frame_id']),int(v['window']),v['method'])==(name,s,500,j,5,old_method)
                           and (method!='lr4-affine' or int(v['rank'])==4))
                    error=float(np.sum(abs(r['predictions'][method]-truth[j].ravel()[g.data])**2))
                    if error!=float(v['error_energy']):raise ValueError('old receiver frame error not exact')
                proof.append(dict(environment=name,snr_db=s,trajectory_id=500,frame_id=j,
                    coefficients_exact=True,five_receiver_error_energies_exact=True))
    return dict(generator_calls=0,statistics_fits=0,all_exact=True,checked_windows=len(proof),rows=proof)


def preflight():
    if Path(sys.prefix).resolve()!=(ROOT/'.venv-cdl').resolve():raise ValueError('frozen .venv-cdl required')
    if (OUT/'t18a_preflight.json').exists():raise ValueError('existing preflight protected')
    accepted();arrays,family,method=frozen_method();env=environments()
    namespace=namespace_audit(protocol.manifest_rows())
    legacy=historical_receiver_regression(arrays,family)
    hashes={};provenance={}
    for name in protocol.ENVIRONMENTS:
        adapter=protocol.GeneralizationCDL(name);provenance[name]=adapter.provenance();hashes[name]={}
        for split,i in (('train_dev',0),('validation_dev',500)):
            h=adapter.generate(i,split=split);again=adapter.generate(i,split=split)
            if array_hash(h)!=array_hash(again):raise ValueError('same-environment deterministic CDL smoke failed')
            hashes[name][split]=array_hash(h)
        if hashes[name]['train_dev']==hashes[name]['validation_dev']:raise ValueError('split H collision')
    tracked=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True)
    if tracked.strip():raise ValueError('historical tracked changes before T18A')
    csvsave(OUT/'t18a_manifest.csv',protocol.manifest_rows())
    paths=subprocess.check_output(['git','ls-files','docs'],cwd=ROOT,text=True).splitlines()
    frozen=history();frozen.update({p:sha(ROOT/p) for p in paths})
    result=dict(task='T18A',all_checks_passed=True,accepted_T17B_commit=T17B_COMMIT,
        T17B_commit_subject='feat: identify cancellation-dominated exact affine interaction',
        startup_pytest=(OUT/'t18a_startup_pytest.txt').read_text(encoding='utf8').strip().splitlines()[-1],
        clean_working_tree_after_T17B_commit=True,tracked_working_tree_still_clean=True,
        frozen_method=method,namespace_audit=namespace,historical_receiver_regression=legacy,
        environment=env,generator_provenance=provenance,small_H_hashes=hashes,
        manifest_sha256=sha(OUT/'t18a_manifest.csv'),source_sha256={p:sha(ROOT/p) for p in SOURCES},
        frozen_history_sha256=frozen,preflight_generator_calls=24,new_unique_preflight_realizations=12,
        formal_test_used=False,formal_test_sealed=True)
    save(OUT/'t18a_preflight.json',result)
    print('T18A preflight PASS: accepted T17B; exact factors/folds/receiver; six deterministic split smoke tests; environment frozen',flush=True)


def aggregate(trajectory_rows):
    pooled=[]
    for name in protocol.ENVIRONMENTS:
        for s in protocol.SNRS:
            for method in protocol.METHODS:
                rows=[r for r in trajectory_rows if (r['environment'],r['snr_db'],r['method'])==(name,s,method)]
                if len(rows)!=100 or [r['trajectory_id'] for r in rows]!=list(range(500,600)) or any(r['frames']!=6 for r in rows):
                    raise ValueError('complete paired 100-trajectory last6 subset required')
                error=sum(r['error_energy'] for r in rows);truth=sum(r['truth_energy'] for r in rows)
                if error<=0 or truth<=0:raise ValueError('positive pooled energies required')
                pooled.append(dict(environment=name,snr_db=s,method=method,trajectories=100,frames=600,
                    subset='frames4..9',error_energy=error,truth_energy=truth,nmse_linear=error/truth,
                    nmse_db=float(10*np.log10(error/truth)),
                    trajectory_nmse_std=float(np.std([r['nmse_linear'] for r in rows],ddof=1)),
                    metric_type='actual_validation',means='target' if method=='matched' else 'source'))
    return pooled


def summarize_weights(raw):
    summaries=[]
    for name in protocol.ENVIRONMENTS:
        for s in protocol.SNRS:
            group=[r for r in raw if (r['environment'],r['snr_db'],r['method'])==(name,s,'affine')]
            if len(group)!=600:raise ValueError('all six causal windows per trajectory required')
            summary=dict(environment=name,snr_db=s,windows=600,independent_trajectories=100,
                outside_simplex_fraction=sum(r['outside_any'] for r in group)/len(group))
            for field in ('alpha','beta','lambda0'):
                x=np.array([r[field] for r in group]);q=np.quantile(x,[.25,.5,.75])
                summary[field]=dict(q25=float(q[0]),median=float(q[1]),q75=float(q[2]),
                    mean=float(np.mean(x)),std=float(np.std(x,ddof=1)),minimum=float(x.min()),maximum=float(x.max()))
            summaries.append(summary)
    return summaries


def execution_preflight():
    """Preserve preparation provenance; freeze added calibration test before run."""
    pre=load(OUT/'t18a_preflight.json');accepted();_,_,method=frozen_method()
    if method!=pre['frozen_method'] or sha(OUT/'t18a_manifest.csv')!=pre['manifest_sha256']:
        raise ValueError('preparation method/data changed')
    original=ROOT/'outputs/t18a_preparation_snapshot'
    for path,h in pre['source_sha256'].items():
        if sha(original/path)!=h:raise ValueError('original preparation source archive changed')
    current=ROOT/'outputs/t18a_execution_snapshot'
    if current.exists() or (OUT/'t18a_execution_preflight.json').exists():raise ValueError('existing execution snapshot protected')
    for path in SOURCES:
        dest=current/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/path).read_bytes())
    log=OUT/'t18a_related_preflight_pytest_attempt03.txt'
    summary=log.read_text(encoding='utf8').strip().splitlines()[-1]
    if not summary.startswith('37 passed, 1 skipped'):raise ValueError('expanded calibration must pass before execution')
    save(OUT/'t18a_execution_preflight.json',dict(all_checks_passed=True,
        original_preflight_sha256=sha(OUT/'t18a_preflight.json'),manifest_sha256=pre['manifest_sha256'],
        source_sha256={p:sha(ROOT/p) for p in SOURCES},source_snapshot_root='outputs/t18a_execution_snapshot',
        original_snapshot_root='outputs/t18a_preparation_snapshot',calibration_summary=summary,
        calibration_transcript_sha256=sha(log),original_preflight_unmodified=True,
        amendment='Added algebraic paired-link test after preparation; new execution snapshot, original hashes preserved.',
        frozen_method_unchanged=True,seeds_data_and_protocol_unchanged=True,scientific_scan_started=False))
    print('Original preparation exact archive + updated execution-source snapshot PASS; frozen method/data unchanged',flush=True)


def evaluate(name,i,h,arrays,family,matched,oracle,g):
    """Generate each communication frame once, then call label-free deployed API."""
    truths=frames(h);pilots=np.empty((3,10,64),np.complex128);pair=[]
    signal=np.zeros(2);unit_noise=np.zeros(2);samples=0
    known=map_pilot_symbols(g.layout,PILOT_SYMBOLS)
    for j,truth in enumerate(truths):
        bits,symbols,z=communication(protocol.seeds(name,'validation_dev',i,j))
        x=assemble_pilot_data_frame(symbols,g.layout);clean=apply_frequency_mimo_channel(truth,x)
        if j>=4:
            mask=g.layout.data_mask[0];signal+=np.sum(abs(clean[:,mask])**2,axis=1)
            unit_noise+=np.sum(abs(z[:,mask])**2,axis=1);samples+=int(mask.sum())
        shared=dict(environment=name,trajectory_id=i,frame_id=j,H_sha256=array_hash(truth),
            bits_sha256=array_hash(bits),symbols_sha256=array_hash(symbols),unit_noise_sha256=array_hash(z))
        for si,s in enumerate(protocol.SNRS):
            y=clean+np.sqrt(protocol.VARIANCES[s])*z
            sparse,_=pilot_ls_estimate(y,known,g.layout.pilot_mask)
            pilots[si,j]=sparse.ravel()[g.selected]
            pair.append(dict(**shared,snr_db=s,Y_sha256=array_hash(y),pilot_sha256=array_hash(pilots[si,j])))
    pilots.flags.writeable=False;raw=[];tr=[];frame_rows=[]
    for si,s in enumerate(protocol.SNRS):
        estimates=protocol.deployed(pilots[si],arrays,s,cv_family(arrays,s),family['P',s,4],family['M',s,4])
        errors={m:[] for m in protocol.METHODS};energies=[]
        for r in estimates:
            j=r['frame_id'];true=truths[j].ravel()[g.data];energy=float(np.sum(abs(true)**2));energies.append(energy)
            if r['dense_coefficients'] is not r['lr4_coefficients']:raise ValueError('Dense/LR4 coefficients not shared')
            # Target references are confined to this post-deployment evaluation.
            predictions=dict(r['predictions'],matched=matched[s].estimate(pilots[si,j]))
            predictions['oracle-affine']=deployment(pilots[si,j],arrays['source_mean_p'],arrays['source_mean_d'],
                arrays[f'Wsource_{s}'],arrays[f'DeltaWP_{s}'],arrays[f'DeltaWM_{s}'],oracle[s],('P','M'))
            for m,pred in predictions.items():
                error=float(np.sum(abs(pred-true)**2));errors[m].append(error)
                frame_rows.append(dict(environment=name,snr_db=s,trajectory_id=i,frame_id=j,method=m,
                    error_energy=error,truth_energy=energy,prediction_sha256=array_hash(pred)))
            for m,c in r['coefficients'].items():
                diag=r['diagnostics'][m]
                raw.append(dict(environment=name,snr_db=s,trajectory_id=i,frame_id=j,window_start=j-4,window=5,
                    method=m,**geometry(c),source_cv_loss=diag['source_cv_loss'],fitted_cv_loss=diag['fitted_cv_loss'],
                    selected_candidate=diag.get('selected_candidate'),active_set=diag.get('selected_active_set'),
                    gram_condition_number=diag.get('condition_number'),coefficients_sha256=array_hash(c),
                    dense_lr4_same_coefficients=(m=='affine')))
        energy=trajectory_energy(energies)
        for method in protocol.METHODS:
            error=trajectory_energy(errors[method]);tr.append(dict(environment=name,snr_db=s,trajectory_id=i,
                method=method,frames=6,subset='frames4..9',error_energy=error,truth_energy=energy,nmse_linear=error/energy))
    return tr,raw,frame_rows,pair,signal,unit_noise,samples


def run():
    if Path(sys.prefix).resolve()!=(ROOT/'.venv-cdl').resolve():raise ValueError('frozen .venv-cdl required')
    if RESULT.exists() or (OUT/'t18a_generalization_nmse.csv').exists():raise ValueError('existing scientific run protected')
    pre=load(OUT/'t18a_preflight.json')
    execution=load(OUT/'t18a_execution_preflight.json')
    if not pre['all_checks_passed'] or not execution['all_checks_passed'] or sha(OUT/'t18a_preflight.json')!=execution['original_preflight_sha256']:
        raise ValueError('original/execution preflight changed')
    if any(sha(ROOT/p)!=v for p,v in execution['source_sha256'].items()):raise ValueError('execution source drift')
    if sha(OUT/'t18a_manifest.csv')!=pre['manifest_sha256']:raise ValueError('manifest changed')
    accepted();arrays,family,proof=frozen_method()
    if proof!=pre['frozen_method']:raise ValueError('method drift')
    started=time.perf_counter();stats_prov={};snr_audit=[];trajectory=[];weights=[];frame_rows=[];pairs=[];hrows=[];oracles=[];runtime={}
    for name in protocol.ENVIRONMENTS:
        adapter=protocol.GeneralizationCDL(name)
        if adapter.provenance()!=pre['generator_provenance'][name]:raise ValueError('generator provenance drift')
        stream=StreamingStatistics(observation_geometry().selected,observation_geometry().data)
        power=TrainReceivePowerAccumulator();batch=[];ids=[];clock={'training_generation_seconds':0.,'validation_generation_seconds':0.}
        tick=time.perf_counter()
        for i in range(500):
            t=time.perf_counter();h=adapter.generate(i,split='train_dev');clock['training_generation_seconds']+=time.perf_counter()-t
            if i==0 and array_hash(h)!=pre['small_H_hashes'][name]['train_dev']:raise ValueError('small train replay changed')
            hrows.append(dict(environment=name,split='train_dev',trajectory_id=i,H_sha256=array_hash(h)))
            power.update(i,h);batch.append(frames(h).reshape(10,4096));ids.append(i)
            if len(batch)==10:stream.update(np.concatenate(batch),ids);batch=[];ids=[]
            if (i+1)%50==0:print(f'{name} matched-only target train {i+1}/500',flush=True)
        target=stream.freeze();reference=power.freeze();clock['training_pipeline_seconds']=time.perf_counter()-tick
        if target.train_trajectory_ids!=tuple(range(500)) or target.samples!=5000:raise ValueError('target train-only protocol drift')
        path=OUT/f't18a_matched_statistics_{name}.npz'
        if path.exists():raise ValueError('existing target-statistics artifact protected')
        np.savez_compressed(path,mean_p=target.mean_p,mean_d=target.mean_d,rpp=target.rpp,rdp=target.rdp,
            train_trajectory_ids=target.train_trajectory_ids,samples=target.samples)
        path.chmod(0o444)
        stats_prov[name]=dict(path=str(path.relative_to(ROOT)).replace('\\','/'),sha256=sha(path),
            component_sha256={k:array_hash(getattr(target,k)) for k in ('mean_p','mean_d','rpp','rdp')},
            namespace=f'T18A_{name}_v1',split='train_dev',training_only=True,trajectories=500,frames=5000,
            covariance_divisor='N population',batch_frames=100,mean_centered=True,
            train_ensemble_power=reference.reference,train_power_per_rx=list(reference.per_rx_signal_power),
            used_for_deployed_estimation=False,validation_used_for_statistics=False)
        matched={s:target.prepare(protocol.VARIANCES[s]) for s in protocol.SNRS};oracle={};oracle_info={}
        for s in protocol.SNRS:
            oracle[s],oracle_info[s]=protocol.oracle_coordinates(target,arrays,s)
            if matched[s].solve_relative_residual>1e-11:raise ValueError('matched Cholesky solve residual failed')
        signal=np.zeros(2);noise=np.zeros(2);samples=0;tick=time.perf_counter();g=observation_geometry()
        for i in range(500,600):
            t=time.perf_counter();h=adapter.generate(i,split='validation_dev');clock['validation_generation_seconds']+=time.perf_counter()-t
            if i==500 and array_hash(h)!=pre['small_H_hashes'][name]['validation_dev']:raise ValueError('small validation replay changed')
            hrows.append(dict(environment=name,split='validation_dev',trajectory_id=i,H_sha256=array_hash(h)))
            h.flags.writeable=False
            tr,raw,fr,paired,sp,npow,count=evaluate(name,i,h,arrays,family,matched,oracle,g)
            trajectory.extend(tr);weights.extend(raw);frame_rows.extend(fr);pairs.extend(paired)
            signal+=sp;noise+=npow;samples+=count
            if (i-499)%10==0:print(f'{name} paired frozen validation {i-499}/100',flush=True)
        clock['validation_pipeline_seconds']=time.perf_counter()-tick;runtime[name]=clock
        for s in protocol.SNRS:
            snr_audit.append(dict(environment=name,snr_db=s,source_reference_power=protocol.POWER,sigma2=protocol.VARIANCES[s],
                target_train_ensemble_power=reference.reference,target_train_power_per_rx=list(reference.per_rx_signal_power),
                target_train_receive_snr_db=float(10*np.log10(reference.reference/protocol.VARIANCES[s])),
                validation_realised_signal_power_per_rx=(signal/samples).tolist(),
                validation_realised_noise_power_per_rx=(protocol.VARIANCES[s]*noise/samples).tolist(),
                validation_realised_receive_snr_db=float(10*np.log10(signal.sum()/(protocol.VARIANCES[s]*noise.sum()))),
                validation_realised_receive_snr_per_rx_db=(10*np.log10(signal/(protocol.VARIANCES[s]*noise))).tolist(),
                subset='frames4..9',audit_only=True,noise_recalibrated=False,target_normalized=False))
            oracles.append(dict(environment=name,snr_db=s,**oracle_info[s]))
        # Checkpoint records retain every finished environment without rerunning it.
        save(OUT/f't18a_progress_{name}.json',dict(completed_environment=name,target_statistics=stats_prov[name],
            runtime=clock,trajectory_rows=[r for r in trajectory if r['environment']==name]))
    pooled=aggregate(trajectory);summary=summarize_weights(weights);conditions=[]
    for name in protocol.ENVIRONMENTS:
        for s in protocol.SNRS:
            values={r['method']:r['nmse_db'] for r in pooled if (r['environment'],r['snr_db'])==(name,s)}
            geo=next(r for r in summary if (r['environment'],r['snr_db'])==(name,s))
            conditions.append(dict(environment=name,snr_db=s,**protocol.metrics(values['source'],values['matched'],values['dense-affine'],
                values['lr4-affine'],values['convex'],values['oracle-affine']),outside_simplex_fraction=geo['outside_simplex_fraction'],
                **{f'nmse_{m.replace("-","_")}_db':v for m,v in values.items()}))
    science=protocol.scientific_gate(conditions)
    oracle_rows=[]
    for row in oracles:
        nmse=next(r for r in pooled if (r['environment'],r['snr_db'],r['method'])==(row['environment'],row['snr_db'],'oracle-affine'))
        oracle_rows.append(dict(row,nmse_db=nmse['nmse_db'],error_energy=nmse['error_energy'],truth_energy=nmse['truth_energy']))
    files={'t18a_generalization_nmse.csv':[r for r in pooled if r['method']!='oracle-affine'],
        't18a_generalization_weights.csv':weights,'t18a_generalization_trajectory.csv':trajectory,
        't18a_oracle_affine.csv':oracle_rows,'t18a_extrapolation.csv':[r for r in conditions if r['environment'] in protocol.EXTRAPOLATION],
        't18a_condition_metrics.csv':conditions,'t18a_frame_metrics.csv':frame_rows,'t18a_paired_input_hashes.csv':pairs,
        't18a_channel_hashes.csv':hrows}
    for filename,rows in files.items():csvsave(OUT/filename,rows)
    for p,h in pre['frozen_history_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('historical drift:'+p)
    result=dict(task='T18A',experiment_type=protocol.EXPERIMENT_TYPE,
        research_question='Does the completely frozen source/modules/Pilot-CV/rank4 pipeline generalize across untouched mobility and composed CDL shifts?',
        method_frozen=True,c6_claim_status=protocol.C6,c7_claim_status=protocol.C7,
        accepted_T17B_commit=T17B_COMMIT,interpolation_environments=list(protocol.INTERPOLATION),
        extrapolation_environments=list(protocol.EXTRAPOLATION),frozen_method_provenance=proof,
        namespace_audit=pre['namespace_audit'],dataset=dict(target_train_trajectories_per_environment=500,
            target_validation_trajectories_per_environment=100,frames_per_trajectory=10,primary_frames=list(protocol.PRIMARY_FRAMES)),
        snr_audit=snr_audit,target_statistics=stats_prov,validation_nmse=pooled,condition_metrics=conditions,
        dense_affine_closure=[{k:r[k] for k in ('environment','snr_db','dense_affine_closure')} for r in conditions],
        pilotcv_efficiency=[{k:r[k] for k in ('environment','snr_db','pilotcv_efficiency')} for r in conditions],
        compression_penalty=[{k:r[k] for k in ('environment','snr_db','compression_penalty_db')} for r in conditions],
        affine_vs_convex_gain=[{k:r[k] for k in ('environment','snr_db','affine_vs_convex_gain_db','lr4_vs_convex_gain_db')} for r in conditions],
        weight_geometry=summary,oracle_affine=oracle_rows,**science,
        scientific_gate_thresholds=dict(pilotcv_efficiency=.80,compression_penalty_db=.50,dense_closure=.70,
            composed_gain_replicated_db=1.,composed_gain_weak_db=.25,outside_fraction=.75,stress_convex_tolerance_db=.50),
        scientific_gate_uses_pooled_db_differences=True,primary_estimator_uses_environment_label=False,
        dense_lr4_coefficients_identical=True,rank_per_module=4,causal_window=5,
        source_means_for_deployed_and_oracle=True,matched_uses_target_means=True,target_statistics_used_for_deployment=False,
        target_normalization=False,per_environment_noise_recalibration=False,new_module_fitted=False,new_svd=False,
        new_model_trained=False,method_parameters_changed=False,BER_evaluations=0,detector_calls=0,
        generator_calls=3600,new_unique_scientific_realizations=3600,preflight_generator_calls=24,
        within_symbol_ICI_modeled=False,stress_interpretation='within the current symbol-wise CDL abstraction',
        formal_test_used=False,formal_test_sealed=True,
        complexity=dict(source=253952,pilot_cv_5frame=46080,two_rank4_corrections=32256,total=332288,
            total_over_source_reported=1.308468,provenance='frozen T16G rank4 proxy',runtime_speedup_inferred=False),
        historical_receiver_regression=pre['historical_receiver_regression'],
        runtime=dict(by_environment=runtime,total_numeric_wall_seconds=time.perf_counter()-started),
        source_sha256=execution['source_sha256'],source_snapshot_root=execution['source_snapshot_root'],
        frozen_history_sha256=pre['frozen_history_sha256'],environment_before=pre['environment'],
        preparation_artifact_sha256={'outputs/t18a_preflight.json':sha(OUT/'t18a_preflight.json'),'outputs/t18a_manifest.csv':sha(OUT/'t18a_manifest.csv'),
            'outputs/t18a_execution_preflight.json':sha(OUT/'t18a_execution_preflight.json')},
        numeric_artifact_sha256={f'outputs/{p}':sha(OUT/p) for p in files},
        all_checks_passed=False,completion_state='actual validation completed; final audit pending',
        warnings=['Pooled NMSE has no confidence interval; trajectory std and window IQR are descriptive with correlated causal windows.',
            'Oracle optimizes centered target-weighted filter error and uses source means, not absolute NMSE/mean-bias optimization.',
            'New T18A H/bits/noise/Y hashes are current evidence; prior missing historical hashes are not claimed.',
            '180 km/h excludes within-symbol ICI in the current symbol-wise abstraction.',
            'C6/C7 negative mechanism claims remain permanent; no physical-additivity or novelty claim.',
            'T17B staged whitespace/newline warnings retained in archived logs; current diff check is separate.',
            'Initial duplicate-key syntax error archived before any scientific run; added paired-link test received a separate execution snapshot.',
            'Optional Triton absent: PyTorch flop-counter warning; no Triton kernel/flop counting is used.',
            'Python startup reports stale base-interpreter location while execution succeeds; frozen environments unchanged.'])
    save(RESULT,result);print(json.dumps(science,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preflight',action='store_true');p.add_argument('--verify-preflight',action='store_true');a=p.parse_args()
    if a.preflight:preflight()
    elif a.verify_preflight:execution_preflight()
    else:run()
