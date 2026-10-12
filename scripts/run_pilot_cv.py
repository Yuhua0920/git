"""T16E uniform pilot-only analytical fitting with exact frozen-input replay."""
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
from src.pilot_cv import (T16D_COMMIT,EXPERIMENT_TYPE,frozen_folds,cross_pilot_operators,PilotCV,
    causal_windows,primary_prediction,deployment,recovery,fitting_gate)
from src.cdl_benchmark import observation_geometry,array_hash,frames
from scripts.run_cdl_mismatch import load,save,sha,csvsave,history,environments
from scripts.run_lowrank_filter import statistics,prepare,BASELINES

OUT=ROOT/'outputs'
SOURCES=('src/pilot_cv.py','scripts/run_pilot_cv.py','scripts/plot_pilot_cv.py',
    'scripts/audit_pilot_cv.py','tests/test_pilot_cv.py','docs/pilot_cv_protocol.md')
RESULT=OUT/'t16e_pilot_cv_results.json'
SNRS=(5,10,20)
TARGETS=('CDL-C-75','CDL-B-120')


def accepted():
    subprocess.check_call(['git','merge-base','--is-ancestor',T16D_COMMIT,'HEAD'],cwd=ROOT)
    old=load(OUT/'t16d_modular_results.json')
    if not old['all_checks_passed'] or old['completion_state']!='completed':raise ValueError('T16D acceptance required')
    for p,h in {**old['source_sha256'],**old['frozen_history_sha256'],**old['numeric_artifact_sha256']}.items():
        if sha(ROOT/p)!=h:raise ValueError(f'historical drift: {p}')
    for p in [*old['source_sha256'],'outputs/t16d_modular_results.json']:
        blob=subprocess.check_output(['git','show',f'{T16D_COMMIT}:{p}'],cwd=ROOT)
        if blob.replace(b'\r\n',b'\n')!=(ROOT/p).read_bytes().replace(b'\r\n',b'\n'):
            raise ValueError('accepted Git snapshot drift')
    return old


def operator_arrays():
    """Only source/B30/C120 statistics; no held-out target statistics or oracle coefficients."""
    old=accepted();_,source,development,variances=statistics()
    estimates,_,seconds=prepare(source,development,variances)
    folds,definition=frozen_folds(observation_geometry().selected)
    arrays=dict(source_mean_p=source.mean_p,source_mean_d=source.mean_d)
    proof=[]
    for s in (5,10,20):
        ws=estimates['CDL-B-30',BASELINES[0],s].weight
        arrays[f'Wsource_{s}']=ws
        kernels={};residuals={};kernel_start=time.perf_counter()
        for label,stats in (('Source',source),('P',development['CDL-B-30']),('M',development['CDL-C-120'])):
            kernels[label],residuals[label]=cross_pilot_operators(stats.rpp,variances[s],folds)
            arrays[f'K{label}_{s}']=np.stack(kernels[label])
        kernel_seconds=time.perf_counter()-kernel_start
        for label,name in (('P','CDL-B-30'),('M','CDL-C-120')):
            wt=estimates[name,BASELINES[1],s].weight;delta=wt-ws
            historical=next(r for r in old['module_provenance']['development_operator_regression'] if r['shift']==name and r['snr_db']==s)
            for a,key in ((ws,'source_W_sha256'),(wt,'target_W_sha256'),(delta,'DeltaW_sha256')):
                if array_hash(a)!=historical[key]:raise ValueError('T16D frozen data-domain module drift')
            arrays[f'DeltaW{label}_{s}']=delta
            arrays[f'DeltaK{label}_{s}']=arrays[f'K{label}_{s}']-arrays[f'KSource_{s}']
        proof.append(dict(snr_db=s,sigma2=variances[s],K_construction_seconds=kernel_seconds,source_rpp_sha256=array_hash(source.rpp),
            profile_rpp_sha256=array_hash(development['CDL-B-30'].rpp),
            mobility_rpp_sha256=array_hash(development['CDL-C-120'].rpp),
            operator_residuals=residuals,
            arrays_sha256={k:array_hash(v) for k,v in arrays.items() if k.endswith(f'_{s}')},
            data_module_T16D_hashes_exact=True))
    return arrays,definition,proof,seconds


def preflight():
    start=time.perf_counter();a,d,p,seconds=operator_arrays()
    foldpath=OUT/'t16e_pilot_folds.json';path=OUT/'t16e_frozen_operators.npz'
    if foldpath.exists() or path.exists():raise ValueError('existing fold/operator artifacts protected')
    save(foldpath,d);foldpath.chmod(0o444)
    np.savez_compressed(path,**a);path.chmod(0o444)
    with np.load(path,allow_pickle=False) as frozen:
        if any(array_hash(frozen[k])!=array_hash(v) for k,v in a.items()):raise ValueError('persisted operator array drift')
    result=dict(task='T16E',stage='frozen_math_preflight_only',all_checks_passed=True,
        accepted_T16D_commit=T16D_COMMIT,pilot_fold_definition=d,
        cross_pilot_operator_provenance=p,
        operator_artifact=dict(path='outputs/t16e_frozen_operators.npz',sha256=sha(path),
            array_sha256={k:array_hash(v) for k,v in a.items()},readonly=True),
        fold_artifact_sha256=sha(foldpath),generator_calls=0,new_channel_realizations=0,
        target_statistics_used_for_fitting=False,oracle_coefficients_used_for_fitting=False,
        source_means_only=True,formal_test_used=False,formal_test_sealed=True,
        decisions_pending=['shift-blind one/two-arm comparison','T16D effective oracle criterion'],
        source_sha256={p:sha(ROOT/p) for p in SOURCES},
        runtime=dict(data_filter_construction_seconds=seconds,total_preflight_seconds=time.perf_counter()-start))
    save(OUT/'t16e_pilot_cv_preflight.json',result)
    print('Frozen 4-fold split / 36 cross-pilot operators / six T16D module hashes / NPZ array roundtrip PASS; actual validation pending protocol decisions')


def frozen_operators():
    pre=load(OUT/'t16e_pilot_cv_preflight.json');info=pre['operator_artifact']
    if sha(ROOT/info['path'])!=info['sha256'] or sha(OUT/'t16e_pilot_folds.json')!=pre['fold_artifact_sha256']:
        raise ValueError('frozen preparation artifact drift')
    with np.load(ROOT/info['path'],allow_pickle=False) as f:arrays={k:f[k].copy(order='K') for k in f.files}
    if any(array_hash(arrays[k])!=h for k,h in info['array_sha256'].items()):raise ValueError('frozen operator array drift')
    for v in arrays.values():v.flags.writeable=False
    return arrays,pre


def execution_preflight():
    """New source/protocol snapshot, without overwriting original preparation."""
    accepted();arrays,pre=frozen_operators();start=time.perf_counter()
    rebuilt,definition,proof,seconds=operator_arrays()
    if definition!=pre['pilot_fold_definition'] or any(not np.array_equal(rebuilt[k],v) for k,v in arrays.items()):
        raise ValueError('original folds/operators not exact')
    r=dict(all_checks_passed=True,task='T16E',source_sha256={p:sha(ROOT/p) for p in SOURCES},
        original_preparation_sha256={p:sha(OUT/p) for p in ('t16e_pilot_folds.json','t16e_frozen_operators.npz',
            't16e_pilot_cv_preflight.json','t16e_preparation_audit.json')},
        pilot_fold_definition=definition,cross_pilot_operator_provenance=proof,
        primary_estimator_uses_environment_label=False,primary_module_model='unified_two_real_coefficients',
        label_assisted_one_module_diagnostic=True,t16d_oracle_scalar_capacity_available=True,
        t16d_oracle_capacity_rule='not scalar_mode_composition_insufficient',
        generator_calls=0,new_channel_realizations=0,
        runtime=dict(K_and_module_verification_seconds=time.perf_counter()-start,data_filter_construction_seconds=seconds,
            K_construction_seconds=sum(v['K_construction_seconds'] for v in proof)))
    save(OUT/'t16e_pilot_cv_execution_preflight.json',r)
    print('Confirmed unified P+M / frozen folds and36 K arrays / six T16D DeltaW hashes EXACT; execution preflight PASS')


def cv_family(arrays,s):
    folds,_=frozen_folds(observation_geometry().selected)
    return PilotCV(folds,arrays['source_mean_p'],tuple(arrays[f'KSource_{s}']),
        tuple(arrays[f'DeltaKP_{s}']),tuple(arrays[f'DeltaKM_{s}']))


def fit_unlabelled(pilots,arrays,s):
    """Primary stage has no target object, H_true, oracle coefficient or label."""
    cv=cv_family(arrays,s);result=[];fit_seconds=0.;inference_seconds=0.
    for window in (1,5):
        for frame,values in causal_windows(pilots,window):
            tick=time.perf_counter();coeff,diag=cv.fit(values,modules=('P','M'));fit_seconds+=time.perf_counter()-tick
            tick=time.perf_counter()
            prediction=deployment(values[-1],arrays['source_mean_p'],arrays['source_mean_d'],arrays[f'Wsource_{s}'],
                arrays[f'DeltaWP_{s}'],arrays[f'DeltaWM_{s}'],coeff,('P','M'))
            inference_seconds+=time.perf_counter()-tick
            result.append(dict(frame_id=frame,window=window,window_start=frame-window+1,
                coefficients=coeff,prediction=prediction,diagnostics=diag))
    return result,fit_seconds,inference_seconds


def attach_reporting_label(result,label):
    """Metadata only, called after estimation; cannot select an estimator."""
    return [dict(**r,shift=label) for r in result]


def readcsv(path):
    with Path(path).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))


def trajectory_energy(values):
    """Preserve T16D's frame-by-frame += arithmetic, including roundoff."""
    total=0.
    for value in values:total+=float(value)
    return total


def checked_reference_energy(errors,truth,prior):
    error=trajectory_energy(errors);energy=trajectory_energy(truth)
    actual=dict(error_energy=error,truth_energy=energy,nmse_linear=error/energy)
    differences={key:value-float(prior[key]) for key,value in actual.items() if value!=float(prior[key])}
    if differences:raise ValueError(f'STOP: historical reference trajectory not exact: {differences}')
    return error,energy


def historical_gate(arrays,old):
    """Verify complete inputs and original source/matched/scalar references first."""
    from scripts.run_modular_correction import cached_H,cached_inputs,target_statistics,frozen_validation_tables
    from src.link import estimate_ls_linear_channel
    start=time.perf_counter();_,tables=frozen_validation_tables()
    targets=target_statistics(load(OUT/'t16c_heldout_results.json'))
    historical=readcsv(OUT/'t16d_trajectory_metrics.csv')
    index={(r['shift'],int(r['trajectory_id']),int(r['snr_db']),r['method']):r for r in historical}
    paired={(r['shift'],r['trajectory_id']):r for r in load(OUT/'t16c_heldout_results.json')['paired_input_evidence']}
    g=observation_geometry();cache_dir=ROOT/'.pytest_cache/t16e_eval_cache';cache_dir.mkdir(exist_ok=True)
    if any(cache_dir.glob('*.npz')):raise ValueError('existing evaluated replay cache protected')
    refs={};rows=[];inference=0.
    for name in TARGETS:
        info=old['validation_replay_gate']['cache'][name]
        if sha(ROOT/info['path'])!=info['file_sha256']:raise ValueError('historical H cache file drift')
        matched={s:targets[name].prepare(old['source_noise_variances'][str(s)]) for s in SNRS}
        corr={}
        for s in SNRS:
            reference=next(r for r in old['alpha_beta'] if r['shift']==name and r['snr_db']==s)
            # Only evaluation references may read historical oracle coefficients.
            corr[s]=reference['beta']*arrays[f'DeltaWM_{s}'] if name=='CDL-C-75' else (
                reference['alpha']*arrays[f'DeltaWP_{s}']+reference['beta']*arrays[f'DeltaWM_{s}'])
        for i in range(500,600):
            h=cached_H(info,i-500);inputs=cached_inputs(info,i)
            if array_hash(h)!=paired[name,i]['H_sha256']:raise ValueError('STOP: original H hash mismatch')
            pilots=np.empty((3,10,64),complex);errors=np.empty((3,3,10));truth_energy=np.empty(10)
            digest=hashlib.sha256()
            for j,truth in enumerate(frames(h)):
                data=truth.ravel()[g.data];truth_energy[j]=float(np.sum(abs(data)**2))
                hashes=[array_hash(a) for a in (truth,inputs['bits'][j],inputs['symbols'][j],inputs['z'][j])]
                for si,s in enumerate(SNRS):
                    y=inputs['y'][j,si];sparse,_=estimate_ls_linear_channel(y,g.layout)
                    z=sparse.ravel()[g.selected];pilots[si,j]=z
                    tick=time.perf_counter()
                    source=arrays['source_mean_d']+(z-arrays['source_mean_p'])@arrays[f'Wsource_{s}'].T
                    target=matched[s].estimate(z)
                    x=z-targets[name].mean_p
                    oracle=targets[name].mean_d+x@arrays[f'Wsource_{s}'].T+x@corr[s].T
                    inference+=time.perf_counter()-tick
                    for mi,value in enumerate((source,target,oracle)):errors[si,mi,j]=float(np.sum(abs(value-data)**2))
                    for v in (*hashes,array_hash(y)):digest.update(v.encode('ascii'))
            if digest.hexdigest()!=paired[name,i]['paired_inputs_sha256']:raise ValueError('STOP: paired input digest mismatch')
            for si,s in enumerate(SNRS):
                for mi,method in enumerate(('source','matched','scalar-real')):
                    prior=index[name,i,s,method]
                    error,energy=checked_reference_energy(errors[si,mi],truth_energy,prior)
                    rows.append(dict(shift=name,trajectory_id=i,snr_db=s,method=method,error_energy=error,truth_energy=energy))
            path=cache_dir/f'{name}_{i}.npz';np.savez_compressed(path,pilots=pilots,errors=errors,truth_energy=truth_energy)
            path.chmod(0o444);refs[name,i]=dict(path=str(path.relative_to(ROOT)).replace('\\','/'),sha256=sha(path),
                pilots_sha256=array_hash(pilots),paired_inputs_sha256=digest.hexdigest())
            if (i-499)%25==0:print(f'{name} original source/matched/oracle replay {i-499}/100',flush=True)
    for name in TARGETS:
        for s in SNRS:
            for method in ('source','matched','scalar-real'):
                selected=[r for r in rows if r['shift']==name and r['snr_db']==s and r['method']==method]
                error=sum(r['error_energy'] for r in selected);energy=sum(r['truth_energy'] for r in selected)
                prior=(next(r for r in old['scalar_mode_results'] if r['shift']==name and r['snr_db']==s) if method=='scalar-real' else
                    next(r for r in tables['baselines'] if r['shift']==name and int(r['snr_db'])==s and r['method']==method))
                if any(float(prior[k])!=v for k,v in (('error_energy',error),('truth_energy',energy),('nmse_linear',error/energy),('nmse_db',float(10*np.log10(error/energy))))):
                    raise ValueError('STOP: historical pooled reference not exact')
    proof=dict(all_checks_passed=True,all_200_H_hashes_exact=True,all_200_paired_digests_exact=True,
        all_1800_reference_trajectory_rows_exact=True,all_18_reference_pooled_rows_exact=True,
        original_input_file_and_tensor_hashes_verified=True,historical_full_tensor_hashes_available=False,
        historical_H_hashes_available=True,historical_combined_input_digests_available=True,
        historical_separate_bits_noise_Y_hashes_available=False,original_cache_reused=True,
        trajectory_energy_reduction='original sequential frame +=; pooled trajectory sum unchanged',
        generator_calls=0,new_channel_realizations=0,
        cache={f'{n}/{i}':v for (n,i),v in refs.items()},
        runtime=dict(historical_gate_seconds=time.perf_counter()-start,historical_reference_inference_seconds=inference))
    save(OUT/'t16e_historical_replay_gate.json',proof)
    print('Historical H/inputs/1800 reference trajectory metrics/18 pooled NMSE EXACT; actual CV permitted',flush=True)
    return refs,proof


def aggregate(frame_rows):
    """Same frames for every method within each comparison table."""
    rows=[];trajectories=[]
    for name in TARGETS:
        for s in SNRS:
            for subset,ids,methods in (('all10',range(10),('source','oracleScalar','matched','pilotCV-1')),
                ('last6',range(4,10),('source','oracleScalar','matched','pilotCV-1','pilotCV-5'))):
                for method in methods:
                    selected=[r for r in frame_rows if r['shift']==name and r['snr_db']==s and r['method']==method and r['frame_id'] in ids]
                    for i in range(500,600):
                        v=[r for r in selected if r['trajectory_id']==i]
                        if len(v)!=len(ids):raise ValueError('same-subset frame count failed')
                        error=trajectory_energy(r['error_energy'] for r in v);energy=trajectory_energy(r['truth_energy'] for r in v)
                        trajectories.append(dict(shift=name,snr_db=s,trajectory_id=i,subset=subset,method=method,
                            frames=len(v),frame_start=ids[0],frame_end=ids[-1],error_energy=error,truth_energy=energy,nmse_linear=error/energy))
                    tr=[r for r in trajectories if r['shift']==name and r['snr_db']==s and r['subset']==subset and r['method']==method]
                    error=sum(r['error_energy'] for r in tr);energy=sum(r['truth_energy'] for r in tr)
                    rows.append(dict(shift=name,snr_db=s,subset=subset,method=method,trajectories=100,frames=100*len(ids),
                        error_energy=error,truth_energy=energy,nmse_linear=error/energy,nmse_db=float(10*np.log10(error/energy)),
                        trajectory_nmse_mean=float(np.mean([r['nmse_linear'] for r in tr])),
                        trajectory_nmse_std=float(np.std([r['nmse_linear'] for r in tr],ddof=1)),metric_type='actual_validation'))
    for r in rows:
        select=lambda m:next(v['nmse_db'] for v in rows if v['shift']==r['shift'] and v['snr_db']==r['snr_db'] and v['subset']==r['subset'] and v['method']==m)
        r.update(recovery(select('source'),select('matched'),select('oracleScalar'),r['nmse_db']))
    return rows,trajectories


def coefficient_summary(rows):
    result=[]
    for name in dict.fromkeys(r['shift'] for r in rows):
        for s in SNRS:
            for window in (1,5):
                selected=[r for r in rows if r['shift']==name and r['snr_db']==s and r['window']==window]
                for key in ('alpha_hat','beta_hat','beta_absolute_error','joint_coefficient_error'):
                    values=np.array([r[key] for r in selected if r[key] is not None],float)
                    if not len(values):continue
                    result.append(dict(shift=name,snr_db=s,window=window,quantity=key,samples=len(values),
                        mean=float(values.mean()),std=float(values.std(ddof=1)),median=float(np.median(values)),
                        q25=float(np.quantile(values,.25)),q75=float(np.quantile(values,.75)),minimum=float(values.min()),maximum=float(values.max())))
    return result


def run():
    from scripts.run_modular_correction import cached_H
    old=accepted();execution=load(OUT/'t16e_pilot_cv_execution_preflight.json')
    if any(sha(ROOT/p)!=h for p,h in execution['source_sha256'].items()):raise ValueError('execution-preflight source drift')
    if RESULT.exists() or (OUT/'t16e_pilot_cv_scan.csv').exists():raise ValueError('existing formal results protected')
    start=time.perf_counter();before=history();env=environments();arrays,pre=frozen_operators()
    refs,gate=historical_gate(arrays,old)
    if not gate['all_checks_passed']:raise ValueError('STOP: failed historical gate')
    g=observation_geometry();frame_rows=[];coeff_rows=[];diag_coeff=[];diag_frames=[]
    fit_seconds=0.;data_seconds=0.;diagnostic_seconds=0.;primary_digest=hashlib.sha256();small=[]
    for name in TARGETS:
        for i in range(500,600):
            info=refs[name,i];path=ROOT/info['path']
            if sha(path)!=info['sha256']:raise ValueError('evaluation cache changed')
            with np.load(path,allow_pickle=False) as f:pilots=f['pilots'];baseline_errors=f['errors'];energies=f['truth_energy']
            if array_hash(pilots)!=info['pilots_sha256']:raise ValueError('paired pilot cache changed')
            h=cached_H(old['validation_replay_gate']['cache'][name],i-500)
            for si,s in enumerate(SNRS):
                # No condition/label/oracle/H is passed to primary estimation.
                inferred,ft,dt=fit_unlabelled(pilots[si],arrays,s);fit_seconds+=ft;data_seconds+=dt
                if len(inferred)!=16 or any(not np.isfinite(r['prediction']).all() for r in inferred):raise ValueError('actual small/large inference failed')
                reported=attach_reporting_label(inferred,name)
                changed=attach_reporting_label(inferred,'tampered-report-label')
                if any(not np.array_equal(a['coefficients'],b['coefficients']) or not np.array_equal(a['prediction'],b['prediction']) for a,b in zip(reported,changed)):
                    raise ValueError('report label altered prediction')
                # Oracle reference accessed AFTER primary estimates are already complete.
                oracle=next(r for r in old['alpha_beta'] if r['shift']==name and r['snr_db']==s)
                for j,truth in enumerate(frames(h)):
                    for mi,method in enumerate(('source','matched','oracleScalar')):
                        frame_rows.append(dict(shift=name,snr_db=s,trajectory_id=i,frame_id=j,method=method,
                            error_energy=float(baseline_errors[si,mi,j]),truth_energy=float(energies[j])))
                for r in reported:
                    j=r['frame_id'];a,b=map(float,r['coefficients']);data=frames(h)[j].ravel()[g.data]
                    frame_rows.append(dict(shift=name,snr_db=s,trajectory_id=i,frame_id=j,method=f"pilotCV-{r['window']}",
                        error_energy=float(np.sum(abs(r['prediction']-data)**2)),truth_energy=float(energies[j])))
                    coeff_rows.append(dict(shift=name,snr_db=s,trajectory_id=i,frame_id=j,window=r['window'],window_start=r['window_start'],
                        alpha_hat=a,beta_hat=b,oracle_alpha=float(oracle['alpha'] or 0.),oracle_beta=oracle['beta'],
                        beta_absolute_error=abs(b-oracle['beta']),
                        joint_coefficient_error=float(np.hypot(a-(oracle['alpha'] or 0.),b-oracle['beta'])),
                        gram_condition_number=r['diagnostics']['condition_number'],normal_residual=r['diagnostics']['normal_equation_relative_residual'],
                        source_cv_loss=r['diagnostics']['source_cv_loss'],fitted_cv_loss=r['diagnostics']['fitted_cv_loss']))
                    primary_digest.update(array_hash(r['coefficients']).encode());primary_digest.update(array_hash(r['prediction']).encode())
                if i==500:small.append(dict(shift=name,snr_db=s,original_trajectory_id=i,primary_estimates=len(inferred),all_finite=True))
                # Label-assisted M-only arm is a separate post-primary C75 diagnostic.
                if name=='CDL-C-75':
                    tick=time.perf_counter();cv=cv_family(arrays,s)
                    for window in (1,5):
                        for j,values in causal_windows(pilots[si],window):
                            coeff,diagnostics=cv.fit(values,modules=('M',))
                            prediction=deployment(values[-1],arrays['source_mean_p'],arrays['source_mean_d'],arrays[f'Wsource_{s}'],
                                arrays[f'DeltaWP_{s}'],arrays[f'DeltaWM_{s}'],coeff,('M',))
                            b=float(coeff[0]);data=frames(h)[j].ravel()[g.data]
                            diag_coeff.append(dict(shift=name,snr_db=s,trajectory_id=i,frame_id=j,window=window,window_start=j-window+1,
                                alpha_hat=None,beta_hat=b,oracle_alpha=None,oracle_beta=oracle['beta'],beta_absolute_error=abs(b-oracle['beta']),
                                joint_coefficient_error=None,gram_condition_number=diagnostics['condition_number'],normal_residual=diagnostics['normal_equation_relative_residual'],
                                source_cv_loss=diagnostics['source_cv_loss'],fitted_cv_loss=diagnostics['fitted_cv_loss']))
                            diag_frames.append(dict(shift=name,snr_db=s,trajectory_id=i,frame_id=j,window=window,
                                error_energy=float(np.sum(abs(prediction-data)**2)),truth_energy=float(energies[j]),primary=False))
                    diagnostic_seconds+=time.perf_counter()-tick
            if (i-499)%25==0:print(f'{name} unified primary CV1/CV5 {i-499}/100',flush=True)
    pooled,trajectory=aggregate(frame_rows);summary=coefficient_summary(coeff_rows)
    diagnostic_summary=coefficient_summary(diag_coeff)
    scientific=fitting_gate(pooled,old['scalar_mode_composition_insufficient'])
    files={'t16e_pilot_cv_scan.csv':pooled,'t16e_pilot_cv_trajectory_metrics.csv':trajectory,
        't16e_pilot_cv_frame_metrics.csv':frame_rows,'t16e_pilot_cv_coefficients.csv':coeff_rows,
        't16e_pilot_cv_coefficient_summary.csv':summary,'t16e_label_assisted_coefficients.csv':diag_coeff,
        't16e_label_assisted_frame_metrics.csv':diag_frames,'t16e_label_assisted_summary.csv':diagnostic_summary}
    for filename,values in files.items():csvsave(OUT/filename,values)
    diagnostic_nmse=[]
    for s in SNRS:
        for window in (1,5):
            selected=[r for r in diag_frames if r['snr_db']==s and r['window']==window]
            error=sum(r['error_energy'] for r in selected);energy=sum(r['truth_energy'] for r in selected)
            diagnostic_nmse.append(dict(shift='CDL-C-75',snr_db=s,window=window,frames=len(selected),
                nmse_db=float(10*np.log10(error/energy)),primary=False,used_for_gate=False))
    if any(sha(ROOT/p)!=h for p,h in before.items()):raise ValueError('historical file changed')
    result=dict(task='T16E',experiment_type=EXPERIMENT_TYPE,literature_gate_passed=True,classical_baseline=True,proposed_method=False,
        accepted_T16D_commit=T16D_COMMIT,primary_estimator_uses_environment_label=False,
        primary_module_model='unified_two_real_coefficients',label_assisted_one_module_diagnostic=True,
        pilot_fold_definition=pre['pilot_fold_definition'],cross_pilot_operator_provenance=execution['cross_pilot_operator_provenance'],
        one_frame_coefficients=[r for r in summary if r['window']==1],five_frame_coefficients=[r for r in summary if r['window']==5],
        coefficient_rows_path='outputs/t16e_pilot_cv_coefficients.csv',validation_nmse=pooled,
        closure=[{k:r[k] for k in ('shift','snr_db','subset','method','closure')} for r in pooled],
        oracle_scalar_efficiency=[{k:r[k] for k in ('shift','snr_db','subset','method','oracle_scalar_efficiency')} for r in pooled],
        label_assisted_diagnostic_nmse=diagnostic_nmse,label_assisted_diagnostic_summary=diagnostic_summary,
        label_assisted_diagnostic_used_for_primary_gate=False,**scientific,
        C75_oracle_alpha_embedded_as_zero_for_joint_error=True,
        source_means_only=True,primary_input_scope='received pilot LS / source means / frozen source-B30-C120 K / original sigma2',
        original_T16D_oracle_reference_target_means_retained=True,
        mean_sensitivity_frozen_T16C_reference='outputs/t16c_heldout_mean_sensitivity.csv',
        historical_replay_gate_passed=True,historical_replay_gate=gate,
        historical_full_tensor_hashes_available=False,deterministic_validation_replay_used=True,
        new_validation_realizations_generated=False,t16e_methods_evaluated_on_same_replayed_realizations=True,
        reporting_label_mutation_exact=True,primary_coefficients_predictions_sha256=primary_digest.hexdigest(),
        actual_small_inference=small,coefficient_clipping=False,fold_search=False,independent_frames_assumed=False,
        generator_calls=0,new_channel_realizations=0,target_statistics_refitted=False,basis_refitted=False,
        BER_evaluations=0,detector_calls=0,new_model_trained=False,formal_test_used=False,formal_test_sealed=True,
        complexity=dict(real_LS_system_size=2,observed_per_fold=48,heldout_per_fold=16,
            prediction_operators_per_fold=3,pilot_objective_MAC_per_frame=4*16*48*3,
            five_frame_objective_MAC_per_estimate=5*4*16*48*3,rolling_reuse_implemented=False,
            source_data_MAC_per_frame=3968*64,primary_module_data_MAC_per_frame=2*3968*64,
            primary_data_total_matrix_vector_MAC_per_frame=3*3968*64,
            primary_scaling_complex_real_multiplies_per_frame=2*3968,
            oracle_scalar_precombined_correction_MAC_per_frame=3968*64,oracle_online_fitting=False),
        runtime=dict(**gate['runtime'],K_verification_seconds=execution['runtime']['K_and_module_verification_seconds'],
            K_construction_seconds=execution['runtime']['K_construction_seconds'],
            coefficient_fitting_seconds=fit_seconds,data_inference_seconds=data_seconds,label_assisted_diagnostic_seconds=diagnostic_seconds,
            total_numeric_wall_seconds=time.perf_counter()-start,target_train_generation_seconds=0.,statistics_fitting_seconds=0.),
        source_sha256={p:sha(ROOT/p) for p in SOURCES},frozen_history_sha256=before,environment_before=env,
        preparation_artifact_sha256=execution['original_preparation_sha256'],
        prior_failed_attempt=dict(path='outputs/t16e_attempt01_gate_failure',actual_PilotCV_started=False,
            reason='built-in float sum differed from original T16D sequential frame += reduction',
            maximum_diagnosed_energy_absolute_error=1.8189894035458565e-12,
            correction='restore original frame accumulation order; keep exact gate and historical artifacts unchanged',
            artifact_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in
                sorted((OUT/'t16e_attempt01_gate_failure').iterdir()) if p.is_file()}),
        numeric_artifact_sha256={f'outputs/{p}':sha(OUT/p) for p in files},
        all_checks_passed=False,completion_state='actual validation complete; final audit pending',
        warnings=['Unconstrained real LS: all weights retained; no clipping or discarded samples.',
            'Attempt01 stopped before actual Pilot-CV: sum versus original sequential += energy roundoff; failed source/log archived, original arithmetic restored and exact gate rerun.',
            'Five correlated frames form a composite prediction loss, not independent likelihood.',
            'T16D oracle reference uses original target means; deployment primary always source means.',
            'Source deployment SNR/noise unchanged; no target normalization; per-symbol abstraction omits within-symbol ICI.',
            'PF-SBNet researcher-provided positioning; original IEEE page inaccessible, not independently primary-source verified.',
            'Original separate bits/noise/Y tensor hashes absent; existing T16D replay-cache hashes are new replay evidence.',
            'Pilot-CV Gram conditioning and extrema saved; no validation-tuned stabilization.'])
    save(RESULT,result);print(scientific,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--verify-preflight',action='store_true');args=parser.parse_args()
    if args.preflight:preflight()
    elif args.verify_preflight:execution_preflight()
    else:run()
