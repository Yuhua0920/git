"""T16F frozen pilot-driven Hard/Convex/Affine comparison; no new channel."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.linalg import cholesky
from src.statistics_mixing import objective_moments,hard_selection,convex_mixing,geometry,gain,scientific_gate
from src.pilot_cv import causal_windows,deployment
from src.modular_correction import ScalarComposition
from src.cdl_benchmark import array_hash,observation_geometry,frames
from scripts.run_cdl_mismatch import load,save,sha,csvsave,history,environments
from scripts.run_pilot_cv import frozen_operators,cv_family,fit_unlabelled,trajectory_energy,readcsv
from scripts.run_modular_correction import cached_H,cached_inputs,target_statistics
from scripts.run_lowrank_filter import statistics

OUT=ROOT/'outputs';RESULT=OUT/'t16f_results.json'
T16E_COMMIT='0b7a90d3cfa5fee82f157f7dd66e5bb0aaa5f7b8'
SNRS=(5,10,20);TARGETS=('CDL-C-75','CDL-B-120')
FOLD_HASH='16594a306157e0b6ac114100ddebbba262f8e4205a98c8f48726e5519e55ab18'
EXPERIMENT_TYPE='closest-classical-baseline gate comparing pilot-driven discrete LMMSE statistics selection, convex mixing, and affine mismatch-correction composition'
SOURCES=('src/statistics_mixing.py','scripts/run_statistics_selection.py',
    'scripts/plot_statistics_selection.py','scripts/audit_statistics_selection.py',
    'tests/test_statistics_selection.py','tests/test_statistics_selection_math.py',
    'tests/test_statistics_selection_figures.py','docs/statistics_selection_protocol.md')


def accepted():
    subprocess.check_call(['git','merge-base','--is-ancestor',T16E_COMMIT,'HEAD'],cwd=ROOT)
    subject=subprocess.check_output(['git','show','-s','--format=%s',T16E_COMMIT],cwd=ROOT,text=True).strip()
    if subject!='feat: add analytical sparse-pilot module-coordinate fitting gate':raise ValueError('accepted T16E commit required')
    e=load(OUT/'t16e_pilot_cv_results.json')
    if not e['all_checks_passed'] or e['completion_state']!='completed':raise ValueError('accepted T16E result required')
    for p,h in {**e['source_sha256'],**e['frozen_history_sha256'],**e['numeric_artifact_sha256'],
        **e['prior_failed_attempt']['artifact_sha256']}.items():
        if sha(ROOT/p)!=h:raise ValueError(f'historical file drift:{p}')
    for p in [*e['source_sha256'],'outputs/t16e_pilot_cv_results.json']:
        blob=subprocess.check_output(['git','show',f'{T16E_COMMIT}:{p}'],cwd=ROOT)
        if blob.replace(b'\r\n',b'\n')!=(ROOT/p).read_bytes().replace(b'\r\n',b'\n'):
            raise ValueError('historical snapshot drift')
    if e['pilot_fold_definition']['sha256']!=FOLD_HASH:raise ValueError('frozen fold drift')
    return e


def build_candidates():
    """Recreate only original frozen filters; no training or target fit."""
    start=time.perf_counter();e=accepted();arrays,_=frozen_operators()
    _,source,development,variances=statistics();old=load(OUT/'t16d_modular_results.json')
    candidate={};proof=[]
    for s in SNRS:
        ws=source.prepare(variances[s]).weight
        if array_hash(ws)!=array_hash(arrays[f'Wsource_{s}']):raise ValueError('source W hash drift')
        for label,name in (('P','CDL-B-30'),('M','CDL-C-120')):
            wt=development[name].prepare(variances[s]).weight
            prior=next(r for r in old['module_provenance']['development_operator_regression'] if r['shift']==name and r['snr_db']==s)
            if array_hash(wt)!=prior['target_W_sha256'] or array_hash(wt-ws)!=array_hash(arrays[f'DeltaW{label}_{s}']):
                raise ValueError('frozen candidate/module hash drift')
            candidate[f'W{label}_{s}']=wt
        proof.append(dict(snr_db=s,sigma2=variances[s],candidate_order=['W0','WP','WM'],
            W_sha256=dict(W0=array_hash(ws),WP=array_hash(candidate[f'WP_{s}']),WM=array_hash(candidate[f'WM_{s}'])),
            K_sha256={name:array_hash(arrays[f'K{label}_{s}']) for name,label in (('K0','Source'),('KP','P'),('KM','M'))},
            candidate_filters_T16D_exact=True,cross_pilot_T16E_exact=True))
    return candidate,proof,time.perf_counter()-start


def frozen_candidates():
    p=load(OUT/'t16f_preflight.json');arrays,_=frozen_operators();info=p['candidate_artifact']
    if sha(ROOT/info['path'])!=info['sha256']:raise ValueError('candidate file changed')
    with np.load(ROOT/info['path'],allow_pickle=False) as f:new={k:f[k].copy(order='K') for k in f.files}
    if any(array_hash(new[k])!=h for k,h in info['array_sha256'].items()):raise ValueError('candidate arrays changed')
    arrays.update(new)
    for a in arrays.values():a.flags.writeable=False
    return arrays,p


def preflight():
    path=OUT/'t16f_frozen_candidates.npz'
    if path.exists():raise ValueError('existing candidate artifact protected')
    candidate,proof,seconds=build_candidates();np.savez_compressed(path,**candidate);path.chmod(0o444)
    save(OUT/'t16f_preflight.json',dict(all_checks_passed=True,accepted_T16E_commit=T16E_COMMIT,
        candidate_provenance=proof,pilot_fold_sha256=FOLD_HASH,
        candidate_artifact=dict(path='outputs/t16f_frozen_candidates.npz',sha256=sha(path),
            array_sha256={k:array_hash(v) for k,v in candidate.items()}),
        original_T16E_operator_file_sha256=sha(OUT/'t16e_frozen_operators.npz'),
        source_sha256={p:sha(ROOT/p) for p in SOURCES},startup_pytest_summary=(ROOT/'.pytest_cache/t16f_startup_pytest.txt').read_text(encoding='utf8').strip().splitlines()[-1],
        hard_tie_order=['W0','WP','WM'],source_means_only=True,
        oracle_controls_use_source_means=True,affine_oracle_matches_deployed_model_class=True,
        b120_affine_oracle_reuses_t16d_coefficients=True,
        c75_affine_oracle_recomputed_as_two_module_target_weighted_optimum=True,
        generator_calls=0,new_channel_realizations=0,statistics_refitted=False,
        runtime=dict(candidate_verification_and_construction_seconds=seconds)))
    print('T16F frozen candidates / original folds / source means / confirmed fair oracle protocol PASS',flush=True)


def execution_preflight():
    arrays,p=frozen_candidates();candidate,proof,seconds=build_candidates()
    if proof!=p['candidate_provenance'] or any(not np.array_equal(arrays[k],v) for k,v in candidate.items()):
        raise ValueError('frozen candidate reconstruction drift')
    save(OUT/'t16f_execution_preflight.json',dict(all_checks_passed=True,
        source_sha256={p:sha(ROOT/p) for p in SOURCES},
        candidate_preflight_sha256=sha(OUT/'t16f_preflight.json'),
        candidate_artifact_sha256=p['candidate_artifact']['sha256'],pilot_fold_sha256=FOLD_HASH,
        candidate_provenance=proof,candidate_verification_seconds=seconds))
    print('T16F frozen candidate recheck and current execution-source snapshot PASS',flush=True)


def fit_primary(pilots,arrays,s):
    """Hard and Convex only; label/H/target/oracle are not estimator inputs."""
    cv=cv_family(arrays,s);result=[];timing=dict(hard_fitting_seconds=0.,convex_fitting_seconds=0.,
        objective_seconds=0.,hard_data_seconds=0.,convex_data_seconds=0.)
    for window in (1,5):
        for j,values in causal_windows(pilots,window):
            tick=time.perf_counter();moments=objective_moments(cv,values);timing['objective_seconds']+=time.perf_counter()-tick
            for method,solver in (('hard',hard_selection),('convex',convex_mixing)):
                tick=time.perf_counter();coeff,diag=solver(*moments);timing[f'{method}_fitting_seconds']+=time.perf_counter()-tick
                tick=time.perf_counter()
                if method=='hard':
                    weight=arrays[('Wsource','WP','WM')[diag['selected_index']]+f'_{s}']
                    prediction=arrays['source_mean_d']+(values[-1]-arrays['source_mean_p'])@weight.T
                else:
                    prediction=deployment(values[-1],arrays['source_mean_p'],arrays['source_mean_d'],arrays[f'Wsource_{s}'],
                        arrays[f'DeltaWP_{s}'],arrays[f'DeltaWM_{s}'],coeff,('P','M'))
                timing[f'{method}_data_seconds']+=time.perf_counter()-tick
                if not np.isfinite(prediction).all():raise ValueError('nonfinite primary estimate')
                result.append(dict(method=method,window=window,frame_id=j,window_start=j-window+1,
                    coefficients=coeff,prediction=prediction,diagnostics=diag))
    return result,timing


def with_reporting_label(estimates,label):return [dict(**r,shift=label) for r in estimates]


def aggregate(frame_rows):
    rows=[];trajectory=[]
    keys=list(dict.fromkeys((r['role'],r['shift'],r['snr_db'],r['window'],r['method']) for r in frame_rows))
    for role,name,s,window,method in keys:
        selected=[r for r in frame_rows if (r['role'],r['shift'],r['snr_db'],r['window'],r['method'])==(role,name,s,window,method)]
        ids=range(10) if window==1 else range(4,10);tr=[]
        for i in range(500,600):
            v=sorted((r for r in selected if r['trajectory_id']==i),key=lambda x:x['frame_id'])
            if [r['frame_id'] for r in v]!=list(ids):raise ValueError('paired causal frame subset failed')
            error=trajectory_energy(r['error_energy'] for r in v);energy=trajectory_energy(r['truth_energy'] for r in v)
            tr.append(dict(role=role,shift=name,snr_db=s,window=window,subset='all10' if window==1 else 'last6',
                method=method,trajectory_id=i,frames=len(v),error_energy=error,truth_energy=energy,nmse_linear=error/energy))
        trajectory.extend(tr);error=sum(r['error_energy'] for r in tr);energy=sum(r['truth_energy'] for r in tr)
        rows.append(dict(role=role,shift=name,snr_db=s,window=window,subset='all10' if window==1 else 'last6',
            method=method,trajectories=100,frames=100*len(ids),error_energy=error,truth_energy=energy,
            nmse_linear=error/energy,nmse_db=float(10*np.log10(error/energy)),
            trajectory_nmse_mean=float(np.mean([r['nmse_linear'] for r in tr])),
            trajectory_nmse_std=float(np.std([r['nmse_linear'] for r in tr],ddof=1)),metric_type='actual_validation'))
    return rows,trajectory


def verify_replay(arrays,e):
    """Affine and original input exact gate; finishes before new primary work."""
    tick=time.perf_counter();old=load(OUT/'t16d_modular_results.json');c=load(OUT/'t16c_heldout_results.json')
    paired={(r['shift'],r['trajectory_id']):r for r in c['paired_input_evidence']}
    coeff={(r['shift'],int(r['snr_db']),int(r['trajectory_id']),int(r['frame_id']),int(r['window'])):r
        for r in readcsv(OUT/'t16e_pilot_cv_coefficients.csv')}
    frame={(r['shift'],int(r['snr_db']),int(r['trajectory_id']),int(r['frame_id']),r['method']):r
        for r in readcsv(OUT/'t16e_pilot_cv_frame_metrics.csv')}
    g=observation_geometry();directory=ROOT/'.pytest_cache/t16f_affine_replay';directory.mkdir(exist_ok=True)
    if any(directory.glob('*.npz')):raise ValueError('existing evaluated replay cache protected')
    references=[];geometry_rows=[];cache={};digest=hashlib.sha256();fit_seconds=0.;data_seconds=0.
    for name in TARGETS:
        info=old['validation_replay_gate']['cache'][name]
        if sha(ROOT/info['path'])!=info['file_sha256']:raise ValueError('historical H cache changed')
        for i in range(500,600):
            original=e['historical_replay_gate']['cache'][f'{name}/{i}'];path=ROOT/original['path']
            if sha(path)!=original['sha256']:raise ValueError('T16E derived pilot cache changed')
            with np.load(path,allow_pickle=False) as f:pilots=f['pilots'];errors=f['errors'];energy=f['truth_energy']
            if array_hash(pilots)!=original['pilots_sha256']:raise ValueError('pilot array drift')
            h=cached_H(info,i-500);inputs=cached_inputs(info,i)
            if array_hash(h)!=paired[name,i]['H_sha256']:raise ValueError('STOP: original H hash mismatch')
            input_digest=hashlib.sha256();truth_frames=frames(h)
            for j,truth in enumerate(truth_frames):
                if float(np.sum(abs(truth.ravel()[g.data])**2))!=float(energy[j]):raise ValueError('truth energy drift')
                hashes=[array_hash(a) for a in (truth,inputs['bits'][j],inputs['symbols'][j],inputs['z'][j])]
                for si,s in enumerate(SNRS):
                    for value in (*hashes,array_hash(inputs['y'][j,si])):input_digest.update(value.encode('ascii'))
            if input_digest.hexdigest()!=paired[name,i]['paired_inputs_sha256']:raise ValueError('STOP: paired digest drift')
            affine_errors=np.empty((3,16));affine_coeff=np.empty((3,16,2))
            for si,s in enumerate(SNRS):
                # No label/target/oracle/H is passed to original primary estimator.
                estimates,ft,dt=fit_unlabelled(pilots[si],arrays,s);fit_seconds+=ft;data_seconds+=dt
                for k,r in enumerate(estimates):
                    j=r['frame_id'];window=r['window'];prior=coeff[name,s,i,j,window]
                    a,b=map(float,r['coefficients']);di=r['diagnostics']
                    comparisons=dict(alpha_hat=a,beta_hat=b,gram_condition_number=di['condition_number'],
                        normal_residual=di['normal_equation_relative_residual'],source_cv_loss=di['source_cv_loss'],fitted_cv_loss=di['fitted_cv_loss'])
                    if any(value!=float(prior[key]) for key,value in comparisons.items()):raise ValueError('STOP: T16E affine coefficients/diagnostics not exact')
                    error=float(np.sum(abs(r['prediction']-truth_frames[j].ravel()[g.data])**2))
                    if error!=float(frame[name,s,i,j,f'pilotCV-{window}']['error_energy']):raise ValueError('STOP: T16E affine frame NMSE not exact')
                    affine_errors[si,k]=error;affine_coeff[si,k]=r['coefficients']
                    digest.update(array_hash(r['coefficients']).encode());digest.update(array_hash(r['prediction']).encode())
                    geometry_rows.append(dict(shift=name,snr_db=s,trajectory_id=i,frame_id=j,window=window,
                        window_start=j-window+1,method='affine',selected_candidate=None,cv_loss=di['fitted_cv_loss'],**geometry(r['coefficients'])))
                    for method,value in (('affine',error),('source',float(errors[si,0,j])),('matched',float(errors[si,1,j]))):
                        if method!='affine' and value!=float(frame[name,s,i,j,method]['error_energy']):raise ValueError('STOP: baseline frame metric drift')
                        references.append(dict(role='primary',shift=name,snr_db=s,trajectory_id=i,frame_id=j,
                            window=window,method=method,error_energy=value,truth_energy=float(energy[j])))
            newpath=directory/f'{name}_{i}.npz'
            np.savez_compressed(newpath,pilots=pilots,affine_errors=affine_errors,affine_coefficients=affine_coeff,
                baseline_errors=errors,truth_energy=energy);newpath.chmod(0o444)
            cache[f'{name}/{i}']=dict(path=str(newpath.relative_to(ROOT)).replace('\\','/'),sha256=sha(newpath))
            if (i-499)%25==0:print(f'{name} exact T16E affine/input replay {i-499}/100',flush=True)
    if digest.hexdigest()!=e['primary_coefficients_predictions_sha256']:raise ValueError('STOP: original affine prediction digest not exact')
    pooled,tr=aggregate(references)
    original_tr=readcsv(OUT/'t16e_pilot_cv_trajectory_metrics.csv')
    historical_tr={(r['shift'],int(r['snr_db']),int(r['trajectory_id']),r['subset'],r['method']):r for r in original_tr}
    for r in tr:
        method=f"pilotCV-{r['window']}" if r['method']=='affine' else r['method']
        prior=historical_tr[r['shift'],r['snr_db'],r['trajectory_id'],r['subset'],method]
        if any(r[k]!=float(prior[k]) for k in ('error_energy','truth_energy','nmse_linear')):raise ValueError('STOP: T16E trajectory energy reduction drift')
    for r in pooled:
        method=f"pilotCV-{r['window']}" if r['method']=='affine' else r['method']
        prior=next(x for x in e['validation_nmse'] if (x['shift'],x['snr_db'],x['subset'],x['method'])==(r['shift'],r['snr_db'],r['subset'],method))
        if any(r[k]!=prior[k] for k in ('error_energy','truth_energy','nmse_linear','nmse_db')):raise ValueError('STOP: T16E pooled metric not exact')
    proof=dict(all_checks_passed=True,all_200_H_hashes_exact=True,all_200_original_paired_digests_exact=True,
        original_input_file_and_tensor_hashes_exact=True,T16E_derived_pilot_cache_hashes_exact=True,
        all_9600_affine_coefficients_and_diagnostics_exact=True,affine_prediction_digest_exact=True,
        affine_frame_errors_exact=True,all_3600_reference_trajectory_rows_exact=True,all_36_reference_pooled_rows_exact=True,
        historical_full_tensor_hashes_available=False,historical_separate_bits_noise_Y_hashes_available=False,
        historical_combined_input_digests_available=True,original_cache_reused=True,
        generator_calls=0,new_validation_realizations_generated=False,cache=cache,
        runtime=dict(historical_gate_seconds=time.perf_counter()-tick,affine_fitting_seconds=fit_seconds,affine_data_seconds=data_seconds))
    save(OUT/'t16f_historical_replay.json',proof)
    print('T16F complete historical input/9600 affine/digest/frame/trajectory/pooled EXACT; new primary evaluation permitted',flush=True)
    return references,geometry_rows,proof


def oracle_models(arrays):
    """Target statistics are used only here; same filter class and source means."""
    tick=time.perf_counter();old=load(OUT/'t16d_modular_results.json')
    targets=target_statistics(load(OUT/'t16c_heldout_results.json'));controls={};proof=[]
    for name in TARGETS:
        for s in SNRS:
            target=targets[name];sigma2=old['source_noise_variances'][str(s)]
            delta=target.prepare(sigma2).weight-arrays[f'Wsource_{s}']
            lower=cholesky(target.rpp+sigma2*np.eye(64),lower=True)
            modules=[arrays[f'DeltaWP_{s}'],arrays[f'DeltaWM_{s}']]
            fit=ScalarComposition.fit(delta,modules,lower)
            gram=np.array(fit.diagnostics['normal_equation_gram']);rhs=np.array(fit.diagnostics['normal_equation_rhs'])
            baseline=fit.diagnostics['correction_energy_per_frame']
            frozen=next(r for r in old['alpha_beta'] if r['shift']==name and r['snr_db']==s)
            historical_coeff=np.array([frozen['alpha'] or 0.,frozen['beta']])
            if name=='CDL-B-120':
                if not np.allclose(fit.coefficients,historical_coeff,rtol=1e-12,atol=1e-12):raise ValueError('B120 original oracle regression failed')
                affine=historical_coeff
            else:affine=fit.coefficients
            excess=lambda a:float(np.vdot((delta-a[0]*modules[0]-a[1]*modules[1])@lower,
                                         (delta-a[0]*modules[0]-a[1]*modules[1])@lower).real)
            original_excess=excess(historical_coeff)
            if name=='CDL-C-75' and excess(affine)>original_excess+1e-11*max(1.,original_excess):
                raise ValueError('C75 two-module oracle worse than frozen single-module reference')
            for method,coeff,diag in [('hard-oracle',*hard_selection(gram,rhs,baseline)),
                ('convex-oracle',*convex_mixing(gram,rhs,baseline)),('affine-oracle',affine,fit.diagnostics)]:
                controls[name,s,method]=np.asarray(coeff)
                proof.append(dict(shift=name,snr_db=s,method=method,alpha=float(coeff[0]),beta=float(coeff[1]),
                    lambda0=float(1-coeff[0]-coeff[1]),weighted_filter_excess_energy_per_frame=excess(coeff),
                    oracle_objective='target-weighted centered filter error; no mean advantage',source_means=True,
                    historical_single_module_excess_energy=original_excess if name=='CDL-C-75' else None,
                    b120_coefficient_recompute_max_abs=float(np.max(abs(fit.coefficients-historical_coeff))) if name=='CDL-B-120' else None,
                    affine_real_coordinate_count=2 if method=='affine-oracle' else None,
                    optimization_diagnostics=diag))
    return controls,proof,time.perf_counter()-tick


def oracle_prediction(pilot,arrays,s,coeff,method):
    if method=='hard-oracle':
        index=0 if np.array_equal(coeff,[0.,0.]) else 1 if np.array_equal(coeff,[1.,0.]) else 2 if np.array_equal(coeff,[0.,1.]) else None
        if index is None:raise ValueError('hard oracle must choose one frozen candidate')
        weight=arrays[('Wsource','WP','WM')[index]+f'_{s}']
        return arrays['source_mean_d']+(pilot-arrays['source_mean_p'])@weight.T
    return deployment(pilot,arrays['source_mean_p'],arrays['source_mean_d'],arrays[f'Wsource_{s}'],
        arrays[f'DeltaWP_{s}'],arrays[f'DeltaWM_{s}'],coeff,('P','M'))


def summarize_geometry(raw):
    weights=[];outside=[];convex=[];selection=[];per_trajectory=[]
    for name in TARGETS:
        for s in SNRS:
            for window in (1,5):
                for method in ('hard','convex','affine'):
                    v=[r for r in raw if (r['shift'],r['snr_db'],r['window'],r['method'])==(name,s,window,method)]
                    count=100*(10 if window==1 else 6)
                    if len(v)!=count:raise ValueError('weight window count failed')
                    key=dict(shift=name,snr_db=s,window=window,method=method,samples=count)
                    for field in ('alpha','beta','lambda0'):
                        values=np.array([r[field] for r in v])
                        weights.append(dict(**key,quantity=field,median=float(np.median(values)),q25=float(np.quantile(values,.25)),
                            q75=float(np.quantile(values,.75)),mean=float(values.mean()),std=float(values.std(ddof=1)),
                            minimum=float(values.min()),maximum=float(values.max())))
                    if method=='affine':
                        outside.append(dict(**key,**{f'{field}_fraction':sum(r[field] for r in v)/count for field in
                            ('outside_alpha','outside_beta','outside_lambda0','outside_any')}))
                    if method=='convex':
                        convex.append(dict(**key,**{f'{kind}_fraction':sum(r['active_kind']==kind for r in v)/count for kind in ('interior','edge','vertex')}))
                    if method=='hard':
                        counts={c:sum(r['selected_candidate']==c for r in v) for c in ('W0','WP','WM')};modal={c:0 for c in counts}
                        for i in range(500,600):
                            trajectory=[r for r in v if r['trajectory_id']==i];n=len(trajectory)
                            cs={c:sum(r['selected_candidate']==c for r in trajectory) for c in counts}
                            modal[min(cs,key=lambda c:-cs[c])]+=1
                            for c in counts:per_trajectory.append(dict(**key,trajectory_id=i,candidate=c,windows=n,count=cs[c],fraction=cs[c]/n))
                        for c in counts:
                            tv=[r['fraction'] for r in per_trajectory if (r['shift'],r['snr_db'],r['window'],r['candidate'])==(name,s,window,c)]
                            selection.append(dict(**key,candidate=c,count=counts[c],frame_window_fraction=counts[c]/count,
                                trajectory_mean_fraction=float(np.mean(tv)),trajectory_std_fraction=float(np.std(tv,ddof=1)),
                                trajectory_modal_count=modal[c],trajectory_modal_fraction=modal[c]/100))
    return weights,outside,convex,selection,per_trajectory


def run():
    e=accepted();arrays,pre=frozen_candidates()
    execution=load(OUT/'t16f_execution_preflight.json')
    if sha(OUT/'t16f_preflight.json')!=execution['candidate_preflight_sha256'] or any(sha(ROOT/p)!=h for p,h in execution['source_sha256'].items()):
        raise ValueError('execution-preflight drift')
    if RESULT.exists() or (OUT/'t16f_selection_mixing.csv').exists():raise ValueError('existing actual results protected')
    start=time.perf_counter();before=history();env=environments()
    frame_rows,raw,gate=verify_replay(arrays,e)
    if not gate['all_checks_passed']:raise ValueError('STOP: failed historical gate')
    controls,oracle_proof,oracle_seconds=oracle_models(arrays)
    g=observation_geometry();timing={};oracle_data_seconds=0.;label_exact=True;small=[]
    old=load(OUT/'t16d_modular_results.json')
    for name in TARGETS:
        for i in range(500,600):
            info=gate['cache'][f'{name}/{i}'];path=ROOT/info['path']
            if sha(path)!=info['sha256']:raise ValueError('verified replay cache drift')
            with np.load(path,allow_pickle=False) as f:pilots=f['pilots'];energy=f['truth_energy']
            h=cached_H(old['validation_replay_gate']['cache'][name],i-500);truth=frames(h)
            for si,s in enumerate(SNRS):
                inferred,clock=fit_primary(pilots[si],arrays,s)
                for key,value in clock.items():timing[key]=timing.get(key,0.)+value
                reports=with_reporting_label(inferred,name);changed=with_reporting_label(inferred,'tampered-report-label')
                if any(not np.array_equal(a['coefficients'],b['coefficients']) or not np.array_equal(a['prediction'],b['prediction']) for a,b in zip(reports,changed)):
                    raise ValueError('report label leakage')
                if i==500:small.append(dict(shift=name,snr_db=s,trajectory_id=i,new_primary_estimates=len(inferred),all_finite=True))
                for r in reports:
                    j=r['frame_id'];window=r['window'];method=r['method'];diag=r['diagnostics']
                    frame_rows.append(dict(role='primary',shift=name,snr_db=s,trajectory_id=i,frame_id=j,
                        window=window,method=method,error_energy=float(np.sum(abs(r['prediction']-truth[j].ravel()[g.data])**2)),truth_energy=float(energy[j])))
                    raw.append(dict(shift=name,snr_db=s,trajectory_id=i,frame_id=j,window=window,window_start=j-window+1,
                        method=method,selected_candidate=diag.get('selected_candidate'),cv_loss=diag['fitted_cv_loss'],**geometry(r['coefficients'])))
                # Oracle inputs are confined to post-primary evaluation below.
                for method in ('hard-oracle','convex-oracle','affine-oracle'):
                    coeff=controls[name,s,method]
                    for j in range(10):
                        tick=time.perf_counter();prediction=oracle_prediction(pilots[si,j],arrays,s,coeff,method);oracle_data_seconds+=time.perf_counter()-tick
                        error=float(np.sum(abs(prediction-truth[j].ravel()[g.data])**2))
                        for window in ((1,5) if j>=4 else (1,)):
                            frame_rows.append(dict(role='oracle_control',shift=name,snr_db=s,trajectory_id=i,frame_id=j,
                                window=window,method=method,error_energy=error,truth_energy=float(energy[j])))
            if (i-499)%25==0:print(f'{name} paired Hard/Convex/fair oracle {i-499}/100',flush=True)
    pooled,tr=aggregate(frame_rows);primary=[r for r in pooled if r['role']=='primary'];oracle_rows=[r for r in pooled if r['role']=='oracle_control']
    for r in primary:
        ref=lambda m:next(v['nmse_db'] for v in primary if (v['shift'],v['snr_db'],v['window'],v['method'])==(r['shift'],r['snr_db'],r['window'],m))
        r.update(gain(ref('hard'),ref('convex'),ref('affine')))
    for r in oracle_rows:
        p=next(v for v in oracle_proof if (v['shift'],v['snr_db'],v['method'])==(r['shift'],r['snr_db'],r['method']))
        deployed=next(v for v in primary if (v['shift'],v['snr_db'],v['window'],v['method'])==(r['shift'],r['snr_db'],r['window'],r['method'].split('-')[0]))
        r.update(source_means=True,alpha=p['alpha'],beta=p['beta'],lambda0=p['lambda0'],
            weighted_filter_excess_energy_per_frame=p['weighted_filter_excess_energy_per_frame'],
            deployed_minus_oracle_nmse_db=deployed['nmse_db']-r['nmse_db'])
    weights,outside,convex,selection,per_trajectory=summarize_geometry(raw)
    gains={(r['shift'],r['snr_db']):{k:r[k] for k in ('affine_vs_hard_gain_db','affine_vs_convex_gain_db')}
        for r in primary if r['method']=='affine' and r['window']==5 and r['snr_db'] in (10,20)}
    fractions={(r['shift'],r['snr_db']):r['outside_any_fraction'] for r in outside if r['window']==5}
    science=scientific_gate(gains,fractions)
    files={'t16f_selection_mixing.csv':primary,'t16f_weight_geometry.csv':raw,'t16f_oracle_controls.csv':oracle_rows,
        't16f_trajectory_metrics.csv':tr,'t16f_frame_metrics.csv':frame_rows,'t16f_weight_summary.csv':weights,
        't16f_selection_trajectory_frequencies.csv':per_trajectory}
    for filename,rows in files.items():csvsave(OUT/filename,rows)
    if any(sha(ROOT/p)!=h for p,h in before.items()):raise ValueError('historical file changed')
    result=dict(task='T16F',experiment_type=EXPERIMENT_TYPE,literature_gate_passed=True,
        closest_classical_baseline_gate=True,proposed_method=False,accepted_T16E_commit=T16E_COMMIT,
        literature_provenance=dict(title='LMMSE channel estimation for OFDM systems with channel correlation function selection',
            year=2021,doi='10.1049/cmu2.12250',source='https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cmu2.12250',
            checked_sections=['Introduction contribution2','Section5.3'],exact_paper_reproduction=False),
        candidate_provenance=pre['candidate_provenance'],pilot_fold_sha256=FOLD_HASH,
        hard_selection=dict(tie_order=['W0','WP','WM'],candidate_count=3,criterion='same held-out cross-pilot loss'),
        convex_mixing=dict(global_active_set=True,constraints='alpha>=0,beta>=0,alpha+beta<=1',black_box_optimizer=False),
        affine_composition=dict(original_T16E_exact=True,unrestricted_real_coefficients=True),
        selection_frequencies=selection,convex_weights=[r for r in weights if r['method']=='convex'],
        convex_active_sets=convex,affine_weights=[r for r in weights if r['method']=='affine'],outside_simplex_fraction=outside,
        validation_nmse=primary,oracle_controls=oracle_rows,oracle_coefficient_optimization=oracle_proof,
        affine_vs_hard_gain=[{k:r[k] for k in ('shift','snr_db','window','affine_vs_hard_gain_db')} for r in primary if r['method']=='affine'],
        affine_vs_convex_gain=[{k:r[k] for k in ('shift','snr_db','window','affine_vs_convex_gain_db')} for r in primary if r['method']=='affine'],
        **science,oracle_controls_use_source_means=True,affine_oracle_matches_deployed_model_class=True,
        b120_affine_oracle_reuses_t16d_coefficients=True,c75_affine_oracle_recomputed_as_two_module_target_weighted_optimum=True,
        historical_t16d_c75_single_module_oracle_preserved=True,
        historical_t16d_single_module_oracle_reference=dict(path='outputs/t16d_modular_results.json',sha256=sha(OUT/'t16d_modular_results.json'),
            rows=[r for r in old['alpha_beta'] if r['shift']=='CDL-C-75'],used_for_primary_or_oracle_comparison=False),
        absolute_theoretical_nmse_available=False,exact_weighted_filter_excess_available=True,
        matched_baseline_uses_original_target_means=True,primary_source_means_only=True,
        primary_estimator_uses_environment_label=False,reporting_label_mutation_exact=label_exact,
        historical_replay_gate_passed=True,historical_replay_gate=gate,actual_small_inference=small,
        deterministic_validation_replay_used=True,new_validation_realizations_generated=False,
        methods_evaluated_on_same_original_realizations=True,historical_full_tensor_hashes_available=False,
        generator_calls=0,new_channel_realizations=0,new_seed_ids=0,statistics_refitted=False,
        lowrank_compression=False,BER_evaluations=0,detector_calls=0,new_model_trained=False,
        formal_test_used=False,formal_test_sealed=True,coefficient_clipping=False,
        complexity=dict(real_coordinate_count=2,hard_candidate_loss_count=3,
            shared_pilot_objective_complex_MAC_per_frame=9216,five_frame_objective_complex_MAC_per_window=46080,
            hard_data_complex_MAC_per_frame=3968*64,convex_data_complex_MAC_per_frame=3*3968*64,
            affine_data_complex_MAC_per_frame=3*3968*64,convex_affine_scaling_complex_real_multiplies=2*3968,
            standalone_costs_include_pilot_objective=True,rolling_objective_reuse=False,dense_implementation=True),
        runtime=dict(**gate['runtime'],**timing,oracle_optimization_seconds=oracle_seconds,
            oracle_data_seconds=oracle_data_seconds,candidate_verification_and_construction_seconds=pre['runtime']['candidate_verification_and_construction_seconds'],
            total_numeric_wall_seconds=time.perf_counter()-start,channel_generation_seconds=0.,statistics_fitting_seconds=0.),
        source_sha256={p:sha(ROOT/p) for p in SOURCES},frozen_history_sha256=before,environment_before=env,
        numeric_artifact_sha256={f'outputs/{p}':sha(OUT/p) for p in files},
        preflight_failure_recovery=dict(path='outputs/t16f_preflight_failures',
            formal_performance_started_before_recovery=False,historical_module_restored_exact=True,
            evidence=load(OUT/'t16f_preflight_failures/recovery_provenance.json'),
            artifact_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in
                sorted((OUT/'t16f_preflight_failures').iterdir()) if p.is_file()}),
        all_checks_passed=False,completion_state='actual validation completed; final audit pending',
        warnings=['Hard/Convex/Affine are pilot-driven model-class comparisons, not new neural or novelty claims.',
            'Preflight syntax error and historical module filename collision were caught before performance; logs/new source archived, old module restored exact and new module renamed statistics_mixing.py.',
            'Five causal frames use correlated composite loss; window IQR and trajectory std are descriptive, no CI.',
            'Oracle controls use source means; matched alone retains original target-mean baseline.',
            'Oracle minimizes centered target-weighted filter excess; not absolute theoretical NMSE or mean-bias optimization.',
            'Original separate bits/noise/Y historical hashes absent; no historical hash evidence invented.',
            'All affine weights/extrema retained; no clipping or discarded samples; dense filters are not called lightweight.'])
    save(RESULT,result);print(science,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preflight',action='store_true')
    p.add_argument('--verify-preflight',action='store_true');args=p.parse_args()
    if args.preflight:preflight()
    elif args.verify_preflight:execution_preflight()
    else:run()
