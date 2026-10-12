"""T16G development-only factorization with frozen T16F coordinates/replay."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.linalg import cholesky
from src.lowrank_modules import (WeightedModule,RankFactors,RANKS,
    factored_affine_predict,compression_metrics,minimum_practical_rank,complexity,scientific_gate)
from src.pilot_cv import deployment
from src.cdl_benchmark import array_hash,frames,observation_geometry
from scripts.run_cdl_mismatch import load,save,sha,csvsave,history,environments
from scripts.run_statistics_selection import frozen_candidates,aggregate
from scripts.run_pilot_cv import readcsv
from scripts.run_lowrank_filter import statistics
from scripts.run_modular_correction import cached_H,cached_inputs,target_statistics

OUT=ROOT/'outputs';RESULT=OUT/'t16g_results.json'
T16F_COMMIT='5c3b9bf1dcdad77f9f76a3684154f18b117bc675'
SNRS=(5,10,20);TARGETS=('CDL-C-75','CDL-B-120')
DEVELOPMENT={'P':'CDL-B-30','M':'CDL-C-120'}
EXPERIMENT_TYPE='development-only weighted low-rank compression gate for pilot-driven affine LMMSE mismatch modules'
SOURCES=('src/lowrank_modules.py','scripts/run_lowrank_modules.py','scripts/plot_lowrank_modules.py',
    'scripts/audit_lowrank_modules.py','tests/test_lowrank_modules.py',
    'tests/test_lowrank_modules_pipeline.py','tests/test_lowrank_modules_figures.py','docs/lowrank_modules_protocol.md')
WINDOW_ORDER=tuple([(1,j) for j in range(10)]+[(5,j) for j in range(4,10)])


def accepted():
    subprocess.check_call(['git','merge-base','--is-ancestor',T16F_COMMIT,'HEAD'],cwd=ROOT)
    subject=subprocess.check_output(['git','show','-s','--format=%s',T16F_COMMIT],cwd=ROOT,text=True).strip()
    if subject!='feat: compare hard selection convex mixing and affine statistics composition':raise ValueError('accepted T16F snapshot required')
    old=load(OUT/'t16f_results.json')
    if not old['all_checks_passed'] or old['completion_state']!='completed':raise ValueError('T16F not accepted')
    hashes={**old['source_sha256'],**old['frozen_history_sha256'],**old['numeric_artifact_sha256'],
        **old['preflight_failure_recovery']['artifact_sha256']}
    for p,h in hashes.items():
        if sha(ROOT/p)!=h:raise ValueError(f'frozen history drift:{p}')
    for p in [*old['source_sha256'],'outputs/t16f_results.json']:
        blob=subprocess.check_output(['git','show',f'{T16F_COMMIT}:{p}'],cwd=ROOT)
        if blob.replace(b'\r\n',b'\n')!=(ROOT/p).read_bytes().replace(b'\r\n',b'\n'):
            raise ValueError('T16F snapshot drift')
    return old


def construct_development_factors(arrays):
    """Only source C30/B30/C120 readers; no held-out statistics reader."""
    started=time.perf_counter();old,source,development,variances=statistics()
    legacy=load(OUT/'t16d_modular_results.json');factors={};proof=[];spectrum=[];capture=[]
    for module,name in DEVELOPMENT.items():
        stats=development[name]
        for snr in SNRS:
            delta=arrays[f'DeltaW{module}_{snr}'];ws=arrays[f'Wsource_{snr}']
            prior=next(v for v in legacy['module_provenance']['development_operator_regression'] if v['shift']==name and v['snr_db']==snr)
            if array_hash(delta)!=prior['DeltaW_sha256'] or array_hash(ws)!=prior['source_W_sha256']:
                raise ValueError('T16D development module drift')
            if delta.shape!=(3968,64) or delta.dtype!=np.complex128:raise ValueError('module dimensions/dtype drift')
            covariance=stats.rpp+variances[snr]*np.eye(64)
            model=WeightedModule.build(delta,covariance)
            full=model.factors(64);restoration=float(np.linalg.norm(full.A@full.Q-delta)/np.linalg.norm(delta))
            if restoration>1e-11:raise ValueError('full-rank math sentinel failed')
            production=model.factors(16)
            factors[f'A_{module}_{snr}']=production.A;factors[f'Q_{module}_{snr}']=production.Q
            factors[f'singular_{module}_{snr}']=model.singular_values
            diagnostics=model.diagnostics()
            if diagnostics['cholesky_relative_residual']>1e-11 or diagnostics['triangular_right_solve_relative_residual']>1e-11:
                raise ValueError('factor solve sentinel failed')
            info=old['matched_statistics_provenance'][name]
            proof.append(dict(module=module,development_condition=name,snr_db=snr,sigma2=variances[snr],
                development_statistics_path=info['path'],development_statistics_sha256=info['sha256'],
                rpp_sha256=array_hash(stats.rpp),DeltaW_sha256=array_hash(delta),Cz_sha256=array_hash(covariance),
                L_sha256=array_hash(model.lower),weighted_svd_input_sha256=array_hash(delta@model.lower),
                module_hash_T16D_exact=True,heldout_statistics_used=False,diagnostics=diagnostics,
                r64_restoration_relative_residual=restoration,r64_validation_performed=False))
            energy=model.singular_values**2;total=float(np.sum(energy))
            for i,value in enumerate(model.singular_values):
                spectrum.append(dict(module=module,development_condition=name,snr_db=snr,mode=i+1,
                    singular_value=float(value),squared_energy=float(energy[i]),total_weighted_energy=total,
                    cumulative_capture=float(np.sum(energy[:i+1])/total)))
            values=[model.capture(rank) for rank in RANKS]
            if any(b<a for a,b in zip(values,values[1:])):raise ValueError('capture not monotonic')
            for rank,value in zip(RANKS,values):capture.append(dict(module=module,development_condition=name,snr_db=snr,rank=rank,capture=value))
    return factors,proof,spectrum,capture,time.perf_counter()-started


def preflight():
    accepted();arrays,_=frozen_candidates();artifact=OUT/'t16g_frozen_module_factors.npz'
    if artifact.exists():raise ValueError('existing factor artifact protected')
    factors,proof,spectrum,capture,seconds=construct_development_factors(arrays)
    metadata=dict(source_commit=T16F_COMMIT,rank_grid=list(RANKS),maximum_production_rank=16,
        dtype='complex128',development_conditions=DEVELOPMENT,construction_protocol_sha256=sha(ROOT/'docs/lowrank_modules_protocol.md'))
    np.savez_compressed(artifact,**factors,metadata_json=np.array(json.dumps(metadata,sort_keys=True)))
    artifact.chmod(0o444)
    with np.load(artifact,allow_pickle=False) as f:
        if any(not np.array_equal(f[k],v) for k,v in factors.items()):raise ValueError('factor persistence mismatch')
    csvsave(OUT/'t16g_module_spectrum.csv',spectrum)
    result=dict(all_checks_passed=True,accepted_T16F_commit=T16F_COMMIT,module_provenance=proof,
        weighted_factorization_definition='M=DeltaW L=U Sigma Vh; A=U_r Sigma_r; solve L.T Q.T=Vh_r.T',
        ordinary_transpose_right_solve=True,explicit_inverse_or_pinv=False,development_only=True,
        heldout_statistics_read_before_factor_freeze=False,factor_artifact=dict(path='outputs/t16g_frozen_module_factors.npz',
            sha256=sha(artifact),array_sha256={k:array_hash(v) for k,v in factors.items()},metadata=metadata),
        development_capture=capture,spectrum_sha256=sha(OUT/'t16g_module_spectrum.csv'),
        source_sha256={p:sha(ROOT/p) for p in SOURCES},
        startup_pytest=(ROOT/'.pytest_cache/t16g_startup_pytest.txt').read_text(encoding='utf8').strip().splitlines()[-1],
        runtime=dict(factor_construction_seconds=seconds,svd_seconds=sum(v['diagnostics']['svd_seconds'] for v in proof)),
        r64_math_only=True,generator_calls=0,coordinate_refits=0)
    save(OUT/'t16g_factor_preflight.json',result)
    print('T16G development-only factors / exact DeltaW / stable right solve / six full-rank sentinels PASS',flush=True)


def frozen_factors():
    pre=load(OUT/'t16g_factor_preflight.json');info=pre['factor_artifact']
    if sha(ROOT/info['path'])!=info['sha256'] or sha(OUT/'t16g_module_spectrum.csv')!=pre['spectrum_sha256']:
        raise ValueError('frozen factors/spectrum changed')
    with np.load(ROOT/info['path'],allow_pickle=False) as f:arrays={k:f[k].copy() for k in info['array_sha256']}
    if any(array_hash(arrays[k])!=h for k,h in info['array_sha256'].items()):raise ValueError('factor array changed')
    family={(m,s,r):RankFactors(arrays[f'A_{m}_{s}'][:,:r],arrays[f'Q_{m}_{s}'][:r])
        for m in DEVELOPMENT for s in SNRS for r in RANKS}
    return family,pre


def execution_preflight():
    """Freeze corrected execution sources; preserve original factors/preflight."""
    accepted();family,pre=frozen_factors()
    recovery=load(OUT/'t16g_preflight_schema_recovery/recovery_provenance.json')
    if sha(OUT/'t16g_frozen_module_factors.npz')!=recovery['original_factor_artifact_sha256'] or sha(OUT/'t16g_factor_preflight.json')!=recovery['original_factor_preflight_sha256']:
        raise ValueError('original factor artifacts changed during schema repair')
    save(OUT/'t16g_execution_preflight.json',dict(all_checks_passed=True,
        source_sha256={p:sha(ROOT/p) for p in SOURCES},
        original_factor_preflight_sha256=sha(OUT/'t16g_factor_preflight.json'),
        original_factor_artifact_sha256=pre['factor_artifact']['sha256'],
        factors_reconstructed_or_refitted=False,coordinate_refits=0,
        schema_serialization_preflight_summary=(ROOT/'.pytest_cache/t16g_schema_recovery_pytest.txt').read_text(encoding='utf8').strip().splitlines()[-1]))
    print('T16G original frozen factors unchanged; corrected execution-source snapshot PASS',flush=True)


def heldout_diagnostic(arrays,family,pre):
    """Target covariance only after the independent factor file/hash freeze."""
    if sha(ROOT/pre['factor_artifact']['path'])!=pre['factor_artifact']['sha256']:raise ValueError('factor freeze gate failed')
    targets=target_statistics(load(OUT/'t16c_heldout_results.json'))
    variances=load(OUT/'t16d_modular_results.json')['source_noise_variances'];rows=[]
    for name in TARGETS:
        for s in SNRS:
            lower=cholesky(targets[name].rpp+variances[str(s)]*np.eye(64),lower=True)
            for module in DEVELOPMENT:
                weighted=arrays[f'DeltaW{module}_{s}']@lower;total=float(np.linalg.norm(weighted)**2)
                for rank in RANKS:
                    factor=family[module,s,rank];error=weighted-factor.A@(factor.Q@lower)
                    rows.append(dict(module=module,shift=name,snr_db=s,rank=rank,
                        heldout_relative_residual=float(np.linalg.norm(error)**2/total),
                        diagnostic_only=True,used_for_factor_construction=False))
    return rows


def aggregate_ranked(rows):
    pooled=[];trajectories=[]
    for rank in (0,*RANKS):
        selected=[r for r in rows if r['rank']==rank]
        if not selected:continue
        p,t=aggregate(selected)
        for group in (p,t):
            for row in group:row['rank']=rank
        pooled.extend(p);trajectories.extend(t)
    return pooled,trajectories


def attach_compression_metrics(pooled):
    """Use one complete schema for baseline and compressed pooled rows."""
    fields=('compression_penalty_db','dense_recovery_db','lowrank_recovery_db',
        'recovery_preservation','lowrank_affine_vs_convex_gain_db')
    for row in pooled:
        reference=lambda method:next(v['nmse_db'] for v in pooled if
            (v['shift'],v['snr_db'],v['window'],v['method'])==(row['shift'],row['snr_db'],row['window'],method))
        if row['method']=='lr-affine':row.update(compression_metrics(reference('source'),reference('affine'),reference('convex'),row['nmse_db']))
        else:row.update({key:None for key in fields})
    return pooled


def historical_gate(arrays,old):
    """Read-only exact replay. Load coefficients; never call PilotCV.fit."""
    start=time.perf_counter();g=observation_geometry();d=load(OUT/'t16d_modular_results.json')
    e=load(OUT/'t16e_pilot_cv_results.json');c=load(OUT/'t16c_heldout_results.json')
    pairs={(v['shift'],v['trajectory_id']):v for v in c['paired_input_evidence']}
    weights={(v['shift'],int(v['snr_db']),int(v['trajectory_id']),int(v['frame_id']),int(v['window'])):v
        for v in readcsv(OUT/'t16f_weight_geometry.csv') if v['method']=='affine'}
    references=[dict(role='primary',shift=v['shift'],snr_db=int(v['snr_db']),trajectory_id=int(v['trajectory_id']),
        frame_id=int(v['frame_id']),window=int(v['window']),method=v['method'],rank=0,
        error_energy=float(v['error_energy']),truth_energy=float(v['truth_energy']))
        for v in readcsv(OUT/'t16f_frame_metrics.csv') if v['role']=='primary']
    frame_index={(v['shift'],v['snr_db'],v['trajectory_id'],v['frame_id'],v['window'],v['method']):v for v in references}
    digest=hashlib.sha256();count=0;cache={};inference=0.
    for name in TARGETS:
        info=d['validation_replay_gate']['cache'][name]
        if sha(ROOT/info['path'])!=info['file_sha256']:raise ValueError('original H cache changed')
        for i in range(500,600):
            frozen=old['historical_replay_gate']['cache'][f'{name}/{i}'];path=ROOT/frozen['path']
            if sha(path)!=frozen['sha256']:raise ValueError('frozen T16F derived cache changed')
            with np.load(path,allow_pickle=False) as f:
                pilots=f['pilots'];coeff=f['affine_coefficients'];errors=f['affine_errors'];energy=f['truth_energy'];baseline=f['baseline_errors']
            original=e['historical_replay_gate']['cache'][f'{name}/{i}']
            if sha(ROOT/original['path'])!=original['sha256'] or array_hash(pilots)!=original['pilots_sha256']:
                raise ValueError('T16E original pilot evidence mismatch')
            h=cached_H(info,i-500);inputs=cached_inputs(info,i);truths=frames(h)
            if array_hash(h)!=pairs[name,i]['H_sha256']:raise ValueError('STOP: historical H mismatch')
            paired=hashlib.sha256()
            for j,truth in enumerate(truths):
                if float(np.sum(abs(truth.ravel()[g.data])**2))!=energy[j]:raise ValueError('truth energy changed')
                input_hashes=[array_hash(a) for a in (truth,inputs['bits'][j],inputs['symbols'][j],inputs['z'][j])]
                for si,s in enumerate(SNRS):
                    for value in (*input_hashes,array_hash(inputs['y'][j,si])):paired.update(value.encode('ascii'))
            if paired.hexdigest()!=pairs[name,i]['paired_inputs_sha256']:raise ValueError('STOP: original paired digest mismatch')
            for si,s in enumerate(SNRS):
                for k,(window,j) in enumerate(WINDOW_ORDER):
                    prior=weights[name,s,i,j,window];a=np.array([float(prior['alpha']),float(prior['beta'])])
                    if not np.array_equal(a,coeff[si,k]):raise ValueError('STOP: coefficients changed')
                    tick=time.perf_counter();prediction=deployment(pilots[si,j],arrays['source_mean_p'],arrays['source_mean_d'],
                        arrays[f'Wsource_{s}'],arrays[f'DeltaWP_{s}'],arrays[f'DeltaWM_{s}'],a,('P','M'));inference+=time.perf_counter()-tick
                    digest.update(array_hash(a).encode());digest.update(array_hash(prediction).encode())
                    error=float(np.sum(abs(prediction-truths[j].ravel()[g.data])**2))
                    if error!=errors[si,k] or error!=frame_index[name,s,i,j,window,'affine']['error_energy']:
                        raise ValueError('STOP: dense Affine frame not exact')
                    for m,b in (('source',0),('matched',1)):
                        if baseline[si,b,j]!=frame_index[name,s,i,j,window,m]['error_energy']:raise ValueError('baseline frame changed')
                    count+=1
            cache[f'{name}/{i}']=frozen
            if (i-499)%25==0:print(f'{name} T16F dense/coordinate/input exact replay {i-499}/100',flush=True)
    if count!=9600 or digest.hexdigest()!=e['primary_coefficients_predictions_sha256']:
        raise ValueError('STOP: complete dense coefficient/prediction digest mismatch')
    pooled,tr=aggregate_ranked(references)
    old_tr={(v['shift'],int(v['snr_db']),int(v['window']),v['method'],int(v['trajectory_id'])):v
        for v in readcsv(OUT/'t16f_trajectory_metrics.csv') if v['role']=='primary'}
    for v in tr:
        original=old_tr[v['shift'],v['snr_db'],v['window'],v['method'],v['trajectory_id']]
        if any(v[k]!=float(original[k]) for k in ('error_energy','truth_energy','nmse_linear')):raise ValueError('STOP: baseline trajectory not exact')
    for v in pooled:
        original=next(x for x in old['validation_nmse'] if (x['shift'],x['snr_db'],x['window'],x['method'])==(v['shift'],v['snr_db'],v['window'],v['method']))
        if any(v[k]!=original[k] for k in ('error_energy','truth_energy','nmse_linear','nmse_db')):raise ValueError('STOP: baseline pooled not exact')
    gate=dict(all_checks_passed=True,all_200_H_hashes_exact=True,all_200_paired_digests_exact=True,
        all_9600_T16F_coefficients_exact=True,coefficients_loaded_not_refitted=True,dense_prediction_digest_exact=True,
        dense_frame_errors_exact=True,all_6000_reference_trajectory_rows_exact=True,all_60_reference_pooled_rows_exact=True,
        original_replay_file_and_tensor_hashes_exact=True,historical_full_tensor_hashes_available=False,
        historical_combined_input_digests_available=True,historical_separate_bits_noise_Y_hashes_available=False,
        dense_predictions_sha256=digest.hexdigest(),cache=cache,
        runtime=dict(T16F_regression_seconds=time.perf_counter()-start,dense_affine_inference_seconds=inference))
    save(OUT/'t16g_historical_replay.json',gate)
    print('T16G complete historical gate EXACT; low-rank actual evaluation permitted',flush=True)
    return references,gate


def run():
    old=accepted();arrays,_=frozen_candidates();family,pre=frozen_factors()
    execution=load(OUT/'t16g_execution_preflight.json')
    if sha(OUT/'t16g_factor_preflight.json')!=execution['original_factor_preflight_sha256'] or any(sha(ROOT/p)!=h for p,h in execution['source_sha256'].items()):
        raise ValueError('execution preflight source drift')
    if RESULT.exists() or (OUT/'t16g_lowrank_affine.csv').exists():raise ValueError('existing actual numerical results protected')
    start=time.perf_counter();before=history();env=environments()
    rows,gate=historical_gate(arrays,old)
    if not gate['all_checks_passed']:raise ValueError('STOP: failed historical replay')
    # Factors are already frozen; target Rpp enters evaluation diagnostics only.
    heldout=heldout_diagnostic(arrays,family,pre);g=observation_geometry();d=load(OUT/'t16d_modular_results.json')
    timing={str(r):0. for r in RANKS};counts={str(r):0 for r in RANKS};small=[]
    for name in TARGETS:
        for i in range(500,600):
            info=gate['cache'][f'{name}/{i}'];path=ROOT/info['path']
            if sha(path)!=info['sha256']:raise ValueError('verified replay cache changed')
            with np.load(path,allow_pickle=False) as f:pilots=f['pilots'];coeff=f['affine_coefficients'];energy=f['truth_energy']
            truths=frames(cached_H(d['validation_replay_gate']['cache'][name],i-500))
            for si,s in enumerate(SNRS):
                for k,(window,j) in enumerate(WINDOW_ORDER):
                    a=coeff[si,k]
                    for rank in RANKS:
                        tick=time.perf_counter();prediction=factored_affine_predict(pilots[si,j],arrays['source_mean_p'],arrays['source_mean_d'],
                            arrays[f'Wsource_{s}'],a,family['P',s,rank],family['M',s,rank]);timing[str(rank)]+=time.perf_counter()-tick;counts[str(rank)]+=1
                        if not np.isfinite(prediction).all():raise ValueError('nonfinite low-rank estimate')
                        error=float(np.sum(abs(prediction-truths[j].ravel()[g.data])**2))
                        rows.append(dict(role='primary',shift=name,snr_db=s,trajectory_id=i,frame_id=j,window=window,
                            method='lr-affine',rank=rank,error_energy=error,truth_energy=float(energy[j])))
                if i==500:small.append(dict(shift=name,snr_db=s,trajectory_id=i,rank_grid=list(RANKS),all_finite=True))
            if (i-499)%25==0:print(f'{name} paired factorized ranks1/2/4/8/16 {i-499}/100',flush=True)
    pooled,tr=aggregate_ranked(rows);attach_compression_metrics(pooled)
    compressed=[v for v in pooled if v['method']=='lr-affine'];minima=[]
    for name in TARGETS:
        for s in SNRS:
            for window in (1,5):
                values={v['rank']:v['compression_penalty_db'] for v in compressed if (v['shift'],v['snr_db'],v['window'])==(name,s,window)}
                minima.append(dict(shift=name,snr_db=s,window=window,**minimum_practical_rank(values)))
    focus={rank:{(v['shift'],v['snr_db']):v for v in compressed if v['rank']==rank and v['window']==5 and v['snr_db'] in (10,20)} for rank in RANKS}
    science=scientific_gate(focus);cost=[complexity(rank) for rank in RANKS]
    files={'t16g_lowrank_affine.csv':pooled,'t16g_heldout_module_residual.csv':heldout,
        't16g_trajectory_metrics.csv':tr,'t16g_complexity.csv':cost,'t16g_frame_metrics.csv':rows}
    for filename,values in files.items():csvsave(OUT/filename,values)
    if any(sha(ROOT/p)!=h for p,h in before.items()):raise ValueError('historical file changed')
    if sha(ROOT/pre['factor_artifact']['path'])!=pre['factor_artifact']['sha256']:raise ValueError('factors changed during evaluation')
    result=dict(task='T16G',experiment_type=EXPERIMENT_TYPE,literature_gate_passed=True,compression_gate=True,proposed_method=False,
        accepted_T16F_commit=T16F_COMMIT,module_provenance=pre['module_provenance'],
        weighted_factorization_definition=pre['weighted_factorization_definition'],factor_artifact=pre['factor_artifact'],
        development_capture=pre['development_capture'],heldout_factor_residual=heldout,validation_nmse=pooled,
        compression_penalty_db=[{k:v[k] for k in ('shift','snr_db','window','rank','compression_penalty_db')} for v in compressed],
        recovery_preservation=[{k:v[k] for k in ('shift','snr_db','window','rank','recovery_preservation')} for v in compressed],
        lowrank_affine_vs_convex_gain_db=[{k:v[k] for k in ('shift','snr_db','window','rank','lowrank_affine_vs_convex_gain_db')} for v in compressed],
        minimum_practical_ranks=minima,r_025db=[{k:v[k] for k in ('shift','snr_db','window','r_025db')} for v in minima],
        r_05db=[{k:v[k] for k in ('shift','snr_db','window','r_05db')} for v in minima],**science,
        rank_grid=list(RANKS),r64_math_only=True,r64_full_validation_performed=False,
        complexity=cost,storage=[{k:v[k] for k in ('rank','factor_complex_coefficient_count','factor_complex64_bytes','factor_complex128_bytes',
            'dense_module_complex_coefficient_count','dense_module_complex64_bytes','dense_module_complex128_bytes')} for v in cost],
        arithmetic_proxy_excludes_real_scalar_products=True,scalar_complex_real_products_per_prediction=7936,
        arithmetic_proxy_includes_5frame_pilot_objective=True,rolling_objective_reuse_assumed=False,
        historical_replay_gate=gate,historical_replay_gate_passed=True,actual_small_inference=small,
        coefficient_refits=0,pilot_coordinate_inference_changed=False,pilot_K_compressed=False,source_W_compressed=False,
        dense_Br_reconstructed_during_inference=False,heldout_covariance_used_for_factor_construction=False,
        primary_estimator_uses_environment_label=False,primary_source_means_only=True,matched_retains_frozen_target_means=True,
        deterministic_validation_replay_used=True,new_validation_realizations_generated=False,new_channel_realizations=0,
        generator_calls=0,new_seed_ids=0,statistics_refitted=False,new_model_trained=False,BER_evaluations=0,detector_calls=0,
        formal_test_used=False,formal_test_sealed=True,historical_full_tensor_hashes_available=False,
        runtime=dict(**pre['runtime'],**gate['runtime'],lowrank_inference_seconds_by_rank=timing,
            predictions_by_rank=counts,total_numeric_wall_seconds=time.perf_counter()-start,channel_generation_seconds=0.),
        source_sha256={p:sha(ROOT/p) for p in SOURCES},frozen_history_sha256=before,environment_before=env,
        numeric_artifact_sha256={f'outputs/{p}':sha(OUT/p) for p in [*files,'t16g_module_spectrum.csv']},
        preparation_artifact_sha256={'outputs/t16g_factor_preflight.json':sha(OUT/'t16g_factor_preflight.json'),
            'outputs/t16g_frozen_module_factors.npz':sha(OUT/'t16g_frozen_module_factors.npz'),
            'outputs/t16g_execution_preflight.json':sha(OUT/'t16g_execution_preflight.json')},
        preflight_schema_recovery=dict(evidence=load(OUT/'t16g_preflight_schema_recovery/recovery_provenance.json'),
            artifact_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted((OUT/'t16g_preflight_schema_recovery').iterdir()) if p.is_file()},
            original_factor_file_and_preflight_unchanged=True),
        all_checks_passed=False,completion_state='actual validation completed; final audit pending',
        warnings=['Low-rank/adaptive-filter combination is established literature; this is a compression gate, not a novelty claim.',
            'Independent review caught CSV schema inconsistency; first execution stopped before historical gate completion or compressed performance, archived, then output-schema corrected without factor/coordinate refit.',
            'Coefficients are loaded exactly, not refitted; pilot fitting and source W are unchanged.',
            'One-frame all10 is secondary; scientific gate uses causal5-frame paired frames4..9 only.',
            'Development capture need not guarantee held-out NMSE; held-out covariance is evaluation-only.',
            'Recovery/penalty/gains are not clipped; correlated-window std is descriptive, not CI.',
            'Matched keeps its original target means; all compressed estimates use source means.',
            'MAC proxy includes pilot CV/source/correction but excludes7936 common real scalar products and small2x2 solver/bookkeeping.',
            'No new historical standalone bits/noise/Y hashes are asserted.'])
    save(RESULT,result);print(science,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--verify-preflight',action='store_true');args=parser.parse_args()
    if args.preflight:preflight()
    elif args.verify_preflight:execution_preflight()
    else:run()
