"""T18B exact T18A replay gate, then five independent frozen-receiver replicates."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from src import validation_replication as p
from src import frozen_affine_validation as old
from src.cdl_benchmark import FrozenStatistics,EmpiricalEstimator,StreamingStatistics,frames,array_hash,observation_geometry
from src.lowrank_modules import WeightedModule
from src.pilot_cv import deployment
from src.statistics_mixing import geometry
from src.link import assemble_pilot_data_frame,apply_frequency_mimo_channel,PILOT_SYMBOLS
from src.pilots import map_pilot_symbols
from src.estimators import pilot_ls_estimate
from scripts.run_cdl_mismatch import load,save,sha,csvsave,history,environments,communication
from scripts.run_pilot_cv import cv_family,readcsv,trajectory_energy
from scripts.run_frozen_generalization import frozen_method
OUT=ROOT/'outputs';RESULT=OUT/'t18b_results.json'
T18A_COMMIT='3063c31cdbc1fba05b522517d04d1ad86f9f62cf'
SOURCES=('src/validation_replication.py','scripts/run_validation_replication.py','scripts/plot_validation_replication.py',
    'scripts/audit_validation_replication.py','tests/test_validation_replication.py','docs/t18b_replication_protocol.md')


def accepted():
    subprocess.check_call(['git','merge-base','--is-ancestor',T18A_COMMIT,'HEAD'],cwd=ROOT)
    if subprocess.check_output(['git','show','-s','--format=%s',T18A_COMMIT],cwd=ROOT,text=True).strip()!='feat: validate frozen affine LMMSE across unseen mobility shifts':
        raise ValueError('accepted T18A snapshot required')
    from scripts.audit_frozen_generalization import verify
    r=load(OUT/'t18a_results.json');verify(r)
    if not r['all_checks_passed'] or not r['affine_composition_replication_partial'] or r['affine_composition_beyond_convex_hull_replicated']:
        raise ValueError('historical T18A PARTIAL result required')
    for path in (*r['source_sha256'],'outputs/t18a_results.json'):
        blob=subprocess.check_output(['git','show',f'{T18A_COMMIT}:{path}'],cwd=ROOT)
        if blob.replace(b'\r\n',b'\n')!=(ROOT/path).read_bytes().replace(b'\r\n',b'\n'):
            raise ValueError('accepted snapshot changed:'+path)
    return r


def targets(r):
    result={}
    for name in p.ENVIRONMENTS:
        info=r['target_statistics'][name]
        if sha(ROOT/info['path'])!=info['sha256']:raise ValueError('frozen target NPZ changed')
        with np.load(ROOT/info['path'],allow_pickle=False) as f:
            values=[f[k].copy() for k in ('mean_p','mean_d','rpp','rdp')]
            if any(array_hash(v)!=info['component_sha256'][k] for v,k in zip(values,('mean_p','mean_d','rpp','rdp'))):raise ValueError('target component changed')
            if not np.array_equal(f['train_trajectory_ids'],np.arange(500)) or int(f['samples'])!=5000:raise ValueError('target training provenance changed')
        for v in values:v.flags.writeable=False
        result[name]=FrozenStatistics(*values,tuple(range(500)),5000)
    return result


def references(r,stored=False):
    target=targets(r);matched={};oracle={};weights={}
    if stored:
        gate=load(OUT/'t18b_historical_replay_gate.json');info=gate['derived_matched_operator_artifact']
        if not gate['all_checks_passed'] or sha(ROOT/info['path'])!=info['sha256']:raise ValueError('replay/derived reference artifact failed')
        with np.load(ROOT/info['path'],allow_pickle=False) as f:
            weights={k:f[k].copy(order='K') for k in f.files}
        if {k:array_hash(v) for k,v in weights.items()}!=info['component_sha256']:raise ValueError('reference W drift')
    for n in p.ENVIRONMENTS:
        matched[n]={};oracle[n]={}
        for s in p.SNRS:
            if stored:
                w=weights[f'{n}_{s}'];w.flags.writeable=False
                matched[n][s]=EmpiricalEstimator(target[n],w,old.VARIANCES[s],info['solve_relative_residual'][f'{n}_{s}'])
            else:
                estimator=target[n].prepare(old.VARIANCES[s])
                if estimator.solve_relative_residual>1e-11:raise ValueError('historical Cholesky residual failed')
                estimator.weight.flags.writeable=False;matched[n][s]=estimator
            v=next(v for v in r['oracle_affine'] if (v['environment'],v['snr_db'])==(n,s))
            oracle[n][s]=np.array([v['alpha'],v['beta']],dtype=np.float64);oracle[n][s].flags.writeable=False
    return matched,oracle


def namespace_audit(rows):
    expected=p.manifest_rows()
    if rows!=expected:raise ValueError('manifest not the frozen seed policy')
    past=old.manifest_rows();past_seeds={v for r in past for k,v in r.items() if k in old.ROLES}
    used=set();by_namespace={}
    for r in rows:
        namespace=r['namespace'];bucket=by_namespace.setdefault(namespace,set())
        for key in old.ROLES:
            if key in old.ROLES[:3] and r['frame_id']!=0:continue
            value=r[key]
            if value in used or value in past_seeds:raise ValueError('individual seed collision')
            used.add(value);bucket.add(value)
    if len(rows)!=20000 or len(used)!=66000 or len(by_namespace)!=20:raise ValueError('incomplete replication manifest')
    return dict(namespaces=list(by_namespace),new_validation_trajectories=2000,frames=20000,
        unique_consumed_individual_seeds=66000,t18a_all_individual_seeds_disjoint=True,
        new_replicates_pairwise_disjoint=True,all_roles_conceptually_separate=True,
        target_train_generation=0,formal_test_namespace_accessed=False)


def evaluate(name,i,h,arrays,family,matched,oracle,seed_function):
    """Unchanged T18A observations/reduction, restricted to exactly10/20 dB."""
    g=observation_geometry();truths=frames(h);pilots=np.empty((2,10,64),np.complex128);pair=[]
    known=map_pilot_symbols(g.layout,PILOT_SYMBOLS)
    for j,truth in enumerate(truths):
        bits,symbols,z=communication(seed_function(j))
        x=assemble_pilot_data_frame(symbols,g.layout);clean=apply_frequency_mimo_channel(truth,x)
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
            if r['dense_coefficients'] is not r['lr4_coefficients']:raise ValueError('Dense/LR4 coefficient identity failed')
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
        for method in p.METHODS:
            error=trajectory_energy(errors[method]);tr.append(dict(environment=name,snr_db=s,trajectory_id=i,
                method=method,frames=6,subset='frames4..9',error_energy=error,truth_energy=energy,nmse_linear=error/energy))
    return tr,raw,frame_rows,pair


def matrix(rows,name,snr,replicate=None):
    group=[r for r in rows if (r['environment'],int(r['snr_db']))==(name,snr) and (replicate is None or int(r['replicate'])==replicate)]
    expected=[(r,i,m) for r in (p.REPLICATES if replicate is None else (replicate,)) for i in range(100) for m in p.METHODS]
    index={(int(r['replicate']),int(r['trajectory_id']),r['method']):r for r in group}
    if set(index)!=set(expected) or len(group)!=len(expected):raise ValueError('exact new trajectory population required; no T18A pooling')
    errors=[];truth=[]
    for r in (p.REPLICATES if replicate is None else (replicate,)):
        for i in range(100):
            vals=[index[r,i,m] for m in p.METHODS]
            if any(int(v['frames'])!=6 or v['subset']!='frames4..9' for v in vals):raise ValueError('primary frame subset changed')
            if len({float(v['truth_energy']) for v in vals})!=1:raise ValueError('unpaired truth denominator')
            errors.append([float(v['error_energy']) for v in vals]);truth.append(float(vals[0]['truth_energy']))
    return np.array(errors),np.array(truth)


def summarize(rows,weights):
    nmse=[];metrics=[];pooled_metrics=[];stability=[];intervals=[];draws={};index_proof={}
    for n in p.ENVIRONMENTS:
        for s in p.SNRS:
            for r in p.REPLICATES:
                err,true=matrix(rows,n,s,r);num,den,db=p.pooled(err,true)
                for j,m in enumerate(p.METHODS):
                    nmse.append(dict(environment=n,snr_db=s,replicate=r,method=m,trajectories=100,frames=600,
                        subset='frames4..9',error_energy=float(num[j]),truth_energy=den,nmse_db=float(db[j]),
                        primary_method=m in p.PRIMARY_METHODS,means='target' if m=='matched' else 'source'))
                metrics.append(dict(environment=n,snr_db=s,replicate=r,**{k:float(v) for k,v in p.metric_columns(db).items()}))
                raw=[v for v in weights if (v['environment'],int(v['snr_db']),int(v['replicate']),v['method'])==(n,s,r,'affine')]
                if len(raw)!=600:raise ValueError('600 correlated causal windows required')
                summary=dict(environment=n,snr_db=s,replicate=r,trajectories=100,windows=600,
                    outside_simplex_fraction=sum(str(v['outside_any'])=='True' for v in raw)/600)
                for field in ('alpha','beta','lambda0'):
                    x=np.array([float(v[field]) for v in raw]);q=np.quantile(x,[.25,.5,.75])
                    summary.update({f'{field}_{k}':float(v) for k,v in zip(('q25','median','q75','mean','std'),(*q,np.mean(x),np.std(x,ddof=1)))})
                stability.append(summary)
            err,true=matrix(rows,n,s);num,den,db=p.pooled(err,true)
            pooled_metrics.append(dict(environment=n,snr_db=s,trajectories=500,frames=3000,
                truth_energy=den,**{k:float(v) for k,v in p.metric_columns(db).items()},
                **{f'error_{m.replace("-","_")}':float(v) for m,v in zip(p.METHODS,num)}))
            cis,proof,values=p.bootstrap(err,true,n,s);intervals.extend(cis);index_proof[f'{n}/{s}']=proof;draws[f'{n}_{s}']=values
    return nmse,metrics,pooled_metrics,stability,intervals,index_proof,draws


def preflight():
    if (OUT/'t18b_preflight.json').exists():raise ValueError('existing preflight protected')
    r=accepted();arrays,family,proof=frozen_method()
    if proof!=r['frozen_method_provenance']:raise ValueError('method not exact')
    references(r);rows=p.manifest_rows();namespace=namespace_audit(rows)
    env=environments()
    if env!=r['environment_after']:raise ValueError('frozen environment drift')
    log=OUT/'t18b_related_preflight_pytest_attempt03.txt';text=log.read_text(encoding='utf8');summary=text.strip().splitlines()[-1]
    if '1 skipped' not in summary or 'failed' in summary or 'error' in summary:raise ValueError('all calibration tests and one explicit missing-result skip required')
    csvsave(OUT/'t18b_manifest.csv',rows)
    for path in SOURCES:
        dest=OUT/'t18b_execution_snapshot'/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/path).read_bytes())
    before=history()
    before.update({v:sha(ROOT/v) for v in subprocess.check_output(['git','ls-files','docs'],cwd=ROOT,text=True).splitlines()})
    save(OUT/'t18b_preflight.json',dict(all_checks_passed=True,accepted_T18A_commit=T18A_COMMIT,
        startup_pytest='1692 passed in 265.61s (0:04:25)',working_tree_clean_after_t18a_commit=True,
        startup_log_sha256=sha(OUT/'t18b_startup_pytest.txt'),source_sha256={v:sha(ROOT/v) for v in SOURCES},
        frozen_history_sha256=before,environment=env,frozen_method=proof,namespace_audit=namespace,
        manifest_sha256=sha(OUT/'t18b_manifest.csv'),target_statistics={n:r['target_statistics'][n] for n in p.ENVIRONMENTS},
        bootstrap_seeds={f'{n}/{s}':p.bootstrap_seed(n,s) for n in p.ENVIRONMENTS for s in p.SNRS},
        bootstrap_samples=p.BOOTSTRAPS,c8_claim_status_before='UNCONFIRMED',
        c8_mapping='support: positive & interpolation & no compression warning; contradiction: positive instability or >=2 replicate threshold failures; geometry excluded',
        calibration_summary=summary,calibration_log_sha256=sha(log),new_validation_started=False))
    print('T18B preparation PASS: original method/target hashes,20 namespaces,66000 separate consumed seeds; new generation remains blocked by historical gate',flush=True)


def check_preflight():
    pre=load(OUT/'t18b_preflight.json')
    if not pre['all_checks_passed'] or Path(sys.prefix).resolve()!=(ROOT/'.venv-cdl').resolve():raise ValueError('frozen CDL interpreter and passed preflight required')
    for group in ('source_sha256','frozen_history_sha256'):
        for path,h in pre[group].items():
            if sha(ROOT/path)!=h:raise ValueError('frozen bytes drift:'+path)
    if sha(OUT/'t18b_manifest.csv')!=pre['manifest_sha256']:raise ValueError('manifest drift')
    _,_,proof=frozen_method()
    if proof!=pre['frozen_method']:raise ValueError('frozen receiver drift')
    return pre


def cell(v):return '' if v is None else str(v)


def compare_rows(actual,index,key):
    for r in actual:
        k=tuple(cell(r[v]) for v in key)
        reference=index[k]
        if {v:cell(x) for v,x in r.items()}!=reference:
            changes=[v for v in r if cell(r[v])!=reference.get(v)]
            raise ValueError(f'historical replay mismatch {k}: {changes}')


def replay():
    if any((OUT/f).exists() for f in ('t18b_historical_replay_gate.json','t18b_frozen_matched_operators.npz','t18b_historical_replay_digests.csv')):raise ValueError('existing historical replay evidence protected')
    pre=check_preflight();r=accepted();arrays,family,_=frozen_method();matched,oracle=references(r)
    datasets={
        'trajectory':('t18a_generalization_trajectory.csv',('environment','snr_db','trajectory_id','method')),
        'weights':('t18a_generalization_weights.csv',('environment','snr_db','trajectory_id','frame_id','method')),
        'frames':('t18a_frame_metrics.csv',('environment','snr_db','trajectory_id','frame_id','method')),
        'pairs':('t18a_paired_input_hashes.csv',('environment','snr_db','trajectory_id','frame_id'))}
    tables={};keys={}
    for label,(filename,key) in datasets.items():
        keys[label]=key;selected=[v for v in readcsv(OUT/filename) if v['environment'] in p.ENVIRONMENTS and int(v['snr_db']) in p.SNRS]
        tables[label]={tuple(v[k] for k in key):v for v in selected}
        if len(tables[label])!=len(selected):raise ValueError('nonunique historical row key')
    hindex={(v['environment'],int(v['trajectory_id'])):v['H_sha256'] for v in readcsv(OUT/'t18a_channel_hashes.csv') if v['split']=='validation_dev' and v['environment'] in p.ENVIRONMENTS}
    counts={label:0 for label in datasets};alltr=[];digests=[];tick=time.perf_counter()
    for n in p.ENVIRONMENTS:
        adapter=old.GeneralizationCDL(n)
        if adapter.provenance()!=load(OUT/'t18a_preflight.json')['generator_provenance'][n]:raise ValueError('generator provenance drift')
        for i in range(500,600):
            h=adapter.generate(i,split='validation_dev')
            if array_hash(h)!=hindex[n,i]:raise ValueError('historical whole-trajectory H hash not exact')
            h.flags.writeable=False
            tr,weights,fr,pair=evaluate(n,i,h,arrays,family,matched[n],oracle[n],lambda j:old.seeds(n,'validation_dev',i,j))
            for label,values in zip(datasets,(tr,weights,fr,pair)):
                compare_rows(values,tables[label],keys[label]);counts[label]+=len(values)
            ordered=[tables['pairs'][tuple(cell(v[k]) for k in keys['pairs'])] for v in pair]
            actual=[{k:cell(v) for k,v in row.items()} for row in pair]
            digest=lambda rows:hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            if digest(ordered)!=digest(actual):raise ValueError('derived historical digest mismatch')
            digests.append(dict(environment=n,trajectory_id=i,H_sha256=array_hash(h),
                derived_historical_pair_records_digest=digest(ordered),replayed_pair_records_digest=digest(actual),exact=True))
            alltr.extend(tr)
            if (i-499)%20==0:print(f'T18A exact replay {n}: {i-499}/100',flush=True)
    for n in p.ENVIRONMENTS:
        for s in p.SNRS:
            values={}
            for m in p.METHODS:
                group=[v for v in alltr if (v['environment'],v['snr_db'],v['method'])==(n,s,m)]
                error=sum(v['error_energy'] for v in group);truth=sum(v['truth_energy'] for v in group)
                reference=next(v for v in r['validation_nmse'] if (v['environment'],v['snr_db'],v['method'])==(n,s,m))
                values[m]=float(10*np.log10(error/truth))
                if (error,truth,values[m])!=(reference['error_energy'],reference['truth_energy'],reference['nmse_db']):raise ValueError('historical pooled metrics mismatch')
            reference=next(v for v in r['condition_metrics'] if (v['environment'],v['snr_db'])==(n,s))
            metric=old.metrics(values['source'],values['matched'],values['dense-affine'],values['lr4-affine'],values['convex'],values['oracle-affine'])
            if any(reference[k]!=v for k,v in metric.items()):raise ValueError('historical gains/metrics mismatch')
    if any(counts[k]!=len(tables[k]) for k in counts):raise ValueError('historical coverage incomplete')
    artifact=OUT/'t18b_frozen_matched_operators.npz'
    weights={f'{n}_{s}':matched[n][s].weight for n in p.ENVIRONMENTS for s in p.SNRS}
    np.savez_compressed(artifact,**weights);artifact.chmod(0o444)
    csvsave(OUT/'t18b_historical_replay_digests.csv',digests)
    save(OUT/'t18b_historical_replay_gate.json',dict(all_checks_passed=True,
        historical_trajectories=400,conditions=8,generator_calls=400,new_realizations=0,
        all_saved_h_input_prediction_weights_energies_pooled_metrics_exact=True,comparisons=counts,
        digest_evidence='derived from immutable T18A per-frame hash records, not a pre-existing combined tensor hash',
        digest_csv_sha256=sha(OUT/'t18b_historical_replay_digests.csv'),historical_matched_W_hashes_available=False,
        derived_matched_operator_artifact=dict(path='outputs/'+artifact.name,sha256=sha(artifact),
            component_sha256={k:array_hash(v) for k,v in weights.items()},
            solve_relative_residual={f'{n}_{s}':matched[n][s].solve_relative_residual for n in p.ENVIRONMENTS for s in p.SNRS},
            derivation='unchanged historical Cholesky from frozen T18A Rpp/Rdp, exact historical matched prediction replay',covariance_refit=False),
        target_statistics_unchanged=True,oracle_coefficients_loaded_not_optimized=True,
        frozen_preflight_sha256=sha(OUT/'t18b_preflight.json'),seconds=time.perf_counter()-tick))
    check_preflight();print('T18A complete four-environment10/20 historical replay EXACT PASS; new validation now permitted',flush=True)


def run():
    if RESULT.exists() or (OUT/'t18b_trajectory_energy.csv').exists() or any(OUT.glob('t18b_checkpoint_*')):raise ValueError('existing full/partial new validation result protected')
    pre=check_preflight();gate=load(OUT/'t18b_historical_replay_gate.json')
    if not gate['all_checks_passed'] or gate['frozen_preflight_sha256']!=sha(OUT/'t18b_preflight.json'):raise ValueError('historical replay gate failed: STOP')
    r=accepted();arrays,family,_=frozen_method();matched,oracle=references(r,stored=True)
    rows=[];weights=[];frames_out=[];pairs=[];hrows=[];runtime=[];tick=time.perf_counter()
    def forbidden(*a,**k):raise AssertionError('no covariance/module/oracle refit in T18B')
    with patch.object(StreamingStatistics,'update',forbidden),patch.object(WeightedModule,'build',forbidden),patch.object(old,'oracle_coordinates',forbidden):
        for n in p.ENVIRONMENTS:
            adapter=p.ReplicationCDL(n)
            if adapter.provenance()!=load(OUT/'t18a_preflight.json')['generator_provenance'][n]:raise ValueError('new generator configuration drift')
            for replicate in p.REPLICATES:
                t=time.perf_counter();newrows=[];newweights=[]
                for i in range(100):
                    h=adapter.generate(i,replicate=replicate);h.flags.writeable=False
                    metadata=dict(replicate=replicate,namespace=f'T18B_{n}_r{replicate}_v1')
                    hrows.append(dict(environment=n,trajectory_id=i,**metadata,H_sha256=array_hash(h)))
                    tr,raw,fr,pair=evaluate(n,i,h,arrays,family,matched[n],oracle[n],lambda j:p.seeds(n,replicate,i,j))
                    for values in (tr,raw,fr,pair):
                        for v in values:v.update(metadata)
                    rows.extend(tr);weights.extend(raw);frames_out.extend(fr);pairs.extend(pair);newrows.extend(tr);newweights.extend(raw)
                csvsave(OUT/f't18b_checkpoint_{n}_r{replicate}_energy.csv',newrows)
                csvsave(OUT/f't18b_checkpoint_{n}_r{replicate}_weights.csv',newweights)
                runtime.append(dict(environment=n,replicate=replicate,seconds=time.perf_counter()-t))
                print(f'T18B {n} replicate{replicate}:100 trajectories completed',flush=True)
    if len({v['H_sha256'] for v in hrows})!=2000:raise ValueError('duplicate new channel realization')
    historical={v['H_sha256'] for v in readcsv(OUT/'t18a_channel_hashes.csv')}
    if historical & {v['H_sha256'] for v in hrows}:raise ValueError('historical H collision')
    numeric_seconds=time.perf_counter()-tick;t=time.perf_counter()
    nmse,metrics,points,stability,cis,index_proof,draws=summarize(rows,weights)
    science=p.scientific_gate(metrics,points,cis,stability)
    files={'t18b_replicate_nmse.csv':nmse,'t18b_replicate_metrics.csv':metrics,
        't18b_trajectory_energy.csv':rows,'t18b_bootstrap_ci.csv':cis,'t18b_weight_stability.csv':stability,
        't18b_pooled_metrics.csv':points,'t18b_window_weights.csv':weights,'t18b_frame_metrics.csv':frames_out,
        't18b_paired_input_hashes.csv':pairs,'t18b_channel_hashes.csv':hrows}
    for name,values in files.items():csvsave(OUT/name,values)
    np.savez_compressed(OUT/'t18b_bootstrap_draws.npz',**draws)
    check_preflight()
    result=dict(task='T18B',experiment_type=p.EXPERIMENT_TYPE,research_question=p.C8,
        accepted_T18A_commit=T18A_COMMIT,method_frozen=True,target_statistics_frozen=True,
        replicate_count=5,trajectories_per_replicate=100,frames_per_trajectory=10,primary_frames=list(old.PRIMARY_FRAMES),
        bootstrap_unit='trajectory',bootstrap_samples=p.BOOTSTRAPS,confidence_interval='95_percentile',
        bootstrap_index_provenance=index_proof,validation_conditions=[dict(environment=n,snr_db=s) for n in p.ENVIRONMENTS for s in p.SNRS],
        primary_methods=list(p.PRIMARY_METHODS),secondary_method='hard',oracle_affine_role='frozen evaluation control only',
        replicate_nmse=nmse,replicate_gains=[{k:v[k] for k in ('environment','snr_db','replicate','affine_vs_convex_gain_db','lr4_vs_convex_gain_db')} for v in metrics],
        replicate_metrics=metrics,pooled_metrics=points,bootstrap_intervals=cis,weight_stability=stability,**science,
        c6_claim_status=old.C6,c7_claim_status=old.C7,c8_claim=p.C8,c8_claim_status_before='UNCONFIRMED',
        t18a_original_composed_gains=[{k:v[k] for k in ('environment','snr_db','lr4_vs_convex_gain_db')} for v in r['condition_metrics'] if v['environment'].startswith('CDL-B') and v['environment'] in p.ENVIRONMENTS and v['snr_db'] in p.SNRS],
        t18a_original_scientific_flags={k:r[k] for k in ('frozen_pipeline_interpolation_supported','affine_composition_beyond_convex_hull_replicated','affine_composition_replication_weak','affine_composition_replication_partial','outside_convex_hull_geometry_replicated','extrapolation_stress_promising')},
        t18a_not_in_new_pooled_sample=True,target_statistics_uncertainty_evaluated=False,
        CI_scope='validation trajectory randomness in these fixed environments, conditional on frozen T18A target statistics; no random-environment population guarantee',
        historical_replay_gate=gate,historical_replay_gate_sha256=sha(OUT/'t18b_historical_replay_gate.json'),
        namespace_audit=dict(pre['namespace_audit'],all2000_new_H_hashes_unique=True,new_H_disjoint_from_all_T18A_train_validation=True),
        frozen_method_provenance=pre['frozen_method'],target_statistics=pre['target_statistics'],
        source_sha256=pre['source_sha256'],source_snapshot_root='outputs/t18b_execution_snapshot',
        frozen_history_sha256=pre['frozen_history_sha256'],environment_before=pre['environment'],
        preparation_artifact_sha256={f'outputs/{v}':sha(OUT/v) for v in ('t18b_preflight.json','t18b_manifest.csv','t18b_historical_replay_gate.json','t18b_historical_replay_digests.csv','t18b_frozen_matched_operators.npz')},
        numeric_artifact_sha256={f'outputs/{v}':sha(OUT/v) for v in (*files,'t18b_bootstrap_draws.npz')},
        dense_lr4_coefficients_identical=True,rank_per_module=4,causal_window=5,source_means_for_deployed_and_oracle=True,
        primary_estimator_uses_environment_label=False,target_statistics_used_for_deployment=False,
        no_covariance_refit=True,no_new_target_train=True,no_basis_refit=True,no_new_method=True,
        target_normalization=False,noise_recalibrated=False,BER_evaluations=0,detector_calls=0,
        formal_test_used=False,formal_test_sealed=True,new_validation_generator_calls=2000,
        runtime=dict(by_replicate=runtime,new_validation_wall_seconds=numeric_seconds,
            bootstrap_and_output_seconds=time.perf_counter()-t,historical_replay_seconds=gate['seconds']),
        all_checks_passed=False,completion_state='replication and bootstrap completed; final regression/figure audit pending',
        warnings=['CI conditional on one frozen target-statistics realization and four fixed environments.',
            'Percentile CIs are marginal; no simultaneous coverage or random-environment population guarantee is claimed.',
            'Six causal windows are correlated within a trajectory; weight quartiles/std are descriptive only.',
            'Matched W was derived from immutable target covariance then proven by exact historical predictions; no invented historical W hash.',
            'T18A PARTIAL materiality remains unchanged; C6/C7 negative mechanism conclusions are permanent.',
            'Archived T18A staged whitespace and Git newline warnings retained unchanged.',
            'Initial AST class-method indentation audit failure archived before any scientific run;33 calibration tests passed after dedent correction.',
            'Stale Python base-interpreter location and optional absent Triton warnings retained; no environment modification.'])
    save(RESULT,result);print(json.dumps(science,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['preflight','replay','run']);args=parser.parse_args()
    {'preflight':preflight,'replay':replay,'run':run}[args.mode]()
