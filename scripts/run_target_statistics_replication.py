"""T19C ten independent train-only CDL populations and frozen-operator audit."""
import argparse
import ast
from contextlib import ExitStack,contextmanager
import inspect
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import time
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.linalg import cholesky
from src import target_statistics_replication as p
from src import operator_space_audit as math
from src import frozen_affine_validation as old
from src.cdl_benchmark import FrozenStatistics,StreamingStatistics,frames,array_hash,observation_geometry
from src.cdl_snr import TrainReceivePowerAccumulator
from src.delay_factor_challenge import EffectiveDelayAudit,FactorCDL
from src.lowrank_modules import WeightedModule
from scripts.run_cdl_mismatch import load,save,sha,csvsave,history,environments
from scripts.run_operator_space_audit import inputs as old_inputs
from scripts.run_pilot_cv import readcsv
OUT=ROOT/'outputs';RESULT=OUT/'t19c_results.json'
COMMIT='8864210069c880d641574dc047106875889a0c51'
SOURCES=('src/target_statistics_replication.py','scripts/run_target_statistics_replication.py',
         'scripts/plot_target_statistics_replication.py','scripts/audit_target_statistics_replication.py',
         'tests/test_target_statistics_replication.py','docs/t19c_statistics_replication_protocol.md')
MANIFESTS=('t15a_cdl_manifest.csv','t15c_cdl_manifest.csv','t16c_heldout_manifest.csv',
           't18a_manifest.csv','t18b_manifest.csv','t19a_manifest.csv')


@contextmanager
def method_guard():
    def forbidden(*args,**kwargs):raise RuntimeError('no module refit, Pilot-CV, BER or communication generation in T19C')
    with ExitStack() as stack:
        stack.enter_context(patch.object(WeightedModule,'build',forbidden))
        stack.enter_context(patch.object(np.linalg,'svd',forbidden))
        from src import frozen_affine_validation,pilot_cv,statistics_mixing,link,numerics
        from scripts import run_cdl_mismatch
        stack.enter_context(patch.object(pilot_cv.PilotCV,'fit',forbidden))
        for module,name in ((frozen_affine_validation,'deployed'),(pilot_cv,'deployment'),(pilot_cv,'primary_prediction'),
                            (statistics_mixing,'objective_moments'),(statistics_mixing,'hard_selection'),(statistics_mixing,'convex_mixing'),
                            (link,'generate_master_qpsk'),(link,'detect_data_zf'),(link,'detect_data_linear_mmse'),
                            (link,'demodulate_data_bits'),(numerics,'complex_gaussian_noise'),(run_cdl_mismatch,'communication')):
            stack.enter_context(patch.object(module,name,forbidden))
        yield


def accepted():
    subprocess.check_call(['git','merge-base','--is-ancestor',COMMIT,'HEAD'],cwd=ROOT)
    from scripts.audit_operator_space_audit import verify
    r=load(OUT/'t19b_results.json')
    if not r['all_checks_passed'] or not verify(r):raise ValueError('accepted T19B required')
    paths=(*r['source_sha256'],*r['post_run_source_sha256'],'outputs/t19b_results.json')
    for path in paths:
        blob=subprocess.check_output(['git','show',f'{COMMIT}:{path}'],cwd=ROOT)
        if blob.replace(b'\r\n',b'\n')!=(ROOT/path).read_bytes().replace(b'\r\n',b'\n'):
            raise ValueError('accepted T19B snapshot changed:'+path)
    return r


def generator_semantics():
    old_tree=ast.parse(textwrap.dedent(inspect.getsource(FactorCDL.generate)))
    new_tree=ast.parse(textwrap.dedent(inspect.getsource(p.StatisticsCDL.generate)))
    def remove_seed_call(tree):
        fn=tree.body[0]
        assignments=[n for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='s' for t in n.targets)]
        if len(assignments)!=1:raise ValueError('exactly one explicit seed call required')
        fn.body.remove(assignments[0]);return ast.dump(tree,include_attributes=False)
    if remove_seed_call(old_tree)!=remove_seed_call(new_tree):raise ValueError('CDL generation changed beyond approved seed call')
    return dict(original_T19A_generator_sha256=sha(ROOT/'src/delay_factor_challenge.py'),
                inherited_physical_constructor=True,export_AST_identical_except_namespace_seed_call=True,
                normalize=False,frames=10,symbols=160,precision='double',profiles=['C'],speed_kmh=30)


def namespace_audit(rows):
    if rows!=p.manifest_rows():raise ValueError('namespace manifest differs from declared policy')
    past=set();historical={}
    for name in MANIFESTS:
        entries=readcsv(OUT/name)
        if not entries or not all(k in entries[0] for k in p.ROLES):raise ValueError('historical manifest schema mismatch:'+name)
        columns=[k for k in entries[0] if k.endswith('_seed') or k.startswith('bits_seed_')]
        for row in entries:
            for key in columns:
                if row[key]!='':past.add(int(row[key]))
        historical[name]=dict(sha256=sha(OUT/name),rows=len(entries),seed_columns=columns)
    used=set();tuples=set();namespace_sets={}
    for row in rows:
        if row['split']!='train_dev' or row['communication_generated']:raise ValueError('validation/communication scope violation')
        if row['frame_id']!=0:continue
        key=tuple(row[k] for k in p.ROLES)
        if key in tuples:raise ValueError('new trajectory stream tuple collision')
        tuples.add(key)
        group=namespace_sets.setdefault(row['namespace'],set())
        for role in p.ROLES:
            seed=row[role]
            if seed in used or seed in past:raise ValueError('new individual seed collision:'+str(seed))
            used.add(seed);group.add(seed)
    if len(rows)!=50000 or len(tuples)!=5000 or len(used)!=15000 or len(namespace_sets)!=10:
        raise ValueError('declared independent population incomplete')
    return dict(passed=True,new_namespaces=list(namespace_sets),manifest_rows=50000,unique_trajectory_streams=5000,
                unique_individual_channel_seeds=15000,historical_individual_seeds_checked=len(past),
                pairwise_disjoint=True,historical_disjoint=True,T19A_data_used_for_replication=False,
                formal_namespace_accessed=False,historical_manifest_provenance=historical)


def frozen_inputs():
    historical=load(OUT/'t19a_results.json')
    arrays,family,source,targets,proof=old_inputs(historical)
    return arrays,family,source,{n:targets[n] for n in p.ENVIRONMENTS},proof


def diagnostic(name,replicate,target,arrays,family):
    rows=[]
    for s in p.SNRS:
        prepared=target.prepare(old.VARIANCES[s]);wt=prepared.weight
        if prepared.solve_relative_residual>1e-11:raise ValueError('matched Cholesky solve failed')
        lt=cholesky(target.rpp+old.VARIANCES[s]*np.eye(64),lower=True)
        ws=arrays[f'Wsource_{s}'];modules=[arrays[f'DeltaW{m}_{s}'] for m in ('P','M')]
        fit=math.operator_fit(ws,wt,lt,modules);source=math.operator_excess(ws,wt,lt)
        if source<=0:raise ValueError('positive finite source mismatch required')
        modules4=[family[m,s,4].A@family[m,s,4].Q for m in ('P','M')]
        rank4=math.operator_fit(ws,wt,lt,modules4)
        dense=1-fit['excess']/source;lr4=1-rank4['excess']/source
        eig=np.linalg.eigvalsh(target.rpp+old.VARIANCES[s]*np.eye(64));eg=np.linalg.eigvalsh(fit['gram'])
        rows.append(dict(environment=name,replicate=replicate,snr_db=s,alpha_op=float(fit['coefficients'][0]),beta_op=float(fit['coefficients'][1]),
            E_op_source=source,E_op_oracle2=fit['excess'],operator_span_capture_dense=dense,
            condition_number_Cz=float(eig[-1]/eig[0]),condition_number_real_Gram=float(eg[-1]/eg[0]),
            matched_operator_frobenius_norm=float(np.linalg.norm(wt)),matched_solve_relative_residual=prepared.solve_relative_residual,
            oracle_normal_relative_residual=fit['normal_relative_residual'],operator_span_capture_lr4=lr4,
            rank4_span_capture_loss=dense-lr4,energy_unit='per-frame data-domain energy;3968 coefficients',evaluation_only=True))
    return rows


def original_reference(arrays,family,targets):
    r=load(OUT/'t19b_results.json');index={(v['environment'],v['snr_db']):v for v in r['operator_span_capture_dense']}
    rows=[]
    for n in p.ENVIRONMENTS:
        replay=diagnostic(n,0,targets[n],arrays,family)
        for value in replay:
            ref=index[n,value['snr_db']]
            if value['operator_span_capture_dense']!=ref['operator_span_capture_dense'] or value['E_op_source']!=ref['E_op_source'] or value['E_op_oracle2']!=ref['E_op_oracle2']:
                raise ValueError('original T19A-statistics operator replay not exact')
            rows.append(dict(environment=n,snr_db=value['snr_db'],operator_span_capture_dense=ref['operator_span_capture_dense'],
                             statistics_source='original T19A500;exact T19B operator reference',exact_replay=True,in_replication_aggregate=False))
    return rows


def verify_protected(pre):
    for group in ('source_sha256','frozen_history_sha256'):
        for path,h in pre[group].items():
            if sha(ROOT/path)!=h:raise ValueError('protected artifact drift:'+path)
    for path,h in pre['source_sha256'].items():
        if sha(ROOT/pre['source_snapshot_root']/path)!=h:raise ValueError('execution archive drift')


def preflight():
    start=time.perf_counter()
    if (OUT/'t19c_preflight.json').exists():raise ValueError('existing preflight protected')
    if Path(sys.prefix).resolve()!=(ROOT/'.venv-cdl').resolve():raise ValueError('frozen CDL interpreter required')
    old_result=accepted();arrays,family,source,targets,proof=frozen_inputs()
    semantics=generator_semantics();manifest=p.manifest_rows();namespace=namespace_audit(manifest)
    small=p.calibration();references=original_reference(arrays,family,targets)
    if not load(OUT/'t19c_fixture_figure_gate.json')['all_checks_passed']:raise ValueError('figure schema/render fixture required')
    summary=(OUT/'t19c_preparation_pytest.txt').read_text(encoding='utf8').strip().splitlines()[-1]
    if '1 skipped' not in summary or 'failed' in summary or 'error' in summary:raise ValueError('preparation regression gate missing')
    generators={};smoke={};smoke_errors=[];g=observation_geometry()
    with method_guard():
        for n in p.ENVIRONMENTS:
            for r in p.REPLICATES:
                key=p.namespace(n,r);adapter=p.StatisticsCDL(n,r)
                gen=adapter.provenance()
                if gen!=load(OUT/'t19a_preflight.json')['generator_provenance'][n]:raise ValueError('physical generator provenance changed')
                generators[key]=gen;h,a,tau=adapter.generate(0,split='train_dev',details=True)
                smoke[key]=dict(H_sha256=array_hash(h),CIR_sha256=array_hash(a),tau_sha256=array_hash(tau))
                x=frames(h).reshape(10,4096);stream=StreamingStatistics(g.selected,g.data)
                stream.update(x,[0]);st=stream.freeze();xc=x-x.mean(axis=0)
                ep=float(np.max(abs(st.rpp-xc[:,g.selected].T@xc[:,g.selected].conj()/10)))
                ed=float(np.max(abs(st.rdp-xc[:,g.data].T@xc[:,g.selected].conj()/10)))
                if max(ep,ed)>1e-11:raise ValueError('real train smoke statistics estimator mismatch')
                smoke_errors.append(dict(namespace=key,Rpp_max_absolute_error=ep,Rdp_max_absolute_error=ed))
    environment=environments()
    if environment!=old_result['environment_after']:raise ValueError('frozen environment changed')
    csvsave(OUT/'t19c_manifest.csv',manifest);csvsave(OUT/'t19c_original_reference.csv',references)
    save(OUT/'t19c_seed_audit.json',namespace)
    snapshot=OUT/'t19c_execution_snapshot';hashes={}
    for path in SOURCES:
        destination=snapshot/path;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/path,destination);hashes[path]=sha(ROOT/path)
    frozen=history()
    for path in subprocess.check_output(['git','ls-files','docs'],cwd=ROOT,text=True).splitlines():frozen[path]=sha(ROOT/path)
    save(OUT/'t19c_preflight.json',dict(passed=True,t19b_commit=COMMIT,source_sha256=hashes,frozen_history_sha256=frozen,
        source_snapshot_root='outputs/t19c_execution_snapshot',method_provenance=proof,generator_semantics=semantics,
        namespace_audit=namespace,manifest_sha256=sha(OUT/'t19c_manifest.csv'),generator_provenance=generators,train_smoke=smoke,
        real_statistics_calibration=smoke_errors,synthetic_statistics_calibration=small,original_reference=references,
        environment_before=environment,preparation_seconds=time.perf_counter()-start,generator_calls=10,
        validation_generated=False,formal_test_sealed=True,c11_before_run='UNCONFIRMED'))
    print('T19C preflight PASS: ten disjoint train namespaces; exact physical generator/estimator; original operator references exact; frozen method; formal sealed',flush=True)


def save_statistics(path,target,*,large=False):
    if path.exists():raise ValueError('existing new statistics protected')
    values=dict(mean_p=target.mean_p,mean_d=target.mean_d,rpp=target.rpp,rdp=target.rdp,samples=target.samples)
    values['train_trajectory_keys' if large else 'train_trajectory_ids']=target.train_trajectory_ids
    np.savez_compressed(path,**values);path.chmod(0o444)
    return dict(path='outputs/'+path.name,sha256=sha(path),samples=target.samples,
                component_sha256={k:array_hash(getattr(target,k)) for k in ('mean_p','mean_d','rpp','rdp')})


def run():
    if RESULT.exists():raise ValueError('existing scientific run protected')
    pre=load(OUT/'t19c_preflight.json');verify_protected(pre)
    if not pre['passed'] or Path(sys.prefix).resolve()!=(ROOT/'.venv-cdl').resolve():raise ValueError('passed preflight and calibrated interpreter required')
    if sha(OUT/'t19c_manifest.csv')!=pre['manifest_sha256']:raise ValueError('manifest changed')
    arrays,family,source,original,proof=frozen_inputs()
    if proof!=pre['method_provenance']:raise ValueError('source/module/fold/means changed')
    start=time.perf_counter();captures=[];stats_rows=[];distances=[];delay_rows=[];hrows=[];records={n:[] for n in p.ENVIRONMENTS};runtimes={}
    g=observation_geometry()
    with method_guard():
        for n in p.ENVIRONMENTS:
            for r in p.REPLICATES:
                tick=time.perf_counter();ns=p.namespace(n,r);adapter=p.StatisticsCDL(n,r)
                if adapter.provenance()!=pre['generator_provenance'][ns]:raise ValueError('generator physical setup drift')
                stream=StreamingStatistics(g.selected,g.data);power=TrainReceivePowerAccumulator();delay=EffectiveDelayAudit();batch=[];ids=[]
                for i in range(500):
                    h,a,tau=adapter.generate(i,split='train_dev',details=True)
                    hashes=dict(H_sha256=array_hash(h),CIR_sha256=array_hash(a),tau_sha256=array_hash(tau))
                    if i==0 and hashes!=pre['train_smoke'][ns]:raise ValueError('first train trajectory exact smoke replay failed:STOP')
                    hrows.append(dict(environment=n,namespace=ns,replicate=r,split='train_dev',trajectory_id=i,**hashes))
                    delay.update(i,a,tau);power.update(i,h);batch.append(frames(h).reshape(10,4096));ids.append(i)
                    if len(batch)==10:stream.update(np.concatenate(batch),ids);batch=[];ids=[]
                    if (i+1)%100==0:print(f'{ns} train-only statistics {i+1}/500',flush=True)
                target=stream.freeze();reference=power.freeze()
                if target.samples!=5000 or target.train_trajectory_ids!=tuple(range(500)):raise ValueError('statistics population drift')
                info=save_statistics(OUT/f't19c_target_statistics_{n}_r{r}.npz',target)
                # Persist a complete population before operator analysis, for recovery.
                csvsave(OUT/f't19c_checkpoint_{n}_r{r}_hashes.csv',hrows[-500:])
                stats_rows.append(dict(environment=n,namespace=ns,replicate=r,trajectories=500,frames=5000,covariance_divisor='N population',
                    batch_frames=100,statistics_estimator='unchanged src.cdl_benchmark.StreamingStatistics',validation_used=False,**info))
                audit=delay.freeze(int(n.split('-')[-1]));delay_rows.append(dict(environment=n,replicate=r,namespace=ns,
                    train_receive_power=reference.reference,train_receive_power_per_rx=list(reference.per_rx_signal_power),
                    SNR_reference_recalibrated=False,**audit))
                comparisons={}
                for part in ('rpp','rdp'):
                    norm=float(np.linalg.norm(getattr(original[n],part)));distance=float(np.linalg.norm(getattr(target,part)-getattr(original[n],part)))
                    comparisons[part+'_distance_fro']=distance;comparisons[part+'_relative_distance_fro']=distance/norm
                distances.append(dict(environment=n,replicate=r,namespace=ns,reference='original T19A same delay500',**comparisons))
                values=diagnostic(n,r,target,arrays,family)
                for v in values:v['statistics_sha256']=info['sha256'];v['statistics_namespace']=ns
                captures+=values;records[n].append((ns,target));runtimes[ns]=time.perf_counter()-tick
        large=[];large_info=[]
        for n in p.ENVIRONMENTS:
            target=p.merge_sufficient_statistics(n,records[n])
            if target.samples!=25000 or len(target.train_trajectory_ids)!=2500:raise ValueError('large-stat weighting/population drift')
            info=save_statistics(OUT/f't19c_large_statistics_{n}.npz',target,large=True)
            large_info.append(dict(environment=n,input_namespaces=[ns for ns,_ in records[n]],trajectories=2500,frames=25000,
                aggregation='sample weighting plus between-mean covariance correction',original_T19A_included=False,used_for_primary_gate=False,**info))
            for value in diagnostic(n,0,target,arrays,family):large.append(dict(value,trajectories=2500,frames=25000,secondary_only=True,in_replication_count=False,statistics_sha256=info['sha256']))
    reference={(v['environment'],v['snr_db']):v['operator_span_capture_dense'] for v in pre['original_reference']}
    summary=p.summarize(captures,reference);flags=p.scientific_gate(captures)
    tables={'statistics_replicates':stats_rows,'operator_capture':captures,'statistics_distances':distances,
            'effective_delay':delay_rows,'large_stat_reference':large,'channel_hashes':hrows}
    for name,values in tables.items():csvsave(OUT/f't19c_{name}.csv',values)
    artifacts={f'outputs/t19c_{n}.csv':sha(OUT/f't19c_{n}.csv') for n in (*tables,'manifest','original_reference')}
    artifacts['outputs/t19c_seed_audit.json']=sha(OUT/'t19c_seed_audit.json')
    artifacts.update({info['path']:info['sha256'] for info in stats_rows+large_info})
    artifacts.update({f'outputs/t19c_checkpoint_{n}_r{r}_hashes.csv':sha(OUT/f't19c_checkpoint_{n}_r{r}_hashes.csv') for n in p.ENVIRONMENTS for r in p.REPLICATES})
    decision=('Condition-specific deficiency replicated with stable50ns control; only a separately assigned residual-subspace/minimal-factor necessity study may be considered; no automatic module.' if flags['delay_operator_deficiency_replication_supported'] and flags['delay50_control_stable'] else
              'Stop the third-direction route; classify the historical deficiency as target-statistics-sensitive observation.' if flags['delay_operator_deficiency_replication_contradicted'] else
              'Do not expand the method; review statistics sample size and covariance-estimator stability in a separately assigned task.')
    verify_protected(pre)
    result=dict(task='T19C',experiment_type=p.EXPERIMENT_TYPE,t19b_commit=COMMIT,method_frozen=True,
        delay_levels_ns=[50,200],statistics_replicates=5,trajectories_per_statistics_replicate=500,frames_per_trajectory=10,
        validation_generated=False,operator_capture_replicates=captures,capture_summary=summary,**flags,
        statistics_artifacts=stats_rows,statistics_distances=distances,effective_delay_audit=delay_rows,
        large_stat_reference=large,large_stat_artifacts=large_info,large_stat_enabled=True,
        original_reference=pre['original_reference'],original_in_replication_aggregate=False,original_in_large_stat_reference=False,
        c11_claim=p.C11,c9_claim_status='INCONCLUSIVE',c10_claim_status='INCONCLUSIVE',c6_claim_status=old.C6,c7_claim_status=old.C7,
        c8_claim_status='SUPPORTED_IN_FIXED_ENVIRONMENTS',t18a_materiality_status='PARTIAL',
        new_module_added=False,new_model_trained=False,new_coefficient_solver=False,rank_per_module=4,
        Pilot_CV_performed=False,BER_evaluations=False,bootstrap_performed=False,source_SNR_reference_recalibrated=False,
        generator_calls=dict(preflight_train_smoke=10,target_train=5000,validation=0),new_train_realizations=5000,
        same_statistics_for_both_SNRs=True,formal_test_used=False,formal_test_sealed=True,
        source_power=old.POWER,source_noise_variances={s:old.VARIANCES[s] for s in p.SNRS},
        namespace_audit=pre['namespace_audit'],method_provenance=proof,generator_semantics=pre['generator_semantics'],
        environment_before=pre['environment_before'],source_sha256=pre['source_sha256'],source_snapshot_root=pre['source_snapshot_root'],
        frozen_history_sha256=pre['frozen_history_sha256'],numeric_artifact_sha256=artifacts,
        calibration=pre['synthetic_statistics_calibration'],train_smoke_exact_replay=True,
        next_step_dependency=decision,runtime=dict(preparation_seconds=pre['preparation_seconds'],statistics_and_operator_seconds=time.perf_counter()-start,per_replicate_seconds=runtimes),
        all_checks_passed=False,completion_state='scientific replication complete; final audits pending')
    save(RESULT,result);print('T19C statistics replication complete; C11='+flags['c11_claim_status']+'; no validation/method changes',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['preflight','run']);args=parser.parse_args()
    {'preflight':preflight,'run':run}[args.mode]()
