"""T19B frozen-statistics exact matrix audit; never generates/replays a channel."""
import argparse
import ast
from contextlib import ExitStack, contextmanager
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.linalg import cholesky
from src import operator_space_audit as p
from src import frozen_affine_validation as old
from src.cdl_benchmark import FrozenStatistics, OfficialCDL, StreamingStatistics, array_hash
from src.delay_factor_challenge import FactorCDL
from src.lowrank_modules import WeightedModule
from scripts.run_cdl_mismatch import load, save, sha, csvsave, history, environments, source_statistics
from scripts.run_frozen_generalization import frozen_method
from scripts.run_pilot_cv import readcsv
OUT = ROOT/'outputs'
RESULT = OUT/'t19b_results.json'
COMMIT = '9b8ccfe80561085f037143bb8f7d1ba3cd8beee1'
SOURCES = ('src/operator_space_audit.py', 'scripts/run_operator_space_audit.py',
           'scripts/plot_operator_space_audit.py', 'scripts/audit_operator_space_audit.py',
           'tests/test_operator_space_audit.py', 'docs/t19b_operator_space_protocol.md')
TABLES = ('exact_mismatch_budget', 'operator_span_capture', 'mean_bias', 'rank4_span',
          'deployed_operator_efficiency', 'deployed_operator_windows',
          'delay_residual_similarity', 'delay_direction_capture')


@contextmanager
def no_generation():
    def forbidden(*args, **kwargs):
        raise RuntimeError('T19B forbids generator/statistics fitting/basis refit')
    with ExitStack() as stack:
        for cls, name in ((OfficialCDL, '__init__'), (OfficialCDL, 'generate'), (FactorCDL, 'generate'),
                          (StreamingStatistics, '__init__'), (StreamingStatistics, 'update'),
                          (StreamingStatistics, 'freeze'), (WeightedModule, 'build')):
            stack.enter_context(patch.object(cls, name, forbidden))
        stack.enter_context(patch.object(np.linalg, 'svd', forbidden))
        yield


def accepted():
    subprocess.check_call(['git', 'merge-base', '--is-ancestor', COMMIT, 'HEAD'], cwd=ROOT)
    from scripts.audit_delay_factor_challenge import verify
    r = load(OUT/'t19a_results.json')
    if not r['all_checks_passed'] or not verify(r):
        raise ValueError('accepted T19A required')
    for path in (*r['source_sha256'], 'outputs/t19a_results.json'):
        blob = subprocess.check_output(['git', 'show', f'{COMMIT}:{path}'], cwd=ROOT)
        if blob.replace(b'\r\n', b'\n') != (ROOT/path).read_bytes().replace(b'\r\n', b'\n'):
            raise ValueError('committed T19A snapshot drift:'+path)
    return r


def mean_policy(r):
    """Actual AST prediction call, implementation, saved output metadata and hashes."""
    runner = ROOT/'scripts/run_delay_factor_challenge.py'
    tree = ast.parse(runner.read_text(encoding='utf8'))
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant)
                              and t.slice.value == 'oracle2-affine' for t in n.targets))
    call = assignment.value
    arguments = [ast.unparse(x) for x in call.args]
    if arguments[1:3] != ["arrays['source_mean_p']", "arrays['source_mean_d']"]:
        raise ValueError('Oracle2 actual means differ from expected source policy')
    deploy = ast.parse((ROOT/'src/pilot_cv.py').read_text(encoding='utf8'))
    function = next(n for n in deploy.body if isinstance(n, ast.FunctionDef) and n.name == 'deployment')
    statements = [ast.unparse(n) for n in function.body if isinstance(n, ast.Assign)]
    if 'x = np.asarray(pilot) - source_mean_p' not in statements or 'base = source_mean_d + x @ source_w.T' not in statements:
        raise ValueError('mean subtraction/addition implementation changed')
    rows = [v for v in readcsv(OUT/'t19a_factor_nmse.csv') if v['method'] == 'oracle2-affine']
    if len(rows) != 16 or any(v['means'] != 'source' for v in rows):
        raise ValueError('saved Oracle2 evaluation policy drift')
    if not r['source_means_for_all_nonmatched'] or not all(v['source_means_used_for_actual_evaluation'] for v in r['oracle2_coordinates']):
        raise ValueError('JSON prediction provenance inconsistent')
    return dict(policy='source_means', actual_call_arguments=arguments,
                deployment_mean_subtraction_and_addition_verified=True,
                historical_runner_sha256=sha(runner), deployment_sha256=sha(ROOT/'src/pilot_cv.py'),
                prediction_metadata_rows=len(rows), committed_snapshot=COMMIT,
                evidence='actual AST call + deployment computation + JSON/CSV + accepted historical hashes')


def inputs(r):
    arrays, family, proof = frozen_method()
    if proof != r['frozen_method_provenance']:
        raise ValueError('frozen P/M and rank4 drift')
    source, source_proof = source_statistics()
    targets = {}
    for n in p.ENVIRONMENTS:
        info = r['target_statistics'][n]
        path = ROOT/info['path']
        if sha(path) != info['sha256']:
            raise ValueError('target statistics byte drift')
        with np.load(path, allow_pickle=False) as f:
            values = [f[k].copy() for k in ('mean_p', 'mean_d', 'rpp', 'rdp')]
            for k, a in zip(('mean_p', 'mean_d', 'rpp', 'rdp'), values):
                if array_hash(a) != info['component_sha256'][k] or a.dtype != np.complex128:
                    raise ValueError('target component hash/dtype drift')
                a.flags.writeable = False
            targets[n] = FrozenStatistics(*values, tuple(map(int, f['train_trajectory_ids'])), int(f['samples']))
    return arrays, family, source, targets, dict(frozen_method=proof, source_statistics=source_proof,
                                               target_statistics=r['target_statistics'])


def preflight():
    tick = time.perf_counter()
    path = OUT/'t19b_preflight.json'
    if path.exists():
        raise ValueError('existing preflight protected')
    if Path(sys.prefix).resolve() != (ROOT/'.venv-cdl').resolve():
        raise ValueError('frozen numerical environment required')
    with no_generation():
        r = accepted(); evidence = mean_policy(r); _, _, _, _, proof = inputs(r)
        small = p.calibration()
    fixture = load(OUT/'t19b_fixture_figure_gate.json')
    if not fixture['all_checks_passed']:
        raise ValueError('small figure calibration required')
    startup=(OUT/'t19b_startup_pytest.txt').read_text(encoding='utf8').strip().splitlines()[-1]
    related=(OUT/'t19b_preparation_pytest_final.txt').read_text(encoding='utf8').strip().splitlines()[-1]
    if not startup.startswith('1769 passed') or not related.startswith('31 passed, 1 skipped'):
        raise ValueError('startup and preparation calibration gates missing')
    environment = environments()
    if environment != r['environment_after']:
        raise ValueError('frozen environment changed')
    frozen = history()
    for path0 in subprocess.check_output(['git', 'ls-files', 'docs'], cwd=ROOT, text=True).splitlines():
        frozen[path0] = sha(ROOT/path0)
    snapshot = OUT/'t19b_execution_snapshot'
    sources = {}
    for source in SOURCES:
        dest = snapshot/source; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/source, dest); sources[source] = sha(ROOT/source)
    save(path, dict(passed=True, t19a_commit=COMMIT, historical_mean_policy=evidence,
                    provenance=proof, calibration=small, environment_before=environment,
                    frozen_history_sha256=frozen, source_sha256=sources,
                    source_snapshot_root='outputs/t19b_execution_snapshot',
                    startup_regression=startup,preparation_regression=related,
                    generator_calls=0, statistics_refits=0, formal_test_sealed=True,
                    c10_before_run='UNCONFIRMED', preparation_seconds=time.perf_counter()-tick))
    print('T19B preflight PASS: actual source means; frozen statistics; independent identities; no generator/refit; formal sealed', flush=True)


def verify_hashes(pre):
    for group in ('frozen_history_sha256', 'source_sha256'):
        for path, h in pre[group].items():
            if sha(ROOT/path) != h:
                raise ValueError('protected artifact drift:'+path)
    for path, h in pre['source_sha256'].items():
        if sha(ROOT/pre['source_snapshot_root']/path) != h:
            raise ValueError('execution snapshot drift')


def scientific():
    tick = time.perf_counter()
    if RESULT.exists():
        raise ValueError('existing scientific results protected')
    pre = load(OUT/'t19b_preflight.json'); verify_hashes(pre)
    with no_generation():
        historical = accepted();arrays,family,source,targets,proof = inputs(historical)
        if proof != pre['provenance']:
            raise ValueError('input provenance changed')
        tables = {n:[] for n in TABLES}; residuals={}; weights={}; models={}; traces=[]
        oracle = {(v['environment'],v['snr_db']):v for v in historical['oracle2_coordinates']}
        old_metrics = {(v['environment'],v['snr_db']):v for v in historical['factor_metrics']}
        saved = readcsv(OUT/'t19a_weights.csv')
        for n in p.ENVIRONMENTS:
            for s in p.SNRS:
                st = targets[n]; prepared = st.prepare(old.VARIANCES[s]);wt = prepared.weight
                ws = arrays[f'Wsource_{s}'];mods=[arrays[f'DeltaW{m}_{s}'] for m in ('P','M')]
                mods4 = [family[m,s,4].A@family[m,s,4].Q for m in ('P','M')]
                lt = cholesky(st.rpp+old.VARIANCES[s]*np.eye(64),lower=True)
                dp=st.mean_p-arrays['source_mean_p'];dd=st.mean_d-arrays['source_mean_d']
                fit=p.operator_fit(ws,wt,lt,mods);fit4=p.operator_fit(ws,wt,lt,mods4)
                total=p.mean_aware_fit(ws,wt,lt,mods,dp,dd)
                opw=ws+sum(c*m for c,m in zip(fit['coefficients'],mods))
                tw=ws+sum(c*m for c,m in zip(total['coefficients'],mods))
                sw=p.mismatch(ws,wt,lt,dp,dd);ow=p.mismatch(opw,wt,lt,dp,dd)
                mw=p.mismatch(wt,wt,lt,dp,dd);tb=p.mismatch(tw,wt,lt,dp,dd)
                capture=1-p.ratio(fit['excess'],sw['E_op']) if sw['E_op']>0 else None
                capture4=1-p.ratio(fit4['excess'],sw['E_op']) if sw['E_op']>0 else None
                key=dict(environment=n,snr_db=s,pure_delay=n in p.PURE)
                baseline=dict(**key,E_op_source=sw['E_op'],E_mean_source=sw['E_mean'],E_total_source=sw['E_total'],
                    mean_fraction_source=p.ratio(sw['E_mean'],sw['E_total']),E_op_oracle2=ow['E_op'],
                    E_mean_oracle2=ow['E_mean'],E_total_oracle_opcoords=ow['E_total'],
                    matched_operator_source_mean_floor=mw['E_mean'],matched_control_operator_excess=mw['E_op'],
                    unit='per-frame data-domain error energy;3968 complex coefficients',
                    absolute_expected_MSE_available=False)
                tables['exact_mismatch_budget'].append(baseline)
                cap=dict(**key,alpha_op=float(fit['coefficients'][0]),beta_op=float(fit['coefficients'][1]),
                         E_op_source=sw['E_op'],E_op_oracle2=fit['excess'],operator_span_capture_dense=capture,
                         historical_validation_representation_closure=old_metrics[n,s]['representation_closure'])
                tables['operator_span_capture'].append(cap)
                tables['mean_bias'].append(dict(**key,alpha_total=float(total['coefficients'][0]),beta_total=float(total['coefficients'][1]),
                    lambda0_total=float(1-total['coefficients'].sum()),E_total_oracle_opcoords=ow['E_total'],
                    E_total_oracle_totalcoords=tb['E_total'],mean_aware_coordinate_gain=(1-p.ratio(tb['E_total'],ow['E_total']) if ow['E_total']>0 else None),
                    E_op_totalcoords=tb['E_op'],E_mean_totalcoords=tb['E_mean'],diagnostic_only=True))
                tables['rank4_span'].append(dict(**key,alpha4_op=float(fit4['coefficients'][0]),beta4_op=float(fit4['coefficients'][1]),
                    E_op_lr4=fit4['excess'],operator_span_capture_lr4=capture4,
                    rank4_span_capture_loss=(capture-capture4 if capture is not None and capture4 is not None else None),
                    rank4_residual_energy_ratio=p.ratio(fit4['excess'],fit['excess']),
                    historical_validation_compression_penalty_db=old_metrics[n,s]['compression_penalty_db'],
                    audit_only_dense_reconstruction=True))
                oldco=np.array([oracle[n,s]['alpha_oracle'],oracle[n,s]['beta_oracle']]);oldex=oracle[n,s]['weighted_filter_excess_energy_per_frame']
                coerr=np.abs(fit['coefficients']-oldco);exerr=abs(fit['excess']-oldex)
                if not np.allclose(fit['coefficients'],oldco,rtol=p.CALIBRATION_RTOL,atol=p.CALIBRATION_RTOL) or not np.isclose(fit['excess'],oldex,rtol=p.CALIBRATION_RTOL,atol=p.CALIBRATION_RTOL):
                    raise ValueError('independent historical Oracle2 regression failed')
                traces.append(dict(**key,coordinates_bitwise_exact=bool(np.array_equal(fit['coefficients'],oldco)),
                    coordinate_max_absolute_error=float(coerr.max()),coordinate_max_relative_error=float((coerr/np.maximum(np.abs(oldco),1e-300)).max()),
                    objective_bitwise_exact=fit['excess']==oldex,objective_absolute_error=exerr,objective_relative_error=exerr/max(abs(oldex),1e-300),
                    normal_relative_residual=fit['normal_relative_residual'],matched_solve_relative_residual=prepared.solve_relative_residual))
                group=[r for r in saved if (r['environment'],int(r['snr_db']),r['method'])==(n,s,'affine')]
                if len(group)!=600 or {(int(r['trajectory_id']),int(r['frame_id'])) for r in group}!={(i,j) for i in range(500,600) for j in range(4,10)}:
                    raise ValueError('all original affine windows required')
                vals=[];maxdirect=0.
                for j,r in enumerate(group):
                    co=np.array([float(r['alpha']),float(r['beta'])])
                    ex=p.deployed_excess(co,fit['coefficients'],fit['gram'],fit['excess'])
                    eff=p.deployed_efficiency(sw['E_op'],ex,fit['excess']);vals.append(eff)
                    # Fixed first/last and every100th window: direct full-matrix regression.
                    if j in (0,599) or j%100==0:
                        direct=p.operator_excess(ws+sum(c*m for c,m in zip(co,mods)),wt,lt)
                        err=abs(direct-ex);maxdirect=max(maxdirect,err)
                        if not np.isclose(direct,ex,rtol=p.CALIBRATION_RTOL,atol=p.CALIBRATION_RTOL):raise ValueError('deployed quadratic sentinel failed')
                    tables['deployed_operator_windows'].append(dict(**key,trajectory_id=int(r['trajectory_id']),frame_id=int(r['frame_id']),
                        alpha_hat=float(co[0]),beta_hat=float(co[1]),E_op_deployed=ex,deployed_operator_efficiency=eff,
                        undefined_reason=('nonpositive/nonfinite source-minus-oracle denominator' if eff is None else '')))
                defined=np.array([v for v in vals if v is not None]);quantile=np.quantile(defined,[.25,.5,.75]) if len(defined) else [None]*3
                tables['deployed_operator_efficiency'].append(dict(**key,windows=600,independent_trajectories=100,
                    q25=quantile[0],median=quantile[1],q75=quantile[2],minimum=float(defined.min()) if len(defined) else None,
                    maximum=float(defined.max()) if len(defined) else None,undefined_windows=600-len(defined),
                    direct_matrix_sentinel_max_absolute_error=maxdirect,uncertainty='descriptive correlated windows;no CI'))
                residual=wt-opw
                residuals[f'R_delay_{n}_{s}']=residual
                weights[f'Wtarget_{n}_{s}']=wt
                models[n,s]=(ws,wt,lt,mods,sw['E_op'],capture,residual)
                print(f'T19B exact operator/bias/compression completed {n}/{s}',flush=True)
        for s in p.SNRS:
            ls=cholesky(source.rpp+old.VARIANCES[s]*np.eye(64),lower=True)
            candidate=models[p.DEVELOPMENT,s][-1]
            residuals[f'D_delay_200_{s}']=candidate.copy();residuals[f'L_source_{s}']=ls
            for n in p.ENVIRONMENTS:residuals[f'Rbar_{n}_{s}']=models[n,s][-1]@ls
            for n in p.HELDOUT:
                ws,wt,lt,mods,source_ex,capture,residual=models[n,s]
                cosine=p.signed_cosine(candidate,residual,ls)
                tables['delay_residual_similarity'].append(dict(environment=n,snr_db=s,candidate_condition=p.DEVELOPMENT,
                    cos_delay=cosine,undefined_reason=('zero/nonfinite residual norm' if cosine is None else ''),
                    weighting='same source Cz Cholesky per SNR',self_condition=False,primary_gate=n in p.PRIMARY))
                joint=p.operator_fit(ws,wt,lt,[*mods,candidate]);capture3=1-p.ratio(joint['excess'],source_ex) if source_ex>0 else None
                tables['delay_direction_capture'].append(dict(environment=n,snr_db=s,alpha3=float(joint['coefficients'][0]),
                    beta3=float(joint['coefficients'][1]),gamma3=float(joint['coefficients'][2]),E_op_3d=joint['excess'],
                    operator_span_capture_dense=capture,operator_span_capture_3d=capture3,
                    delay_direction_incremental_capture=(capture3-capture if capture3 is not None and capture is not None else None),
                    primary_gate=n in p.PRIMARY,self_condition=False,oracle_only=True,joint_fit=True,
                    normal_relative_residual=joint['normal_relative_residual']))
        dense=p.dense_gate(tables['operator_span_capture']);reuse=p.reuse_gate(tables['delay_direction_capture'])
        for name,rows in tables.items():csvsave(OUT/f't19b_{name}.csv',rows)
        for name,values in (('delay_residual_directions',residuals),('derived_matched_operators',weights)):
            path=OUT/f't19b_{name}.npz'
            if path.exists():raise ValueError('existing arrays protected')
            np.savez(path,**values);path.chmod(0o444)
        artifacts={f'outputs/t19b_{n}.csv':sha(OUT/f't19b_{n}.csv') for n in TABLES}
        artifacts.update({f'outputs/t19b_{n}.npz':sha(OUT/f't19b_{n}.npz') for n in ('delay_residual_directions','derived_matched_operators')})
        decision=('Dense sufficient: do not research a third module; inspect mean/metric/compression.' if dense['dense_pm_operator_space_delay_sufficient'] else
                  'Only a separately assigned T19C minimal third-direction necessity gate may be considered.' if dense['condition_specific_operator_deficiency'] and reuse['delay_residual_direction_reusable'] else
                  'Do not add a fixed single delay module: the tested direction has weak cross-context reuse.' if dense['condition_specific_operator_deficiency'] and reuse['delay_direction_reuse_weak'] else
                  'Partial evidence: no third module authorized; retain condition-specific operator/compression diagnostics.')
        r=dict(task='T19B',experiment_type=p.EXPERIMENT_TYPE,t19a_commit=COMMIT,
            t19a_oracle2_mean_policy='source_means',oracle2_mean_policy_provenance=pre['historical_mean_policy'],
            exact_identity_calibration=pre['calibration'],historical_oracle2_regression=traces,
            operator_excess=tables['exact_mismatch_budget'],mean_bias=tables['mean_bias'],
            operator_span_capture_dense=tables['operator_span_capture'],operator_span_capture_lr4=tables['rank4_span'],
            rank4_span_capture_loss=tables['rank4_span'],deployed_operator_efficiency=tables['deployed_operator_efficiency'],
            delay_residual_cosines=tables['delay_residual_similarity'],operator_span_capture_3d=tables['delay_direction_capture'],
            delay_direction_incremental_capture=tables['delay_direction_capture'],**dense,**reuse,
            c9_claim_status='INCONCLUSIVE',c10_claim=p.C10,c6_claim_status=old.C6,c7_claim_status=old.C7,
            c8_claim_status='SUPPORTED_IN_FIXED_ENVIRONMENTS',t18a_materiality_status='PARTIAL',
            t19a_delay_factor_gate_partial=True,t19a_rank4_out_of_factor_warning=True,
            new_channel_generated=False,deterministic_channel_replay_used=False,generator_calls=0,statistics_refits=0,
            new_module_added=False,new_model_trained=False,third_module_deployment=False,gamma_pilot_fitting=False,
            formal_test_used=False,formal_test_sealed=True,inference_implementation_unchanged=True,
            audit_only_dense_reconstruction=True,absolute_expected_MSE_available=False,
            target_matched_W_historical_array_hash_available=False,
            target_matched_W_evidence='new deterministic Cholesky reconstruction from immutable T19A Rpp/Rdp; historical coordinates/objectives regressed',
            array_metadata=dict(dtype='complex128',order='data3968 by pilot64',energy_unit='per-frame data-domain coefficient error energy',
                residual_array_sha256={k:array_hash(v) for k,v in residuals.items()},derived_W_array_sha256={k:array_hash(v) for k,v in weights.items()},
                candidate_source_condition=p.DEVELOPMENT,candidate_evaluation_only=True,
                source_statistics=proof['source_statistics'],target_statistics=proof['target_statistics']),
            source_sha256=pre['source_sha256'],source_snapshot_root=pre['source_snapshot_root'],
            frozen_history_sha256=pre['frozen_history_sha256'],numeric_artifact_sha256=artifacts,
            environment_before=pre['environment_before'],next_step_dependency=decision,
            runtime=dict(preparation_seconds=pre['preparation_seconds'],matrix_audit_seconds=time.perf_counter()-tick),
            all_checks_passed=False,completion_state='scientific audit complete; final regression pending')
        verify_hashes(pre);save(RESULT,r)
        print('T19B matrix audit complete; C10='+reuse['c10_claim_status']+'; no receiver/module deployment',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['preflight','run']);args=parser.parse_args()
    {'preflight':preflight,'run':scientific}[args.mode]()
