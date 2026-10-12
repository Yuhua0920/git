"""T19D frozen-statistics pure-delay PCA and held-out operator transfer."""
import argparse
from contextlib import contextmanager,ExitStack
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.linalg import cholesky
from src import delay_residual_subspace as p
from src import operator_space_audit as math
from src import frozen_affine_validation as old
from src.cdl_benchmark import OfficialCDL,StreamingStatistics,array_hash
from src.delay_factor_challenge import FactorCDL
from src.target_statistics_replication import StatisticsCDL
from src.lowrank_modules import WeightedModule
from scripts.run_cdl_mismatch import load,save,sha,csvsave,history,environments
from scripts.run_operator_space_audit import inputs as old_inputs
from scripts.audit_target_statistics_replication import read_statistics
OUT=ROOT/'outputs';RESULT=OUT/'t19d_results.json'
COMMIT='80bdee1dfef0f87ade489a3256c4c87f1cfed742'
SOURCES=('src/delay_residual_subspace.py','scripts/run_delay_residual_subspace.py',
         'scripts/audit_delay_residual_subspace.py','scripts/plot_delay_residual_subspace.py',
         'tests/test_delay_residual_subspace.py','docs/t19d_residual_subspace_protocol.md')
TABLES=('residual_pca','loro_stability','heldout_transfer','direction_similarity','direction_comparison','development_reconstruction')


@contextmanager
def guard():
    def forbidden(*args,**kwargs):raise RuntimeError('T19D forbids channel generation/statistics refit/deployed fitting/BER')
    with ExitStack() as stack:
        for cls,name in ((OfficialCDL,'__init__'),(OfficialCDL,'generate'),(FactorCDL,'generate'),(StatisticsCDL,'generate'),
                         (StreamingStatistics,'__init__'),(StreamingStatistics,'update'),(StreamingStatistics,'freeze'),(WeightedModule,'build')):
            stack.enter_context(patch.object(cls,name,forbidden))
        from src import pilot_cv,link,numerics
        for module,name in ((pilot_cv,'deployment'),(pilot_cv,'primary_prediction'),(link,'generate_master_qpsk'),
                            (link,'detect_data_zf'),(link,'detect_data_linear_mmse'),(link,'demodulate_data_bits'),(numerics,'complex_gaussian_noise')):
            stack.enter_context(patch.object(module,name,forbidden))
        stack.enter_context(patch.object(pilot_cv.PilotCV,'fit',forbidden))
        yield


def accepted():
    subprocess.check_call(['git','merge-base','--is-ancestor',COMMIT,'HEAD'],cwd=ROOT)
    from scripts.audit_target_statistics_replication import verify
    r=load(OUT/'t19c_results.json')
    if not r['all_checks_passed'] or not verify(r):raise ValueError('accepted T19C required')
    for path in (*r['source_sha256'],*r['claim_document_sha256'],'outputs/t19c_results.json'):
        blob=subprocess.check_output(['git','show',f'{COMMIT}:{path}'],cwd=ROOT)
        if blob.replace(b'\r\n',b'\n')!=(ROOT/path).read_bytes().replace(b'\r\n',b'\n'):
            raise ValueError('committed T19C snapshot drift:'+path)
    return r


def inputs():
    c=load(OUT/'t19c_results.json');a=load(OUT/'t19a_results.json')
    arrays,_,source,targets,proof=old_inputs(a)
    dev={}
    for info in c['statistics_artifacts']:
        key=(info['environment'],info['replicate']);dev[key]=read_statistics(info)
    if set(dev)!={(n,r) for n in p.DEVELOPMENT for r in p.REPLICATES}:raise ValueError('ten frozen development statistics required')
    return arrays,source,dev,{n:targets[n] for n in p.HELDOUT},dict(frozen_method=proof['frozen_method'],source_statistics=proof['source_statistics'],
        development_statistics=c['statistics_artifacts'],heldout_statistics={n:a['target_statistics'][n] for n in p.HELDOUT},
        no_large_stat_or_original_pure_delay_in_basis=True)


def operator(st,s,arrays):
    prepared=st.prepare(old.VARIANCES[s]);lower=cholesky(st.rpp+old.VARIANCES[s]*np.eye(64),lower=True)
    if prepared.solve_relative_residual>1e-11:raise ValueError('matched solve failed')
    source=arrays[f'Wsource_{s}'];mods=[arrays[f'DeltaW{m}_{s}'] for m in ('P','M')]
    fit=math.operator_fit(source,prepared.weight,lower,mods)
    residual=prepared.weight-(source+sum(c*m for c,m in zip(fit['coefficients'],mods)))
    excess=math.operator_excess(source,prepared.weight,lower)
    if excess<=0:raise ValueError('positive operator mismatch required')
    return source,prepared.weight,lower,mods,fit,residual,excess


def regression(arrays,source,dev,held):
    b=load(OUT/'t19b_results.json');c=load(OUT/'t19c_results.json')
    bi={(v['environment'],v['snr_db']):v for v in b['operator_span_capture_dense']}
    ci={(v['environment'],v['replicate'],v['snr_db']):v for v in c['operator_capture_replicates']}
    di={(v['environment'],v['snr_db']):v for v in b['operator_span_capture_3d']}
    evidence=[]
    with np.load(OUT/'t19b_derived_matched_operators.npz',allow_pickle=False) as f:weights={k:f[k] for k in f.files}
    with np.load(OUT/'t19b_delay_residual_directions.npz',allow_pickle=False) as f:directions={s:f[f'D_delay_200_{s}'] for s in p.SNRS}
    for (n,r),st in dev.items():
        for s in p.SNRS:
            ws,wt,lt,mods,fit,res,ex=operator(st,s,arrays);ref=ci[n,r,s]
            if 1-fit['excess']/ex!=ref['operator_span_capture_dense'] or ex!=ref['E_op_source'] or fit['excess']!=ref['E_op_oracle2']:
                raise ValueError('T19C primary operator reconstruction not exact')
    for n,st in held.items():
        for s in p.SNRS:
            ws,wt,lt,mods,fit,res,ex=operator(st,s,arrays);ref=bi[n,s]
            if not np.array_equal(wt,weights[f'Wtarget_{n}_{s}']):raise ValueError('historical matched W reconstruction drift')
            if 1-fit['excess']/ex!=ref['operator_span_capture_dense'] or fit['excess']!=ref['E_op_oracle2'] or ex!=ref['E_op_source']:
                raise ValueError('held-out q0 historical replay not exact')
            joint=math.operator_fit(ws,wt,lt,[*mods,directions[s]]);oldref=di[n,s]
            if 1-joint['excess']/ex!=oldref['operator_span_capture_3d'] or joint['excess']!=oldref['E_op_3d']:
                raise ValueError('historical T19B D200 replay not exact')
            evidence.append(dict(environment=n,snr_db=s,q0_exact=True,matched_W_array_exact=True,D200_capture_exact=True,D200_objective_exact=True))
    return dict(T19C_development_conditions_exact=20,T19B_heldout_conditions_exact=12,heldout_regression=evidence,
        D200_file_sha256=sha(OUT/'t19b_delay_residual_directions.npz'),historical_core_sha256=sha(ROOT/'src/operator_space_audit.py'))


def verify_protected(pre):
    for group in ('source_sha256','frozen_history_sha256'):
        for path,h in pre[group].items():
            if sha(ROOT/path)!=h:raise ValueError('protected artifact drift:'+path)
    for path,h in pre['source_sha256'].items():
        if sha(ROOT/pre['source_snapshot_root']/path)!=h:raise ValueError('execution snapshot drift')


def preflight():
    tick=time.perf_counter();path=OUT/'t19d_preflight.json'
    if path.exists():raise ValueError('existing preflight protected')
    if Path(sys.prefix).resolve()!=(ROOT/'.venv-cdl').resolve():raise ValueError('frozen CDL numerical interpreter required')
    with guard():
        c=accepted();arrays,source,dev,held,proof=inputs();small=p.calibration();replay=regression(arrays,source,dev,held)
    if not load(OUT/'t19d_fixture_figure_gate.json')['all_checks_passed']:raise ValueError('small figure calibration required')
    startup=(OUT/'t19d_startup_pytest.txt').read_text(encoding='utf8').strip().splitlines()[-1]
    preparation=(OUT/'t19d_preparation_pytest_final.txt').read_text(encoding='utf8').strip().splitlines()[-1]
    if not startup.startswith('1830 passed') or '1 skipped' not in preparation or 'failed' in preparation or 'error' in preparation:
        raise ValueError('actual startup/preparation gates missing')
    environment=environments()
    if environment!=c['environment_after']:raise ValueError('frozen numerical environment changed')
    frozen=history()
    for file in subprocess.check_output(['git','ls-files','docs'],cwd=ROOT,text=True).splitlines():frozen[file]=sha(ROOT/file)
    snapshot=OUT/'t19d_execution_snapshot';hashes={}
    for file in SOURCES:
        dst=snapshot/file;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/file,dst);hashes[file]=sha(ROOT/file)
    save(path,dict(passed=True,t19c_commit=COMMIT,calibration=small,historical_regression=replay,method_provenance=proof,
        source_sha256=hashes,source_snapshot_root='outputs/t19d_execution_snapshot',frozen_history_sha256=frozen,
        environment_before=environment,startup_regression=startup,preparation_regression=preparation,preparation_seconds=time.perf_counter()-tick,
        generator_calls=0,statistics_refits=0,formal_test_sealed=True,c12_before_run='UNCONFIRMED'))
    print('T19D preflight PASS: T19C committed; exact development/q0/D200 regression; real geometry; no generator/refit; formal sealed',flush=True)


def compute(arrays,source,dev,held):
    tables={name:[] for name in TABLES};residual_arrays={};basis_arrays={};basis_metadata={};similarity_summary=[]
    historical=load(OUT/'t19b_results.json');oldindex={(v['environment'],v['snr_db']):v for v in historical['operator_span_capture_3d']}
    with np.load(OUT/'t19b_delay_residual_directions.npz',allow_pickle=False) as f:d200={s:f[f'D_delay_200_{s}'] for s in p.SNRS}
    for s in p.SNRS:
        ls=cholesky(source.rpp+old.VARIANCES[s]*np.eye(64),lower=True);residuals={}
        for key,st in dev.items():
            ws,wt,lt,mods,fit,res,ex=operator(st,s,arrays);residuals[key]=res
            n,r=key;v,norm=p.unit(p.augment(res@ls))
            residual_arrays[f'R_{n}_r{r}_{s}']=res;residual_arrays[f'Z_{n}_r{r}_{s}']=res@ls
            residual_arrays[f'u_{n}_r{r}_{s}']=v
        residual_arrays[f'L_source_{s}']=ls
        for k in p.REPLICATES:
            training={key:value for key,value in residuals.items() if key[1]!=k};fold=p.learn(training,ls,excluded_pair=k)
            for n in p.DEVELOPMENT:
                for q in (1,2):
                    tables['loro_stability'].append(dict(snr_db=s,heldout_pair=k,environment=n,replicate=k,q=q,
                        source_weighted_projection_capture=p.projection_capture(residuals[n,k],ls,fold['vectors'],q),
                        construction_keys=[list(key) for key in fold['construction_keys']],construction_rows=8,heldout_rows=2,used_for_transfer_gate=False))
        basis=p.learn(residuals,ls)
        basis_metadata[str(s)]=dict(construction_keys=[list(key) for key in basis['construction_keys']],
            sign_sentinels=basis['sign_sentinels'],construction=basis['construction'],centered=basis['centered'])
        for i in (0,1):basis_arrays[f'D{i+1}_{s}']=basis['directions'][i]
        basis_arrays[f'weighted_real_vectors_{s}']=basis['vectors'];basis_arrays[f'L_source_{s}']=ls
        basis_arrays[f'singular_values_{s}']=basis['singular_values'];basis_arrays[f'input_norms_{s}']=basis['input_norms']
        for i,(value,ratio,cumulative) in enumerate(zip(basis['singular_values'],basis['explained_energy_ratio'],basis['cumulative_energy']),1):
            tables['residual_pca'].append(dict(snr_db=s,component=i,singular_value=float(value),explained_energy_ratio=float(ratio),
                cumulative_energy=float(cumulative),construction_rows=10,real_augmented=True,direction_normalized=True,centered=False,used_for_transfer_gate=False))
        keys=basis['construction_keys'];normed=np.stack([p.unit(p.augment(residuals[key]@ls))[0] for key in keys])
        cross=[]
        for i,key in enumerate(keys):
            for j in range(i+1,len(keys)):
                other=keys[j];cos=float(normed[i]@normed[j]);between=key[0]!=other[0]
                if between:cross.append(cos)
                tables['direction_similarity'].append(dict(snr_db=s,environment_a=key[0],replicate_a=key[1],environment_b=other[0],replicate_b=other[1],
                    cross_delay=between,signed_cosine=cos,weighting='common source Cz per SNR',diagnostic_only=True))
        similarity_summary.append(dict(snr_db=s,cross_delay_pairs=25,minimum=float(min(cross)),median=float(np.median(cross)),
            maximum=float(max(cross)),mean=float(np.mean(cross)),negative_cosine_count=sum(v<0 for v in cross),physical_delay_law_claimed=False))
        for key in keys:
            tables['development_reconstruction'].append(dict(environment=key[0],replicate=key[1],snr_db=s,
                projection_capture_q1=p.projection_capture(residuals[key],ls,basis['vectors'],1),
                projection_capture_q2=p.projection_capture(residuals[key],ls,basis['vectors'],2),development_only=True,transfer_evidence=False))
        for n in p.HELDOUT:
            ws,wt,lt,mods,fit0,res,ex=operator(held[n],s,arrays)
            fit1=p.joint_fit(ws,wt,lt,mods,basis['directions'],1);fit2=p.joint_fit(ws,wt,lt,mods,basis['directions'],2)
            c0=1-fit0['excess']/ex;c1=1-fit1['excess']/ex;c2=1-fit2['excess']/ex
            gram1=np.linalg.eigvalsh(fit1['gram']);gram2=np.linalg.eigvalsh(fit2['gram'])
            if fit1['excess']>fit0['excess']+1e-11*max(fit0['excess'],1) or fit2['excess']>fit1['excess']+1e-11*max(fit1['excess'],1):
                raise ValueError('nested joint LS numerical monotonicity failed; no clipping')
            row=dict(environment=n,snr_db=s,capture_q0=c0,capture_q1=c1,capture_q2=c2,increment_q1=c1-c0,increment_q2=c2-c0,marginal_q2=c2-c1,
                E_op_source=ex,E_op_q0=fit0['excess'],E_op_q1=fit1['excess'],E_op_q2=fit2['excess'],
                alpha_q0=float(fit0['coefficients'][0]),beta_q0=float(fit0['coefficients'][1]),
                alpha_q1=float(fit1['coefficients'][0]),beta_q1=float(fit1['coefficients'][1]),gamma1_q1=float(fit1['coefficients'][2]),
                alpha_q2=float(fit2['coefficients'][0]),beta_q2=float(fit2['coefficients'][1]),gamma1_q2=float(fit2['coefficients'][2]),gamma2_q2=float(fit2['coefficients'][3]),
                q1_normal_relative_residual=fit1['normal_relative_residual'],q2_normal_relative_residual=fit2['normal_relative_residual'],
                q1_Gram_condition=float(gram1[-1]/gram1[0]),q2_Gram_condition=float(gram2[-1]/gram2[0]),
                q1_promising=c1>=.70 and c1-c0>=.10,q2_promising=c2>=.70 and c2-c0>=.10,
                joint_real_LS=True,oracle_only=True,in_basis_construction=False,energy_unit='per-frame data-domain energy;3968 coefficients')
            tables['heldout_transfer'].append(row)
            historical_fit=math.operator_fit(ws,wt,lt,[*mods,d200[s]]);ref=oldindex[n,s]
            if 1-historical_fit['excess']/ex!=ref['operator_span_capture_3d']:raise ValueError('T19B D200 comparison drift')
            tables['direction_comparison'].append(dict(environment=n,snr_db=s,T19B_D200_capture=ref['operator_span_capture_3d'],T19D_PCA_D1_capture=c1,
                PCA_minus_D200_capture=c1-ref['operator_span_capture_3d'],T19B_exact_regression=True,diagnostic_only=True))
        print(f'T19D SNR{s}: five excluded-pair folds; final ten-row real PCA; six held-out contexts',flush=True)
    return tables,residual_arrays,basis_arrays,basis_metadata,similarity_summary


def run():
    if RESULT.exists():raise ValueError('existing scientific result protected')
    pre=load(OUT/'t19d_preflight.json');verify_protected(pre)
    if not pre['passed'] or Path(sys.prefix).resolve()!=(ROOT/'.venv-cdl').resolve():raise ValueError('passed preflight/pinned interpreter required')
    tick=time.perf_counter()
    with guard():
        arrays,source,dev,held,proof=inputs()
        if proof!=pre['method_provenance']:raise ValueError('method/statistics provenance drift')
        replay=regression(arrays,source,dev,held)
        if replay!=pre['historical_regression']:raise ValueError('historical replay drift')
        tables,residual_arrays,basis_arrays,metadata,similarity=compute(arrays,source,dev,held)
    artifacts={};array_metadata={}
    for name,values in tables.items():
        path=OUT/f't19d_{name}.csv';csvsave(path,values);artifacts['outputs/'+path.name]=sha(path)
    for name,values in (('pure_delay_residuals',residual_arrays),('delay_basis',basis_arrays)):
        path=OUT/f't19d_{name}.npz'
        if path.exists():raise ValueError('existing arrays protected')
        np.savez_compressed(path,**values);path.chmod(0o444);artifacts['outputs/'+path.name]=sha(path)
        array_metadata[name]={k:dict(sha256=array_hash(v),dtype=str(v.dtype),shape=list(v.shape)) for k,v in values.items()}
    flags=p.transfer_gate(tables['heldout_transfer'])
    decision=('Only a separately assigned minimal third-module deployment necessity test may be considered.' if flags['single_delay_direction_transfer_supported'] else
        'Review method complexity/dimensionality before any two-module addition.' if flags['two_direction_delay_subspace_transfer_supported'] else
        'Stop fixed delay-module expansion: stable deficiency has weak tested low-dimensional cross-context transfer.' if flags['delay_residual_subspace_transfer_weak'] else
        'Do not expand methods; review profile/delay contexts of transfer success and failure.')
    result=dict(task='T19D',experiment_type=p.EXPERIMENT_TYPE,t19c_commit=COMMIT,method_frozen=True,new_channel_generated=False,
        generator_calls=0,statistics_refits=0,development_conditions=['C30/50','C30/200'],heldout_conditions=['C90/50','C90/200','B30/50','B30/200','B90/50','B90/200'],
        module_space_dimensions=[1,2],dimension_terminology='additional module-space dimension q;not matrix rank',
        real_augmented_mapping='C-order [vec(Re Z);vec(Im Z)]',primary_normalization='unit norm per residual;equal weight;uncentered real SVD',
        source_common_weighting=True,target_weighting_used_for_basis=False,development_heldout_separation=True,
        pca_explained_energy=tables['residual_pca'],loro_stability=tables['loro_stability'],heldout_transfer=tables['heldout_transfer'],
        direction_similarity=tables['direction_similarity'],cross_delay_cosine_summary=similarity,direction_comparison=tables['direction_comparison'],
        development_reconstruction=tables['development_reconstruction'],basis_metadata=metadata,array_metadata=array_metadata,
        **{k:[v[k] for v in tables['heldout_transfer']] for k in ('capture_q0','capture_q1','capture_q2','increment_q1','increment_q2','marginal_q2')},**flags,
        c12_claim=p.C12,c11_claim_status='SUPPORTED_IN_C30_200_10_STATISTICS_REPLICATION',c9_claim_status='INCONCLUSIVE',c10_claim_status='INCONCLUSIVE',
        c6_claim_status=old.C6,c7_claim_status=old.C7,c8_claim_status='SUPPORTED_IN_FIXED_ENVIRONMENTS',t18a_materiality_status='PARTIAL',
        t19a_rank4_out_of_factor_warning=True,new_module_added=False,new_model_trained=False,rank_compression_performed=False,
        gamma_pilot_fitting=False,deployed_fitting=False,validation_NMSE=False,BER_evaluations=False,formal_test_used=False,formal_test_sealed=True,
        method_provenance=proof,historical_regression=replay,calibration=pre['calibration'],next_step_dependency=decision,
        source_sha256=pre['source_sha256'],source_snapshot_root=pre['source_snapshot_root'],frozen_history_sha256=pre['frozen_history_sha256'],
        numeric_artifact_sha256=artifacts,environment_before=pre['environment_before'],
        runtime=dict(preparation_seconds=pre['preparation_seconds'],operator_analysis_seconds=time.perf_counter()-tick),
        all_checks_passed=False,completion_state='oracle operator analysis complete; final audits pending')
    verify_protected(pre);save(RESULT,result);print('T19D operator study complete; C12='+flags['c12_claim_status'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['preflight','run']);args=parser.parse_args()
    {'preflight':preflight,'run':run}[args.mode]()
