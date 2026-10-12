"""T17B exact frozen-statistics decomposition; no channel or performance replay."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from src.finite_shift_resolvent import ExactWienerMap, TERMS, weighted_bookkeeping, decision, relative
from src.wiener_perturbation import energy, spectrum, principal_angles
from src.cdl_benchmark import array_hash
from scripts.run_cdl_mismatch import load, save, sha, history, environments
from scripts.audit_wiener_mechanism import verify as verify_t17a

OUT=ROOT/'outputs';RESULT=OUT/'t17b_results.json';PREFLIGHT=OUT/'t17b_execution_preflight_attempt02.json'
COMMIT='776d663d3faa0873dc158b9d7f6a3a344f7d90ee'
SNRS=(5,10,20);CONDITIONS={'0':'C30','P':'B30','M':'C120','PM':'B120'}
C6='CONTRADICTED_BY_CURRENT_MECHANISM_TEST'
TYPE='exact finite-shift resolvent decomposition gate for affine LMMSE mismatch-composition interaction'
QUESTION='Why is W_B120 approximately W_B30+W_C120-W_C30 accurate under a finite, non-perturbative shift: individually modest exact interactions, or cancellation of large terms?'
HYPOTHESIS='Finite profile and mobility shifts are approximately separable in the exact Wiener map; the small mixed interaction should not require severe cancellation among large exact contributions.'
DOCS=('docs/t17b_resolvent_protocol.md','docs/t17b_claim_evidence_update.md','docs/t17b_reviewer_update.md')
SOURCES=('src/finite_shift_resolvent.py','scripts/run_finite_shift_resolvent.py',
    'scripts/plot_finite_shift_resolvent.py','scripts/audit_finite_shift_resolvent.py',
    'tests/test_finite_shift_resolvent.py','tests/test_finite_shift_pipeline.py',*DOCS)


def accepted():
    subprocess.check_call(['git','merge-base','--is-ancestor',COMMIT,'HEAD'],cwd=ROOT)
    subject=subprocess.check_output(['git','show','-s','--format=%s',COMMIT],cwd=ROOT,text=True).strip()
    if subject!='feat: test and reject local Wiener perturbation explanation':raise ValueError('T17A committed snapshot required')
    old=load(OUT/'t17a_results.json');verify_t17a(old)
    if not old['all_checks_passed'] or old['c6_claim_status']!=C6:raise ValueError('accepted T17A rejection must remain frozen')
    for p in [*old['source_sha256'],'outputs/t17a_results.json']:
        blob=subprocess.check_output(['git','show',f'{COMMIT}:{p}'],cwd=ROOT)
        if blob.replace(b'\r\n',b'\n')!=(ROOT/p).read_bytes().replace(b'\r\n',b'\n'):raise ValueError('T17A snapshot changed:'+p)
    return old


def frozen_history():
    value=history()
    for p in subprocess.check_output(['git','ls-files','docs'],cwd=ROOT,text=True).splitlines():value[p]=sha(ROOT/p)
    return value


def load_statistics():
    old=accepted();a=load(OUT/'t15a_cdl_results.json');c=load(OUT/'t15c_cdl_mismatch_results.json');h=load(OUT/'t16c_heldout_results.json')
    infos={'0':a['empirical_lmmse_definition']['snapshots']['500'],
        'P':c['matched_statistics_provenance']['CDL-B-30'],
        'M':c['matched_statistics_provenance']['CDL-C-120'],
        'PM':h['target_statistics_provenance']['CDL-B-120']}
    d,p,proof={},{},[]
    for key,info in infos.items():
        path=ROOT/info['path']
        if sha(path)!=info['sha256']:raise ValueError('statistics file hash changed')
        with np.load(path,allow_pickle=False) as f:
            d[key]=np.array(f['rdp'],dtype=np.complex128,copy=True)
            p[key]=np.array(f['rpp'],dtype=np.complex128,copy=True)
            if tuple(map(int,f['train_trajectory_ids']))!=tuple(range(500)) or int(f['samples'])!=5000:
                raise ValueError('frozen training protocol changed')
        for label,value in [('rdp',d[key]),('rpp',p[key])]:
            if array_hash(value)!=old['statistics_provenance'][CONDITIONS[key]][label]:raise ValueError('historical covariance array hash changed')
            value.flags.writeable=False
        proof.append(dict(condition=CONDITIONS[key],path=info['path'],file_sha256=sha(path),
            Rdp_sha256=array_hash(d[key]),Rpp_sha256=array_hash(p[key]),historical_hash_exact=True,
            arrays_loaded=['rdp','rpp'],means_used=False,statistics_refitted=False))
    variances={s:float(c['snr_policy']['production_variances'][str(s)]) for s in SNRS}
    return old,d,p,variances,proof


def prepare(d,p,variances,old):
    maps={};proof=[]
    for s in SNRS:
        m=ExactWienerMap.build(d,p,variances[s]);maps[s]=m
        for key,name in CONDITIONS.items():
            recorded=next(v for v in old['operator_provenance'] if v['condition']==name and v['snr_db']==s)
            if array_hash(m.w[key])!=recorded['W_sha256']:raise ValueError('frozen T17A W hash changed')
            residual=relative(d[key],m.w[key]@(p[key]+variances[s]*np.eye(p[key].shape[0])))
            if residual>1e-11:raise ValueError('filter equation residual')
            proof.append(dict(condition=name,snr_db=s,W_sha256=array_hash(m.w[key]),
                exact_T17A_recorded_hash=True,T17A_original_historical_hash_available=recorded['historical_raw_W_hash_available'],
                Cholesky_equation_relative_residual=residual))
    return maps,proof


def csvwrite(path,records):
    if path.exists():raise ValueError('existing numeric CSV protected:'+str(path))
    fields=list(dict.fromkeys(k for r in records for k in r))
    with path.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        w.writerows({k:r.get(k) for k in fields} for r in records)


def evaluate(maps,old):
    summaries=[];terms=[];crossrows=[];spectra=[];ranks=[];angles=[];counter=[];regress=[]
    times=dict(exact_terms_seconds=0.,bookkeeping_seconds=0.,counterfactual_seconds=0.,SVD_subspace_seconds=0.)
    for s,m in maps.items():
        tick=time.perf_counter();observed,c,identities=m.decomposition();lower=m.lower()
        resolvents=m.resolvent_audit(m.d['0'])
        if max(identities.values())>1e-11 or max(r['action_relative_residual'] for r in resolvents)>1e-11:raise ValueError('exact arithmetic identity failed')
        times['exact_terms_seconds']+=time.perf_counter()-tick
        tick=time.perf_counter();correction=m.w['PM']-m.w['0'];row,termrows=weighted_bookkeeping(observed,correction,c,lower)
        if row['energy_identity_relative_residual']>1e-11:raise ValueError('Gram energy identity failed')
        historical=next(v for v in old['mechanism_closure'] if v['snr_db']==s and v['prediction']=='exact_module_sum')
        legacy_residual=correction-((m.w['P']-m.w['0'])+(m.w['M']-m.w['0']))
        legacy_energy=energy(legacy_residual@lower);legacy_closure=1.-legacy_energy/energy(correction@lower)
        if legacy_energy!=historical['residual_weighted_energy'] or legacy_closure!=historical['closure']:raise ValueError('original T17A affine reduction not exact')
        if abs(row['exact_affine_closure']-legacy_closure)>2e-13:raise ValueError('affine parenthesization drift')
        regress.append(dict(snr_db=s,historical_closure=legacy_closure,historical_residual_energy=legacy_energy,
            original_reduction_exact=True,current_formula_difference=row['exact_affine_closure']-legacy_closure))
        row.update(snr_db=s,**identities,resolvent_identities=resolvents)
        summaries.append(row);terms.extend(dict(snr_db=s,**t) for t in termrows)
        terms.append(dict(snr_db=s,term='R_exact',weighted_energy=row['interaction_energy'],weighted_norm=float(np.sqrt(row['interaction_energy'])),
            relative_to_complete_correction=row['interaction_norm_ratio'],relative_to_interaction=1.,
            normalized_energy_relative_to_complete_correction=row['interaction_energy']/row['complete_correction_energy']))
        for i in range(4):
            for j in range(4):crossrows.append(dict(snr_db=s,term_i=TERMS[i],term_j=TERMS[j],
                gram_real=row['weighted_gram_real'][i][j],gram_imag=row['weighted_gram_imag'][i][j],
                cross_energy=row['cross_energy_matrix'][i][j],
                normalized_cross_energy=row['cross_energy_matrix'][i][j]/row['complete_correction_energy'],
                used_once_in_energy_identity=i<j))
        times['bookkeeping_seconds']+=time.perf_counter()-tick
        tick=time.perf_counter();cf=dict(snr_db=s,**m.counterfactual(observed,lower));counter.append(cf)
        if cf['additive_statistics_counterfactual_available'] and max(cf['decomposition_relative_residual'],cf['energy_identity_relative_residual'])>1e-11:raise ValueError('counterfactual identity failed')
        times['counterfactual_seconds']+=time.perf_counter()-tick
        tick=time.perf_counter()
        for weighting in ('raw','target_weighted'):
            vectors={}
            for name,value in {'R_exact':observed,**c}.items():
                u,sv,capture,diagnostic=spectrum(value if weighting=='raw' else value@lower)
                vectors[name]=u;ranks.append(dict(snr_db=s,operator=name,weighting=weighting,**diagnostic))
                spectra.extend(dict(snr_db=s,operator=name,weighting=weighting,mode=i+1,singular_value=float(v),cumulative_capture=float(capture[i])) for i,v in enumerate(sv))
                if name=='R_exact':
                    historical_sv=[v['singular_value'] for v in old['interaction_spectrum'] if v['snr_db']==s and v['operator']=='R_exact' and v['weighting']==weighting]
                    if not np.array_equal(sv,np.array(historical_sv)):raise ValueError('exact historical R spectrum drift')
            for name in TERMS:
                for q in (4,8):angles.append(dict(snr_db=s,reference='R_exact',prediction=name,weighting=weighting,top=q,
                                                angles_degrees=principal_angles(vectors['R_exact'],vectors[name],q)))
        times['SVD_subspace_seconds']+=time.perf_counter()-tick
    a,b=(next(r for r in summaries if r['snr_db']==s) for s in (10,20))
    stability=dict(descriptive_only=True,threshold_added=False,dominant_contribution_10=a['dominant_contribution'],dominant_contribution_20=b['dominant_contribution'],
        cancellation_index_20_minus_10=b['cancellation_index']-a['cancellation_index'],
        contribution_share_20_minus_10=[y-x for x,y in zip(a['contribution_energy_shares'],b['contribution_energy_shares'])],
        normalized_energy_20_minus_10=[y-x for x,y in zip(a['normalized_contribution_energies'],b['normalized_contribution_energies'])],
        interaction_norm_ratio_20_minus_10=b['interaction_norm_ratio']-a['interaction_norm_ratio'],
        cross_energy_sign_pattern_10=np.sign(a['cross_energy_matrix']).astype(int).tolist(),
        cross_energy_sign_pattern_20=np.sign(b['cross_energy_matrix']).astype(int).tolist())
    return dict(four_term_decomposition=summaries,contribution_energies=terms,cross_energy_matrix=crossrows,
        interaction_spectrum=spectra,rank_diagnostics=ranks,subspace_diagnostics=angles,counterfactual_results=counter,
        exact_affine_regression=regress,snr_stability=stability,runtime_components=times,**decision(summaries))


def c7_status(r):
    if r['finite_shift_separability_supported']:return 'SUPPORTED_IN_B120'
    if r['affine_additivity_cancellation_dominated']:return 'CONTRADICTED/CANCELLATION_DOMINATED'
    return 'PARTIAL'


def preflight():
    if PREFLIGHT.exists():raise ValueError('preflight protected')
    startup=(OUT/'t17b_startup_pytest.txt').read_text(encoding='utf8').strip().splitlines()[-1]
    if not startup.startswith('1625 passed'):raise ValueError('startup full regression missing')
    old,d,p,v,stats=load_statistics();maps,w=prepare(d,p,v,old)
    if any('UNCONFIRMED' not in (ROOT/doc).read_text(encoding='utf8') for doc in DOCS[:2]):raise ValueError('C7 pre-run status absent')
    frozen=frozen_history();sources={path:sha(ROOT/path) for path in SOURCES}
    snapshot=OUT/'t17b_preparation_snapshot_attempt02'
    for path in SOURCES:
        target=snapshot/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/path,target)
    preparation={str(path.relative_to(ROOT)).replace('\\','/'):sha(path) for path in OUT.rglob('*')
                 if path.is_file() and path.relative_to(OUT).parts[0].startswith('t17b_')}
    save(PREFLIGHT,dict(accepted_T17A_commit=COMMIT,startup_full_pytest_summary=startup,startup_full_pytest_passed=True,
        startup_full_pytest_count=1625,working_tree_clean_after_T17A_commit=True,
        startup_log_sha256=sha(OUT/'t17b_startup_pytest.txt'),historical_staged_whitespace_report_sha256=sha(OUT/'t17b_t17a_staged_diffcheck.txt'),
        statistics_provenance=stats,operator_provenance=w,source_sha256=sources,
        source_snapshot_root=str(snapshot.relative_to(ROOT)).replace('\\','/'),frozen_history_sha256=frozen,
        preparation_artifact_sha256=preparation,
        C6=C6,C7='UNCONFIRMED',generator_calls=0,statistics_refits=0,new_H=False,formal_test_sealed=True,all_checks_passed=True))
    print('T17B preflight PASS: committed T17A, 4 frozen covariances,12 W hashes exact; no mechanism evaluated')


def run():
    if RESULT.exists():raise ValueError('existing result protected')
    pre=load(PREFLIGHT)
    for p,h in {**pre['source_sha256'],**pre['frozen_history_sha256']}.items():
        if sha(ROOT/p)!=h:raise ValueError('preflight drift:'+p)
    start=time.perf_counter();tick=start;env=environments();runtime=dict(environment_audit_seconds=time.perf_counter()-tick)
    tick=time.perf_counter();old,d,p,v,stats=load_statistics();runtime['statistics_load_seconds']=time.perf_counter()-tick
    tick=time.perf_counter();maps,w=prepare(d,p,v,old);runtime['filter_regression_seconds']=time.perf_counter()-tick
    r=evaluate(maps,old);runtime.update(r['runtime_components'])
    artifacts={}
    for filename,key in [('t17b_exact_interaction_terms.csv','contribution_energies'),('t17b_cross_energy_matrix.csv','cross_energy_matrix'),
        ('t17b_interaction_spectrum.csv','interaction_spectrum')]:
        path=OUT/filename;csvwrite(path,r[key]);artifacts[str(path.relative_to(ROOT)).replace('\\','/')]=sha(path)
    if any(x['additive_statistics_counterfactual_available'] for x in r['counterfactual_results']):
        path=OUT/'t17b_counterfactual_additivity.csv';csvwrite(path,r['counterfactual_results']);artifacts[str(path.relative_to(ROOT)).replace('\\','/')]=sha(path)
    runtime['total_numeric_wall_seconds']=time.perf_counter()-start
    for path,h in pre['frozen_history_sha256'].items():
        if sha(ROOT/path)!=h:raise ValueError('historical payload changed')
    r.update(task='T17B',experiment_type=TYPE,research_question=QUESTION,hypothesis=HYPOTHESIS,
        accepted_T17A_commit=COMMIT,statistics_provenance=stats,operator_provenance=w,
        exact_affine_closure=[dict(snr_db=x['snr_db'],value=x['exact_affine_closure']) for x in r['four_term_decomposition']],
        cancellation_index=[dict(snr_db=x['snr_db'],value=x['cancellation_index']) for x in r['four_term_decomposition']],
        cross_fraction=[dict(snr_db=x['snr_db'],value=x['cross_fraction']) for x in r['four_term_decomposition']],
        interaction_norm_ratio=[dict(snr_db=x['snr_db'],value=x['interaction_norm_ratio']) for x in r['four_term_decomposition']],
        additive_statistics_counterfactual_available={str(x['snr_db']):x['additive_statistics_counterfactual_available'] for x in r['counterfactual_results']},
        scientific_gate_thresholds=dict(closure=.90,structural_index=2.,structural_size=.50,cancellation_index=10.,cancellation_size=1.),
        c6_claim_status=C6,c7_claim_status=c7_status(r),novelty_claim_made=False,new_channel_generated=False,new_model_trained=False,
        generator_calls=0,statistics_refits=0,coefficient_refits=0,Taylor_terms_used=False,
        formal_test_used=False,formal_test_sealed=True,source_sha256=pre['source_sha256'],
        pre_run_source_snapshot_root=pre['source_snapshot_root'],frozen_history_sha256=pre['frozen_history_sha256'],
        numeric_artifact_sha256=artifacts,preparation_artifact_sha256=pre['preparation_artifact_sha256'],
        preflight_sha256=sha(PREFLIGHT),environment_before=env,runtime=runtime,
        completion_state='numeric_complete_audit_pending',all_checks_passed=False,
        warnings=['Exact contributions are bookkeeping, not independent physical causes.',
            'Cancellation decomposition is coordinate-dependent; gate applies only to predefined B12010/20.',
            'Historical staged whitespace preserved; no original logs or archive payload rewritten.',
            'T17A raw target W hashes were replay hashes; T17B verifies the subsequently committed T17A recorded hashes.'])
    save(RESULT,r);print('T17B exact artifacts saved; final audit pending; C7='+r['c7_claim_status'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preflight',action='store_true');a=parser.parse_args()
    preflight() if a.preflight else run()
