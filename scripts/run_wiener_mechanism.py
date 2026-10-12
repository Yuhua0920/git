"""T17A frozen covariance/operator mechanism; no channel or validation replay."""
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
from scipy.linalg import cholesky
from src.wiener_perturbation import WienerSensitivity,alignment,compare,energy,real_scalar,spectrum,principal_angles,mechanism_gate
from src.cdl_benchmark import array_hash
from scripts.run_lowrank_filter import statistics
from scripts.run_modular_correction import target_statistics
from scripts.run_cdl_mismatch import load,save,sha,csvsave,history,environments

OUT=ROOT/'outputs'
RESULT=OUT/'t17a_results.json'
SUBJECT='feat: add weighted low-rank affine LMMSE correction modules'
SNRS=(5,10,20)
TYPE='closest-work red-team and covariance-perturbation mechanism gate for low-rank affine LMMSE mismatch composition'
QUESTION='Why are profile/mobility corrections approximately affine-additive for B120: covariance additivity, local Wiener sensitivity, or coincidence among current environments?'
HYPOTHESIS='The Wiener operator is locally first-order linear in Rdp/Rpp perturbations; approximately additive covariance perturbations should yield approximately additive filter corrections, with residual interaction due to covariance non-additivity and higher-order inverse terms.'
DOCS=('docs/t17a_mechanism_protocol.md','docs/t17a_closest_work_matrix.md','docs/t17a_novelty_ledger.md',
      'docs/t17a_claim_evidence_matrix.md','docs/t17a_reviewer_red_team.md')
SOURCES=('src/wiener_perturbation.py','scripts/run_wiener_mechanism.py',
         'scripts/plot_wiener_mechanism.py','scripts/audit_wiener_mechanism.py',
         'tests/test_wiener_perturbation.py','tests/test_wiener_mechanism_pipeline.py',
         'tests/test_wiener_mechanism_figures.py',*DOCS)
FORMAL_PREFLIGHT=OUT/'t17a_formal_execution_preflight.json'
CLAIM= 'docs/t17a_claim_evidence_matrix.md'


def accepted():
    commits=subprocess.check_output(['git','log','--all','--format=%H%x09%s'],cwd=ROOT,text=True).splitlines()
    candidates=[row.split('\t')[0] for row in commits if row.split('\t',1)[1]==SUBJECT]
    if len(candidates)!=1:raise ValueError('T16G must be committed with the prescribed subject first')
    commit=candidates[0]
    subprocess.check_call(['git','merge-base','--is-ancestor',commit,'HEAD'],cwd=ROOT)
    g=load(OUT/'t16g_results.json')
    if not g['all_checks_passed'] or g['completion_state']!='completed':raise ValueError('accepted T16G required')
    hashes={**g['source_sha256'],**g['frozen_history_sha256'],**g['numeric_artifact_sha256'],
            **g['preparation_artifact_sha256'],**g['preflight_schema_recovery']['artifact_sha256']}
    for path,digest in hashes.items():
        if sha(ROOT/path)!=digest:raise ValueError('historical file drift:'+path)
    for path in [*g['source_sha256'],'outputs/t16g_results.json']:
        old=subprocess.check_output(['git','show',f'{commit}:{path}'],cwd=ROOT)
        if old.replace(b'\r\n',b'\n')!=(ROOT/path).read_bytes().replace(b'\r\n',b'\n'):
            raise ValueError('accepted T16G snapshot drift')
    return commit,g


def load_frozen_statistics():
    _,source,development,variances=statistics()
    targets=target_statistics(load(OUT/'t16c_heldout_results.json'))
    stats={'C30':source,'B30':development['CDL-B-30'],'C120':development['CDL-C-120'],
           'B120':targets['CDL-B-120'],'C75':targets['CDL-C-75']}
    if set(development)!={'CDL-B-30','CDL-C-120'} or set(targets)!={'CDL-C-75','CDL-B-120'}:
        raise ValueError('frozen statistics scope differs')
    proof={name:{key:array_hash(getattr(s,key)) for key in ('mean_p','mean_d','rpp','rdp')} for name,s in stats.items()}
    return stats,variances,proof


def historical_objective_sentinel(delta,modules,lower,reference):
    """Re-evaluate frozen coefficients/objectives; never solve or refit them."""
    target=delta@lower;values=[x@lower for x in modules]
    gram=np.array([[np.vdot(x,y).real for y in values] for x in values],float)
    rhs=np.array([np.vdot(x,target).real for x in values],float)
    coefficients=([reference['beta']] if len(modules)==1 else [reference['alpha'],reference['beta']])
    correction=sum(c*w for c,w in zip(coefficients,modules))
    residual=delta-correction
    observed=dict(normal_equation_gram=gram.tolist(),normal_equation_rhs=rhs.tolist(),
                  weighted_residual=energy(residual@lower),correction_energy_per_frame=energy(target))
    observed['weighted_residual_fraction']=observed['weighted_residual']/observed['correction_energy_per_frame']
    if any(observed[k]!=reference[k] for k in observed):
        raise ValueError('historical target operator quadratic sentinel differs')
    return observed,residual


def prepare_operators(stats,variances):
    f=load(OUT/'t16f_preflight.json');g=load(OUT/'t16g_results.json');d=load(OUT/'t16d_modular_results.json')
    weights={};proof=[];sentinels=[]
    for snr in SNRS:
        frozen=next(x for x in f['candidate_provenance'] if x['snr_db']==snr)['W_sha256']
        for name,key in [('C30','W0'),('B30','WP'),('C120','WM'),('B120',None),('C75',None)]:
            operator=stats[name].prepare(variances[snr])
            if operator.solve_relative_residual>1e-11:raise ValueError('historical Cholesky solve residual')
            # Preserve the historical readonly() C-order copy before reductions.
            weight=np.asarray(operator.weight,dtype=np.complex128).copy()
            weight.flags.writeable=False;weights[name,snr]=weight
            if key and array_hash(weight)!=frozen[key]:raise ValueError('T16F exact W hash failed')
            proof.append(dict(condition=name,snr_db=snr,W_sha256=array_hash(weight),
                historical_raw_W_hash_available=key is not None,
                historical_raw_W_hash_exact=True if key else None,
                historical_snapshot_and_statistics_replay=True,solve_relative_residual=operator.solve_relative_residual))
        wp=weights['B30',snr]-weights['C30',snr];wm=weights['C120',snr]-weights['C30',snr]
        for label,value in [('P',wp),('M',wm)]:
            expected=next(x for x in g['module_provenance'] if x['module']==label and x['snr_db']==snr)
            if array_hash(value)!=expected['DeltaW_sha256']:raise ValueError('T16G exact module hash failed')
        for name,label,modules in [('B120','CDL-B-120',[wp,wm]),('C75','CDL-C-75',[wm])]:
            old=next(x for x in d['alpha_beta'] if x['shift']==label and x['snr_db']==snr)
            lower=cholesky(stats[name].rpp+variances[snr]*np.eye(64),lower=True)
            diagnostics,_=historical_objective_sentinel(weights[name,snr]-weights['C30',snr],modules,lower,old)
            sentinels.append(dict(condition=name,snr_db=snr,all_fields_exact=True,fields=diagnostics))
    return weights,proof,sentinels


def covariance_additivity(factual,fadd,snr,component):
    interaction=factual-fadd
    row=dict(snr_db=snr,component=component,actual_norm=float(np.linalg.norm(factual)),
             additive_norm=float(np.linalg.norm(fadd)),interaction_norm=float(np.linalg.norm(interaction)),
             additive_to_actual_norm_ratio=float(np.linalg.norm(fadd)/np.linalg.norm(factual)),
             relative_interaction_norm=float(np.linalg.norm(interaction)/np.linalg.norm(factual)))
    row.update(alignment(factual,fadd))
    return row


def evaluate(stats,weights,variances):
    additive=[];terms=[];closures=[];spectra=[];ranks=[];angles=[];interactions=[];auxiliary=[]
    d=load(OUT/'t16d_modular_results.json')
    f=load(OUT/'t16f_results.json')
    timings={k:0. for k in ('covariance_diagnostics_seconds','first_order_terms_seconds',
        'second_order_terms_seconds','SVD_subspace_seconds','metric_and_auxiliary_seconds')}
    for snr in SNRS:
        tick=time.perf_counter()
        source=stats['C30'];w0=weights['C30',snr]
        model=WienerSensitivity.build(source.rdp,source.rpp,variances[snr],historical_w0=w0)
        fp=stats['B30'].rdp-source.rdp;ep=stats['B30'].rpp-source.rpp
        fm=stats['C120'].rdp-source.rdp;em=stats['C120'].rpp-source.rpp
        fpm=stats['B120'].rdp-source.rdp;epm=stats['B120'].rpp-source.rpp
        fi=fpm-(fp+fm);ei=epm-(ep+em)
        additive += [covariance_additivity(fpm,fp+fm,snr,'Rdp'),covariance_additivity(epm,ep+em,snr,'Rpp')]
        timings['covariance_diagnostics_seconds']+=time.perf_counter()-tick
        lower=cholesky(stats['B120'].rpp+variances[snr]*np.eye(64),lower=True)
        delta_p=weights['B30',snr]-w0;delta_m=weights['C120',snr]-w0;delta_pm=weights['B120',snr]-w0
        tick=time.perf_counter()
        lp,lm=model.first(fp,ep),model.first(fm,em)
        ladd=lp+lm;lint=model.first(fi,ei)
        actualfirst=model.first(fpm,epm)
        timings['first_order_terms_seconds']+=time.perf_counter()-tick
        tick=time.perf_counter()
        cross=model.cross(fp,ep,fm,em)
        qadd=model.second(fp+fm,ep+em);qactual=model.second(fpm,epm)
        cross_residual=qadd-model.second(fp,ep)-model.second(fm,em)-cross
        cross_identity_relative_residual=float(np.linalg.norm(cross_residual)/np.linalg.norm(cross))
        if cross_identity_relative_residual>1e-11:raise ValueError('actual Q_cross arithmetic identity failed')
        timings['second_order_terms_seconds']+=time.perf_counter()-tick
        tick=time.perf_counter()
        observed=weights['B120',snr]-(weights['B30',snr]+weights['C120',snr]-w0)
        corrected=ladd+lint+cross
        predictions={'exact_correction':delta_pm,'exact_module_sum':delta_p+delta_m,
            'first_order_additive':ladd,'first_order_actual_perturbation':actualfirst,
            'second_order_additive':ladd+qadd,
            'second_order_actual_perturbation':actualfirst+qactual,
            'interaction_corrected':corrected}
        comparisons={name:compare(delta_pm,value,lower) for name,value in predictions.items()}
        for name,value in comparisons.items():closures.append(dict(snr_db=snr,prediction=name,**value))
        for name,exact,pred,target in [('profile_first_order',delta_p,lp,'B30'),
             ('mobility_first_order',delta_m,lm,'C120'),
             ('PM_actual_first_order',delta_pm,actualfirst,'B120'),
             ('PM_additive_first_order',delta_pm,ladd,'B120')]:
            l=cholesky(stats[target].rpp+variances[snr]*np.eye(64),lower=True)
            # Current instruction requires B120 weighting for all four comparisons.
            # Keep the earlier individual-condition weighting as a named diagnostic.
            local={f'local_{k}':v for k,v in compare(exact,pred,l).items()}
            terms.append(dict(snr_db=snr,term=name,target_weight='B120',local_target_weight=target,
                              **compare(exact,pred,lower),**local))
        families={'R_exact':observed,'R_pred_1':lint,'Q_cross':cross,'R_pred_2':lint+cross}
        report=dict(snr_db=snr,R_pred_1=compare(observed,lint,lower),
                    R_pred_2=compare(observed,lint+cross,lower),
                    linearity_relative_residual=float(np.linalg.norm(actualfirst-ladd-lint)/np.linalg.norm(actualfirst)),
                    neumann_radius_profile=model.neumann_radius(ep),neumann_radius_mobility=model.neumann_radius(em),
                    neumann_radius_actual_PM=model.neumann_radius(epm),neumann_radius_additive=model.neumann_radius(ep+em),
                    first_order_additive_closure=comparisons['first_order_additive']['closure'],
                    interaction_corrected_closure=comparisons['interaction_corrected']['closure'])
        report['Q_cross_identity_relative_residual']=cross_identity_relative_residual
        report['second_order_additive_closure']=comparisons['second_order_additive']['closure']
        report['second_order_actualpert_closure']=comparisons['second_order_actual_perturbation']['closure']
        report['interaction_closure_Lint']=report['R_pred_1']['closure']
        report['interaction_closure_Lint_Qcross']=report['R_pred_2']['closure']
        old=next(x for x in d['alpha_beta'] if x['shift']=='CDL-B-120' and x['snr_db']==snr)
        _,old_residual=historical_objective_sentinel(delta_pm,[delta_p,delta_m],lower,old)
        report['historical_T16D_oracle_residual_not_unit_coordinate_interaction']=True
        interactions.append(report)
        timings['metric_and_auxiliary_seconds']+=time.perf_counter()-tick
        tick=time.perf_counter()
        for weighting in ['raw','target_weighted']:
            decomposition={}
            for name,value in families.items():
                a=value if weighting=='raw' else value@lower
                u,s,capture,diag=spectrum(a);decomposition[name]=(u,diag)
                ranks.append(dict(snr_db=snr,operator=name,weighting=weighting,**diag))
                for mode,(singular,cumulative) in enumerate(zip(s,capture),1):
                    spectra.append(dict(snr_db=snr,operator=name,weighting=weighting,mode=mode,
                        singular_value=float(singular),squared_energy=float(singular**2),cumulative_capture=float(cumulative)))
            for name in ['R_pred_1','Q_cross','R_pred_2']:
                for q in (4,8):
                    defined=min(decomposition['R_exact'][1]['numerical_rank'],decomposition[name][1]['numerical_rank'])>=q
                    angles.append(dict(snr_db=snr,reference='R_exact',prediction=name,weighting=weighting,top=q,
                        defined=defined,angles_degrees=principal_angles(decomposition['R_exact'][0],decomposition[name][0],q) if defined else None))
            old_u,_,_,old_diag=spectrum(old_residual if weighting=='raw' else old_residual@lower)
            for q in (4,8):
                defined=min(old_diag['numerical_rank'],decomposition['R_exact'][1]['numerical_rank'])>=q
                angles.append(dict(snr_db=snr,reference='historical_T16D_oracle_residual',prediction='R_exact',
                    weighting=weighting,top=q,defined=defined,
                    angles_degrees=principal_angles(old_u,decomposition['R_exact'][0],q) if defined else None))
        timings['SVD_subspace_seconds']+=time.perf_counter()-tick
        tick=time.perf_counter()
        f75=stats['C75'].rdp-source.rdp;e75=stats['C75'].rpp-source.rpp
        l75=cholesky(stats['C75'].rpp+variances[snr]*np.eye(64),lower=True)
        ad75,ad120=model.right_solve(f75)@l75,model.right_solve(fm)@l75
        ap75,ap120=model.right_solve(w0@e75)@l75,model.right_solve(w0@em)@l75
        gd=real_scalar(ad75,ad120);gp=real_scalar(ap75,ap120)
        rd,rp=real_scalar(f75,fm),real_scalar(e75,em)
        beta=next(x['beta'] for x in d['alpha_beta'] if x['shift']=='CDL-C-75' and x['snr_db']==snr)
        fair=next(x for x in f['oracle_coefficient_optimization'] if x['shift']=='CDL-C-75'
                  and x['snr_db']==snr and x['method']=='affine-oracle')
        auxiliary.append(dict(snr_db=snr,gamma_D_weighted=gd['gamma'],gamma_P_weighted=gp['gamma'],
            residual_D_weighted=gd['relative_residual'],residual_P_weighted=gp['relative_residual'],
            cosine_D_weighted=alignment(ad75,ad120)['cosine_real'],
            cosine_P_weighted=alignment(ap75,ap120)['cosine_real'],
            gamma_D_fro=rd['gamma'],gamma_P_fro=rp['gamma'],residual_D_fro=rd['relative_residual'],
            residual_P_fro=rp['relative_residual'],historical_T16D_filter_beta=beta,
            cosine_D_fro=alignment(f75,fm)['cosine_real'],cosine_P_fro=alignment(e75,em)['cosine_real'],
            historical_T16F_two_module_alpha=fair['alpha'],historical_T16F_two_module_beta=fair['beta'],
            beta_model_classes_distinguished=True,
            distinct_objectives=True,used_for_primary_gate=False))
        timings['metric_and_auxiliary_seconds']+=time.perf_counter()-tick
    return dict(covariance_additivity=additive,perturbation_terms=terms,mechanism_closure=closures,
                interaction_spectrum=spectra,rank_diagnostics=ranks,principal_angles=angles,
                interaction_decomposition=interactions,C75_auxiliary=auxiliary,runtime_components=timings,
                **mechanism_gate(interactions))


def c6_status(values):
    if values['wiener_perturbation_mechanism_supported']:return 'SUPPORTED_IN_TESTED_B120_CONDITIONS'
    if values['wiener_perturbation_mechanism_contradicted']:return 'CONTRADICTED_BY_CURRENT_MECHANISM_TEST'
    return 'PARTIALLY_SUPPORTED'


def formal_preflight():
    """Final source lock after engineering additions; never evaluate mechanism."""
    if FORMAL_PREFLIGHT.exists():raise ValueError('existing formal preflight protected')
    commit,_=accepted()
    if commit!='37c6536096aca288a0cd1db3b030bfbb12789ea3':raise ValueError('accepted commit differs')
    permission=load(OUT/'t17a_permission_recovery/permission_preflight.json')
    raw=load(ROOT/permission['raw_provenance_record'])
    if not permission['all_checks_passed'] or sha(ROOT/permission['raw_provenance_record'])!=permission['raw_provenance_sha256']:
        raise ValueError('permission recovery gate failed')
    startup=OUT/'t17a_permission_recovery/t17a_startup_accepted_pytest.txt'
    if sha(startup)!=raw['startup_log_sha256'] or raw['startup_pytest_counts']['passed']!=1610:
        raise ValueError('actual1610-test startup differs')
    log=OUT/'t17a_related_preflight_pytest.txt'
    summary=log.read_text(encoding='utf8').strip().splitlines()[-1]
    if 'passed' not in summary or any(x in summary for x in ('failed','error','skipped')):
        raise ValueError('related preparation tests must pass without skip')
    original=load(OUT/'t17a_permission_recovery/before.json')
    # Literature facts and pre-gate C6 remain at the researcher-accepted versions.
    for p in DOCS:
        if p in original['preparation_sha256'] and sha(ROOT/p)!=original['preparation_sha256'][p]:
            raise ValueError('pre-run documentation drift:'+p)
    if 'UNCONFIRMED' not in next(x for x in (ROOT/CLAIM).read_text(encoding='utf8').splitlines() if x.startswith('| C6 ')):
        raise ValueError('C6 upgraded before gate')
    stats,var,stat_proof=load_frozen_statistics();_,operator_proof,sentinels=prepare_operators(stats,var)
    for path,digest in raw['frozen_history_sha256'].items():
        if sha(ROOT/path)!=digest:raise ValueError('permission-verified history drift')
    preparation={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in OUT.rglob('*')
                 if p.is_file() and p.relative_to(OUT).parts[0].startswith('t17a_')}
    snapshot=OUT/'t17a_preparation_snapshot'
    if snapshot.exists():raise ValueError('existing source snapshot protected')
    for p in SOURCES:
        target=snapshot/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/p).read_bytes())
    source={p:sha(ROOT/p) for p in SOURCES}
    save(FORMAL_PREFLIGHT,dict(task='T17A',stage='final frozen source lock; no mechanism evaluated',
        accepted_T16G_commit=commit,startup_full_pytest_passed=True,startup_full_pytest_count=1610,
        permission_recovery_completed=True,permission_record_sha256=sha(OUT/'t17a_permission_recovery/permission_preflight.json'),
        startup_log_path=str(startup.relative_to(ROOT)).replace('\\','/'),startup_log_sha256=sha(startup),
        preparation_tests_summary=summary,preparation_tests_log_sha256=sha(log),
        statistics_provenance=stat_proof,operator_provenance=operator_proof,
        historical_operator_quadratic_sentinels=sentinels,source_sha256=source,
        preparation_artifact_sha256=preparation,source_snapshot_root='outputs/t17a_preparation_snapshot',
        formal_test_sealed=True,formal_test_used=False,generator_calls=0,new_statistics_fit=False,
        C6='UNCONFIRMED',all_checks_passed=True))
    print('T17A final source lock PASS; startup1610; no generator/refit; C6 UNCONFIRMED')


def sentinel():
    m=WienerSensitivity.build([[3]],[[4]],1.)
    if not np.isclose(m.first([[2]],[[1]])[0,0],.28):raise ValueError('first-order sentinel')
    if not np.isclose(m.second([[2]],[[1]])[0,0],-.056):raise ValueError('second-order sentinel')
    log=OUT/'t17a_math_preflight_pytest.txt'
    if not log.exists() or '25 passed' not in log.read_text(encoding='utf8'):raise ValueError('actual mathematical tests required')
    save(OUT/'t17a_math_preflight.json',dict(stage='independent mathematical preparation only',
        mathematical_sentinel_passed=True,pytest_summary=log.read_text(encoding='utf8').strip().splitlines()[-1],
        main_mechanism_run_performed=False,accepted_T16G_commit_pending=True,new_channel_generated=False,
        formal_test_used=False,source_sha256={p:sha(ROOT/p) for p in SOURCES}))
    print('T17A independent mathematical preparation PASS; historical startup/commit gate remains pending')


def run():
    start=time.perf_counter();commit,g=accepted()
    if RESULT.exists():raise ValueError('existing mechanism output protected')
    preflight=load(FORMAL_PREFLIGHT);startup=ROOT/preflight['startup_log_path']
    if not preflight['all_checks_passed'] or sha(startup)!=preflight['startup_log_sha256']:
        raise ValueError('successful current full pytest and frozen preflight required')
    for p,digest in preflight['source_sha256'].items():
        if sha(ROOT/p)!=digest:raise ValueError('source changed after execution preflight:'+p)
    before=history();tick=time.perf_counter();env=environments();environment_seconds=time.perf_counter()-tick
    tick=time.perf_counter();stats,var,stat_proof=load_frozen_statistics();load_seconds=time.perf_counter()-tick
    tick=time.perf_counter();weights,operator_proof,operator_sentinel=prepare_operators(stats,var)
    prepare_seconds=time.perf_counter()-tick
    tick=time.perf_counter();values=evaluate(stats,weights,var);mechanism_seconds=time.perf_counter()-tick
    files={'t17a_covariance_additivity.csv':values['covariance_additivity'],
        't17a_perturbation_terms.csv':values['perturbation_terms'],
        't17a_mechanism_closure.csv':values['mechanism_closure'],
        't17a_interaction_spectrum.csv':values['interaction_spectrum'],
        't17a_c75_interpolation.csv':values['C75_auxiliary']}
    for name,rows in files.items():
        if (OUT/name).exists():raise ValueError('existing numerical artifact protected')
        csvsave(OUT/name,rows)
    if before!=history():raise ValueError('historical bytes changed')
    result=dict(task='T17A',experiment_type=TYPE,research_question=QUESTION,hypothesis=HYPOTHESIS,
        accepted_T16G_commit=commit,literature_gate_completed=True,closest_work_matrix_completed=True,
        startup_full_pytest_passed=True,startup_full_pytest_count=1610,permission_recovery_completed=True,
        formal_execution_preflight_sha256=sha(FORMAL_PREFLIGHT),
        novelty_claim_made=False,claim_evidence_matrix_completed=True,reviewer_red_team_completed=True,
        statistics_provenance=stat_proof,operator_provenance=operator_proof,
        historical_operator_quadratic_sentinels=operator_sentinel,
        historical_raw_target_W_hash_available=False,historical_full_W_hashes_claimed=False,
        first_order_results=values['perturbation_terms'],second_order_results=[x for x in values['mechanism_closure'] if x['prediction'].startswith('second_order')],
        **values,gate_stage1='L_add',gate_stage2='L_add + L_interaction + Q_cross',gate_reference='W_B120-W0',
        subspace_diagnostics=values['principal_angles'],c75_interpolation=values['C75_auxiliary'],
        c6_claim_status=c6_status(values),scientific_gate_thresholds=dict(strong_first=.70,strong_second=.90,
            contradicted_first=.30,contradicted_second=.50),
        validation_NMSE_rerun=False,frozen_T16F_T16G_NMSE_artifacts_hash_verified=True,
        new_model_trained=False,new_channel_generated=False,generator_calls=0,new_statistics_fit=False,
        coefficient_refits=0,formal_test_used=False,formal_test_sealed=True,
        source_sha256={p:sha(ROOT/p) for p in SOURCES},frozen_history_sha256=before,
        preparation_artifact_sha256=preflight['preparation_artifact_sha256'],
        pre_run_source_snapshot_root=preflight['source_snapshot_root'],
        numeric_artifact_sha256={f'outputs/{name}':sha(OUT/name) for name in files},environment_before=env,
        runtime=dict(statistics_load_seconds=load_seconds,filter_regression_seconds=prepare_seconds,
                     environment_audit_seconds=environment_seconds,**values['runtime_components'],mechanism_seconds=mechanism_seconds,
                     total_numeric_wall_seconds=time.perf_counter()-start),all_checks_passed=False,
        completion_state='mechanism artifacts saved; final engineering audit pending',
        warnings=['Local expansion may fail at finite shifts; closure is not clipped.',
            'C75 gammas and filter beta have distinct objectives and do not enter gate.',
            'Raw target filter hashes absent historically: snapshot/statistics/quadratic sentinels are distinct evidence.',
            'T16D fitted oracle residual differs from unit-coordinate exact interaction.',
            'Literature UNKNOWN fields remain unresolved; no novelty conclusion.'])
    save(RESULT,result)
    print('T17A mechanism artifacts saved; final audit pending')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--sentinel',action='store_true')
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    if args.sentinel:sentinel()
    elif args.preflight:formal_preflight()
    else:run()
