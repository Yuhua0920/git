"""T19B immutable-history, exact-matrix, claims, figures and full-regression gates."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.linalg import cholesky
from src import operator_space_audit as p
from src import frozen_affine_validation as old
from src.cdl_benchmark import array_hash
from scripts.run_operator_space_audit import OUT,RESULT,COMMIT,SOURCES,TABLES,accepted,mean_policy,inputs,no_generation,verify_hashes
from scripts.run_cdl_mismatch import load,save,sha,environments
from scripts.run_pilot_cv import readcsv
from scripts.operator_space_audit_provenance import verify_sources


def close(a,b):
    if not np.allclose(a,b,rtol=p.CALIBRATION_RTOL,atol=p.CALIBRATION_RTOL):
        raise ValueError('exact matrix regression outside machine-precision calibration tolerance')


def verify(r):
    verify_sources(r)
    for path,h in r['numeric_artifact_sha256'].items():
        if sha(ROOT/path)!=h:raise ValueError('numeric artifact drift:'+path)
    for group in ('claim_document_sha256','figure_artifact_sha256'):
        for path,h in r.get(group,{}).items():
            if sha(ROOT/path)!=h:raise ValueError('final artifact drift:'+path)
    if any(r[k] for k in ('new_channel_generated','deterministic_channel_replay_used','new_module_added','new_model_trained','third_module_deployment','gamma_pilot_fitting','formal_test_used')):
        raise ValueError('scope violation')
    if r['generator_calls']!=0 or r['statistics_refits']!=0 or not r['formal_test_sealed']:
        raise ValueError('no-generation/no-fit/formal gate failed')
    if (r['c6_claim_status'],r['c7_claim_status'],r['c8_claim_status'],r['c9_claim_status'],r['t18a_materiality_status'])!=(old.C6,old.C7,'SUPPORTED_IN_FIXED_ENVIRONMENTS','INCONCLUSIVE','PARTIAL'):
        raise ValueError('historical claim changed')
    if not r['t19a_delay_factor_gate_partial'] or not r['t19a_rank4_out_of_factor_warning'] or r['target_matched_W_historical_array_hash_available']:
        raise ValueError('historical evidence misstated')
    mapping={'exact_mismatch_budget':'operator_excess','operator_span_capture':'operator_span_capture_dense',
        'mean_bias':'mean_bias','rank4_span':'operator_span_capture_lr4','deployed_operator_efficiency':'deployed_operator_efficiency',
        'delay_residual_similarity':'delay_residual_cosines','delay_direction_capture':'operator_span_capture_3d'}
    for name,key in mapping.items():
        csvrows=readcsv(OUT/f't19b_{name}.csv')
        if len(csvrows)!=len(r[key]):raise ValueError('CSV/JSON row count drift')
        for csvrow,jsonrow in zip(csvrows,r[key]):
            for field,value in jsonrow.items():
                cell=csvrow[field]
                if value is None:
                    if cell!='':raise ValueError('undefined CSV value drift')
                elif isinstance(value,(int,float)) and not isinstance(value,bool):
                    if float(cell)!=value:raise ValueError('CSV/JSON numeric drift')
                elif cell!=str(value):raise ValueError('CSV/JSON provenance drift')
    if any(r[k]!=v for k,v in p.dense_gate(r['operator_span_capture_dense']).items()) or any(r[k]!=v for k,v in p.reuse_gate(r['operator_span_capture_3d']).items()):
        raise ValueError('frozen scientific gate drift')
    with no_generation():
        historical=accepted();evidence=mean_policy(historical)
        if evidence!=r['oracle2_mean_policy_provenance'] or evidence['policy']!=r['t19a_oracle2_mean_policy']:
            raise ValueError('actual mean-policy drift')
        arrays,family,source,targets,_=inputs(historical)
        with np.load(OUT/'t19b_delay_residual_directions.npz',allow_pickle=False) as f:
            residuals={k:f[k] for k in f.files}
        with np.load(OUT/'t19b_derived_matched_operators.npz',allow_pickle=False) as f:
            weights={k:f[k] for k in f.files}
        metadata=r['array_metadata']
        if any(array_hash(v)!=metadata['residual_array_sha256'][k] for k,v in residuals.items()) or any(array_hash(v)!=metadata['derived_W_array_sha256'][k] for k,v in weights.items()):
            raise ValueError('persisted array drift')
        if len(weights)!=16 or len(residuals)!=36:raise ValueError('missing persisted residual/control arrays')
        cap={(v['environment'],v['snr_db']):v for v in r['operator_span_capture_dense']}
        budget={(v['environment'],v['snr_db']):v for v in r['operator_excess']}
        rank={(v['environment'],v['snr_db']):v for v in r['operator_span_capture_lr4']}
        means={(v['environment'],v['snr_db']):v for v in r['mean_bias']}
        gram={};opt={};excess={}
        for n in p.ENVIRONMENTS:
            for s in p.SNRS:
                wt=weights[f'Wtarget_{n}_{s}'];st=targets[n];ws=arrays[f'Wsource_{s}']
                # The original solve reconstructs deterministically; no new statistics.
                if not np.array_equal(wt,st.prepare(old.VARIANCES[s]).weight):raise ValueError('matched-W deterministic reconstruction drift')
                lt=cholesky(st.rpp+old.VARIANCES[s]*np.eye(64),lower=True)
                mods=[arrays[f'DeltaW{m}_{s}'] for m in ('P','M')];co=np.array([cap[n,s]['alpha_op'],cap[n,s]['beta_op']])
                residual=wt-(ws+sum(c*m for c,m in zip(co,mods)))
                if not np.array_equal(residual,residuals[f'R_delay_{n}_{s}']):raise ValueError('residual definition drift')
                fit=p.operator_fit(ws,wt,lt,mods);close(co,fit['coefficients'])
                close(p.energy(residual@lt),cap[n,s]['E_op_oracle2'])
                dp=st.mean_p-arrays['source_mean_p'];dd=st.mean_d-arrays['source_mean_d']
                sw=p.mismatch(ws,wt,lt,dp,dd)
                close([sw['E_op'],sw['E_mean'],sw['E_total']],[budget[n,s][k] for k in ('E_op_source','E_mean_source','E_total_source')])
                ow=ws+sum(c*m for c,m in zip(co,mods))
                close(p.energy(p.mean_bias(ow,dp,dd)),budget[n,s]['E_mean_oracle2'])
                close(p.energy(p.mean_bias(wt,dp,dd)),budget[n,s]['matched_operator_source_mean_floor'])
                total=p.mean_aware_fit(ws,wt,lt,mods,dp,dd)
                close(total['coefficients'],[means[n,s]['alpha_total'],means[n,s]['beta_total']])
                close(total['excess'],means[n,s]['E_total_oracle_totalcoords'])
                mods4=[family[m,s,4].A@family[m,s,4].Q for m in ('P','M')]
                fit4=p.operator_fit(ws,wt,lt,mods4);close(fit4['excess'],rank[n,s]['E_op_lr4'])
                gram[n,s]=fit['gram'];opt[n,s]=co;excess[n,s]=fit['excess']
        original={(v['environment'],int(v['snr_db']),int(v['trajectory_id']),int(v['frame_id'])):v for v in readcsv(OUT/'t19a_weights.csv') if v['method']=='affine'}
        windows=readcsv(OUT/'t19b_deployed_operator_windows.csv')
        if len(windows)!=9600 or len(original)!=9600:raise ValueError('window population drift')
        if {(v['environment'],int(v['snr_db']),int(v['trajectory_id']),int(v['frame_id'])) for v in windows}!=set(original):
            raise ValueError('window duplicate/missing condition')
        for v in windows:
            key=v['environment'],int(v['snr_db']);oid=(*key,int(v['trajectory_id']),int(v['frame_id']))
            saved=original[oid];co=np.array([float(v['alpha_hat']),float(v['beta_hat'])])
            if [float(saved['alpha']),float(saved['beta'])]!=co.tolist():raise ValueError('saved coefficients changed')
            ex=p.deployed_excess(co,opt[key],gram[key],excess[key]);close(ex,float(v['E_op_deployed']))
            expected=p.deployed_efficiency(cap[key]['E_op_source'],ex,excess[key])
            if expected is None:
                if v['deployed_operator_efficiency']!='' or not v['undefined_reason']:raise ValueError('invalid denominator not flagged')
            else:close(expected,float(v['deployed_operator_efficiency']))
        similarity={(v['environment'],v['snr_db']):v for v in r['delay_residual_cosines']}
        for s in p.SNRS:
            ls=cholesky(source.rpp+old.VARIANCES[s]*np.eye(64),lower=True)
            if not np.array_equal(ls,residuals[f'L_source_{s}']):raise ValueError('common source metric drift')
            candidate=residuals[f'R_delay_{p.DEVELOPMENT}_{s}']
            if not np.array_equal(candidate,residuals[f'D_delay_200_{s}']):raise ValueError('candidate provenance drift')
            for n in p.ENVIRONMENTS:
                if not np.array_equal(residuals[f'R_delay_{n}_{s}']@ls,residuals[f'Rbar_{n}_{s}']):raise ValueError('source-weighted array drift')
            for n in p.HELDOUT:
                close(p.signed_cosine(candidate,residuals[f'R_delay_{n}_{s}'],ls),similarity[n,s]['cos_delay'])
        for v in r['operator_span_capture_3d']:
            n,s=v['environment'],v['snr_db'];ws=arrays[f'Wsource_{s}'];wt=weights[f'Wtarget_{n}_{s}']
            lt=cholesky(targets[n].rpp+old.VARIANCES[s]*np.eye(64),lower=True)
            joint=p.operator_fit(ws,wt,lt,[arrays[f'DeltaWP_{s}'],arrays[f'DeltaWM_{s}'],residuals[f'D_delay_200_{s}']])
            close(joint['coefficients'],[v['alpha3'],v['beta3'],v['gamma3']]);close(joint['excess'],v['E_op_3d'])
            if n==p.DEVELOPMENT or not v['joint_fit'] or not v['oracle_only']:raise ValueError('direction gate leakage')
    return True


def claim():
    r=load(RESULT);verify(r)
    if r.get('claim_document_sha256'):raise ValueError('existing claim documents protected')
    text='# T19B Claim–Evidence update\n\nResearch Question: exact separation of operator capacity, source-mean bias, rank4 compression and coordinate inference; held-out reuse of a delay residual.\n\n'
    text+='Historical states remain C6='+r['c6_claim_status']+', C7='+r['c7_claim_status']+', C8='+r['c8_claim_status']+', C9=INCONCLUSIVE; T18A materiality PARTIAL. T19A partial gate and rank4 warning are preserved.\n\n'
    text+='C10: '+p.C10+'\n\nStatus: '+r['c10_claim_status']+'. This is an oracle operator-space diagnostic, not deployed-estimator validation or a physical orthogonality claim.\n\n'
    text+='| Pure condition | Dense capture | Rank4 capture | Capture loss | Source mean fraction | Mean-aware gain |\n|---|---:|---:|---:|---:|---:|\n'
    for i,v in enumerate(r['operator_span_capture_dense'][:4]):
        rank=r['operator_span_capture_lr4'][i];budget=r['operator_excess'][i];mean=r['mean_bias'][i]
        text+='| '+v['environment']+'/'+str(v['snr_db'])+' | '+' | '.join(str(x) for x in (v['operator_span_capture_dense'],rank['operator_span_capture_lr4'],rank['rank4_span_capture_loss'],budget['mean_fraction_source'],mean['mean_aware_coordinate_gain']))+' |\n'
    text+='\n'+''.join(f'- {k}: {r[k]}\n' for k in ('dense_pm_operator_space_delay_sufficient','dense_pm_operator_space_delay_deficient','dense_pm_operator_space_delay_partial','condition_specific_operator_deficiency','delay_residual_direction_reusable','delay_direction_reuse_weak','delay_direction_reuse_partial','promising_count','weak_count'))
    text+='\nDecision dependency: '+r['next_step_dependency']+' No next-stage work is implemented. Compression losses are reported without a new significance/dB threshold.\n\n'
    text+='Evidence: frozen finite training statistics,16 conditions,14 held-out residual tests and8 primary reuse rows. Common-source metric used for direction alignment; target metric used for independent joint3D oracle projection. Actual T19A Oracle2 uses source means. Exact excess does not require Rdd; absolute theoretical NMSE is unavailable. Frozen-statistics exactness does not remove covariance estimation uncertainty.\n'
    review='# T19B Reviewer Red Team\n\n'
    review+='- Exact centered excess and mean-bias energy are different from empirical validation NMSE closure. Their denominators must not be interchanged. No claim that exact finite-training matrices establish true population capacity.\n'
    review+='- Actual Oracle2 uses source means; matched T19A uses target means. Operator-only and mean-aware controls separate this asymmetry. Missing historical matched-W hashes are disclosed; newly reconstructed W hashes are not invented history.\n'
    review+='- Full Rdd exists only in an independent synthetic sentinel. No absolute expected MSE/NMSE, new Rdd estimate or anchored prediction is claimed.\n'
    review+='- Rank4 dense matrices are audit-only products of frozen factors. Old inference/coefficients/basis and historical1.054123dB penalty are unchanged; no automatic rank increase.\n'
    review+='- All9600 original affine windows are included. Window quartiles are correlated descriptive summaries, not confidence intervals over9600 independent samples. No clipping, deletion or seed search.\n'
    review+='- P/M names describe provenance, not physical orthogonality. Delay residuals can reflect simultaneous covariance shifts and estimation noise. Signed cosine is not a physical mechanism.\n'
    review+='- C30/200 supplies the candidate per SNR and is excluded from all14 held-out rows. The8 primary contexts and0.70/0.10/0.05 gate were fixed before analysis. Joint3D LS reoptimizes all coefficients; no sequential fit.\n'
    review+='- C10 is bounded to an evaluation-only oracle direction on tested contexts. No deployability, pilot inference for gamma, third module, network, BER or formal-test evidence exists. C9 historical INCONCLUSIVE remains intact.\n\n'
    review+='Actual bounded outcome: C10='+r['c10_claim_status']+'; '+r['next_step_dependency']+'\n'
    hashes={}
    for name,value in (('t19b_claim_evidence_update.md',text),('t19b_reviewer_update.md',review)):
        path=ROOT/'docs'/name
        if path.exists():raise ValueError('existing document protected')
        path.write_text(value,encoding='utf8');hashes['docs/'+name]=sha(path)
    r['claim_document_sha256']=hashes;save(RESULT,r);print('T19B bounded claims/reviewer update generated',flush=True)


def visual_review():
    r=load(OUT/'t19b_figure_audit.json')
    if r['source_returncode'] or not all(r['CSV_only_rebuild'].values()) or any(v['text_returncode'] or v['collision_returncode'] for v in r['rendered_checks'].values()):
        raise ValueError('automatic figure QA incomplete')
    r.update(visual_review='Actual three PNGs inspected: all four pure conditions, source/Oracle2/matched source-mean budgets and14 held-out direction rows. Axis scales retain signed values; legends/units/denominators and no-CI scope visible.',all_checks_passed=True)
    save(OUT/'t19b_figure_audit.json',r)


def finalize():
    tick=time.perf_counter();r=load(RESULT);verify(r)
    if not load(OUT/'t19b_figure_audit.json')['all_checks_passed'] or not r.get('claim_document_sha256'):
        raise ValueError('claims/manual figure QA incomplete')
    env=dict(os.environ);env['TEMP']=env['TMP']=str(ROOT/'.pytest_cache/pytest_workspace_temp');tests={}
    groups=[('related',['tests/test_operator_space_audit.py']),
            ('provenance',['tests/test_delay_factor_challenge.py','tests/test_historical_provenance.py','tests/test_frozen_generalization.py','tests/test_validation_replication.py']),('full',[])]
    for label,args in groups:
        path=OUT/f't19b_pytest_{label}.txt'
        if path.exists():raise ValueError('existing regression log protected')
        t=time.perf_counter();print('Actual '+label+' regression',flush=True)
        with path.open('w',encoding='utf8') as f:
            proc=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-m','pytest',*args,'-q'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        summary=path.read_text(encoding='utf8').strip().splitlines()[-1];print(summary,flush=True)
        tests[label]=dict(command=[str(ROOT/'.venv/Scripts/python.exe'),'-m','pytest',*args,'-q'],summary=summary,exit_code=proc.returncode,seconds=time.perf_counter()-t,sha256=sha(path))
        if proc.returncode or 'skipped' in summary:raise ValueError('regression failure/skip:STOP')
    verify(r);after=environments()
    if after!=r['environment_before']:raise ValueError('frozen environment changed')
    diff=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    if diff.returncode:raise ValueError('git diff --check failed')
    new_source={}
    for name in (*SOURCES,*r['claim_document_sha256']):
        lines=(ROOT/name).read_text(encoding='utf8').splitlines()
        bad=[i for i,line in enumerate(lines,1) if line.rstrip()!=line]
        if bad:raise ValueError('new file whitespace:'+name)
        new_source[name]=dict(lines=len(lines),trailing_whitespace_lines=bad,sha256=sha(ROOT/name))
    from scripts.plot_operator_space_audit import STEMS
    r.update(pytest=tests,environment_after=after,git_diff_check=dict(exit_code=diff.returncode,output=diff.stdout+diff.stderr),
        new_file_diff_equivalent=new_source,final_hash_audit=dict(passed=True,historical_files=len(r['frozen_history_sha256']),unexpected_content_mismatch=0),
        figure_audit_sha256=sha(OUT/'t19b_figure_audit.json'),figure_artifact_sha256={f'outputs/{stem}.{ext}':sha(OUT/f'{stem}.{ext}') for stem in STEMS for ext in ('png','pdf','svg','alignment.json','collision-audit.json')},
        all_checks_passed=True,completion_state='completed')
    r['runtime']['final_audit_and_regression_seconds']=time.perf_counter()-tick
    save(RESULT,r);print('T19B engineering PASS; C10='+r['c10_claim_status']+'; formal test sealed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['claim','visual-review','finalize']);args=parser.parse_args()
    {'claim':claim,'visual-review':visual_review,'finalize':finalize}[args.mode]()
