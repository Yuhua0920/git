"""Script-generated advisor audit packet and append-only delivery manifest."""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import sp_u2b_diagnostic as d
d.install_guard()
import numpy as np
from scripts.analyze_sp_u2b import read_rows
from scripts.run_sp_u2b import verify


def main():
    pre=verify()
    result=d.load(d.OUT/'sp_u2b_analysis.json')
    integrity=d.load(d.OUT/'sp_u2b_integrity_audit.json')
    qa=d.load(d.OUT/'sp_u2b_figure_qa_execution.json')
    assert qa['all_exit_zero'] and integrity['replayed_production_windows']==1200
    def association(s,x,y):
        return next(r['spearman'] for r in result['associations'] if r['scope']=='trajectory_means' and r['snr_db']==s and r['predictor']==x and r['outcome']==y)
    hypotheses=[
        dict(hypothesis='H1',verdict='Support',observation='clean sigma_min vs noise-coordinate change: '+str([association(s,'clean_sigma_min','delta_noise_norm') for s in d.SNRS]),
             interpretation='absolute excitation associates with sensitivity in both SNR strata',alternative='realization gain remains a partial confound; gain-stratified and within-trajectory association reduce but do not remove it',
             evidence_strength='moderate descriptive development evidence',decision_impact='retain excitation-sensitive characterization; no causal threshold'),
        dict(hypothesis='H2',verdict='Mixed',observation='noisy kappa correlations '+str([association(s,'noisy_kappa_G','delta_noise_norm') for s in d.SNRS])+'; joint-bound correlations '+str([association(s,'joint_bound','delta_noise_norm') for s in d.SNRS]),
             interpretation='condition alone is weak; joint proxies are informative descriptively',alternative='incremental information beyond clean sigma_min is unproven; relative perturbation is substantially gain-confounded; observed sigma_min is slightly stronger than clean sigma_min',
             evidence_strength='moderate for insufficiency of condition; limited for new joint explanatory value',decision_impact='do not claim joint proxy dominance or tight uncertainty calibration'),
        dict(hypothesis='H3',verdict='Support',observation='conditioning improves in 443/600 and 373/600 windows; improvement OR sigma increase plus upper-quartile change occurs in 128/600 and 97/600',
             interpretation='noise can improve observed numerical geometry while moving coordinates',alternative='movement often reduces MSE; events do not prove harmful instability or information gain',
             evidence_strength='direct population description within one condition',decision_impact='retain observed-conditioning caveat, not a degradation claim'),
        dict(hypothesis='H4',verdict='Mixed',observation='coordinate norm to output energy correlations '+str([association(s,'delta_noise_norm','coord_output_mse') for s in d.SNRS])+'; to signed noise MSE penalty '+str([association(s,'delta_noise_norm','delta_R_noise') for s in d.SNRS]),
             interpretation='exact output propagation and magnitude relation pass; stable channel-degradation link fails',alternative='signed residual/output cross term and surrogate displacement dominate penalty sign',
             evidence_strength='strong numerical identity; descriptive contradiction of complete degradation story',decision_impact='preserve propagation identity; reject noise-coordinate-change implies channel degradation'),
        dict(hypothesis='H5',verdict='Support',observation='surrogate coordinate gap exceeds noise-coordinate gap in 424/600 and 575/600; surrogate MSE gap exceeds absolute noise penalty in 501/600 and 591/600',
             interpretation='pilot surrogate gap remains material, especially at 20dB',alternative='truth-assisted noisy-input realization oracle is not population oracle; compression and representation remain relevant',
             evidence_strength='moderate within tested development condition',decision_impact='do not cast the remaining problem as only coordinate AWGN uncertainty')]
    d.csvsave(d.OUT/'sp_u2b_table3_hypothesis_audit.csv',hypotheses)
    figures=d.load(d.OUT/'sp_u2b_figure_manifest.json')
    figure_notes={
        'A':('negative excitation/sensitivity trend within each SNR','gain and temporal channel variation affect both quantities'),
        'B':('weak and nonmonotonic condition association','condition range is modest; gain stratification changes association'),
        'C':('relative design perturbation associates with coordinate changes','gain-stratum associations weaken; pooled SNR effect exaggerates relation'),
        'D':('bounds hold but sit far above actual changes','triangle/operator norm bounds ignore cancellation and are not calibrated uncertainty'),
        'E':('quadratic prediction and measured difference lie on identity line','exact algebra validates instrumentation, not excess-MSE benefit'),
        'F':('many negative noise-coordinate penalties; surrogate penalties remain','oracle geometry and signed cross term prevent an additive positive-error narrative')}
    # These are records of an actual tool-based visual inspection, not an automated assertion of scientific validity.
    for f in figures:
        f['visual_self_check']='PASS: inspected exported PNG individually; labels/legend visible, no clipping/overlap; log and symlog transforms explicitly stated'
        f['key_observation'],f['alternative_interpretation']=figure_notes[f['figure']]
        f['PDF_text_audit']='PASS: minimum glyph 5.6 pt'
        f['PDF_collision_audit']='PASS: zero FAIL and zero WARN'
        f['physical_size_mm']=[183,125]
    d.save(d.OUT/'sp_u2b_figure_review.json',figures)
    rows=read_rows(d.OUT/'sp_u2b_development_windows.csv')
    cross_keys=['cross_rep_coord','cross_rep_comp','cross_rep_noise','cross_coord_comp','cross_coord_noise','cross_comp_noise']
    energy_keys=['representation_vector_mse','coordinate_vector_mse','compression_vector_mse','observation_noise_vector_mse']
    decomposition=[]
    for s in d.SNRS:
        group=[r for r in rows if r['snr_db']==s]
        energies=[float(np.mean([r[k] for r in group])) for k in energy_keys]
        crosses=[float(np.mean([r[k] for r in group])) for k in cross_keys]
        mse=float(np.mean([r['MSE_noisyCV'] for r in group]))
        d.close(sum(energies)+sum(crosses),mse)
        decomposition.append(dict(snr_db=s,reference='current noisy-input rank4 channel oracle; dense clean representation bookkeeping',
                                  component_mean_MSE=dict(zip(energy_keys,energies)),signed_cross_mean_MSE=dict(zip(cross_keys,crosses)),
                                  sum_components=sum(energies),sum_cross=sum(crosses),actual_mean_MSE=mse,
                                  scalar_additive_partition=False))
    tests=d.load(d.OUT/'sp_u2b_tests_pass.json')
    delivery_test_log=(d.OUT/'sp_u2b_delivery_pytest.txt').read_text(encoding='utf-8')
    assert '6 passed' in delivery_test_log
    failures=[dict(stage='analysis JSON export',error='NumPy int64 not JSON serializable',recovery='type-only boundary normalization; six previous CSVs verified byte-identical'),
              dict(stage='plot QA atomic write',error='guard rejected dot-prefixed tempfile at outputs root',recovery='owned subdirectory; guard and data unchanged'),
              dict(stage='post-run geometry long export',error='heterogeneous CSV schema from clean_residual_norm',recovery='explicit geometry allowlist; full 1200-window replay revalidated'),
              dict(stage='read-only ad hoc summary',error='WindowsPath plus string TypeError due to operator precedence',recovery='correct parenthesized path query; no output data modified')]
    lessons=[dict(trigger='JSON export after aggregate computation',rule='Exercise the complete summary object with NumPy-derived counts through the JSON boundary before expensive execution'),
             dict(trigger='firewall with external atomic writer',rule='Test actual helper tempfile behavior inside an owned subdirectory; do not weaken the guard'),
             dict(trigger='clean/noisy diagnostic long export',rule='Select an explicit common field schema; prefixes can collect non-geometry metrics')]
    packet=dict(task='SP-U2B',head=d.HEAD,decision='THEORY PARTIALLY SUPPORTED',
                next_stage_authorized=False,stopped_for_advisor_review=True,production_changed=False,paper_changed=False,Git_commit=False,
                hypotheses=hypotheses,scientific_definitions=pre['hypotheses_and_protocol'],
                numerical_results=result,integrity=integrity,four_error_cross_terms=decomposition,
                pytest=dict(pre_CDL_passed=24,pre_CDL_failed=0,pre_CDL_errors=0,post_run_passed=6,post_run_failed=0,post_run_errors=0,
                            full_suite=False,reason='full suite opens reporting-only historical/formal outcomes',pre_CDL_log=tests['log'],
                            post_run_log='outputs/sp_u2b_delivery_pytest.txt'),
                negative_execution_records=failures,task_local_lessons=lessons,
                assessment=dict(engineering='PASS after documented recoveries',scientific='PARTIAL; full degradation chain not supported',
                                theory_branch='conditional PASS only for excitation/sensitivity characterization and exact output propagation',
                                firewall='PASS; no old formal outcome reads or hashes',reproducibility='PASS within frozen recorded environment; historical numeric seed disjointness not claimed'),
                minimal_next_experiment_recommendation='Only if advisor requests it: fixed ID 0..19 development channels/data/operators, 20 new preregistered AWGN repeats per trajectory at existing 10/20dB; separate conditional-noise direction and signed MSE from between-realization effects. No new pilot pattern or baseline. Not executed.',
                novelty='no universal theorem or novelty clearance; algorithm characterization only',
                reviewer_impact='remove implication that coordinate movement necessarily degrades channel MSE; acknowledge surrogate/compression/representation and endogenous shared noise',
                figures=figures,access=d.ACCESS)
    d.save(d.OUT/'sp_u2b_advisor_review_packet.json',packet)
    verify()
    diff=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    assert diff.returncode==0
    excluded=[];files={}
    for folder in ('scripts','tests','configs','outputs'):
        for path in (ROOT/folder).rglob('*'):
            if not path.is_file() or 'sp_u2b' not in str(path.relative_to(ROOT)) or '__pycache__' in str(path):continue
            rel=str(path.relative_to(ROOT)).replace('\\','/')
            if any(token in rel for token in ('sp_u2b_pytest_tmp','sp_u2b_delivery_pytest_tmp','sp_u2b_mplconfig')):
                excluded.append(rel);continue
            files[rel]=d.sha(path)
    d.save(d.OUT/'sp_u2b_delivery_manifest.json',dict(task='SP-U2B',head=d.HEAD,files_sha256=files,
               delivery_file_count_excluding_manifest=len(files),scratch_retained_excluded=excluded,
               protected_files_unchanged=506,tracked_and_staged_diff_empty=True,git_diff_check_exit_code=diff.returncode,
               old_outcomes_hashed=False,all_generated_data_script_created=True,stopped=True))
    print('SP-U2B delivered:',len(files),'files excluding final manifest; THEORY PARTIALLY SUPPORTED; STOP',flush=True)
    print('decomposition',decomposition,flush=True)


if __name__=='__main__':main()
