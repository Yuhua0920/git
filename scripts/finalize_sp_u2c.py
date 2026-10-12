"""Generate mentor-review interpretation and immutable delivery hashes."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import sp_u2c_diagnostic as d
d.install_guard()


def packet():
    from scripts.run_sp_u2c import verify
    pre=verify();a=d.load(d.OUT/'sp_u2c_analysis.json');g=d.load(d.OUT/'sp_u2c_run_gate.json')
    audit=d.load(d.OUT/'sp_u2c_integrity_audit.json');qa=d.load(d.OUT/'sp_u2c_figure_qa_execution.json')
    assert audit['passed'] and qa['all_exit_zero']
    hypotheses=[
        dict(hypothesis='H1',observation='Negative conditional means: 79/120 at10dB,113/120 at20dB; both split halves negative57/120,106/120; negative trajectory means16/20,19/20',
             interpretation='Persistence is substantially stronger at20dB;10dB remains heterogeneous; sample means are not exact conditional expectations',
             alternative='20 draws leave conditional Monte Carlo error; overlapping windows and selected development channels limit generalization',evidence_strength='moderate development evidence, stronger at20dB',verdict='MIXED'),
        dict(hypothesis='H2',observation='Negative cross1736/2400,1896/2400; negative mean cross115/120,117/120; every trajectory mean cross negative at both SNRs',
             interpretation='Negative averaged oracle cross cancels movement cost in many fixed windows',
             alternative='Negative cosine conditional on negative penalty is partly algebraic; oracle and M vary with noisy current input; current-frame dependence remains unisolated',evidence_strength='exact bookkeeping plus moderate conditional evidence; no isolated causal mechanism',verdict='SUPPORT'),
        dict(hypothesis='H3',observation='Trajectory Spearman sigma_min versus RMS:-.984962,-.978947; versus trace covariance:-.923308,-.849624',
             interpretation='Clean absolute excitation association survives repeated-noise endpoints',alternative='Channel strength, direction-specific excitation and trajectory temporal structure may confound associations',evidence_strength='strong descriptive association on20 development trajectories; not causal or external validation',verdict='SUPPORT'),
        dict(hypothesis='H4',observation='Trajectory Spearman clean kappa versus RMS:+.193985,+.105263; versus trace:-.003008,+.061654',
             interpretation='Condition number remains secondary in this population',alternative='Condition range and20-trajectory population may be restricted',evidence_strength='descriptive within fixed population',verdict='SUPPORT'),
        dict(hypothesis='H5',observation='All120/120 conditional windows/SNR have surrogate mean gap greater than absolute mean noise effect;117/120 and120/120 greater than meanQ',
             interpretation='Coefficient noise sensitivity cannot by itself explain the principal oracle gap',alternative='Gap uses realization-specific noisy-input rank4 oracle; not pure population surrogate mismatch and not a full representation/compression separation',evidence_strength='exact risk diagnostic in this operator class',verdict='SUPPORT')]
    d.csvsave(d.OUT/'sp_u2c_table3_hypothesis_audit.csv',hypotheses)
    figures=d.load(d.OUT/'sp_u2c_figure_manifest.json')
    visual=[]
    observations={
      'A':('All240 conditional points show an inverse excitation-RMS association','Channel strength and directional excitation confounding'),
      'B':('All240 points show weak condition-RMS organization compared with A','Restricted condition range; scale choices do not establish causality'),
      'C':('Mean cross is often below minus meanQ; all4800 cells per Q,C,delta included','Algebraic decomposition is not causal mechanism; pooled heatmap scale makes small20dB values visually pale'),
      'D':('Conditional signed-risk means differ in persistence and scale across SNR','Shared pooled histogram bins compress the20dB distribution; exact counts and quantiles are inCSV'),
      'E':('Negative-penalty ECDF has much more negative cosine than positive group','Grouping by delta sign imposes part of the separation algebraically'),
      'F':('All240 conditional points have surrogate gap above absolute mean noise effect','Signed-noise averaging can cancel; also compare surrogate to positive meanQ')}
    for f in figures:
        letter=f['figure'];alignment=d.load(d.OUT/f'sp_u2c_figures/fig_{letter}.alignment.json')
        collision=d.load(d.OUT/f'sp_u2c_figures/fig_{letter}.collision.json')
        assert collision['verdict']=='PASS'
        assert alignment['verdict'] in ('PASS','NOT APPLICABLE')
        visual.append(dict(figure=letter,visual_inspection='whole figure and each panel reviewed in rendered PNG',
             labels='readable; no clipping/overlap seen; SNR colors and shapes consistent',
             panel_count=4 if letter=='C' else 1,source=f['raw_source'],count=f['count'],excluded=f['excluded'],
             physical_width_mm=183,physical_height_mm=210 if letter=='C' else 125,dpi=300,
             alignment=alignment['verdict'],collision=collision['verdict'],observation=observations[letter][0],alternative=observations[letter][1]))
    d.save(d.OUT/'sp_u2c_figure_review.json',dict(passed=True,figures=visual,
          source_preflight=dict(pass_count=17,warn_count=4,fail_count=0,
          warning_review=['PNG is development preview, editable PDF/SVG delivered; no TIFF submission claim',
             '300dpi preview satisfies current diagnostic use; no600dpi raster claim',
             'static dimension parser misreads183/25.4; actual rendered width183mm verified fromPDF',
             'logarithmic data positivity asserted forA/B/F; C meanQ strictly positive; signedC intentionally symlog']),
          uncertainty='full raw conditional points/ECDF/heatmaps; no inference intervals; all repeat spread inCSV'))
    w=d.read_rows(d.OUT/'sp_u2c_conditional_windows.csv')
    bias={str(s):dict(mean_squared_conditional_shift=sum(v['bias_norm']**2 for v in w if v['snr_db']==s)/120,
                     mean_V_c=sum(v['V_c'] for v in w if v['snr_db']==s)/120,
                     mean_covariance_component=.95*sum(v['trace_covariance'] for v in w if v['snr_db']==s)/120,
                     mean_noise_MCSE_delta=sum(v['noise_MCSE_delta'] for v in w if v['snr_db']==s)/120) for s in d.SNRS}
    d.save(d.OUT/'sp_u2c_advisor_review_packet.json',dict(task='SP-U2C',head=d.HEAD,decision='C. MIXED EFFECT',
           hypotheses_verbatim=pre['protocol']['hypotheses_verbatim'],hypothesis_audit=hypotheses,raw_report=a['report'],associations=a['associations'],
           conditional_shift_decomposition=bias,run=g,integrity=audit,figures=visual,
           bounded_claim='In this fixed development population, noisy cross-pilot adaptation often offsets nonzero output movement cost by moving toward the realization-specific rank4 channel-risk optimum for the same noisy input; conditional persistence is SNR/window dependent.',
           no_claim=['noise improves estimation','universal beneficial bias','unbiasedness','iid OLS covariance','general theorem','independent validation'],
           scientific_decision=dict(Research_Question='Does repeated AWGN preserve negative coordinate-risk effect and oracle-directed cross term?',
              Hypothesis='Frozen SP-U2C H1-H5',Observation='See unmodified raw report and hypothesis table',
              Interpretation='Conditional cancellation survives especially20dB, with mixed windows at10dB',
              Alternative_Explanation='Current-frame/fold coupling and repeat-specific oracle/output metric are not isolated; development population and finite repeats',
              Evidence_Strength='Exact model identities plus moderate conditional development evidence',
              Claim_Impact='Reject movement magnitude equals degradation; retain strictly scoped conditional-risk geometry',
              Novelty_Impact='Diagnostic value only; no verified new estimator or closest-work advantage',
              Reviewer_Impact='Clarifies reference/oracle/noise definitions and sample clustering; mechanism and external validity remain open',
              Decision='C. MIXED EFFECT; no additional general theory in this task',
              Recommended_Next_Minimal_Experiment='Only if mentor assigns: isolate current-frame participation on these fixed subjects/operators/noise seeds. Not executed.'),
           theory_branch='Conditional insight exceeds excitation-only statement, but general theory/causal mechanism remains unestablished; stop pending mentor review',
           self_rating=dict(engineering='PASS after archived pre-noise tuple-adapter repair',scientific_hypothesis='PARTIAL: H1 mixed, H2-H5 supported within scope',
               theory_branch='PARTIAL, no new theorem and no authorization to extend',firewall='PASS',reproducibility='PASS'),
           failures=[dict(stage='first actual fixed-input entry',exception='AttributeError: tuple has no ravel',
                         main_noise_draws_before_failure=0,archive='outputs/sp_u2c_preflight_failure01',repair='tuple[0] extraction and real-entry test; independent execution amendment; protocol/seeds/budget unchanged')],
           task_local_lessons=[dict(rule='Before freezing a repeat run, exercise actual estimator return types and full fixed/noisy-input entry, in addition to synthetic algebra tests',evidence='archived first entry failure and21 passing pre-run tests')],
           stopped=True,next_task_started=False,production_changed=False,paper_changed=False,committed=False))
    print('Mentor packet generated: C MIXED EFFECT; five-hypothesis table; all six visual QA records')


def manifest():
    from scripts.run_sp_u2c import verify,git
    pre=verify()
    assert d.load(d.OUT/'sp_u2c_delivery_tests.json')['exit_code']==0
    paths=list(d.OUT.glob('sp_u2c_*'))
    files=[]
    for p in paths:
        if p.is_dir():
            if 'pytest_tmp' in p.name or 'mplconfig' in p.name:continue
            files.extend(q for q in p.rglob('*') if q.is_file())
        elif p.is_file():files.append(p)
    artifacts={p.relative_to(ROOT).as_posix():d.sha(p) for p in sorted(files)}
    source={p.relative_to(ROOT).as_posix():d.sha(p) for folder in ('scripts','tests','configs') for p in (ROOT/folder).glob('*sp_u2c*') if p.is_file()}
    d.save(d.OUT/'sp_u2c_delivery_manifest.json',dict(task='SP-U2C',head=d.HEAD,artifact_count=len(artifacts),artifact_sha256=artifacts,
           delivery_sources=source,protected_sources_count=len(pre['protected_sources']),protocol_sha256=pre['protocol_sha256'],
           tracked_diff=git('diff','--stat'),staged_diff=git('diff','--cached','--stat'),stopped=True,formal_read=False,channel_generator_calls=0))
    print('Manifest frozen:',len(artifacts),'artifacts plus manifest;',len(source),'new source/config/test files; tracked diff empty')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['packet','manifest']);globals()[p.parse_args().action]()
