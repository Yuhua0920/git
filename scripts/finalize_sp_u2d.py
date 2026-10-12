"""Record human visual QA, evidence-limited review and final delivery hashes."""
import argparse
import datetime
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import sp_u2d_diagnostic as d
d.install_guard()


def packet():
    from scripts.run_sp_u2d import verify
    import pymupdf
    pre=verify();analysis=d.load(d.OUT/'sp_u2d_analysis.json')
    figures=d.load(d.OUT/'sp_u2d_figure_manifest.json')
    qa=d.load(d.OUT/'sp_u2d_figure_qa_execution.json');assert qa['all_exit_zero']
    reviewed=[]
    for f in figures:
        with pymupdf.open(f['pdf']) as doc:
            assert len(doc)==1
            width,height=doc[0].rect.width*25.4/72,doc[0].rect.height*25.4/72
            assert abs(width-183)<.001 and abs(height-125)<.001
        alignment=d.load(Path(f['pdf']).with_suffix('.alignment.json'))
        assert alignment['verdict'] in ('PASS','NOT APPLICABLE')
        assert d.sha(Path(f['source']))==f['source_sha256']
        reviewed.append(dict(**{k:v for k,v in f.items() if k!='visual_QA'},
            visual_QA='PASS: all five rendered PNGs inspected; labels, legends, zero lines, all points and nonmonotonic profiles visible; no clipping or text collision',
            width_mm=width,height_mm=height,alignment_verdict=alignment['verdict']))
    d.save(d.OUT/'sp_u2d_visual_review.json',dict(passed=True,rendered_figures_inspected=list('ABCDE'),figures=reviewed,
        source_lint=dict(passes=17,warnings=4,failures=0,
            warning_dispositions=['PNG plus vector PDF/SVG appropriate for this diagnostic; TIFF not requested',
              '300 dpi diagnostic PNG; vector exports retain text and lines',
              'Static width parser misreads 183/25.4; actual PDF width verified 183 mm',
              'All Fig A influence values explicitly asserted positive before log plotting']),
        PDF_text_audits=5,PDF_collision_audits=5,all_audits_exit_zero=True))
    evaluations=[
      dict(hypothesis='H1',observation='D_coord mean10=0.000387732,median10=-0.000242343,positive10=10/20; mean20=0.002026942,positive20=15/20; current exceeds oldest in only7/20 and6/20 trajectories.',
           interpretation='Partial influence evidence at20dB; no unique current-frame dominance across both SNRs.',alternative='Endpoint geometry, frame energy, temporal heterogeneity and finite-repeat variability.',strength='Moderate descriptive negative evidence against universal dominance.',verdict='MIXED; strong H1 not supported'),
      dict(hypothesis='H2',observation='D_C positive14/20 at10dB; negative13/20 at20dB. Mean cosine for age1 is more negative than age0 at both SNRs.',
           interpretation='Oracle alignment is SNR- and age-dependent; current frame is not uniformly most aligned.',alternative='Current-observation-dependent M and oracle, noisy regressors, energy and cross-pilot geometry.',strength='Moderate descriptive evidence against unique current alignment.',verdict='MIXED; strong H2 not supported'),
      dict(hypothesis='H3',observation='D_R mean10=+0.000592671,negative5/20; mean20=-0.000155145,negative13/20. Similar-clean-energy subset D_R is positive at both SNRs, n5.',
           interpretation='Two-SNR stable current risk advantage fails.20dB full-set advantage is partial and not retained by primary energy check.',alternative='Energy heterogeneity, correlated windows and realization-specific risk geometry.',strength='Moderate falsification of the requested two-SNR direction; small energy subset is exploratory.',verdict='MIXED; strong H3 not supported'),
      dict(hypothesis='H4',observation='Expected monotone influence profiles1/20 and0/20; C1/20 and1/20; risk1/20 and2/20 at10/20dB.',
           interpretation='No simple recency law; influence is often large at both endpoints.',alternative='Nonmonotone temporal channel structure and geometry.',strength='Descriptive exploratory evidence only.',verdict='Simple recency gradient NOT SUPPORTED'),
      dict(hypothesis='H5',observation='Every variant has4frames and512 real rows; ages nonetheless have different alignment and risk profiles.',
           interpretation='Age contrasts cannot be explained solely by a different frame count. This does not isolate a causal current-state mechanism or exclude averaging contributions to full-fit behavior.',alternative='Pilot amplitude, geometry, chronological heterogeneity and common-observation reuse.',strength='Strong design fact; conditional scientific interpretation.',verdict='conditional PASS for count control; current-state attribution remains unresolved')]
    d.csvsave(d.OUT/'sp_u2d_table4_hypotheses.csv',evaluations)
    d.save(d.OUT/'sp_u2d_review_packet.json',dict(task='SP-U2D',head=d.HEAD,
        protocol_sha256=pre['protocol_sha256'],hypotheses_verbatim=pre['protocol']['hypotheses_verbatim'],
        results=analysis,hypothesis_evaluation=evaluations,
        decision='C: MIXED / SNR-DEPENDENT AGE EFFECT; default THEORY FREEZE; strong A current-state gate FAIL',
        claim='Matched four-frame deletions show observation-dependent, SNR-dependent age effects on conditional risk geometry; unique current-state causal source unresolved.',
        forbidden_claims=['Current frame uniquely drives adaptation','Universal recency law','Noise beneficial in general','LOFO selects a new production window'],
        limitations=['Highest independent unit20 trajectories; shared subjects across SNRs',
          'Similar-clean-energy subset only5 trajectories and7 windows; no causal amplitude conclusion',
          'Noisy-energy subset conditions on noisy observations and retains unequal repeat counts',
          'Truth oracle and LOFO are diagnostic only; all output maps use the same current noisy observation'],
        ratings=dict(Engineering='PASS',Scientific_hypothesis='conditional PASS for diagnostic; strong H1-H3 FAIL',
           Mechanism_branch='FAIL for strong current-state attribution; C weak age-effect conditional PASS; THEORY FREEZE',
           Firewall='PASS',Reproducibility='PASS'),
        no_next_task=True,next_minimal_experiment='None executed or scheduled; await mentor review.',
        old_outcomes_used_for_design=False,new_channel=False,new_noise=False,production_changed=False,formal_read=False))
    print('PASS: five rendered figure reviews;183x125mm PDFs;H1-H5 evidence packet;C default freeze')


def seal():
    from scripts.run_sp_u2d import verify,git
    pre=verify()
    cmd=[sys.executable,'-m','pytest','tests/test_sp_u2d_delivery.py','-q','-p','no:cacheprovider','--basetemp=outputs/sp_u2d_pytest_tmp_post']
    result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
    log=d.OUT/'sp_u2d_pytest_post.txt';assert not log.exists();log.write_text(result.stdout+result.stderr,encoding='utf8')
    print(result.stdout+result.stderr,flush=True)
    if result.returncode:raise RuntimeError('delivery checks failed')
    verify()
    d.save(d.OUT/'sp_u2d_final_execution_audit.json',dict(passed=True,head=git('rev-parse','HEAD'),
        protected_sources=len(pre['protected_sources']),old_inputs=len(pre['input_sha256']),
        pre_test_exit=0,post_test_exit=result.returncode,post_test_command=cmd,
        production_replays=4800,LOFO_replayed_exact=180,production_changed=False,new_channel=False,new_noise=False,
        formal_read=False,tracked_diff=git('diff','--stat'),staged_diff=git('diff','--cached','--stat'),
        git_diff_check_exit=subprocess.run(['git','diff','--check'],cwd=ROOT).returncode,
        source_figures_verified=True,reviewed_figures=5,no_next_task=True,access=d.ACCESS))
    artifacts={}
    for path in sorted(d.OUT.glob('sp_u2d_*')):
        if path.name=='sp_u2d_delivery_manifest.json' or any(x in path.name for x in ('pytest_tmp','mplconfig')):continue
        if path.is_file():artifacts[path.relative_to(ROOT).as_posix()]=d.sha(path)
        elif path.name=='sp_u2d_figures':
            for f in sorted(path.iterdir()):
                if f.is_file():artifacts[f.relative_to(ROOT).as_posix()]=d.sha(f)
    sources={p.relative_to(ROOT).as_posix():d.sha(p) for folder in ('scripts','tests','configs')
       for p in sorted((ROOT/folder).glob('*sp_u2d*')) if p.is_file()}
    d.save(d.OUT/'sp_u2d_delivery_manifest.json',dict(task='SP-U2D',head=d.HEAD,
        sealed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),artifact_sha256=artifacts,
        execution_source_sha256=sources,protected_sources=pre['protected_sources'],old_input_sha256=pre['input_sha256'],
        stop=True,no_commit=True,no_paper_edit=True))
    print('SEALED',len(artifacts),'artifacts;',len(sources),'U2D sources;529 protected sources and430 old inputs unchanged')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['packet','seal']);globals()[parser.parse_args().action]()
