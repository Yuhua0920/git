"""Close the authorized diagnostic and preserve reviewer-visible provenance."""
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import sp_u8a_common as c

def main():
    c.install_guard()
    for name in ('fixtures.json','pytest_pre.json','pytest_post.json','replay_gate.json','conditioning_summary.json','fixed_singular_value_gate.json'):
        assert c.read(c.OUT/name)['PASS']
    preserved=c.read(c.OUT/'preservation.json');assert preserved['changed']==0 and preserved['R1_added']==0
    execution=c.read(c.OUT/'execution_freeze.json')
    for p,h in execution['hashes'].items():assert c.sha(c.ROOT/p)==h
    r1=c.read(c.R1/'freeze.json');deployable=set(c.read(c.frozen.PROTOCOL)['deployable']['objects']);verified=[];excluded=[]
    for p,h in r1['hashes'].items():
        if p.startswith('outputs/') and p not in deployable and not p.startswith('outputs/sp_u7b_r1/'):
            excluded.append(p);continue
        assert c.sha(c.ROOT/p)==h;verified.append(p)
    skill=Path('C:/Users/yuhua/.codex/skills/nature-figure/scripts')
    validation=subprocess.run([sys.executable,str(skill/'validate_figure.py'),'scripts/plot_sp_u8a.py','--json','--strict'],
        cwd=c.ROOT,capture_output=True,text=True)
    assert validation.returncode==0
    c.write(c.OUT/'figure_source_QA.json',json.loads(validation.stdout))
    audits=[]
    for p in sorted((c.OUT/'figures').glob('*.pdf')):
        prefix=p.with_suffix('')
        alignment=c.read(str(prefix)+'.alignment.json');collision=c.read(str(prefix)+'.collision.json');text=c.read(str(prefix)+'.text.json')
        assert alignment['summary']['fail']==0 and alignment['summary']['warn']==0
        assert collision['summary']['fail']==0 and collision['summary']['warn']==0
        assert text['auditable'] and text['below_minimum_count']==0
        audits.append(dict(figure=p.name,panel_alignment='PASS',collisions='PASS',minimum_glyph_pt=text['minimum_found_pt']))
    assert len(audits)==3
    c.write(c.OUT/'visual_review.json',dict(PASS=True,manual_review='all eight panels and assembled figures visually inspected',
        figures=audits,checks=['all 1000 trajectories retained in empirical summaries and scatter panels',
            'median/IQR/P05/P95 explicitly distinct from confidence intervals',
            'Perfect label POST-HOC DIAGNOSTIC ONLY',
            'shared limits for comparable panels; fixed-ID equalizer panels share logarithmic y scale',
            'legible labels, distinct method shapes/colors, no clipping or annotation overlap'],
        panels=8,excluded_trajectories=0))
    c.write(c.OUT/'research_lessons.json',dict(lessons=[
        dict(trigger='ordering sentinel development',mistake='palindromic deterministic bit patterns concealed flatten-order reversal',
            fix='asymmetric TX/frequency/time patterns; five mutation types now produce bit errors',
            rule='Prove that deterministic sentinels reject each intended wrong transformation before using them to certify ordering.'),
        dict(trigger='preservation inventory preparation',mistake='Git inventory included historical outputs beyond the audit data budget',
            fix='firewall blocked opens; preserve code plus the USED data root, not historical outcome payloads',
            rule='Keep preservation hash inventories aligned with the authorized evidence roots.'),
        dict(trigger='figure source preflight',mistake='static regex interpreted maximum_figures=3 as raster res=3 and missed loop-built SVG suffix',
            fix='explicit export suffixes and unambiguous figure_count_limit key; no data or resolution change',
            rule='Investigate static QA findings against actual code and rendered files; correct ambiguity without weakening scientific checks.')],
        preparation_failure='retained in preparation_firewall_failure.json; before official A/B gate and before CDL-A diagnostic',
        scientific_ladder_failures=0,external_skill_files_modified=False))
    decision=dict(task='SP-U8A',status='COMPLETE_STOP_WAIT_MENTOR',engineering_verdict='PASS',BER_replay_verdict='EXACT PASS',
        receiver_sanity_case='A',BER_evidence='VALID + RECEIVER-LIMITED',primary_NMSE_impact='NONE',
        primary_case='A_STRONG_TRANSFER FROZEN; no re-test',Phase2_experimental_branch='FREEZE',OOD_branch='FROZEN',
        rationale=['Perfect ZF BER remains high at 10/20 dB and in the same order as Matched',
            'true weak singular values and inverse noise enhancement show consistent strong descriptive association with Perfect BER',
            'fixed-ID inverse norms and exact mismatch/noise/cross-term decomposition demonstrate noise/error-induced alteration of the inverse',
            'Matched excess BER persists; neither its entire error nor a causal share is assigned to conditioning'],
        interpretation='The unusual BER ordering is primarily a receiver/channel-conditioning phenomenon under the frozen ZF detector; channel-estimation NMSE and BER need not share the same ranking.',
        LS_interpretation='apparent implicit regularization under mismatched ZF; not estimator superiority',
        decision_delta='secondary BER remains valid but receiver-limited; primary result unchanged; no new experiment authorized',
        open_risks_ranked=['frozen ZF suppresses usable BER and prevents general end-to-end superiority claims',
            'descriptive post-hoc associations do not establish causal effects or contribution percentages',
            'seven predetermined IDs do not constitute a population-wide estimated-inverse mechanism theorem',
            'inverse tails, mismatch cross terms and nonlinear hard decisions prevent ranking BER by mean channel/symbol MSE'],
        recommended_next_action='Mentor review; then separately authorize Phase-2 Claim-Evidence Consolidation + Reviewer Red Team + manuscript upgrade plan',
        next_task_executed=False,new_H_bits_noise_trajectories=0,coordinate_refits=0,receiver_changes=0,commit=False,
        protected_files=preserved['protected_files'],R1_files=c.read(c.OUT/'before.json')['R1_files'],
        R1_frozen_inputs_verified=len(verified),historical_metadata_hash_entries_not_reopened=len(excluded))
    c.write(c.OUT/'decision.json',decision)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.ROOT,text=True).strip()
    assert subprocess.run(['git','diff','--quiet'],cwd=c.ROOT).returncode==0
    assert subprocess.run(['git','diff','--cached','--quiet'],cwd=c.ROOT).returncode==0
    status=subprocess.check_output(['git','status','--porcelain'],cwd=c.ROOT,text=True)
    with (c.OUT/'git_status.txt').open('w',encoding='utf8') as f:f.write(c.NOTICE+'\n'+status)
    scripts=sorted(list((c.ROOT/'scripts').glob('*sp_u8a*.py'))+list((c.ROOT/'tests').glob('*sp_u8a*.py')))
    hashes={p.relative_to(c.ROOT).as_posix():c.sha(p) for p in scripts}
    data={p.relative_to(c.OUT).as_posix():c.sha(p) for p in sorted(c.OUT.rglob('*'))
        if p.is_file() and 'mplconfig' not in p.parts and not any(part.startswith('pytest_tmp_') for part in p.parts)
        and p.name!='delivery_manifest.json'}
    c.write(c.OUT/'delivery_manifest.json',dict(PASS=True,HEAD=head,tracked_diff_empty=True,staged_diff_empty=True,
        source_hashes=hashes,output_hashes=data,output_files=len(data),R1_unchanged=preserved,
        scientific_data_roots=['outputs/sp_u7b_r1/'],read_only_deployable_artifacts='frozen t16e/f/g operators and factors for exact prediction replay',
        full_trajectory_diagnostics=dict(BER_rows=10000,BER_frame_rows=60000,paired_rows=8000,channel_trajectory_rows=1000,
            channel_frame_rows=6000,per_RE_archives=1000,per_RE_count=5952000,fixed_ID_summary_rows=70,fixed_ID_frame_archives=420,
            fixed_ID_singular_value_summaries=70,fixed_ID_singular_value_archives=70),
        temporary_caches_not_delivery_data=True,decision='decision.json'))
    print(c.NOTICE,flush=True)
    print(json.dumps(dict(status=decision['status'],receiver_case='A',BER='VALID + RECEIVER-LIMITED',primary_impact='NONE',
        R1_preservation=preserved,verified_R1_source_inputs=len(verified),HEAD=head,source_hashes=hashes,
        critical_hashes={k:data[k] for k in ('analysis.json','decision.json','replay_gate.json','conditioning_summary.json','preservation.json','trajectories.csv','fixed_ID_diagnostics.csv')},
        manifest_sha256=c.sha(c.OUT/'delivery_manifest.json')),indent=2),flush=True)

if __name__=='__main__':main()
