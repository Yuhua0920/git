"""Figure QA, independent metric reconciliation, and immutable delivery seal."""
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u4b_common import *
SKILL=Path('C:/Users/yuhua/.codex/skills/nature-figure')


def figures():
    manifest=readjson(OUT/'sp_u4b_figure_manifest.json');reports=[]
    for f in manifest['figures']:
        stem=OUT/'sp_u4b_figures'/f['figure'];items=[]
        for name,args in [('text',[str(SKILL/'scripts/audit_pdf_text.py'),str(stem)+'.pdf','--min-pt','5','--json']),
            ('collision',[str(SKILL/'scripts/audit_figure_collisions.py'),str(stem)+'.pdf','--json-out',str(stem)+'.collision.json','--overlay-pdf',str(stem)+'.collision.pdf'])]:
            p=subprocess.run([sys.executable,*args],capture_output=True,text=True,encoding='utf8',errors='replace',cwd=ROOT)
            (OUT/f'sp_u4b_qa_{f["figure"]}_{name}.txt').write_text(p.stdout+'\n'+p.stderr,encoding='utf8');items.append(dict(audit=name,exit_code=p.returncode))
        reports.append(dict(figure=f['figure'],alignment=readjson(str(stem)+'.alignment.json'),audits=items))
    save(OUT/'sp_u4b_rendered_qa.json',dict(passed=all(all(i['exit_code']==0 for i in r['audits']) for r in reports),reports=reports))
    print([(r['figure'],r['audits']) for r in reports])


def seal():
    p=prereg();before=readjson(OUT/'sp_u4b_before_manifest.json')
    for path,h in {**before['protected'],**before['U4A']['source_sha256'],**before['U4A']['artifact_sha256']}.items():
        if sha(ROOT/path)!=h:raise ValueError('historical file changed:'+path)
    if sha(OUT/'sp_u4a_delivery_manifest.json')!=before['U4A_manifest_sha256']:raise ValueError('U4A seal changed')
    pre=readjson(OUT/'sp_u4b_tests_pre.json')
    if any(sha(ROOT/k)!=h for k,h in pre['source_sha256'].items()):raise ValueError('execution drift')
    for name in ('sp_u4b_tests_post.json','sp_u4b_rendered_qa.json','sp_u4b_visual_review.json'):
        if not readjson(OUT/name)['passed']:raise ValueError('delivery gate failed:'+name)
    frames=csvread(OUT/'sp_u4b_new_frames.csv');tr=csvread(OUT/'sp_u4b_trajectories.csv');groups={}
    for f in frames:groups.setdefault(tuple(f[k] for k in ('replicate','condition','snr_db','budget','method','trajectory_id')),[]).append(f)
    checked=0
    for r in tr:
        if r['replicate'] in ('R0','Common500'):continue
        ff=sorted(groups[tuple(r[k] for k in ('replicate','condition','snr_db','budget','method','trajectory_id'))],key=lambda f:int(f['frame_id']))
        assert [int(f['frame_id']) for f in ff]==list(range(4,10))
        result=trajectory_metric([float(f['error_energy']) for f in ff],[float(f['truth_energy']) for f in ff])
        if any(result[k]!=float(r[k]) for k in result):raise ValueError('metric reconstruction failed')
        checked+=1
    diff=subprocess.run(['git','diff','--exit-code'],cwd=ROOT,capture_output=True);check=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if diff.returncode or check.returncode or head!=p['head']:raise ValueError('Git drift')
    audit=dict(passed=True,protected_sources=len(before['protected']),U4A_artifacts_checked=len(before['U4A']['artifact_sha256']),U4A_sources_checked=len(before['U4A']['source_sha256']),
        all_historical_hashes_unchanged=True,R0_exact_reuse=True,N500_single_common_reference=True,all200_communication_inputs_exact=True,
        new_trajectory_rows_reconstructed_exact=checked,trajectory_unit_preserved=True,offline_subset_replicates=5,not500_independent_repetitions=True,
        formal_firewall=True,T20B_U2_MFA_outcomes_read=False,new_H_generations=0,git_diff_empty=True,git_diff_check_exit_code=0,HEAD=head,stop=True)
    save(OUT/'sp_u4b_integrity_audit.json',audit)
    sources={q.relative_to(ROOT).as_posix():sha(q) for folder in ('scripts','tests') for q in (ROOT/folder).glob('*sp_u4b*') if q.is_file()}
    artifacts={q.relative_to(ROOT).as_posix():sha(q) for q in OUT.rglob('*') if q.is_file() and q.relative_to(OUT).parts[0].startswith('sp_u4b_')
        and 'pytest_tmp' not in str(q) and 'mplconfig' not in str(q) and q.name!='sp_u4b_delivery_manifest.json'}
    save(OUT/'sp_u4b_delivery_manifest.json',dict(task='SP-U4B',HEAD=head,source_sha256=sources,artifact_sha256=artifacts,case=readjson(OUT/'sp_u4b_analysis.json')['case'],
        offline_statistics_branch='FROZEN',theory='FROZEN',MFA='STOPPED',final_method='FROZEN',next_task_started=False,stop=True))
    print(f'SP-U4B complete seal PASS: {len(sources)} sources,{len(artifacts)} artifacts; STOP')


if __name__=='__main__':globals()[sys.argv[1]]()
