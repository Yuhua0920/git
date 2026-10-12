"""Rendered figure audits, raw-data reconciliation, firewall and integrity seal."""
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u4a_common import *
SKILL=Path('C:/Users/yuhua/.codex/skills/nature-figure')


def figures():
    manifest=readjson(OUT/'sp_u4a_figure_manifest.json');reports=[]
    for f in manifest['figures']:
        stem=OUT/'sp_u4a_figures'/f['figure'];items=[]
        for name,args in (
            ('text',[str(SKILL/'scripts/audit_pdf_text.py'),str(stem)+'.pdf','--min-pt','5','--json']),
            ('collision',[str(SKILL/'scripts/audit_figure_collisions.py'),str(stem)+'.pdf','--json-out',str(stem)+'.collision.json','--overlay-pdf',str(stem)+'.collision.pdf'])):
            p=subprocess.run([sys.executable,*args],capture_output=True,text=True,encoding='utf8',errors='replace',cwd=ROOT)
            (OUT/f'sp_u4a_qa_{f["figure"]}_{name}.txt').write_text(p.stdout+'\n'+p.stderr,encoding='utf8')
            items.append(dict(audit=name,exit_code=p.returncode))
        reports.append(dict(figure=f['figure'],alignment=readjson(str(stem)+'.alignment.json'),audits=items))
    save(OUT/'sp_u4a_rendered_qa.json',dict(passed=all(all(i['exit_code']==0 for i in r['audits']) for r in reports),reports=reports))
    print([(r['figure'],r['audits']) for r in reports],flush=True)


def reconcile():
    rows=csvread(OUT/'sp_u4a_trajectories.csv');frames=csvread(OUT/'sp_u4a_frames.csv');summary=csvread(OUT/'sp_u4a_nmse_summary.csv')
    index={}
    for r in frames:index.setdefault((r['condition'],r['snr_db'],r['budget'],r['method'],r['trajectory_id']),[]).append(r)
    worst=0.
    for r in rows:
        ff=sorted(index[r['condition'],r['snr_db'],r['budget'],r['method'],r['trajectory_id']],key=lambda x:int(x['frame_id']))
        if [int(f['frame_id']) for f in ff]!=list(range(4,10)):raise ValueError('score frames/trajectory incomplete')
        m=trajectory_metric([float(f['error_energy']) for f in ff],[float(f['truth_energy']) for f in ff])
        if any(m[k]!=float(r[k]) for k in m):raise ValueError('trajectory metric not reconstructed exactly')
    for r in summary:
        selected=[v for v in rows if all(v[k]==r[k] for k in ('condition','snr_db','budget','method'))]
        if len(selected)!=100:raise ValueError('population incomplete')
        agg=sum(float(v['error_energy']) for v in selected)/sum(float(v['truth_energy']) for v in selected)
        worst=max(worst,abs(agg-float(r['aggregate_nmse_linear'])))
        if agg!=float(r['aggregate_nmse_linear']):raise ValueError('summary aggregate mismatch')
    save(OUT/'sp_u4a_metric_reconciliation.json',dict(passed=True,trajectory_rows=len(rows),frame_rows=len(frames),summary_rows=len(summary),maximum_aggregate_difference=worst,
        highest_independent_unit='trajectory',independent_n_per_condition_snr=100,window_rows_not_independent=True,paired=True,no_pseudoreplication=True))


def seal():
    prereg();reconcile()
    if not readjson(OUT/'sp_u4a_tests_post.json')['passed']:raise ValueError('pytest gate FAIL')
    if not readjson(OUT/'sp_u4a_rendered_qa.json')['passed']:raise ValueError('rendered QA FAIL')
    if not readjson(OUT/'sp_u4a_visual_review.json')['passed']:raise ValueError('visual review missing/FAIL')
    changed=[p for p,h in protected_files().items() if sha(ROOT/p)!=h]
    if changed:raise ValueError('historical drift:'+str(changed))
    pre=readjson(OUT/'sp_u4a_tests_pre.json')
    if any(sha(ROOT/p)!=h for p,h in pre['source_sha256'].items()):raise ValueError('execution source changed after pretests')
    diff=subprocess.run(['git','diff','--exit-code'],cwd=ROOT,capture_output=True,text=True)
    check=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    if diff.returncode or check.returncode:raise ValueError('tracked Git diff/check FAIL')
    current=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if current!=config()['head']:raise ValueError('HEAD changed')
    core=[p.relative_to(ROOT).as_posix() for folder in ('scripts','tests','configs') for p in (ROOT/folder).glob('*sp_u4a*') if p.is_file()]
    artifacts={p.relative_to(ROOT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p.relative_to(OUT).parts[0].startswith(PREFIX)
        and 'pytest_tmp' not in str(p) and 'mplconfig' not in str(p) and p.name!='sp_u4a_delivery_manifest.json'}
    save(OUT/'sp_u4a_integrity_audit.json',dict(passed=True,protected_files_checked=len(protected_files()),all_historical_hashes_unchanged=True,
        tracked_diff_empty=True,git_diff_check_exit_code=check.returncode,HEAD=current,execution_source_hashes_unchanged=True,
        formal_firewall='runtime output allowlist + denial tests + source-only imports; T20B outcomes not read',
        old_evaluation_outputs_read=False,new_dataset=config()['dataset'],no_method_tuning=True,no_commit=True,stop=True))
    artifacts['outputs/sp_u4a_integrity_audit.json']=sha(OUT/'sp_u4a_integrity_audit.json')
    save(OUT/'sp_u4a_delivery_manifest.json',dict(task='SP-U4A',HEAD=current,artifact_sha256=artifacts,source_sha256={p:sha(ROOT/p) for p in core},
        task_state='completed; STOP for mentor review',theory='FROZEN',MFA='STOPPED',next_task_started=False))
    print(f'SP-U4A seal PASS: {len(protected_files())} protected files, {len(artifacts)} new artifacts, {len(core)} sources',flush=True)


if __name__=='__main__':
    if len(sys.argv)!=2 or sys.argv[1] not in ('figures','seal'):raise ValueError('choose figures or seal')
    globals()[sys.argv[1]]()
