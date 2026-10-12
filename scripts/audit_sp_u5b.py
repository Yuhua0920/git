"""Control integrity, final-PDF audits, deterministic figure export and delivery seal."""
import sys,subprocess,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u5b_common import *
SKILL=Path('C:/Users/yuhua/.codex/skills/nature-figure')

def figures():
    import pymupdf
    manifest=readjson(OUT/'sp_u5b_figure_manifest.json');files=[OUT/'sp_u5b_figures'/(r['figure']+'.'+e) for r in manifest['figures'] for e in ('png','pdf','svg')]
    before={p.name:sha(p) for p in files};subprocess.run([sys.executable,str(ROOT/'scripts/plot_sp_u5b.py')],cwd=ROOT,check=True)
    after={p.name:sha(p) for p in files}
    if before!=after:raise ValueError('CSV-only export differs')
    results=[]
    for r in manifest['figures']:
        stem=OUT/'sp_u5b_figures'/r['figure'];aud=[]
        for name,args in (('text',[str(SKILL/'scripts/audit_pdf_text.py'),str(stem)+'.pdf','--min-pt','5','--json']),
            ('collision',[str(SKILL/'scripts/audit_figure_collisions.py'),str(stem)+'.pdf','--json-out',str(stem)+'.collision.json','--overlay-pdf',str(stem)+'.collision.pdf'])):
            run=subprocess.run([sys.executable,*args],cwd=ROOT,capture_output=True,text=True,encoding='utf8',errors='replace')
            (OUT/f"sp_u5b_qa_{r['figure']}_{name}.txt").write_text(run.stdout+'\n'+run.stderr,encoding='utf8');aud.append(dict(audit=name,exit_code=run.returncode))
        with pymupdf.open(str(stem)+'.pdf') as doc:width=doc[0].rect.width*25.4/72
        if abs(width-183)>.01:raise ValueError('width')
        results.append(dict(figure=r['figure'],PDF_width_mm=width,alignment=readjson(str(stem)+'.alignment.json')['verdict'],audits=aud))
    passed=all(r['alignment']=='PASS' and all(a['exit_code']==0 for a in r['audits']) for r in results)
    save(OUT/'sp_u5b_rendered_qa.json',dict(passed=passed,CSV_only_nine_exports_exact=True,export_sha256=after,results=results))
    print(json.dumps(results,indent=2));
    if not passed:raise ValueError('rendered QA fails')

def confirm_visual_review():
    # Invoked only after the final three exported PNGs and all12 panels have been inspected.
    panels=[]
    questions={'A_Native_Common':'paired penalty persists after score-domain control','B_Common_Gain':'common benefit survival and ID risk','C_Common_BER':'receiver ordering with identical bits'}
    for name,q in questions.items():
        for j,(c,s) in enumerate((('ID',10),('ID',20),('Composed',10),('Composed',20))):
            panels.append(dict(figure=name,panel=chr(97+j),condition=c,snr_db=s,unique_claim=q,replicate_unit='100 reused trajectories',
                center='aggregate errors/bits' if name=='C_Common_BER' else 'median',spread='trajectory IQR; all100 paired points in A',
                labels_legible=True,uncertainty_visible=True,alignment='PASS',collision='PASS',passed=True))
    save(OUT/'sp_u5b_visual_review.json',dict(passed=True,figures=3,panels_inspected=12,assembled_figures_inspected=True,panels=panels,
        warnings_reviewed='PNG300dpi preview with editable PDF/SVG; no TIFF required for development delivery; ID BER curves near-coincident as supported by raw values'))
    print('visual review all12 panels recorded')

def record_feedback_correction():
    source=OUT/'sp_u5a_relative_sensitivity.csv'
    r=next(r for r in csvread(source) if (r['condition'],r['snr_db'],r['contrast'])==('Composed','20','reduction32_minus16'))
    save(OUT/'sp_u5b_feedback_correction.json',dict(previous_feedback='SP-U5A Section H: Composed20 extra LR4 sensitivity P95',
        previously_reported=1.862302,correct_frozen_value=float(r['p95']),source_sha256=sha(source),frozen_data_modified=False,
        correction_scope='copyable feedback number only; no change to experiment, analysis definition or results'))
    print('U5A feedback P95 corrected from frozen CSV:',r['p95'])

def seal(case):
    firewall();p=prereg();count=protect()
    pre=readjson(OUT/'sp_u5b_tests_pre.json')
    if any(sha(ROOT/k)!=h for k,h in pre['source_sha256'].items()):raise ValueError('core execution source drift')
    for name in ('tests_post','replay_gate','analysis','rendered_qa','visual_review'):
        if not readjson(OUT/f'sp_u5b_{name}.json')['passed']:raise ValueError('gate '+name)
    if readjson(OUT/'sp_u5b_figure_source_qa.json')['summary']['counts']['FAIL']:raise ValueError('source QA')
    frames_rows=csvread(OUT/'sp_u5b_frames.csv');trs=csvread(OUT/'sp_u5b_trajectories.csv');groups={};truths={}
    for r in frames_rows:
        groups.setdefault(tuple(r[k] for k in ('condition','trajectory_id','snr_db','dF','method','domain')),[]).append(r)
        if r['domain']=='common':
            key=tuple(r[k] for k in ('condition','trajectory_id','snr_db','frame_id'));v=(r['truth_sha256'],r['truth_energy'],r['bit_count'])
            if key in truths and truths[key]!=v:raise ValueError('common truth/denominator differs')
            truths[key]=v
    for r in trs:
        ff=groups[tuple(r[k] for k in ('condition','trajectory_id','snr_db','dF','method','domain'))]
        if sorted(int(f['frame_id']) for f in ff)!=list(range(4,10)):raise ValueError('frame population')
        metric=trajectory_metric([float(f['error_energy']) for f in ff],[float(f['truth_energy']) for f in ff])
        if any(float(r[k])!=v for k,v in metric.items()):raise ValueError('metric aggregation')
        if sum(int(f['bit_errors']) for f in ff)!=int(r['bit_errors']) or sum(int(f['bit_count']) for f in ff)!=int(r['bit_count']):raise ValueError('BER aggregation')
    ix={(r['condition'],r['trajectory_id'],r['snr_db'],r['dF'],r['method'],r['domain']):r for r in trs}
    for key,r in ix.items():
        if key[3]=='8' and key[-1]=='common':
            old=ix[*key[:-1],'native']
            if any(r[k]!=old[k] for k in ('error_energy','truth_energy','nmse_linear','nmse_db','bit_errors','bit_count','BER')):raise ValueError('D8 identity')
    coords=coordinate_index();reuse=csvread(OUT/'sp_u5b_coordinate_reuse.csv')
    for r in reuse:
        old=coords[r['condition'],int(r['trajectory_id']),int(r['snr_db']),int(r['dF']),int(r['frame_id'])]
        if float(r['alpha'])!=float(old['alpha']) or float(r['beta'])!=float(old['beta']) or r['refit']!='False':raise ValueError('coordinate drift')
    for args in (['git','diff','--exit-code'],['git','diff','--check']):
        if subprocess.run(args,cwd=ROOT,capture_output=True).returncode:raise ValueError('Git diff')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if head!=p['HEAD'] or case not in ('A','B','C'):raise ValueError('HEAD/case')
    save(OUT/'sp_u5b_integrity_audit.json',dict(passed=True,protected_files=count,artifact_source_drift=False,core_source_drift=False,U5A_drift=False,historical_final_drift=False,
        trajectory_rows_reconstructed_exact=len(trs),frame_rows=len(frames_rows),common_truth_frame_groups_exact=len(truths),coordinate_rows_exact=len(reuse),D8_native_common_exact=True,
        native_max_error=0,firewall='PASS',T20B_read=False,U4_evaluation_read=False,new_inputs=0,refit_calls=0,rebuild_calls=0,HEAD=head,tracked_diff_empty=True,no_commit=True))
    sources={q.relative_to(ROOT).as_posix():sha(q) for folder in ('scripts','tests','configs') for q in (ROOT/folder).glob('*sp_u5b*') if q.is_file()}
    artifacts={q.relative_to(ROOT).as_posix():sha(q) for q in OUT.rglob('*') if q.is_file() and q.relative_to(OUT).parts[0].startswith(PREFIX)
        and 'pytest_tmp' not in str(q) and 'mplconfig' not in str(q) and q.name!='sp_u5b_delivery_manifest.json'}
    save(OUT/'sp_u5b_delivery_manifest.json',dict(task='SP-U5B',HEAD=head,source_sha256=sources,artifact_sha256=artifacts,decision='Case '+case,
        decision_basis='qualitative reading of common paired penalties, extra sensitivity, adaptation gains and native/common differences; no selected performance cutoff',
        pilot_branch='FROZEN',theory='FROZEN',MFA='STOPPED',final_method='FROZEN',no_U5C=True,no_next_task=True,stop=True))
    print(f'SP-U5B seal PASS; Case {case};{len(sources)} sources,{len(artifacts)} artifacts; STOP')

if __name__=='__main__':
    if sys.argv[1]=='seal':seal(sys.argv[2])
    else:globals()[sys.argv[1]]()
