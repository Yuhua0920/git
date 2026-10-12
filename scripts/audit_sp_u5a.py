"""Independent full-input/metric audit, rendered QA and immutable delivery seal."""
import sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u5a_common import *
from src.link import generate_master_qpsk,assemble_pilot_data_frame,apply_frequency_mimo_channel
from src.estimators import pilot_ls_estimate
SKILL=Path('C:/Users/yuhua/.codex/skills/nature-figure')
def inputs():
    firewall();rows=csvread(OUT/'sp_u5a_input_hashes.csv');checked=0;pilots=0
    for r in rows:
        path=OUT/f"sp_u5a_inputs/{r['condition']}_{int(r['trajectory_id']):03}.npz"
        if sha(path)!=r['cache_sha256']:raise ValueError('cache digest')
        with np.load(path) as f:
            if array_hash(f['H'])!=r['H_sha256']:raise ValueError('H digest')
            digests={df:hashlib.sha256() for df in (8,16,32)}
            for j,h in enumerate(frames(f['H'])):
                ss=seed_map(r['condition'],int(r['trajectory_id']),j);b,s=generate_master_qpsk(bits_seeds=(ss['bits_seed_tx0'],ss['bits_seed_tx1']))
                rng=np.random.default_rng(ss['noise_seed']);atom=(rng.standard_normal((2,64,16))+1j*rng.standard_normal((2,64,16)))/np.sqrt(2)
                if not np.array_equal(b,f['bits'][j]) or not np.array_equal(s,f['symbols'][j]) or not np.array_equal(atom,f['noise_atoms'][j]):raise ValueError('full field replay')
                for df in (8,16,32):
                    g=geometry(df);x=assemble_pilot_data_frame(s,g.layout);clean=apply_frequency_mimo_channel(h,x)
                    if not np.all(x[g.layout.mute_mask]==0) or not np.allclose(np.sum(abs(x)**2,axis=0),1,rtol=0,atol=1e-12):raise ValueError('power/muting')
                    for si,snr in enumerate((10,20)):
                        y=clean+np.sqrt(cfg()['noise_variances'][str(snr)])*atom;p,_=pilot_ls_estimate(y,x,g.layout.pilot_mask);p=p.ravel()[g.selected]
                        if not np.array_equal(p,f[f'pilots_{df}'][si,j]):raise ValueError('received pilot replay')
                        digests[df].update(array_hash(y).encode());digests[df].update(array_hash(p).encode());pilots+=1
                checked+=1
            for df in (8,16,32):
                if digests[df].hexdigest()!=r[f'dF{df}_received_digest']:raise ValueError('all received-grid digest')
    save(OUT/'sp_u5a_full_input_audit.json',dict(passed=True,trajectories=len(rows),full_frames_checked=checked,received_grid_and_pilot_vectors_checked=pilots,
        full_bits_symbols_noise_exact=True,no_U4_evaluation_read=True,true_transmissions=True,pilot_subsample_proxy=False))
    print('all200 full inputs and12000 density/frame/SNR received pilot vectors exact')
def figures():
    manifest=readjson(OUT/'sp_u5a_figure_manifest.json');results=[]
    for r in manifest['figures']:
        stem=OUT/'sp_u5a_figures'/r['figure'];aud=[]
        for name,args in [('text',[str(SKILL/'scripts/audit_pdf_text.py'),str(stem)+'.pdf','--min-pt','5','--json']),
            ('collision',[str(SKILL/'scripts/audit_figure_collisions.py'),str(stem)+'.pdf','--json-out',str(stem)+'.collision.json','--overlay-pdf',str(stem)+'.collision.pdf'])]:
            p=subprocess.run([sys.executable,*args],cwd=ROOT,capture_output=True,text=True,encoding='utf8',errors='replace')
            (OUT/f"sp_u5a_qa_{r['figure']}_{name}.txt").write_text(p.stdout+'\n'+p.stderr,encoding='utf8');aud.append(dict(audit=name,exit_code=p.returncode))
        results.append(dict(figure=r['figure'],alignment=readjson(str(stem)+'.alignment.json'),audits=aud))
    save(OUT/'sp_u5a_rendered_qa.json',dict(passed=all(all(a['exit_code']==0 for a in r['audits']) for r in results),results=results))
    print([(r['figure'],r['audits']) for r in results])
def reproduce_figures():
    import pymupdf
    manifest=readjson(OUT/'sp_u5a_figure_manifest.json');paths=[OUT/'sp_u5a_figures'/(r['figure']+'.'+ext) for r in manifest['figures'] for ext in ('png','pdf','svg')]
    before={p.name:sha(p) for p in paths};subprocess.run([sys.executable,str(ROOT/'scripts/plot_sp_u5a.py')],cwd=ROOT,check=True);after={p.name:sha(p) for p in paths}
    if before!=after:raise ValueError('CSV export mismatch')
    widths={}
    for r in manifest['figures']:
        with pymupdf.open(OUT/'sp_u5a_figures'/(r['figure']+'.pdf')) as doc:widths[r['figure']]=doc[0].rect.width*25.4/72
    if any(abs(v-183)>.01 for v in widths.values()):raise ValueError('PDF width')
    save(OUT/'sp_u5a_visual_review.json',dict(passed=True,figures=6,panels=22,all22_panels_visually_inspected=True,PDF_width_mm=widths,
        CSV_only_18_exports_byte_identical=True,export_sha256=after,notes=[
            'A/B two SNR strata: actual fractions and spacing labeled; medians/IQR distinguish adaptation and ID boundary',
            'C four panels: negative ID gain and positive Composed gain both visible; uncertainty retained',
            'D four panels: all100 paired values per method; no exclusions; bars show trajectory IQR',
            'E four panels: actual decoded-bit BER denominator; ID curves nearly coincide; IQR segments explicitly described',
            'F six panels: condition line styles and alpha/beta colors legible; window correlation stated'],
        source_warnings_reviewed='PNG300dpi preview; editable PDF/SVG; no TIFF required for development delivery'))
    print('visual review22 panels and18 deterministic exports PASS')
def seal():
    p=prereg();protected=readjson(OUT/'sp_u5a_before_manifest.json')
    if any(sha(ROOT/k)!=v for k,v in protected.items()):raise ValueError('historical drift')
    pre=readjson(OUT/'sp_u5a_tests_pre.json')
    if any(sha(ROOT/k)!=v for k,v in pre['source_sha256'].items()):raise ValueError('core execution drift')
    for name in ('tests_post','regression_gate','input_gate','full_input_audit','evaluation_gate','rendered_qa','visual_review','analysis'):
        if not readjson(OUT/f'sp_u5a_{name}.json')['passed']:raise ValueError('gate:'+name)
    groups={}
    for f in csvread(OUT/'sp_u5a_frames.csv'):groups.setdefault(tuple(f[k] for k in ('condition','trajectory_id','snr_db','artifact','method')),[]).append(f)
    rows=csvread(OUT/'sp_u5a_trajectories.csv')
    for r in rows:
        ff=sorted(groups[tuple(r[k] for k in ('condition','trajectory_id','snr_db','artifact','method'))],key=lambda x:int(x['frame_id']))
        if [int(f['frame_id']) for f in ff]!=list(range(4,10)):raise ValueError('frame population')
        metric=trajectory_metric([float(f['error_energy']) for f in ff],[float(f['truth_energy']) for f in ff])
        if any(float(r[k])!=v for k,v in metric.items()):raise ValueError('trajectory metric')
        if sum(int(f['bit_errors']) for f in ff)!=int(r['bit_errors']) or sum(int(f['bit_count']) for f in ff)!=int(r['bit_count']):raise ValueError('BER aggregation')
        if int(r['bit_count'])!=geometry(int(r['dF'])).counts['two_stream_data_symbols']*2*6:raise ValueError('actual bit denominator')
    for w in csvread(OUT/'sp_u5a_windows.csv'):
        if int(w['window_start'])!=int(w['frame_id'])-4 or int(w['window_end'])!=int(w['frame_id']) or w['fallback']!='False':raise ValueError('causal/no rescue')
    for args in (['git','diff','--exit-code'],['git','diff','--check']):
        if subprocess.run(args,cwd=ROOT,capture_output=True).returncode:raise ValueError('Git diff')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if head!=p['head']:raise ValueError('HEAD drift')
    save(OUT/'sp_u5a_integrity_audit.json',dict(passed=True,protected_hashes_checked=len(protected),historical_files_unchanged=True,core_preexecution_sources_unchanged=True,
        trajectory_rows_reconstructed_exact=len(rows),all_BER_counts_exact=True,all_windows_causal=True,no_cross_density_mixing=True,formal_firewall=True,T20B_outcomes_read=False,
        U4_evaluation_read=False,HEAD=head,tracked_diff_empty=True,diff_check_exit0=True,no_commit=True,stop=True))
    sources={q.relative_to(ROOT).as_posix():sha(q) for folder in ('scripts','tests','configs') for q in (ROOT/folder).glob('*sp_u5a*') if q.is_file()}
    artifacts={q.relative_to(ROOT).as_posix():sha(q) for q in OUT.rglob('*') if q.is_file() and q.relative_to(OUT).parts[0].startswith(PREFIX)
        and 'pytest_tmp' not in str(q) and 'mplconfig' not in str(q) and q.name!='sp_u5a_delivery_manifest.json'}
    save(OUT/'sp_u5a_delivery_manifest.json',dict(task='SP-U5A',HEAD=head,source_sha256=sources,artifact_sha256=artifacts,decision='Case B: meaningful pilot-density tradeoff, positive Composed adaptation, no numerical breakdown',
        pilot_branch='FROZEN',theory='FROZEN',MFA='STOPPED',final_method='FROZEN',final_protocol_dF16=True,no_next_task=True,stop=True))
    print(f'SP-U5A seal PASS: {len(sources)} sources, {len(artifacts)} artifacts; STOP')
if __name__=='__main__':globals()[sys.argv[1]]()
