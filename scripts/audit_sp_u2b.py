"""Final bounded QA, cached-input replay, mean diagnostic and integrity audit."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import sp_u2b_diagnostic as d
d.install_guard()
import numpy as np
from scripts.analyze_sp_u2b import read_rows
from scripts.run_sp_u2b import verify
from src.cdl_benchmark import frames
from src.pilot_cv import causal_windows
from src.lowrank_modules import factored_affine_predict
from src.spatial_information import observation_geometry

SKILL=Path('C:/Users/yuhua/.codex/skills/nature-figure/scripts')


def figure_qa():
    records=[]
    commands=[('source',[str(SKILL/'validate_figure.py'),'scripts/plot_sp_u2b.py','--json'])]
    for letter in 'ABCDEF':
        path=d.OUT/'sp_u2b_figures'/f'fig_{letter}.pdf'
        commands.extend([(f'{letter}_text',[str(SKILL/'audit_pdf_text.py'),str(path),'--min-pt','5','--json']),
                         (f'{letter}_collision',[str(SKILL/'audit_figure_collisions.py'),str(path),
                                                '--json-out',str(path.with_suffix('.collision.json')),
                                                '--overlay-pdf',str(path.with_suffix('.collision.pdf'))])])
    for label,argv in commands:
        start=time.perf_counter()
        run=subprocess.run([sys.executable,*argv],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
        path=d.OUT/f'sp_u2b_qa_{label}.txt'
        assert not path.exists()
        path.write_text(run.stdout+'\nSTDERR:\n'+run.stderr,encoding='utf-8')
        records.append(dict(label=label,argv=argv,exit_code=run.returncode,elapsed_seconds=time.perf_counter()-start,
                            log=str(path.relative_to(ROOT)),log_sha256=d.sha(path)))
        print(label,'exit',run.returncode,run.stdout[:1000],flush=True)
    d.save(d.OUT/'sp_u2b_figure_qa_execution.json',dict(records=records,all_exit_zero=all(r['exit_code']==0 for r in records),
          source_warnings_review='PNG 300dpi is intended development preview; no submission TIFF requested. Width parser reads 183/25.4 as 183 inches; actual PDF width audited directly.'))
    if any(r['exit_code'] for r in records):raise RuntimeError('figure QA needs repair/review')


def replay_and_mean():
    pre=verify()
    arrays,cvs,factors,proof=d.frozen_method()
    rows=read_rows(d.OUT/'sp_u2b_development_windows.csv')
    g=observation_geometry()
    lookup={(int(r['trajectory_id']),int(r['snr_db']),int(r['frame_id'])):r for r in rows}
    meanp=np.zeros(64,complex);meand=np.zeros(3968,complex)
    replay_count=0
    cache_hashes=[]
    # Persisted-input replay: no CDL calls and no old data.
    for i in range(100):
        path=d.OUT/'sp_u2b_development_cache'/f'trajectory_{i:03d}.npz'
        with np.load(path,allow_pickle=False) as f:
            h=f['H'];pilots=f['pilots']
            truths=frames(h)
            for j in range(4,10):
                meanp+=pilots[0,j];meand+=truths[j].ravel()[g.data]
            for si,s in enumerate(d.SNRS,1):
                for j,window in causal_windows(pilots[si],5):
                    c,diag=cvs[s].fit(window,modules=('P','M'))
                    c0=cvs[s].fit(pilots[0,j-4:j+1],modules=('P','M'))[0]
                    assert np.array_equal(c,f[f'c_{s}_{j}']) and np.array_equal(c0,f[f'c0_{s}_{j}'])
                    G,z=d.explicit_design(cvs[s],window)
                    assert np.array_equal(G,f[f'G_{s}_{j}']) and np.array_equal(z,f[f'z_{s}_{j}'])
                    pred=factored_affine_predict(window[-1],arrays['source_mean_p'],arrays['source_mean_d'],
                                                arrays[f'Wsource_{s}'],c,factors['P',s],factors['M',s])
                    assert d.array_hash(pred)==lookup[i,s,j]['prediction_sha256']
                    replay_count+=1
        cache_hashes.append(dict(trajectory_id=i,sha256=d.sha(path)))
    meanp/=600;meand/=600
    mean_info=[]
    for s in d.SNRS:
        bias=arrays['source_mean_d']+arrays[f'Wsource_{s}']@(meanp-arrays['source_mean_p'])-meand
        mean_info.append(dict(snr_db=s,source_mean_p_norm=float(np.linalg.norm(arrays['source_mean_p'])),
                              source_mean_d_rms=float(np.linalg.norm(arrays['source_mean_d'])/np.sqrt(3968)),
                              development_mean_p_norm=float(np.linalg.norm(meanp)),
                              development_mean_d_rms=float(np.linalg.norm(meand)/np.sqrt(3968)),
                              sample_mean_source_bias_MSE=float(np.vdot(bias,bias).real/3968),
                              interpretation='posthoc sample mean over dependent windows of 100 trajectories; finite-sample variation and representation mismatch not isolated population mean mismatch',
                              used_for_method=False))
    long=[]
    for row in rows:
        for prefix in ('clean','noisy'):
            item=dict(trajectory_id=int(row['trajectory_id']),frame_id=int(row['frame_id']),snr_db=int(row['snr_db']),
                      clean_flag=(prefix=='clean'))
            item.update({key[len(prefix)+1:]:value for key,value in row.items() if key.startswith(prefix+'_') and key not in ('clean_flag','clean_channel_mean_power','clean_abs_rho')})
            long.append(item)
    d.csvsave(d.OUT/'sp_u2b_geometry_long.csv',long)
    verify()
    gate=d.load(d.OUT/'sp_u2b_development_gate.json')
    assert [r['cache_sha256'] for r in gate['power_and_channel_hashes']]==[r['sha256'] for r in cache_hashes]
    d.save(d.OUT/'sp_u2b_integrity_audit.json',dict(head=pre['head'],protected_files=len(pre['protected_sha256']),
              protected_files_changed=0,frozen_execution_sources_changed=0,tracked_diff_empty=True,staged_diff_empty=True,
              replayed_production_windows=replay_count,coordinates_bit_exact=True,design_bit_exact=True,prediction_hashes_exact=True,
              new_generator_calls=0,cache_hashes=cache_hashes,mean_diagnostic=mean_info,
              formal_outcomes_accessed=False,access=d.ACCESS))
    print('All 1200 cached production windows replayed exactly; protected source/paper intact; no CDL calls',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('figures','integrity'));a=p.parse_args()
    if a.mode=='figures':figure_qa()
    else:replay_and_mean()
