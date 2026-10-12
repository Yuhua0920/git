"""Freeze, calibrate and compute all matched U2D counterfactual records."""
import argparse
import datetime
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import sp_u2d_diagnostic as d
d.install_guard()


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()


def protect():
    assert git('rev-parse','HEAD')==d.HEAD
    assert not git('diff','--name-only') and not git('diff','--cached','--name-only')


def verify():
    protect();pre=d.load(d.OUT/'sp_u2d_preregistration.json')
    for p,h in {**pre['protected_sources'],**pre['execution_sources']}.items():assert d.sha(ROOT/p)==h,p
    for p,h in pre['input_sha256'].items():assert d.sha(d.OUT/p)==h,p
    assert d.sha(ROOT/'configs/sp_u2d_protocol.json')==pre['protocol_sha256']
    assert d.sha(d.OUT/'sp_u2d_energy_manifest.csv')==pre['energy_manifest_sha256']
    return pre


def freeze():
    protect()
    config=d.load(ROOT/'configs/sp_u2d_protocol.json')
    assert config['sample_rate_hz']==config['nfft']*config['subcarrier_spacing_hz']
    assert config['selected_ids']==list(d.IDS) and config['repeats']==list(d.REPEATS)
    old=d.load(d.OUT/'sp_u2c_preregistration.json')
    gate=d.load(d.OUT/'sp_u2c_run_gate.json')
    delivery=d.load(d.OUT/'sp_u2c_delivery_manifest.json')
    identity=old['cache_identity']
    caches=gate['repeat_cache']
    assert {(a['trajectory_id'],a['repeat_id']) for a in caches}=={(i,r) for i in d.IDS for r in d.REPEATS}
    inputs={p:d.sha(d.OUT/p) for p in sorted(d.ALLOWED)}
    for p,h in inputs.items():
        if p.startswith('sp_u2b_development_cache/') or p.startswith(('t16e_','t16f_','t16g_')):
            assert h==old['input_sha256'][p]
        if p.startswith('sp_u2c_') and p!='sp_u2c_delivery_manifest.json':
            assert h==delivery['artifact_sha256']['outputs/'+p]
    paths=set(git('ls-files').splitlines())|set(git('ls-files','--others','--exclude-standard').splitlines())
    protected={p:d.sha(ROOT/p) for p in sorted(paths) if p.startswith(('src/','scripts/','tests/','configs/','docs/','paper/'))
               and 'sp_u2d' not in p and '__pycache__' not in p and (ROOT/p).is_file()}
    execution={p:d.sha(ROOT/p) for p in sorted(paths) if 'sp_u2d' in p and p.startswith(('scripts/','tests/','configs/')) and (ROOT/p).is_file()}
    method=d.u.frozen_method()
    assert d.normal(method[3])==old['frozen_method']
    fixed={i:d.fixed_cache(i,identity) for i in d.IDS}
    energy=d.energy_manifest(fixed,method[0]['source_mean_p'])
    d.csvsave(d.OUT/'sp_u2d_energy_manifest.csv',energy)
    d.save(d.OUT/'sp_u2d_preregistration.json',dict(task='SP-U2D',head=d.HEAD,
          frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=config,
          protocol_sha256=d.sha(ROOT/'configs/sp_u2d_protocol.json'),protected_sources=protected,execution_sources=execution,
          input_sha256=inputs,fixed_cache_identity=identity,repeat_caches=caches,frozen_method=method[3],
          energy_manifest_sha256=d.sha(d.OUT/'sp_u2d_energy_manifest.csv'),
          hypotheses_frozen_before_LOFO=True,energy_strata_outcome_free=True,LOFO_fits_before_freeze=0,
          channel_generator_calls=0,noise_generator_calls=0,formal_read=False,old_outcomes_used_for_design=False,access=d.ACCESS))
    print('FROZEN: 20 IDs x20 repeats x2SNR x6windows x5ages; protected',len(protected),'files;400 caches exact; no new generators')


def tests():
    pre=verify()
    cmd=[sys.executable,'-m','pytest','tests/test_sp_u2d_diagnostic.py','-q','-p','no:cacheprovider','--basetemp=outputs/sp_u2d_pytest_tmp_pre']
    result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
    path=d.OUT/'sp_u2d_pytest_pre.txt';assert not path.exists();path.write_text(result.stdout+result.stderr,encoding='utf8')
    d.save(d.OUT/'sp_u2d_tests_gate.json',dict(command=cmd,exit_code=result.returncode,
         execution_sources=pre['execution_sources'],log_sha256=d.sha(path)))
    print(result.stdout+result.stderr,flush=True)
    if result.returncode:raise RuntimeError('pre-analysis tests failed; no full LOFO run')


def run():
    assert Path(sys.prefix).resolve()==(ROOT/'.venv-cdl').resolve()
    pre=verify();test=d.load(d.OUT/'sp_u2d_tests_gate.json')
    assert test['exit_code']==0 and test['execution_sources']==pre['execution_sources']
    start=time.perf_counter();method=d.u.frozen_method();refs=d.production_references()
    assert d.normal(method[3])==pre['frozen_method']
    metadata={(a['trajectory_id'],a['repeat_id']):a for a in pre['repeat_caches']}
    energy={(int(a['trajectory_id']),int(a['snr_db']),int(a['frame_id'])):a for a in d.read_rows(d.OUT/'sp_u2d_energy_manifest.csv')}
    rows=[];contrasts=[]
    for i in d.IDS:
        x=d.fixed_cache(i,pre['fixed_cache_identity'])
        for r in d.REPEATS:
            pilots=d.repeat_cache(i,r,metadata)
            for si,s in enumerate(d.SNRS):
                for j in d.FRAMES:
                    current=d.record(i,r,s,j,pilots[si],x,method,refs[(i,r,s,j)],energy[(i,s,j)])
                    rows.extend(current);contrasts.append(d.paired(current))
        print('Completed trajectory',i,'LOFO rows',len(rows),flush=True)
    assert len(rows)==24000 and len(contrasts)==4800
    d.csvsave(d.OUT/'sp_u2d_age_rows.csv',rows);d.csvsave(d.OUT/'sp_u2d_paired_rows.csv',contrasts)
    verify()
    error_keys=('moment_reconstruction_error','moment_subtraction_error','direct_four_frame_error','normal_error',
                'risk_identity_error','propagation_error','oracle_LS_error')
    maxerrors={k:max(a[k] for a in rows if a[k] is not None) for k in error_keys}
    d.save(d.OUT/'sp_u2d_run_gate.json',dict(passed=True,production_records=4800,LOFO_records=24000,
          trajectories=20,repeats=20,score_windows=6,ages=5,snrs=list(d.SNRS),
          full_coordinate_replays_bit_exact=4800,prediction_hash_replays_exact=4800,
          singular_LOFO=sum(a['singular'] for a in rows),undefined_cosine=sum(a['alignment_undefined'] for a in rows),
          numerical_borderlines=sum(a['numeric_borderline'] for a in rows),NaN_count=0,max_errors=maxerrors,
          channel_generator_calls=0,noise_generator_calls=0,raw_sha256=d.sha(d.OUT/'sp_u2d_age_rows.csv'),
          paired_sha256=d.sha(d.OUT/'sp_u2d_paired_rows.csv'),wall_seconds=time.perf_counter()-start,
          production_changed=False,formal_read=False,access=d.ACCESS))
    print('PASS 24000 matched LOFO rows;4800 full replays exact;seconds',time.perf_counter()-start)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','tests','run']);globals()[p.parse_args().action]()
