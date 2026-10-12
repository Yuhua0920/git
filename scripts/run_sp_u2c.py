"""Preregister, test and run exactly 20 x 20 fixed-channel noise repeats."""
import argparse
import datetime
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import sp_u2c_diagnostic as d
d.install_guard()
import numpy as np


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()


def protect():
    assert git('rev-parse','HEAD') == d.HEAD
    assert not git('diff','--name-only') and not git('diff','--cached','--name-only')


def verify():
    protect()
    p = d.load(d.OUT/'sp_u2c_preregistration.json')
    amendment=d.OUT/'sp_u2c_execution_amendment01.json'
    execution=d.load(amendment)['execution_sources'] if amendment.exists() else p['execution_sources']
    for name,h in {**p['protected_sources'],**execution}.items():
        assert d.sha(ROOT/name) == h, name
    assert d.sha(ROOT/'configs/sp_u2c_protocol.json') == p['protocol_sha256']
    assert d.sha(d.OUT/'sp_u2c_seed_manifest.csv') == p['seed_manifest_sha256']
    for name,h in p['input_sha256'].items():
        assert d.sha(d.OUT/name) == h, name
    p['actual_execution_sources']=execution
    return p


def freeze():
    protect()
    paths = set(git('ls-files').splitlines()) | set(git('ls-files','--others','--exclude-standard').splitlines())
    protected = {p:d.sha(ROOT/p) for p in sorted(paths) if p.startswith(('src/','scripts/','tests/','configs/','docs/','paper/'))
                 and 'sp_u2c' not in p and '__pycache__' not in p and (ROOT/p).is_file()}
    sources = {p:d.sha(ROOT/p) for p in sorted(paths) if 'sp_u2c' in p and p.startswith(('scripts/','tests/','configs/')) and (ROOT/p).is_file()}
    config = d.load(ROOT/'configs/sp_u2c_protocol.json')
    assert config['sample_rate_hz'] == config['nfft']*config['subcarrier_spacing_hz']
    assert config['selected_ids'] == list(d.IDS) and config['repeats'] == d.REPEATS
    records = d.load(d.OUT/'sp_u2b_development_gate.json')['power_and_channel_hashes']
    manifest = [{k:records[i][k] for k in ('stage','trajectory_id','H_sha256','cache_sha256')} for i in d.IDS]
    seeds = [dict(namespace='SP_U2C_RECEIVER_AWGN_v1', role='unit_receiver_AWGN', trajectory_id=i,repeat_id=r,
                  frame_id=j,paired_snrs='10,20',seed=d.noise_seed(i,r,j)) for i in d.IDS for r in range(20) for j in range(10)]
    new = {v['seed'] for v in seeds}
    with (d.OUT/'sp_u2b_seed_manifest.csv').open(encoding='utf8') as f:
        import csv
        old = {int(row[k]) for row in csv.DictReader(f) for k in ('sionna_seed','torch_seed','numpy_seed','bits_seed_tx0','bits_seed_tx1','noise_seed')}
    assert len(new)==4000 and not new.intersection(old)
    method = d.frozen_method()
    for i in d.IDS:
        d.cache(i, manifest)
    d.csvsave(d.OUT/'sp_u2c_seed_manifest.csv',seeds)
    inputs = {p:d.sha(d.OUT/p) for p in sorted(d.ALLOWED)}
    d.save(d.OUT/'sp_u2c_preregistration.json',dict(task='SP-U2C',head=d.HEAD,
           frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=config,
           protocol_sha256=d.sha(ROOT/'configs/sp_u2c_protocol.json'),protected_sources=protected,execution_sources=sources,
           input_sha256=inputs,cache_identity=manifest,frozen_method=method[3],
           seed_manifest_sha256=d.sha(d.OUT/'sp_u2c_seed_manifest.csv'),unique_noise_seeds=4000,
           old_seed_values=len(old),seed_overlap=0,noise_generated=0,channel_generator_calls=0,
           old_formal_read=False,old_performance_used=False,access=d.ACCESS))
    print('FROZEN; protected sources',len(protected),'new seeds 4000; overlap 0; no channel generation',flush=True)


def tests():
    p = verify()
    cmd=[sys.executable,'-m','pytest','tests/test_sp_u2c_diagnostic.py','-q','-p','no:cacheprovider',
         '--basetemp=outputs/sp_u2c_pytest_tmp_pre02']
    result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
    log=d.OUT/'sp_u2c_pytest_pre02.txt'
    assert not log.exists()
    log.write_text(result.stdout+result.stderr,encoding='utf8')
    d.save(d.OUT/'sp_u2c_tests_gate02.json',dict(command=cmd,exit_code=result.returncode,
           execution_sources=p['actual_execution_sources'],log_sha256=d.sha(log)))
    print(result.stdout+result.stderr,flush=True)
    if result.returncode: raise RuntimeError('pre-run pytest failed; no scientific repeats')


def run():
    assert Path(sys.prefix).resolve() == (ROOT/'.venv-cdl').resolve()
    p=verify()
    t=d.load(d.OUT/'sp_u2c_tests_gate02.json')
    assert t['exit_code']==0 and t['execution_sources']==p['actual_execution_sources']
    start=time.perf_counter()
    method=d.frozen_method()
    assert d.normal(method[3]) == p['frozen_method']
    rows,noise_rows,fixed_rows,cache_records=[],[],[],[]
    from scripts.sp_u2b_diagnostic import explicit_design
    for i in d.IDS:
        x=d.cache(i,p['cache_identity'])
        useful,fixed=d.fixed_inputs(i,x)
        fixed_rows.extend(fixed)
        for s in d.SNRS:
            for j in d.FRAMES:
                c0,_=method[1][s].fit(x['clean_pilots'][j-4:j+1],modules=('P','M'))
                assert np.array_equal(c0,x[f'c0_{s}_{j}'])
                G,z=explicit_design(method[1][s],x['clean_pilots'][j-4:j+1])
                assert np.array_equal(G,x[f'G0_{s}_{j}']) and np.array_equal(z,x[f'z0_{s}_{j}'])
        for r in range(20):
            pilots,nr=d.noisy_inputs(i,r,useful)
            current=d.diagnose(i,r,pilots,x,method)
            rows.extend(current);noise_rows.extend(nr)
            path=d.OUT/f'sp_u2c_repeat_cache/trajectory_{i:03d}_repeat_{r:02d}.npz'
            path.parent.mkdir(exist_ok=True)
            assert not path.exists()
            np.savez_compressed(path,pilots=pilots)
            cache_records.append(dict(trajectory_id=i,repeat_id=r,path=path.relative_to(d.OUT).as_posix(),sha256=d.sha(path)))
        print('Completed fixed trajectory',i,'20 paired-noise repeats; records',len(rows),flush=True)
    assert len(rows)==4800 and len(noise_rows)==4000 and len(fixed_rows)==200
    for i in d.IDS:
        for s in d.SNRS:
            for j in d.FRAMES:
                group=[a for a in rows if a['trajectory_id']==i and a['snr_db']==s and a['frame_id']==j]
                assert len(group)==20 and len({a['clean_coordinate_sha256'] for a in group})==1
    d.csvsave(d.OUT/'sp_u2c_repeat_rows.csv',rows)
    d.csvsave(d.OUT/'sp_u2c_noise_atoms.csv',noise_rows)
    d.csvsave(d.OUT/'sp_u2c_fixed_input_hashes.csv',fixed_rows)
    verify()
    d.save(d.OUT/'sp_u2c_run_gate.json',dict(task='SP-U2C',passed=True,trajectories=20,score_windows_per_trajectory=6,
           repeats=20,snrs=[10,20],rows=4800,standard_noise_atoms=4000,channel_generator_calls=0,
           oracle_failures=0,rank_failures=0,NaN_count=0,undefined_cosines=sum(a['alignment_undefined'] for a in rows),
           borderlines=sum(a['numeric_borderline'] for a in rows),cosine_roundoff_flags=sum(a['cosine_roundoff_outside_unit'] for a in rows),
           max_identity_error=max(a['identity_error_max'] for a in rows),max_oracle_LS_error=max(a['oracle_LS_error'] for a in rows),
           rows_sha256=d.sha(d.OUT/'sp_u2c_repeat_rows.csv'),repeat_cache=cache_records,
           wall_seconds=time.perf_counter()-start,python=sys.version,numpy=np.__version__,access=d.ACCESS,
           fixed_clean_coordinate_hash_invariance=True,production_changed=False,formal_read=False))
    print('PASS 4800 rows; seconds',time.perf_counter()-start,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','tests','run'])
    globals()[p.parse_args().action]()
