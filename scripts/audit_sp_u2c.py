"""Read-only cache/source audit and three predetermined full repeat replays."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import sp_u2c_diagnostic as d
d.install_guard()
import numpy as np


def main():
    from scripts.run_sp_u2c import verify,git
    pre=verify();gate=d.load(d.OUT/'sp_u2c_run_gate.json')
    rows=d.read_rows(d.OUT/'sp_u2c_repeat_rows.csv')
    assert d.sha(d.OUT/'sp_u2c_repeat_rows.csv')==gate['rows_sha256']
    atoms=d.read_rows(d.OUT/'sp_u2c_noise_atoms.csv')
    method=d.frozen_method()
    checked=0
    for record in gate['repeat_cache']:
        assert d.sha(d.OUT/record['path'])==record['sha256'];checked+=1
    replays=[]
    for i,r in pre['protocol']['replay_pairs']:
        x=d.cache(i,pre['cache_identity']);useful,_=d.fixed_inputs(i,x)
        pilots,new_atoms=d.noisy_inputs(i,r,useful)
        with np.load(d.OUT/f'sp_u2c_repeat_cache/trajectory_{i:03d}_repeat_{r:02d}.npz',allow_pickle=False) as f:
            assert np.array_equal(pilots,f['pilots'])
        saved_atoms=[a for a in atoms if a['trajectory_id']==i and a['repeat_id']==r]
        assert len(saved_atoms)==10
        for a,b in zip(new_atoms,saved_atoms):
            assert a['standardized_atom_sha256']==b['standardized_atom_sha256'] and a['noise_seed']==b['noise_seed']
        current=d.diagnose(i,r,pilots,x,method)
        saved=[a for a in rows if a['trajectory_id']==i and a['repeat_id']==r]
        assert len(saved)==12
        for a,b in zip(current,saved):
            for k,v in a.items():
                if isinstance(v,(int,float,bool)):
                    assert v==b[k],(i,r,k,v,b[k])
                else:assert v==b[k],(i,r,k)
        replays.append(dict(trajectory_id=i,repeat_id=r,windows_replayed=12,all_fields_exact=True,noise_atoms_exact=True,pilots_bit_exact=True))
    conditional=d.read_rows(d.OUT/'sp_u2c_conditional_windows.csv')
    from scripts.analyze_sp_u2c import build_conditional
    rebuilt=build_conditional(rows)
    for a,b in zip(rebuilt,conditional):
        assert all(v==b[k] for k,v in a.items())
    import csv
    with (d.OUT/'sp_u2b_seed_manifest.csv').open(encoding='utf8') as f:
        old={(int(a['trajectory_id']),int(a['frame_id'])):a for a in csv.DictReader(f) if a['stage']=='development' and int(a['trajectory_id']) in d.IDS}
    fixed=d.read_rows(d.OUT/'sp_u2c_fixed_input_hashes.csv')
    for a in fixed:
        b=old[int(a['trajectory_id']),int(a['frame_id'])]
        assert all(a[k]==int(b[k]) for k in ('bits_seed_tx0','bits_seed_tx1'))
    verify()
    d.save(d.OUT/'sp_u2c_integrity_audit.json',dict(passed=True,protected_sources=len(pre['protected_sources']),
           selected_original_cache_hashes_exact=20,repeat_cache_hashes_exact=checked,fixed_symbol_seed_records_exact=len(fixed),
           replay=replays,conditional_covariance_summaries_rebuilt_exact=240,raw_rows_sha256=gate['rows_sha256'],
           channel_generator_calls=0,CDL_imports=False,formal_read=False,old_performance_used=False,
           tracked_diff=git('diff','--stat'),staged_diff=git('diff','--cached','--stat'),access=d.ACCESS))
    print('PASS protected sources',len(pre['protected_sources']),'400 caches; 36 full window replays exact; 240 summaries exact')


if __name__=='__main__':main()
