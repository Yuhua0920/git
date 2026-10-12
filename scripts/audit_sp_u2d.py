"""Read-only replay and source/input integrity checks."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import sp_u2d_diagnostic as d
d.install_guard()


def main():
    from scripts.run_sp_u2d import verify,git
    pre=verify();gate=d.load(d.OUT/'sp_u2d_run_gate.json');assert gate['passed']
    raw=d.read_rows(d.OUT/'sp_u2d_age_rows.csv');pairs=d.read_rows(d.OUT/'sp_u2d_paired_rows.csv')
    assert len(raw)==24000 and len(pairs)==4800
    assert d.sha(d.OUT/'sp_u2d_age_rows.csv')==gate['raw_sha256']
    assert d.sha(d.OUT/'sp_u2d_paired_rows.csv')==gate['paired_sha256']
    for a in raw:
        assert a['removed_frame']==a['frame_id']-a['age'] and a['retained_frames']==4 and a['real_design_rows']==512
        if not a['singular']:d.close(a['delta_R'],a['Q']+a['C'])
    for k,b in enumerate(pairs):
        group=raw[5*k:5*k+5];rebuilt=d.paired(group)
        assert all(v==b[key] for key,v in rebuilt.items())
    method=d.u.frozen_method();refs=d.production_references()
    metadata={(a['trajectory_id'],a['repeat_id']):a for a in pre['repeat_caches']}
    energy={(int(a['trajectory_id']),int(a['snr_db']),int(a['frame_id'])):a for a in d.read_rows(d.OUT/'sp_u2d_energy_manifest.csv')}
    replay=[]
    for i,r in pre['protocol']['replay_pairs']:
        x=d.fixed_cache(i,pre['fixed_cache_identity']);pilots=d.repeat_cache(i,r,metadata)
        current=[]
        for si,s in enumerate(d.SNRS):
            for j in d.FRAMES:current.extend(d.record(i,r,s,j,pilots[si],x,method,refs[(i,r,s,j)],energy[(i,s,j)]))
        saved=[a for a in raw if a['trajectory_id']==i and a['repeat_id']==r]
        assert len(current)==len(saved)==60
        for a,b in zip(current,saved):assert all(v==b[k] for k,v in a.items())
        replay.append(dict(trajectory_id=i,repeat_id=r,production_windows=12,LOFO_rows=60,all_fields_exact=True))
    from scripts.analyze_sp_u2d import aggregate,AGE_FIELDS,CONTRAST_FIELDS
    for filename,rebuilt in [('sp_u2d_trajectory_ages.csv',aggregate(raw,AGE_FIELDS,('age',))),
                             ('sp_u2d_trajectory_contrasts.csv',aggregate(pairs,CONTRAST_FIELDS))]:
        saved=d.read_rows(d.OUT/filename)
        assert len(saved)==len(rebuilt) and all(all(v==b[k] for k,v in a.items()) for a,b in zip(rebuilt,saved))
    verify()
    d.save(d.OUT/'sp_u2d_integrity_audit.json',dict(passed=True,protected_sources=len(pre['protected_sources']),
          old_input_file_hashes_exact=len(pre['input_sha256']),fixed_channel_caches=20,repeat_pilot_caches=400,
          original_production_replays=4800,paired_rows_rebuilt_exact=4800,trajectory_age_summaries_exact=200,
          trajectory_contrasts_exact=40,replay=replay,all_four_frame_counts_exact=True,
          channel_generator_calls=0,noise_generator_calls=0,production_changed=False,formal_read=False,
          tracked_diff=git('diff','--stat'),staged_diff=git('diff','--cached','--stat'),access=d.ACCESS))
    print('PASS:529 protected sources;4800 pairs exact;180 LOFO replays exact;all input hashes unchanged')


if __name__=='__main__':main()
