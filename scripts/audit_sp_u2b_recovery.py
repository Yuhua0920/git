"""Preserve the post-run audit schema failure; select explicit geometry fields."""
from pathlib import Path
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import sp_u2b_diagnostic as d
d.install_guard()
from scripts.analyze_sp_u2b import read_rows

GEOMETRY_KEYS={'rank_real','sigma_min','sigma_max','kappa_G','gram_condition','lambda_min','lambda_max',
               'column_norm_P','column_norm_M','rho','determinant','excitation_fro',
               'condition_square_relative_error','near_singular'}


def long_rows(rows):
    out=[]
    for row in rows:
        for prefix in ('clean','noisy'):
            item=dict(trajectory_id=int(row['trajectory_id']),frame_id=int(row['frame_id']),snr_db=int(row['snr_db']),
                      clean_flag=(prefix=='clean'))
            item.update({key[len(prefix)+1:]:value for key,value in row.items()
                         if key.startswith(prefix+'_') and key[len(prefix)+1:] in GEOMETRY_KEYS})
            assert set(item)==GEOMETRY_KEYS|{'trajectory_id','frame_id','snr_db','clean_flag'}
            out.append(item)
    return out


def main():
    source=ROOT/'scripts/audit_sp_u2b.py'
    archive=d.OUT/'sp_u2b_integrity_failure01'
    archive.mkdir(exist_ok=False)
    (archive/'audit_sp_u2b.py.txt').write_bytes(source.read_bytes())
    # Reproduce the observed serializer rejection using a real row, without replaying or generating CDL.
    row=read_rows(d.OUT/'sp_u2b_development_windows.csv')[0]
    defective=[]
    for prefix in ('clean','noisy'):
        item=dict(trajectory_id=0,frame_id=4,snr_db=10,clean_flag=prefix=='clean')
        item.update({key[len(prefix)+1:]:value for key,value in row.items()
                     if key.startswith(prefix+'_') and key not in ('clean_flag','clean_channel_mean_power','clean_abs_rho')})
        defective.append(item)
    try:d.csvsave(archive/'rejected_geometry.csv',defective)
    except ValueError:
        (archive/'reproduced_failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
    else:raise AssertionError('schema failure did not reproduce')
    original_save=d.csvsave
    def schema_adapter(path,rows):
        if Path(path).name=='sp_u2b_geometry_long.csv':
            # Only export schema changes; original numerical replay and mean calculation remain exact.
            rows=long_rows(read_rows(d.OUT/'sp_u2b_development_windows.csv'))
        return original_save(path,rows)
    d.csvsave=schema_adapter
    from scripts.audit_sp_u2b import replay_and_mean
    replay_and_mean()
    d.save(d.OUT/'sp_u2b_integrity_recovery.json',dict(error='clean_residual_norm was captured as geometry by a broad clean_ prefix',
            correction='explicit geometry field allowlist for long CSV; no diagnostic values changed',
            original_audit_source_sha256=d.sha(source),recovery_source_sha256=d.sha(Path(__file__)),
            original_source_preserved=True,CDL_rerun=False,production_unchanged=True))
    print('Geometry export schema recovery PASS; full cached-input replay exact',flush=True)


if __name__=='__main__':main()
