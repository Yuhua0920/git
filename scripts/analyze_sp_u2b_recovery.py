"""Serialization-only recovery; original analysis source and CSVs are immutable."""
import csv
import io
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import sp_u2b_diagnostic as d
d.install_guard()
import numpy as np
from scripts import analyze_sp_u2b as original


def normalize(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {key:normalize(v) for key,v in value.items()}
    if isinstance(value,(list,tuple)):return [normalize(v) for v in value]
    return value


def verify_or_create_csv(path,rows):
    path=Path(path)
    if not path.exists():
        return ORIGINAL_CSVSAVE(path,rows)
    fields=list(rows[0])
    assert all(set(r)==set(fields) for r in rows)
    buffer=io.StringIO(newline='')
    writer=csv.DictWriter(buffer,fieldnames=fields)
    writer.writeheader();writer.writerows(rows)
    if buffer.getvalue().encode('utf-8')!=path.read_bytes():
        raise RuntimeError('serialization recovery changed existing CSV bytes: '+str(path))
    VERIFIED.append(dict(path=str(path.relative_to(ROOT)),sha256=d.sha(path)))


ORIGINAL_CSVSAVE=d.csvsave
ORIGINAL_SAVE=d.save
VERIFIED=[]


def main():
    pre=d.load(d.OUT/'sp_u2b_preregistration.json')
    source=ROOT/'scripts/analyze_sp_u2b.py'
    assert d.sha(source)==pre['execution_source_sha256']['scripts/analyze_sp_u2b.py']
    archive=d.OUT/'sp_u2b_analysis_failure01'
    archive.mkdir(exist_ok=False)
    (archive/'analyze_sp_u2b.py.txt').write_bytes(source.read_bytes())
    files=('sp_u2b_trajectory_means.csv','sp_u2b_statistics.csv','sp_u2b_associations.csv',
           'sp_u2b_quartiles.csv','sp_u2b_table1_geometry.csv','sp_u2b_table2_gaps.csv')
    before={name:d.sha(d.OUT/name) for name in files}
    d.csvsave=verify_or_create_csv
    d.save=lambda path,value:ORIGINAL_SAVE(path,normalize(value))
    original.main()
    assert before=={name:d.sha(d.OUT/name) for name in files}
    ORIGINAL_SAVE(d.OUT/'sp_u2b_analysis_recovery.json',dict(error='NumPy int64 descriptive outlier count at JSON boundary',
                  original_source_preserved=True,original_source_sha256=d.sha(source),
                  correction='normalize NumPy scalar types only at JSON boundary; no numerical changes',
                  recovery_source_sha256=d.sha(Path(__file__)),existing_CSVs_verified_exact=VERIFIED,
                  raw_development_csv_sha256=d.sha(d.OUT/'sp_u2b_development_windows.csv'),
                  channel_or_method_rerun=False,protocol_changed=False))
    print('Serialization-only recovery PASS; six CSV files byte-identical; no CDL rerun',flush=True)


if __name__=='__main__':main()
