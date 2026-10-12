"""Remove generated import caches only, then verify every immutable source byte."""
import json,shutil
from scripts.sp_u12b5a_common import ROOT,REF,OUT,read,write,sha,firewall
def main():
    firewall();removed=[]
    for cache in REF.rglob('__pycache__'):
        target=cache.resolve()
        assert target.is_relative_to(REF.resolve()) and target.name=='__pycache__'
        assert all(v.suffix=='.pyc' for v in target.rglob('*') if v.is_file())
        removed.append(str(target.relative_to(ROOT)))
        shutil.rmtree(target)
    historical=read(OUT/'historical_reference.json')
    metadata=read(ROOT/'outputs/sp_u12b5a_mfa_reference_metadata.json')
    checks=[]
    for version,repo,path,files in [(2023,k,REF/'paper_period'/k,v['files']) for k,v in historical.items()]+[(2026,v['repository'].rsplit('/',1)[-1],REF/v['repository'].rsplit('/',1)[-1],v['files']) for v in metadata['repositories']]:
        actual={p.relative_to(path).as_posix():sha(p) for p in sorted(path.rglob('*')) if p.is_file()}
        checks.append(dict(version=version,repository=repo,byte_exact=actual==files,files=len(actual),source_modified=False))
    write(OUT/'reference_integrity.json',dict(PASS=all(v['byte_exact'] for v in checks),checks=checks,generated_pyc_removed=removed,original_source_bytes_modified=0))
    assert all(v['byte_exact'] for v in checks)
    print(json.dumps(dict(PASS=True,checks=checks,source_modified=0)))
if __name__=='__main__':main()
