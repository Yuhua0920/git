"""Completed confirmation evidence ZIP, actual extraction and package review."""
from pathlib import Path
import sys,csv,json,hashlib,zipfile,uuid,subprocess,os
ROOT=Path(r'D:\Users\yuhua\Desktop\Project');OUT=ROOT/'outputs/revision_mr08a_independent_confirmation'
SCRATCH=Path(r'C:\Users\yuhua\.codex\visualizations\2026\10\04\01a10645-29c1-7fa3-9f56-11e4543f2416\mr03_work')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    if not read(OUT/'reports/completion.json')['PASS']:raise ValueError('COMPLETION_REQUIRED')
    certification=OUT/'reports/independent_numerical_review.json';ch=sha(certification)
    if not read(certification)['PASS']:raise ValueError('FULL_DRAW_REVIEW_REQUIRED')
    zip_path=OUT/'results_review.zip';manifest=OUT/'results_review_file_manifest.csv'
    files=[];omitted=[]
    for p in sorted(OUT.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or '.pytest_cache' in p.parts:continue
        rel=p.relative_to(OUT).as_posix()
        if p.name in ('results_review.zip','results_review.zip.sha256','results_review_file_manifest.csv'):continue
        if rel=='data/estimators_7snr.npz' or (rel.startswith('data/bootstrap/formal/') and p.name in ('indices.npy','bootstrap_linear.npy')):
            omitted.append(dict(path=rel,bytes=p.stat().st_size,SHA256=sha(p),reason='retained intact in project; lightreview uses sufficientenergies andprimarydistributions; certifiedfullactualindices review referenced'))
        else:files.append(p)
    omitted_path=OUT/'reports/package_omitted_large_assets.json'
    write(omitted_path,dict(omitted=omitted,all_omitted_files_remain_in_project=True,full_actual_draw_certification_SHA256=ch,package_only_does_not_recheck_omitted_arrays=True))
    files.append(omitted_path)
    entries=[dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,SHA256=sha(p)) for p in sorted(files)]
    with manifest.open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=('path','bytes','SHA256'));w.writeheader();w.writerows(entries)
    with zipfile.ZipFile(zip_path,'x',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for r in entries:z.write(OUT/r['path'],r['path'])
        z.write(manifest,manifest.name)
    zh=sha(zip_path)
    with zip_path.with_suffix('.zip.sha256').open('x',encoding='ascii') as f:f.write(zh+'  '+zip_path.name+'\n')
    extracted=SCRATCH/('mr08a_review_'+uuid.uuid4().hex[:12]);extracted.mkdir()
    with zipfile.ZipFile(zip_path) as z:
        if z.testzip() or len(z.namelist())!=len(set(z.namelist())):raise ValueError('ZIP_CRC_DUPLICATE')
        for name in z.namelist():
            if Path(name).is_absolute() or not (extracted/name).resolve().is_relative_to(extracted.resolve()):raise ValueError('ZIP_PATH_ESCAPE')
        z.extractall(extracted)
    for r in entries:
        p=extracted/r['path']
        if p.stat().st_size!=r['bytes'] or sha(p)!=r['SHA256']:raise ValueError('EXTRACTED_HASH:'+r['path'])
    if sha(extracted/manifest.name)!=sha(manifest):raise ValueError('MANIFEST_HASH')
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['OPENBLAS_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
    report=extracted/'reports/unpacked_review.json'
    proc=subprocess.run([sys.executable,'-B',str(extracted/'src/review_package_mr08a.py'),'--out',str(extracted),'--full-certification-sha256',ch,'--report',str(report)],cwd=extracted,env=env,capture_output=True,text=True,encoding='utf8')
    with (OUT/'reports/package_review.log').open('x',encoding='utf8') as f:f.write(proc.stdout+'\n'+proc.stderr)
    if proc.returncode:raise RuntimeError('UNPACKED_REVIEW_FAILED:'+proc.stderr[-2500:])
    package=read(report)
    if not package['PASS']:raise ValueError('UNPACKED_PACKAGE_NOT_PASS')
    sys.path.insert(0,str(OUT/'src'));from run_mr08a import verify
    preservation=verify()
    answer=dict(PASS=True,ZIP=str(zip_path),SHA256=zh,ZIP_bytes=zip_path.stat().st_size,ZIP_MiB=zip_path.stat().st_size/2**20,archive_members=len(entries)+1,payload_files=len(entries),uncompressed_payload_bytes=sum(r['bytes'] for r in entries)+manifest.stat().st_size,CRC_PASS=True,all_extracted_SHA256_PASS=True,manifest_SHA256=sha(manifest),actual_extracted_directory=str(extracted),full_project_draw_review_SHA256=ch,unpacked_package_only_review=package,omitted_large_files=omitted,omitted_files_not_deleted=True,preservation=preservation,verification_report_outside_original_archive=True)
    write(OUT/'reports/package_verification.json',answer)
    print(json.dumps({k:answer[k] for k in ('PASS','ZIP','SHA256','ZIP_bytes','ZIP_MiB','archive_members','preservation')},ensure_ascii=False))
if __name__=='__main__':main()
