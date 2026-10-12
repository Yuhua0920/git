"""Package existing MR03 evidence; independently inspect ZIP/extracted statistics."""
from pathlib import Path
import hashlib, json, csv, zipfile, subprocess, os, uuid, sys
ROOT=Path(r'D:\Users\yuhua\Desktop\Project')
OUT=ROOT/'outputs/revision_mr03_ppca_capacity_controls'
SCRATCH=Path(r'C:\Users\yuhua\.codex\visualizations\2026\10\04\01a10645-29c1-7fa3-9f56-11e4543f2416\mr03_work')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,obj):
    with p.open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)
def main():
    zip_path=OUT/'results_review.zip'
    if zip_path.exists():raise FileExistsError(zip_path)
    env=dict(os.environ,GIT_OPTIONAL_LOCKS='0',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
    commands={}
    for action in (['status','--porcelain','--untracked-files=no'],['diff','--stat'],['diff','--check']):
        r=subprocess.run(['git','-C',str(ROOT),*action],text=True,capture_output=True,env=env)
        commands[' '.join(action)]={'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
        if r.returncode:raise ValueError('GIT_READONLY_CHECK_FAILED:'+str(commands))
    expected={'paper/appendix/evidence_slots.tex','paper/bibliography/references.bib','paper/main.tex','paper/metadata.tex','paper/sections/01_introduction.tex','paper/sections/02_related_work.tex','paper/sections/03_system_model.tex','paper/sections/04_proposed_method.tex','paper/sections/05_experimental_setup.tex','paper/sections/06_results.tex','paper/sections/07_discussion.tex','paper/sections/08_conclusion.tex','paper/tables/02_nmse.tex','paper/tables/03_ber.tex','paper/tables/06_summary.tex','paper/tables/notation.tex'}
    actual={line[3:] for line in commands['status --porcelain --untracked-files=no']['stdout'].splitlines()}
    # Initial session explicitly established these 16 prior paper changes. Never reset them.
    if actual!=expected:raise ValueError('TRACKED_STATUS_DIFFERS_FROM_INITIAL_SESSION:'+str(actual.symmetric_difference(expected)))
    write(OUT/'reports/git_status_after.json',dict(readonly_optional_locks=True,new_tracked_changes=0,preexisting_modified_files=sorted(actual),checks=commands,no_git_commit=True))
    exclude={'data/conditional_operators.npz'}
    files=[p for p in OUT.rglob('*') if p.is_file() and p.relative_to(OUT).as_posix() not in exclude]
    if any('__pycache__' in p.parts for p in files):raise ValueError('UNEXPECTED_PYC')
    manifest=OUT/'package_manifest.csv'
    with manifest.open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['relative_path','bytes','sha256']);w.writeheader()
        for p in sorted(files):w.writerow(dict(relative_path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    files.append(manifest)
    with zipfile.ZipFile(zip_path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):z.write(p,'results_review/'+p.relative_to(OUT).as_posix())
    zsha=sha(zip_path)
    with (OUT/'results_review.zip.sha256').open('x',encoding='ascii') as f:f.write(zsha+'  results_review.zip\n')
    staging=SCRATCH/('extracted_review_'+uuid.uuid4().hex[:12])
    staging.mkdir()
    with zipfile.ZipFile(zip_path) as z:
        if z.testzip() is not None:raise ValueError('ZIP_CRC_FAILURE')
        with z.open('results_review/package_manifest.csv') as f:
            records=list(csv.DictReader(__import__('io').TextIOWrapper(f,encoding='utf8')))
        for record in records:
            name='results_review/'+record['relative_path']
            info=z.getinfo(name)
            if info.file_size!=int(record['bytes']):raise ValueError('ZIP_SIZE:'+name)
            with z.open(name) as f:
                if hashlib.file_digest(f,'sha256').hexdigest()!=record['sha256']:raise ValueError('ZIP_HASH:'+name)
        expected_members={'results_review/'+r['relative_path'] for r in records}|{'results_review/package_manifest.csv'}
        if set(z.namelist())!=expected_members:raise ValueError('ZIP_MEMBERS')
        for info in z.infolist():
            target=(staging/info.filename).resolve()
            if not target.is_relative_to(staging.resolve()) or info.is_dir():raise ValueError('ZIP_PATH_BOUNDARY')
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as src,target.open('xb') as dst:
                import shutil;shutil.copyfileobj(src,dst)
    extracted=staging/'results_review'
    cmd=[str(ROOT/'.venv-cdl/Scripts/python.exe'),'-B',str(extracted/'src_or_scripts/independent_review_capacity.py'),'--out',str(extracted),'--mr01-bootstrap',str(extracted/'review_reference/exploratory_bootstrap_indices.npz'),'--report',str(staging/'independent_review.json')]
    r=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=120)
    with (OUT/'reports/package_extracted_review.log').open('x',encoding='utf8') as f:f.write(r.stdout+'\nSTDERR:\n'+r.stderr+'\nEXIT_CODE: '+str(r.returncode)+'\n')
    if r.returncode:raise ValueError('EXTRACTED_REVIEW_FAILED:'+r.stderr+r.stdout)
    review=json.loads((staging/'independent_review.json').read_text(encoding='utf8'))
    if not review['PASS']:raise ValueError('EXTRACTED_REVIEW_NOT_PASS')
    write(OUT/'reports/package_verification.json',dict(PASS=True,ZIP_SHA256=zsha,zip_bytes=zip_path.stat().st_size,zip_members=len(files),manifest_authenticated_files=len(records),all_CRC_PASS=True,all_manifest_size_SHA256_PASS=True,staging=str(staging),extracted_statistical_review=review,source_data_modified=False,new_inference=False,new_random_draws=0,excluded_cached_operators='data/conditional_operators.npz',verification_record_outside_ZIP_to_avoid_circular_hash=True))
    info=dict(PASS=True,path=str(zip_path),bytes=zip_path.stat().st_size,MiB=zip_path.stat().st_size/1024**2,files=len(files),SHA256=zsha,sidecar=str(OUT/'results_review.zip.sha256'),manifest=str(manifest),report=str(OUT/'reports/package_verification.json'))
    write(OUT/'package_info.json',info)
    print(json.dumps(info,indent=2))
if __name__=='__main__':main()
