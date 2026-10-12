"""Small completed-results ZIP and actual unpacked review; no model inference."""
from pathlib import Path
import csv,json,hashlib,zipfile,uuid,subprocess,sys,os,math
ROOT=Path(r'D:\Users\yuhua\Desktop\Project')
OUT=ROOT/'outputs/revision_mr03c_prepared_gmm'
SCRATCH=Path(r'C:\Users\yuhua\.codex\visualizations\2026\10\04\01a10645-29c1-7fa3-9f56-11e4543f2416\mr03_work')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rows(p):
    with Path(p).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
with (OUT/'reports/root_visual_review.json').open('x',encoding='utf8') as f:
    json.dump(dict(PASS=True,final_pngs_actually_opened_by_primary_reviewer=5,caption_scope='alreadydiscloseddevelopment; unadjustedexploratory95%',readability_and_no_clipping=True,negative_cells_visible=True,posterior_normalized_entropy_visual_only=True,figures=['gmm_absolute_V_new','gmm_absolute_D_eval','gmm_paired_V_new','gmm_paired_D_eval','gmm_posterior_diagnostics']),f,indent=2)
zip_path=OUT/'results_review.zip'
manifest=OUT/'results_review_file_manifest.csv'
excluded={zip_path.name,manifest.name,'results_review.zip.sha256'}
files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name not in excluded and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts)
entries=[dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,SHA256=sha(p)) for p in files]
with manifest.open('x',encoding='utf8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['path','bytes','SHA256']);w.writeheader();w.writerows(entries)
with zipfile.ZipFile(zip_path,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files:z.write(p,p.relative_to(OUT).as_posix())
    z.write(manifest,manifest.name)
zh=sha(zip_path)
with zip_path.with_suffix('.zip.sha256').open('x',encoding='ascii') as f:f.write(zh+'  '+zip_path.name+'\n')
unpacked=SCRATCH/('mr03c_review_'+uuid.uuid4().hex[:12]);unpacked.mkdir()
with zipfile.ZipFile(zip_path) as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    for name in z.namelist():
        target=(unpacked/name).resolve()
        assert target.is_relative_to(unpacked.resolve()) and not Path(name).is_absolute()
    z.extractall(unpacked)
for r in entries:
    p=unpacked/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['SHA256'],r['path']
assert sha(unpacked/manifest.name)==sha(manifest)
# New descriptive trajectory differences checked separately against validated
# packaged trajectory energies. These are not additional bootstrap contrasts.
et=rows(unpacked/'tables/trajectory_energies.csv');index={(r['dataset'],r['condition'],r['trajectory_id'],r['snr_db'],r['method']):r for r in et}
diff=rows(unpacked/'tables/trajectory_paired_differences.csv');assert len(diff)==15000
identities=set();max_error=0
for r in diff:
    k=tuple(r[n] for n in ('dataset','condition','trajectory_id','snr_db'));full=k+(r['second'],)
    assert full not in identities;identities.add(full)
    a=index[k+('GMM_PREPARED_3',)];b=index[k+(r['second'],)];ea=float(a['error_energy']);eb=float(b['error_energy']);q=float(a['truth_energy']);assert q==float(b['truth_energy'])
    vals={'GMM_error_energy':ea,'comparator_error_energy':eb,'truth_energy':q,'GMM_NMSE_linear':ea/q,'comparator_NMSE_linear':eb/q,'GMM_minus_comparator_db':10*math.log10(ea/eb)}
    for n,v in vals.items():
        err=abs(float(r[n])-v);assert err<=1e-12*(1+abs(v));max_error=max(max_error,err)
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['OPENBLAS_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
review_path=unpacked/'reports/unpacked_review.json'
r=subprocess.run([sys.executable,'-B',str(unpacked/'src/independent_review_gmm_standalone.py'),'--report',str(review_path)],cwd=unpacked,env=env,capture_output=True,text=True,encoding='utf8')
with (OUT/'package_review.log').open('x',encoding='utf8') as f:f.write(r.stdout+'\n'+r.stderr)
if r.returncode:raise RuntimeError('UNPACKED_REVIEW_FAILED:'+r.stderr[-2000:])
review=json.loads(review_path.read_text(encoding='utf8'));assert review['PASS'] and review['external_files_read']==0
# Confirm protected originals once more after artifact assembly.
sys.path.insert(0,str(OUT/'src'));from run_gmm3 import verify
preservation=verify()
write(OUT/'reports/package_verification.json',dict(PASS=True,ZIP=str(zip_path),SHA256=zh,ZIP_bytes=zip_path.stat().st_size,ZIP_MiB=zip_path.stat().st_size/2**20,payload_files=len(entries),archive_members=len(entries)+1,uncompressed_bytes=sum(r['bytes'] for r in entries)+manifest.stat().st_size,CRC_PASS=True,all_extracted_file_hashes_PASS=True,manifest_SHA256=sha(manifest),actual_unpacked_directory=str(unpacked),unpacked_independent_review=review,additional_trajectory_differences_checked=len(diff),additional_trajectory_difference_max_error=max_error,preservation=preservation,post_packaging_verification_reports_outside_archive=True,no_predictions_or_new_draws=True))
print(json.dumps(dict(PASS=True,ZIP=str(zip_path),SHA256=zh,ZIP_bytes=zip_path.stat().st_size,ZIP_MiB=zip_path.stat().st_size/2**20,archive_members=len(entries)+1,unpacked_independent_comparisons=review['numeric_comparisons'],additional_trajectory_differences=len(diff),preservation=preservation),ensure_ascii=False))
