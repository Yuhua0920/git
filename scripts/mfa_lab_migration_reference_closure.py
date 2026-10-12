"""Include the three public author Git metadata files for exact tree hashes."""
from pathlib import Path
import sys,json,csv,shutil,subprocess,hashlib
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.mfa_lab_migration_common import OUT,read,write,sha
from scripts.mfa_lab_migration_prepare import STAGE

moved=Path(read(OUT/'relocation_validation.json')['relocated_root'])
names=['reference/sp_u12b5a/paper_period/'+n for n in
       ('cplx-mfa/.gitignore','MFA_estimator/.gitignore','MFA_estimator/.gitmodules')]
manifest=read(STAGE/'migration_manifest.json');rows=read(OUT/'file_inventory.json')
tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode('utf8').split('\0'))
for name in names:
    assert name not in manifest['files']
    assert (ROOT/name).is_file()
    for root in(STAGE,moved):shutil.copyfile(ROOT/name,root/name)
    row={'path':name,'category':'FROZEN_AUTHOR_REFERENCE','bytes':(ROOT/name).stat().st_size,
         'sha256':sha(ROOT/name),'git_state':'TRACKED_CLEAN'if name in tracked else'UNTRACKED'}
    rows.append(row);manifest['files'][name]={k:v for k,v in row.items()if k!='path'}
rows.sort(key=lambda x:x['path']);write(OUT/'file_inventory.json',rows)
with(OUT/'necessary_files.csv').open('w',encoding='utf-8-sig',newline='')as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for root in(STAGE,moved):
    shutil.copyfile(OUT/'necessary_files.csv',root/'necessary_files.csv')
    provenance=read(root/'migration_provenance.json');provenance['source_file_count']=len(rows)
    write(root/'migration_provenance.json',provenance)
for name in('necessary_files.csv','migration_provenance.json'):
    manifest['files'][name].update(bytes=(STAGE/name).stat().st_size,sha256=sha(STAGE/name))
for root in(STAGE,moved):
    write(root/'migration_manifest.json',manifest)
    (root/'SHA256SUMS.txt').write_text(''.join(f"{v['sha256']}  {n}\n"for n,v in sorted(manifest['files'].items()))+
        sha(root/'migration_manifest.json')+'  migration_manifest.json\n',encoding='utf8')
ready=read(OUT/'stage_ready.json');ready.update(payload_files=len(manifest['files']),archive_expected_files=len(manifest['files'])+2,
    payload_bytes=sum(v['bytes']for v in manifest['files'].values()),manifest_sha256=sha(STAGE/'migration_manifest.json'))
write(OUT/'stage_ready.json',ready)
audit=read(OUT/'engineering_audit.json');audit.update(source_payload_files=len(rows),payload_untracked=sum(r['git_state']=='UNTRACKED'for r in rows))
audit['untracked_and_modified_research_files_included'] += [r['path']for r in rows if r['path']in names and r['git_state']!='TRACKED_CLEAN']
write(OUT/'engineering_audit.json',audit)
metadata=read(ROOT/'outputs/sp_u12b5a/historical_reference.json');actual={}
for repo,entry in metadata.items():
    for name,h in entry['files'].items():
        assert sha(moved/'reference/sp_u12b5a/paper_period'/repo/name)==h
        actual[repo+'/'+name]=h
tree=hashlib.sha256(json.dumps(actual,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert tree=='e6c27180ba2abb07c699e943ba581290371a4dea04583b19b63037c552c3db5f'
old=read(OUT/'package_report.json');archive=Path(old['archive']).resolve();dest=(OUT/'superseded_before_reference_metadata_closure').resolve()
assert archive.is_relative_to(OUT.resolve())and dest.is_relative_to(OUT.resolve())and not dest.exists()
assert sha(archive)==old['sha256'];dest.mkdir()
for name in('Project_MFA_Lab_v1.zip','Project_MFA_Lab_v1.zip.sha256','package_report.json'):
    source=(OUT/name).resolve();target=(dest/name).resolve()
    assert source.is_relative_to(OUT.resolve())and target.is_relative_to(dest)
    source.rename(target)
write(OUT/'reference_tree_closure.json',{'PASS':True,'reason':'Preserve complete public author tree, including its .gitignore/.gitmodules provenance.',
    'added_public_files':names,'complete_reference_files':len(actual),'combined_tree_sha256':tree,
    'scientific_code_changes':0,'superseded_package_preserved':str(dest),'new_expected_archive_files':ready['archive_expected_files']})
print(json.dumps({'PASS':True,'author_tree_files':len(actual),'archive_files':ready['archive_expected_files']}))
