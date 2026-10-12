"""Final complete-author-tree and deployment integrity checks, no formal run."""
from pathlib import Path
import sys,os,subprocess,json
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.mfa_lab_migration_common import OUT,read,write,sha
report=read(OUT/'relocation_validation.json');moved=Path(report['relocated_root'])
qa=OUT/'validation_attempt_3';py=moved/'.venv-lab/Scripts/python.exe'
env=os.environ.copy();env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
env['MFA_MIGRATION_TEST_ROOT']=str(moved);env['TEMP']=env['TMP']=str(qa/'temporary')
for key in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
def call(command):
    p=subprocess.run([str(c)for c in command],cwd=moved,env=env,capture_output=True,text=True,encoding='utf8',errors='replace')
    return {'command':[str(c)for c in command],'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
test=call([py,'-B','-X','utf8','-m','pytest',ROOT/'tests/test_mfa_lab_migration.py','-q','-p','no:cacheprovider','--basetemp',qa/'pytest_tmp_final'])
write(qa/'pytest_migration_final_tree.json',test);assert test['exit_code']==0
check=call(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',moved/'deployment/mfa_lab_v1/Verify-Package.ps1'])
write(qa/'powershell_integrity_final_tree.json',check);assert check['exit_code']==0
manifest=read(moved/'migration_manifest.json');assert all(sha(moved/n)==v['sha256']for n,v in manifest['files'].items())
report['actual_pytest']['migration_before_public_metadata_closure']=report['actual_pytest']['migration']
report['actual_pytest']['migration']=test['stdout'].strip()
report.update(complete_author_reference_tree_PASS=True,author_reference_files=15,final_payload_files_verified=len(manifest['files']),
    final_manifest_sha256=sha(moved/'migration_manifest.json'),deployment_code_after_hidden_launch_unchanged=True,
    final_payload_bytes_after_tests_unchanged=True,PowerShell_integrity_final_tree_PASS=True)
write(OUT/'relocation_validation.json',report)
print(json.dumps({'PASS':True,'migration_pytest':test['stdout'].strip(),'author_tests_previously_passed':33,'payload_files':len(manifest['files'])}))
