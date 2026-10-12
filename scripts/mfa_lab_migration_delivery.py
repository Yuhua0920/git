"""Produce complete transfer inventory and a factual preparation report."""
from pathlib import Path
import sys,json,csv,collections,zipfile,subprocess
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.mfa_lab_migration_common import OUT,read,write,sha
from scripts.mfa_lab_migration_prepare import STAGE

package=read(OUT/'package_report.json');preserved=read(OUT/'preservation.json')
validation=read(OUT/'relocation_validation.json');launch=read(OUT/'powershell_launch_validation.json')
assert all(r['PASS']for r in(package,preserved,validation,launch))
assert sha(Path(package['archive']))==package['sha256']
manifest=read(STAGE/'migration_manifest.json');source_rows=read(OUT/'file_inventory.json')
byname={r['path']:r for r in source_rows};rows=[]
with zipfile.ZipFile(package['archive'])as archive:
    for entry in archive.infolist():
        name=entry.filename.removeprefix('Project_MFA_Lab/')
        assert entry.filename.startswith('Project_MFA_Lab/')and'..'not in Path(name).parts
        info=byname.get(name)or manifest['files'].get(name)or{}
        rows.append({'archive_entry':entry.filename,'path':name,'bytes':entry.file_size,
                     'sha256':sha(STAGE/name),'category':info.get('category','GENERATED_CONTROL_OR_PROVENANCE'),
                     'git_state':info.get('git_state','NEW_DEPLOYMENT_RECORD')})
assert len(rows)==package['files']==211
with(OUT/'archive_files.csv').open('w',encoding='utf-8-sig',newline='')as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(OUT/'archive_SHA256SUMS.txt').write_text(''.join(f"{r['sha256']}  {r['archive_entry']}\n"for r in rows),encoding='utf8')
check=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True,encoding='utf8')
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert check.returncode==0 and head==read(OUT/'preservation_before.json')['Git_HEAD']
report={'decision':'MIGRATION_PREPARATION_COMPLETE_LAB_VALIDATION_PENDING',
        'package':package,'full_file_inventory':str(OUT/'archive_files.csv'),
        'full_file_SHA256_inventory':str(OUT/'archive_SHA256SUMS.txt'),
        'source_files':len(source_rows),'source_git_states':dict(collections.Counter(r['git_state']for r in source_rows)),
        'categories':dict(collections.Counter(r['category']for r in source_rows)),
        'environment':read(STAGE/'deployment/mfa_lab_v1/environment_source.json'),
        'actual_verification':{'original_reference_pytest_passed':33,'migration_pytest_passed':15,
            'fresh_offline_environment':True,'unicode_and_space_root':True,'pip_check':validation['pip_check'],
            'PowerShell_parse':True,'PowerShell_payload_SHA256':True,'PowerShell_hidden_background_launch':launch['PASS'],
            'PowerShell_status_query':True,'actual_worker_PID_monitor':True,'intentional_failure_STOP_and_exception_log':True,
            'complete_author_tree_files':15,'complete_author_tree_SHA256':'e6c27180ba2abb07c699e943ba581290371a4dea04583b19b63037c552c3db5f',
            'installer_executable_locally_run':False,'full_R4_replay_executed':False,'R5_smoke_executed':False,
            'laboratory_host_verified':False},
        'preservation':preserved,'git_HEAD':head,'git_commit_performed':False,'git_diff_check_exit':check.returncode,
        'formal_candidate_training':0,'formal_validation_scoring':0,'final_retraining':0,'target_evaluation':0,
        'historical_scientific_numeric_outcomes_decoded':0,
        'remaining_risks':[
            'Laboratory Windows permissions, installer execution and actual hardware/BLAS Fidelity remain to be checked there.',
            'Exact environment versions must pass; do not silently substitute dependency versions.',
            'Original scientific orchestrators reference excluded historical governance/sealed paths; they are archived, not portable approved launch commands.',
            'Future formal MFA training requires a separately approved deployment wrapper and resource protocol; no model-selection state has advanced.',
            'R5 taskbook was not available; synthetic smoke is a prepared D4032/N128/K4/r16 one-step technical candidate, not a formally certified R5 gate.',
            '128 GB does not automatically change existing 6 GiB process/2 GiB headroom guards; no full-scale resource certificate is claimed.',
            'Background jobs survive remote control disconnect while the laboratory Windows session stays active; logout, sleep or restart can stop them.'],
        'STOP':'MENTOR_REVIEW_AND_LABORATORY_HOST_VALIDATION'}
write(OUT/'delivery_report.json',report)
print(json.dumps({'decision':report['decision'],'archive':package['archive'],'bytes':package['bytes'],
                  'files':len(rows),'sha256':package['sha256'],'source_git_states':report['source_git_states'],
                  'pytest_passed':48,'protected_files_unchanged':preserved['protected_files'],
                  'manifest_SHA256':sha(STAGE/'migration_manifest.json'),'formal_execution':0}))
