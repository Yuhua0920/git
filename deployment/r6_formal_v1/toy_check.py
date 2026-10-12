"""Actual pytest and task-script entry; refuses to reuse toy attempt artifacts."""
from pathlib import Path
import sys,os,subprocess,json,time
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from common import ROOT,LOCAL,OUT,write,read,sha,environment

def main():
    LOCAL.mkdir(parents=True,exist_ok=True)
    marker=LOCAL/'background_probe_result.json'
    if marker.exists() and read(marker).get('test_receipt'):
        fixtures=LOCAL/'export_fixture_archive';fixtures.mkdir(exist_ok=True)
        marker.rename(fixtures/('labelled_background_fixture_'+time.strftime('%Y%m%d_%H%M%S')+'.json'))
    snapshot=LOCAL/('source_snapshot_'+time.strftime('%Y%m%d_%H%M%S'));snapshot.mkdir()
    for source in HERE.iterdir():
        if source.is_file():(snapshot/source.name).write_bytes(source.read_bytes())
    # Explicit new attempts retain failures; never delete their evidence.
    target=LOCAL/'toy_models'
    if target.exists():
        archived=LOCAL/('toy_attempt_archive_'+time.strftime('%Y%m%d_%H%M%S'))
        archived.mkdir()
        for name in ('toy_models','corrupt_model','monitor','toy_test_report.json'):
            p=LOCAL/name
            if p.exists():p.rename(archived/name)
    env=os.environ.copy();env['TEMP']=env['TMP']=str(LOCAL/'temporary');(LOCAL/'temporary').mkdir(exist_ok=True)
    env['PYTHONDONTWRITEBYTECODE']='1';env['OMP_NUM_THREADS']='1'
    result=subprocess.run([sys.executable,'-B','-X','utf8','-m','pytest',str(HERE/'test_r6.py'),'-q','-p','no:cacheprovider','--basetemp',str(LOCAL/('pytest_tmp_'+time.strftime('%Y%m%d_%H%M%S')))],cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf8',errors='replace')
    path=LOCAL/('pytest_'+time.strftime('%Y%m%d_%H%M%S')+'.log');path.write_text(result.stdout+'\nSTDERR\n'+result.stderr,encoding='utf8')
    value={'PASS':result.returncode==0,'exit_code':result.returncode,'pytest_stdout':result.stdout,'pytest_log':str(path.relative_to(ROOT)),
           'tested_package_manifest_sha256':sha(HERE/'package_manifest.json'),'environment':environment(),
           'synthetic_scope':{'toy_fit_N':96,'toy_fit_D':32,'nine_structures':True,'real_candidate_fits':0,'real_validation_scores':0,'real_final_fits':0},
           'formal_candidates_directory_present':(OUT/'candidates').exists(),'formal_selection_present':(OUT/'SELECTION_MANIFEST.json').exists(),
           'formal_final_model_present':(OUT/'FINAL_MODEL_MANIFEST.json').exists()}
    write(LOCAL/'toy_test_report.json',value);print(json.dumps(value,indent=2));return result.returncode

if __name__=='__main__':sys.exit(main())
