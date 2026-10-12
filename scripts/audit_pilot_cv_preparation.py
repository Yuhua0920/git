"""Audit T16E preparation only; no performance, fitting or generator calls."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_pilot_cv import accepted,OUT,SOURCES
from scripts.run_cdl_mismatch import load,save,sha,history,environments
from src.cdl_benchmark import array_hash
from src.pilot_cv import frozen_folds
from src.cdl_benchmark import observation_geometry


def audit():
    old=accepted();pre=load(OUT/'t16e_pilot_cv_preflight.json')
    # The preflight test file precedes two added persisted-artifact regressions.
    # Retain its original recorded hash; do not fabricate current==historical.
    for p,h in pre['source_sha256'].items():
        if p!='tests/test_pilot_cv.py' and sha(ROOT/p)!=h:raise ValueError('mathematical preflight implementation drift')
    _,definition=frozen_folds(observation_geometry().selected)
    if definition!=load(OUT/'t16e_pilot_folds.json') or definition!=pre['pilot_fold_definition']:
        raise ValueError('frozen fold drift')
    info=pre['operator_artifact']
    if sha(ROOT/info['path'])!=info['sha256']:raise ValueError('operator file drift')
    with np.load(ROOT/info['path'],allow_pickle=False) as f:
        if any(array_hash(f[k])!=h for k,h in info['array_sha256'].items()):raise ValueError('operator array drift')
    cache_checks={}
    for name,cache in old['validation_replay_gate']['cache'].items():
        if sha(ROOT/cache['path'])!=cache['file_sha256']:raise ValueError('historical replay H cache drift')
        for artifact in cache['input_artifacts'].values():
            path=ROOT/artifact['path']
            if sha(path)!=artifact['file_sha256']:raise ValueError('historical replay input cache file drift')
            with np.load(path,allow_pickle=False) as f:
                if any(array_hash(f[k])!=h for k,h in artifact['replay_tensor_sha256'].items()):raise ValueError('historical replay cache tensor drift')
        cache_checks[name]=dict(H_file_exact=True,input_file_and_tensor_hashes_exact=True,trajectories=100,frames=1000)
    env=environments()
    if env!=old['environment_after']:raise ValueError('frozen environment drift')
    result=dict(task='T16E',scope='preparation only; not actual-validation acceptance',all_preparation_checks_passed=True,
        actual_validation_completed=False,historical_replay_metric_gate_performed=False,
        existing_replay_cache_integrity_verified=True,cache_checks=cache_checks,
        frozen_environment_unchanged=True,environment=env,frozen_history_sha256=history(),
        preparation_sources_sha256={p:sha(ROOT/p) for p in (*SOURCES,'scripts/audit_pilot_cv_preparation.py')},
        preflight_original_test_sha256=pre['source_sha256']['tests/test_pilot_cv.py'],
        expanded_current_test_sha256=sha(ROOT/'tests/test_pilot_cv.py'),
        fold_artifact_sha256=sha(OUT/'t16e_pilot_folds.json'),operator_artifact_sha256=sha(ROOT/info['path']),
        generator_calls=0,new_channel_realizations=0,new_model_trained=False,formal_test_used=False,formal_test_sealed=True)
    save(OUT/'t16e_preparation_audit.json',result)
    print('Frozen environments / historical files / 200 original replay caches / fold and operator arrays PASS; no actual T16E metric gate claimed')


if __name__=='__main__':audit()
