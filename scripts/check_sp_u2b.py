"""Run guarded pre-CDL pytest and preserve actual process logs."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import sp_u2b_diagnostic as d
d.install_guard()


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True,type=int);a=p.parse_args()
    pre=d.load(d.OUT/'sp_u2b_preregistration.json')
    temp=d.OUT/f'sp_u2b_pytest_tmp_attempt{a.attempt:02d}'
    assert not temp.exists() and temp.resolve().parent==d.OUT.resolve()
    tests=['tests/test_sp_u2b_diagnostic.py',
           'tests/test_pilot_cv.py::test_fold_geometry_partition_and_no_heldout_observed_overlap',
           'tests/test_pilot_cv.py::test_real_LS_independent_stacked_reference_and_brute_force',
           'tests/test_pilot_cv.py::test_no_environment_or_oracle_features_in_fit_interfaces',
           'tests/test_pilot_cv.py::test_causal_windows_and_future_frame_perturbation',
           'tests/test_pilot_cv.py::test_unconstrained_extreme_coefficients_are_not_clipped']
    argv=['-q','-p','no:cacheprovider','--basetemp',str(temp),*tests]
    code='import sys;from scripts.sp_u2b_diagnostic import install_guard;install_guard();import pytest;sys.exit(pytest.main('+repr(argv)+'))'
    start=time.perf_counter()
    run=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
    log=d.OUT/f'sp_u2b_pytest_attempt{a.attempt:02d}.txt'
    if log.exists():raise RuntimeError('existing test log protected')
    log.write_text(run.stdout+'\nSTDERR:\n'+run.stderr,encoding='utf-8')
    report=dict(exit_code=run.returncode,elapsed_seconds=time.perf_counter()-start,argv=argv,
                full_suite=False,reason='full suite reads frozen historical/formal outcomes; bounded guarded nodes only',
                log=str(log.relative_to(ROOT)),log_sha256=d.sha(log),
                execution_source_sha256=pre['execution_source_sha256'])
    d.save(d.OUT/f'sp_u2b_test_execution_attempt{a.attempt:02d}.json',report)
    print(run.stdout,run.stderr,flush=True)
    if run.returncode==0:d.save(d.OUT/'sp_u2b_tests_pass.json',report)
    sys.exit(run.returncode)


if __name__=='__main__':main()
