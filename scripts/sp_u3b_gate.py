"""SP-U3B: mandatory estimated-resource gate, before native full-data allocation.

The frozen author EM expression materializes an N*r*S complex ndarray before
np.dot. A necessary allocation exceeding the user cap is sufficient to FAIL.
No fit, corpus replay, grid or development evaluation may start after this FAIL.
"""
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
OUT = ROOT / 'outputs'
ACCESS = []
OLD_INPUTS = {
    't15a_cdl_manifest.csv', 't15c_cdl_manifest.csv',
    't15a_cdl_statistics_500.npz', 't15c_cdl_statistics_CDL-B-30.npz',
    't15c_cdl_statistics_CDL-C-120.npz',
    't15b_gmm_probe_dataset_manifest.json', 't16c_heldout_manifest.csv',
    't16e_pilot_cv_preflight.json', 't16e_frozen_operators.npz',
    't16f_preflight.json', 't16f_frozen_candidates.npz',
    't16g_factor_preflight.json', 't16g_frozen_module_factors.npz'
}


def guard(event, args):
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    try:
        rel = path.relative_to(OUT.resolve()).as_posix()
    except ValueError:
        return
    flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
    writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
    own = rel.split('/')[0].startswith('sp_u3b_')
    allowed = own or (not writing and (rel.split('/')[0].startswith('sp_u3a_') or rel in OLD_INPUTS))
    ACCESS.append(dict(path=rel, writing=writing, allowed=allowed))
    if not allowed:
        raise PermissionError('SP-U3B firewall: ' + rel)


sys.addaudithook(guard)


def sha(path):
    # Bounded hashing: do not materialize a whole corpus or model for provenance.
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, value):
    path = Path(path)
    assert not path.exists(), str(path)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + '\n', encoding='utf8')


def protocol():
    return load(ROOT / 'configs/sp_u3b_protocol.json')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def verify_u3a():
    seal = load(OUT / 'sp_u3a_delivery_manifest.json')
    groups = ('artifact_sha256', 'execution_source_sha256', 'protected_sources')
    for group in groups:
        for path, digest in seal[group].items():
            assert sha(ROOT / path) == digest, path
    for path, digest in seal['frozen_input_sha256'].items():
        assert sha(OUT / path) == digest, path
    assert git('rev-parse', 'HEAD') == 'ca83982b1d0e4bdca0215a3ecbe3a7a8f8f8ec55'
    assert not git('diff', '--name-only') and not git('diff', '--cached', '--name-only')
    assert load(OUT / 'sp_u3a_baseline_gate.json')['gate'] == 'PASS'
    pins = []
    for name, commit in protocol()['author_commits'].items():
        root = OUT / 'sp_u3a_sources' / name
        pin = load(root / 'repository_pin.json')
        assert pin['commit'] == commit
        assert all(sha(root / path) == digest for path, digest in pin['files'].items())
        pins.append(dict(repository=name, commit=commit, files=len(pin['files']), unchanged=True))
    return dict(passed=True, counts={g: len(seal[g]) for g in groups},
                frozen_input_hashes=len(seal['frozen_input_sha256']), authors=pins)


def author_allocation():
    path = OUT / 'sp_u3a_sources/cplx-mfa/src/cplx_mfa/mfa.py'
    source = path.read_text(encoding='utf8')
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ComplexMFA')
    em = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_em_step')
    stmt = next(n for n in em.body if isinstance(n, ast.For))
    assignment = next(n for n in stmt.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'weighted_cross_covariance' for t in n.targets))
    call = assignment.value
    assert isinstance(call, ast.Call) and ast.unparse(call.func) == 'np.dot'
    product = call.args[0]
    assert isinstance(product, ast.BinOp) and isinstance(product.op, ast.Mult)
    assert ast.unparse(product.left) == 'zeroed_data[:, None, :]'
    assert ast.unparse(product.right) == 'latents[None, :, :].conj()'
    # Establish N,S and r,S from the exact same native component-loop statements.
    assigned = {t.id: n.value for n in stmt.body if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Name)}
    assert ast.unparse(assigned['zeroed_data']) == 'data.T - self.means_[component, :, None]'
    assert ast.unparse(assigned['latents']) == 'betas[component] @ centered_data'
    assert ast.unparse(assigned['centered_data']) == 'data.T - self.means_[component, :, None]'
    run = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_run_em')
    assert 'iteration > 5 and relative_change < self.tol' in ast.unparse(run)
    assert 'converged = True' in ast.unparse(run)
    assert not any(isinstance(n, ast.Attribute) and n.attr == 'converged_' for n in ast.walk(cls))
    return dict(path=path.relative_to(ROOT).as_posix(), source_sha256=sha(path),
                assignment_line=assignment.lineno, product_line=product.lineno,
                expression=ast.get_source_segment(source, assignment),
                eager_argument=True, chunking=False,
                convergence_state='local converged flag in _run_em; no public converged_; infer from trace and nonconvergence warning only for executed fits')


def allocation_bound(K, rank, N=4096, samples=12000):
    import numpy as np
    assert np.broadcast_shapes((N, 1, samples), (1, rank, samples)) == (N, rank, samples)
    dtype = np.result_type(np.complex128, np.complex128)
    assert dtype == np.dtype('complex128')
    temporary = N * rank * samples * dtype.itemsize
    data = N * samples * dtype.itemsize
    dense = 2 * K * N * N * dtype.itemsize
    # All three are live at the expression: data is the fit input and the native
    # model still owns covariance and precision arrays. Other live arrays omitted.
    return dict(product_shape=[N, rank, samples], dtype=str(dtype),
                temporary_bytes=temporary, temporary_GiB=temporary / 2**30,
                training_data_bytes=data, covariance_precision_bytes=dense,
                necessary_process_array_bytes=temporary + data + dense,
                necessary_process_array_GiB=(temporary + data + dense) / 2**30,
                excluded='Python/dependency overhead, means/loadings, centered_data, zeroed_data, loadings_latents, workspace and other temporaries; lower bound only')


def grid_status():
    cfg = protocol()
    return [dict(variant=v, K=k, rank=r, seed=s,
                 status='NOT_RUN_PHASE_A_RESOURCE_FAIL', iterations=None,
                 converged=None, validation_NLL=None)
            for v in cfg['variants'] for k in cfg['K'] for r in cfg['rank'] for s in cfg['seeds']]


def gate():
    started = time.perf_counter()
    integrity = verify_u3a()
    cfg = protocol()
    proof = author_allocation()
    rows = []
    for sentinel in cfg['sentinels']:
        bound = allocation_bound(sentinel['K'], sentinel['rank'])
        exceeded = bound['necessary_process_array_bytes'] > cfg['max_process_bytes']
        rows.append(dict(**sentinel, training_trajectories=1200, training_frames=12000,
                         allocation=bound,
                         estimated_resource_gate='FAIL' if exceeded else 'LOWER_BOUND_WITHIN_CAP_NOT_A_PASS',
                         fit_started=False, iterations=None, converged=None,
                         fit_wall_seconds=None, observed_peak_memory_bytes=None,
                         fit_warnings=[], lower_bound_trace=[], finite_parameters=None,
                         minimum_Psi=None, maximum_Psi=None, weights=None, occupancy=None,
                         termination_reason='NOT_STARTED_GLOBAL_RESOURCE_PREFLIGHT_FAIL'))
    assert rows[1]['allocation']['temporary_bytes'] > cfg['max_process_bytes']
    save(OUT / 'sp_u3b_resource_gate.json', dict(
        task='SP-U3B', phase='A', gate='FAIL', reason='NECESSARY_NATIVE_TEMPORARY_EXCEEDS_ESTIMATED_8GiB_CAP',
        cap_bytes=cfg['max_process_bytes'], cap_GiB=8,
        author_expression=proof, sentinels=rows, integrity_before=integrity,
        protocol_sha256=sha(ROOT / 'configs/sp_u3b_protocol.json'),
        script_sha256=sha(__file__), preflight_wall_seconds=time.perf_counter() - started,
        scientific_fits_executed=0, failed_executed_fits=0,
        full_corpus_replayed=False, grid_executed=False, B120_generated=False,
        phase_B='BLOCKED', phase_C='BLOCKED', stop=True,
        interpretation='Native implementation feasibility limitation under fixed cap; no channel-estimation performance inference',
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), access=ACCESS))
    save(OUT / 'sp_u3b_grid_status.json', dict(rows=grid_status(),
        selection_started=False, winner=None, final_refit=None,
        all_unexecuted_reason='PHASE A estimated resource FAIL; not nonconverged candidate fits'))
    print('PHASE A FAIL: sentinel B native temporary', rows[1]['allocation']['temporary_GiB'],
          'GiB; necessary live arrays at least', rows[1]['allocation']['necessary_process_array_GiB'], 'GiB', flush=True)
    print('A/B fits NOT STARTED; no corpus replay, search, B120 comparison or performance values', flush=True)


def tests():
    gate_result = load(OUT / 'sp_u3b_resource_gate.json')
    assert gate_result['gate'] == 'FAIL' and gate_result['stop']
    results = []
    # Separate processes: retain U3A's original immutable access boundary.
    for label, path in [('u3b', 'tests/test_sp_u3b_gate.py'), ('u3a_regression', 'tests/test_sp_u3a_mfa.py')]:
        command = [sys.executable, '-m', 'pytest', path, '-q', '-ra', '-p', 'no:cacheprovider',
                   '--basetemp=outputs/sp_u3b_pytest_tmp_' + label]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        log = OUT / ('sp_u3b_pytest_' + label + '.txt')
        assert not log.exists()
        log.write_text(result.stdout + result.stderr, encoding='utf8')
        print(result.stdout + result.stderr, flush=True)
        results.append(dict(label=label, command=command, exit_code=result.returncode, log_sha256=sha(log)))
        if result.returncode:
            raise RuntimeError('Test failure; preserve evidence and stop')
    integrity = verify_u3a()
    diff = subprocess.run(['git', 'diff', '--check'], cwd=ROOT, capture_output=True, text=True)
    assert diff.returncode == 0
    save(OUT / 'sp_u3b_test_integrity.json', dict(passed=True, tests=results,
         prior_integrity=integrity, git_HEAD=git('rev-parse', 'HEAD'),
         tracked_diff=git('diff', '--stat'), staged_diff=git('diff', '--cached', '--stat'),
         diff_check_exit=diff.returncode, allowed_old_inputs_use='hash verification only',
         formal_outcomes_read=False, old_U2_outcomes_read=False,
         author_source_modified=False, production_modified=False,
         no_performance_or_convergence_claim=True, stop=True, access=ACCESS))
    sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in
               [ROOT / 'scripts/sp_u3b_gate.py', ROOT / 'tests/test_sp_u3b_gate.py', ROOT / 'configs/sp_u3b_protocol.json']}
    artifacts = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(OUT.glob('sp_u3b_*')) if p.is_file()}
    save(OUT / 'sp_u3b_delivery_manifest.json', dict(task='SP-U3B', stop=True,
         gate='FAIL', artifacts=artifacts, sources=sources,
         next_action='mentor review required; no automatic continuation'))
    print('SP-U3B sealed resource FAIL; tests complete; stopped', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['gate', 'tests'])
    globals()[parser.parse_args().action]()
