"""Exercise the shipped PowerShell hidden launcher with no MFA step."""
from pathlib import Path
import sys, os, subprocess, time, json, traceback
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.mfa_lab_migration_common import OUT, read, write

def call(command, cwd, env):
    p = subprocess.run([str(x) for x in command], cwd=cwd, env=env,
                       capture_output=True, text=True, encoding='utf8', errors='replace')
    return {'command': [str(x) for x in command], 'exit_code': p.returncode,
            'stdout': p.stdout, 'stderr': p.stderr}

def main():
    qa = OUT / 'validation_attempt_3'
    moved = Path(read(OUT / 'relocation_validation.json')['relocated_root'])
    parent_python = qa / 'fresh_environment/Scripts/python.exe'
    local_env = moved / '.venv-lab'
    assert not local_env.exists()
    env = os.environ.copy()
    env.pop('PYTHONHOME', None)
    env.pop('PYTHONPATH', None)
    env['TEMP'] = env['TMP'] = str(qa / 'temporary')
    result = call([parent_python, '-B', '-X', 'utf8', '-m', 'venv', local_env], moved, env)
    write(qa / 'launcher_venv_creation.json', result)
    assert result['exit_code'] == 0
    py = local_env / 'Scripts/python.exe'
    dep = moved / 'deployment/mfa_lab_v1'
    result = call([py, '-m', 'pip', '--isolated', 'install', '--no-index', '--find-links',
                   dep / 'wheelhouse', '--require-hashes', '-r', dep / 'requirements-win-cpu.lock'], moved, env)
    write(qa / 'launcher_offline_install.json', result)
    assert result['exit_code'] == 0
    result = call(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                   dep / 'Start-LabJob.ps1', '-Mode', 'self-test', '-Threads', '1'], moved, env)
    write(qa / 'actual_PowerShell_hidden_launch.json', result)
    assert result['exit_code'] == 0
    latest = read(moved / 'lab_runs/latest.json')
    job = moved / latest['job']
    assert latest['action'] == 'self-test'
    assert job.is_relative_to(moved / 'lab_runs')
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        status = read(job / 'status.json')
        if status['state'] in ('COMPLETE', 'STOPPED'):
            break
        time.sleep(.5)
    assert status['state'] == 'COMPLETE' and status['PASS'], status
    assert (job / 'resources.jsonl').stat().st_size > 0
    assert read(job / 'self_test.json')['formal_fit_calls'] == 0
    query = call(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                  dep / 'Get-LabStatus.ps1'], moved, env)
    write(qa / 'actual_PowerShell_status_query.json', query)
    assert query['exit_code'] == 0 and 'COMPLETE' in query['stdout']
    report = {'PASS': True, 'actual_hidden_PowerShell_launch_tested': True,
              'actual_status_query_tested': True, 'job': str(job), 'status': status,
              'R5_smoke_executed': False, 'formal_candidate_training': 0,
              'validation_scoring': 0, 'target_evaluation': 0}
    write(OUT / 'powershell_launch_validation.json', report)
    print(json.dumps({'PASS': True, 'PowerShell_hidden_launch': True, 'status_query': True}))

if __name__ == '__main__':
    try:
        main()
    except Exception:
        write(OUT / 'powershell_launch_failure.json', {'traceback': traceback.format_exc(), 'formal_fits': 0})
        raise
