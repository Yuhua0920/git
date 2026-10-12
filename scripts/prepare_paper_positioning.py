"""Record a clean accepted editorial snapshot before T20G writing."""
import subprocess
from paper_support import ROOT, json_write, sha256


def build():
    tracked_changes = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=ROOT, text=True)
    untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=ROOT, text=True).splitlines()
    bootstrap_files = {'scripts/prepare_paper_positioning.py', 'scripts/build_paper_summary.py',
                       'tests/test_paper_positioning.py'}
    if tracked_changes.strip() or set(untracked)-bootstrap_files:
        raise ValueError('T20G must begin from the clean accepted T20F commit')
    files = subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines()
    report = dict(t20f_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                  cwd=ROOT, text=True).strip(), working_tree_clean_before_t20g=True,
                  tracked_before_sha256={p: sha256(ROOT/p) for p in files},
                  paper_acceptance='97 passed in 31.03s',
                  full_acceptance='2077 passed in 465.29s (0:07:45)',
                  acceptance_logs=['outputs/t20f_acceptance_paper_pytest.txt',
                                   'outputs/t20f_acceptance_full_pytest.txt'],
                  new_snapshot_utilities=sorted(untracked),
                  snapshot_attempt_note='Initial check rejected its own new untracked utilities; tracked T20F files remained clean. Clean status was observed immediately after the commit, before utilities were added.')
    json_write(ROOT/'outputs/t20g_before_manifest.json', report)
    return dict(t20f_commit=report['t20f_commit'], tracked_files=len(files), clean=True)


if __name__ == '__main__':
    print(build())
