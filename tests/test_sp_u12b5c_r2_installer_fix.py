"""Only native Windows wrapper regressions; no model/seed/evaluation replay."""
from pathlib import Path
import hashlib
import json
import os
import sys

import pytest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import sp_u12b5c_r2_installer_fix as f


@pytest.fixture
def lab(tmp_path):
    root, frozen = f.relocate(tmp_path)
    return root, frozen


def run_install(root, frozen, mode='omitted', expected=None, label='install'):
    r = f.install(root, f.PACKAGE, f.OUT / 'Install-R2.ps1', mode, expected)
    (root.parent / (label + '.stdout.log')).write_text(r.stdout, encoding='utf8')
    (root.parent / (label + '.stderr.log')).write_text(r.stderr, encoding='utf8')
    f.unchanged(root, frozen)
    return r


@pytest.mark.parametrize('mode', ('omitted', 'empty', 'whitespace', 'explicit'))
def test_default_empty_whitespace_and_explicit_python_install(lab, mode):
    root, frozen = lab
    r = run_install(root, frozen, mode)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'PASS' in r.stdout and 'True' in r.stdout
    assert (root / 'deployment/b5c_paired_r2_v1/package_manifest.json').is_file()
    assert 'MissingArgument' not in r.stderr


@pytest.mark.parametrize('mode', ('omitted', 'missing_explicit'))
def test_missing_interpreter_rejected_before_any_deployment(lab, mode):
    root, frozen = lab
    if mode == 'omitted':
        # Preserve the synthetic interpreter under a different test-only name.
        p = root / '.venv-lab/Scripts/python.exe'
        assert p.resolve().is_relative_to(root.resolve())
        p.rename(p.with_name('unselected-python.exe'))
    r = run_install(root, frozen, mode)
    assert r.returncode != 0 and 'Existing LAB Python interpreter missing' in r.stderr
    assert not (root / 'deployment/b5c_paired_r2_v1').exists()


def test_bad_zip_hash_still_rejected(lab):
    root, frozen = lab
    r = run_install(root, frozen, expected='0' * 64)
    assert r.returncode != 0 and 'R2 ZIP SHA256 mismatch' in r.stderr
    assert not (root / 'deployment/b5c_paired_r2_v1').exists()


def test_duplicate_install_still_rejected_without_overwrite(lab):
    root, frozen = lab
    assert run_install(root, frozen).returncode == 0
    deploy = root / 'deployment/b5c_paired_r2_v1'
    installed = {p: f.sha(p) for p in deploy.rglob('*') if p.is_file()}
    r = run_install(root, frozen, label='duplicate')
    assert r.returncode != 0 and 'R2 namespace already exists' in r.stderr
    assert all(f.sha(p) == h for p, h in installed.items())


def test_only_installer_and_derived_manifest_change():
    archive = f.OUT / 'original_r2_deployment'
    changed = [p.name for p in f.DEPLOY.iterdir() if p.is_file() and p.read_bytes() != (archive / p.name).read_bytes()]
    assert set(changed) == {'Install-R2.ps1', 'package_manifest.json'}
    before = f.read(f.OUT / 'before.json')['files']
    assert f.sha(f.OLD_OUT / 'SP_U12B5C_R2_Execution.zip') == before['outputs/sp_u12b5c_r2/SP_U12B5C_R2_Execution.zip']['sha256']
    assert (f.DEPLOY / 'science.py').read_bytes() == (ROOT / 'deployment/b5c_paired_r1_v1/science.py').read_bytes()


def test_original_native_empty_argument_failure_was_actually_reproduced():
    r = f.read(f.OUT / 'original_failure_reproduction.json')
    assert r['PASS'] and r['expected_failure_reproduced'] and r['exit_code'] == 1
    assert r['default_python_exists'] and r['formal_predictions'] == 0
    log = Path(r['stderr']).read_text(encoding='utf8')
    assert 'Python' in log and 'MissingArgument' in log
