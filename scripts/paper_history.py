"""Validate editorial history against its committed snapshot, never rewrite hashes."""
import subprocess
import types
import hashlib
from paper_support import ROOT, json_read


def prior_paper_commit():
    return json_read(ROOT / "outputs/t20f_before_manifest.json")["t20e_commit"]


def historical_bytes(commit, path):
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def historical_hash_evidence(commit, path, expected_sha256):
    """Require the recorded byte hash, using Git's checkout filters if needed."""
    raw = historical_bytes(commit, path)
    raw_hash = hashlib.sha256(raw).hexdigest()
    checked = raw
    representation = "git_blob"
    if raw_hash != expected_sha256:
        checked = subprocess.check_output(
            ["git", "cat-file", "--filters", f"{commit}:{path}"], cwd=ROOT)
        representation = "git_checkout_filters"
    checked_hash = hashlib.sha256(checked).hexdigest()
    if checked_hash != expected_sha256:
        raise ValueError(f"Historical SHA256 mismatch: {commit}:{path}")
    return checked, dict(commit=commit, path=path, expected_sha256=expected_sha256,
        git_blob_sha256=raw_hash, verified_bytes_sha256=checked_hash,
        representation=representation, exact_match=True)


def historical_bytes_exact(commit, path, expected_sha256):
    return historical_hash_evidence(commit, path, expected_sha256)[0]


def historical_module(commit, path):
    module = types.ModuleType("historical_paper_formatter")
    module.__file__ = str(ROOT / path)
    exec(compile(historical_bytes(commit, path), module.__file__, "exec"), module.__dict__)
    return module
