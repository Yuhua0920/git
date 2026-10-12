"""Versioned source checks approved for the T11C-A detector extension.

Historical hashes are never rewritten. Only the two extended numerical modules
and the explicitly listed provenance plumbing use Git snapshots. Other sources,
configuration and result files still require their current whole-file digest.
"""

from functools import lru_cache
import ast
import hashlib
from pathlib import Path
import subprocess


T11B_COMMIT = "763e24c0ce60bc9c93fec3d22ba761415affb337"
# Ordered accepted stage snapshots, including the stage which committed a result.
STAGE_COMMITS = {
    "t09a": "57a020bf06f333ff320b09576d2b6da314a420d9",
    "t09b": "9583cfe7b0c44ff049240650bc53f0ca88e8aeb5",
    "t10a": "7f2405d4a14118a713d926a1eb375f99c8d67095",
    "t10b": "b638f7fda47304309f2a2e34bf8b08cddafb0a9a",
    "t10c_pre": "c0a1ac3525cde5d90a9fe26b75b00170d2416caf",
    "t10c": "d51dfb4df7c40298e5edb164a46168fe07111dc6",
    "t10d": "6129743ca9940ad41a5d9de2f70005c9160822c2",
    "t11a": "9fe2f165f9f511c9fe1c90585f2ec5023ef3c951",
    "t11b": T11B_COMMIT,
}
EXTENDED_MODULES = frozenset({"src/detectors.py", "src/link.py"})
PROVENANCE_PLUMBING = frozenset({
    "scripts/run_tdl_awgn_scan.py", "scripts/validate_lmmse_oracle.py",
    "scripts/run_oracle_lmmse_scan.py", "scripts/validate_multiframe_trajectory.py",
    "scripts/validate_lmmse_train.py", "scripts/run_lmmse_train_validation.py",
    "scripts/validate_nearest_interpolation.py", "src/nearest_validation.py",
    "scripts/finalize_nearest_validation_provenance.py",
    "tests/test_lmmse.py", "tests/test_oracle_lmmse_scan.py",
    "tests/test_lmmse_train.py", "tests/test_lmmse_train_validation.py",
    "tests/test_trajectories.py", "tests/test_nearest_interpolation.py",
    "tests/test_nearest_validation.py", "tests/test_historical_provenance.py",
})
GUARD_FUNCTIONS = {
    "scripts/run_tdl_awgn_scan.py": {"frozen_source_checks"},
    "scripts/validate_lmmse_oracle.py": {"frozen_source_checks"},
    "scripts/run_oracle_lmmse_scan.py": {"frozen_source_checks"},
    "scripts/validate_lmmse_train.py": {"source_checks"},
    "scripts/run_lmmse_train_validation.py": {"source_checks"},
    "scripts/validate_nearest_interpolation.py": {"accepted_provenance"},
    "src/nearest_validation.py": {"verify_accepted_sources"},
    "scripts/finalize_nearest_validation_provenance.py": {"verify_historical_t11a", "finalize"},
}


def plumbing_semantics_unchanged(root, path):
    """Require every non-guard definition/constant to equal the accepted snapshot.

    Only the provenance import addition and listed guard functions are excluded. The older
multi-frame entry has its guard inline in main; normalize that one expression
to the accepted expression, leaving all other main statements exact.
"""
    root = Path(root)
    old = ast.parse(git_blob(str(root), T11B_COMMIT, path))
    new = ast.parse((root/path).read_bytes())
    def normalized(tree, current):
        kept = []
        for node in tree.body:
            if (isinstance(node, ast.ImportFrom) and node.module == "src.provenance"
                    and all(alias.name == "recorded_source_matches" and alias.asname is None for alias in node.names)):
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in GUARD_FUNCTIONS.get(path, set()):
                continue
            if current and path == "scripts/validate_multiframe_trajectory.py" and isinstance(node, ast.FunctionDef) and node.name == "main":
                old_main = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == "main")
                prior = next(n for n in old_main.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "frozen" for t in n.targets))
                for i, statement in enumerate(node.body):
                    if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "frozen" for t in statement.targets):
                        node.body[i] = prior
            kept.append(ast.dump(node, include_attributes=False))
        return kept
    return normalized(old, False) == normalized(new, True)


@lru_cache(maxsize=None)
def git_blob(root, commit, path):
    """Read exact binary Git blob bytes; no shell newline conversion."""
    return subprocess.run(["git", "show", f"{commit}:{path}"], cwd=root,
                          check=True, capture_output=True).stdout


def legacy_prefix_unchanged(root, path):
    """T11C-A is append-only: every pre-existing numerical byte is preserved."""
    if path not in EXTENDED_MODULES:
        raise ValueError("legacy-prefix check only applies to detector/link modules")
    root = Path(root)
    return (root/path).read_bytes().startswith(git_blob(str(root), T11B_COMMIT, path))


def recorded_source_matches(root, path, digest, stage="t11b", *, current_hash=None):
    """Verify recorded provenance at its stage; retain strict current core checks.

current_hash is an optional caller's raw hash callback, also used by negative
controls. It does not replace or misrepresent the ordinary SHA256 operation.
Missing historical blobs are errors; there is no working-tree fallback.
"""
    root = Path(root)
    if path in EXTENDED_MODULES or path in PROVENANCE_PLUMBING:
        blob = git_blob(str(root), STAGE_COMMITS[stage], path)
        matches = hashlib.sha256(blob).hexdigest() == digest
        if path in EXTENDED_MODULES:
            matches = matches and legacy_prefix_unchanged(root, path)
        elif not path.startswith("tests/"):
            matches = matches and plumbing_semantics_unchanged(root, path)
        return matches
    actual = (current_hash(root/path) if current_hash else
              hashlib.sha256((root/path).read_bytes()).hexdigest())
    return actual == digest
