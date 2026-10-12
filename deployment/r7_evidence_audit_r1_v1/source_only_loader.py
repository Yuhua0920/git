"""Audit-process import policy: authenticate and compile the same source bytes.

No reference/adapter mathematics are replaced. SourceFileLoader is intercepted
only inside this disposable audit process; the filesystem guard is unchanged.
"""
from pathlib import Path
import hashlib
import importlib.machinery
import sys

class SourceAuthenticationFailure(RuntimeError):
    pass

_original = importlib.machinery.SourceFileLoader.get_code
_policy = None
observations = []

def install(root, expected):
    global _policy
    root = Path(root).resolve()
    exact = {}
    for name, record in expected.items():
        if not name.endswith('.py'):
            continue
        path = (root / name).resolve()
        if not path.is_relative_to(root) or '..' in Path(name).parts:
            raise SourceAuthenticationFailure('SOURCE_UNSAFE_PATH:' + name)
        exact[path] = record['sha256']
    policy = (root, exact)
    if _policy is not None:
        if _policy != policy:
            raise SourceAuthenticationFailure('SOURCE_POLICY_REBIND_DENIED')
        return
    _policy = policy
    importlib.machinery.SourceFileLoader.get_code = authenticated_code

def authenticated_code(loader, fullname):
    root, exact = _policy
    path = Path(loader.path).resolve()
    if path in exact:
        # Read once. Compile precisely the bytes just authenticated; no TOCTOU
        # between hash check and a second importlib read; no bytecode-cache API.
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != exact[path]:
            raise SourceAuthenticationFailure('SOURCE_SHA256_MISMATCH:' + str(path))
        code = loader.source_to_code(data, loader.path)
        observations.append({'path': str(path), 'sha256': actual,
                             'compiled_filename': code.co_filename,
                             'cache_read': False, 'cache_write': False})
        return code
    if path.is_relative_to(root):
        # The interpreter's existing environment is not project/reference code.
        runtime = [Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()]
        if not any(p != root and path.is_relative_to(p) for p in runtime):
            raise SourceAuthenticationFailure('SOURCE_NOT_REGISTERED:' + str(path))
    return _original(loader, fullname)
