"""Thin, fixed-2023 author-code interface. No project-native MFA mathematics.

Only namespace / NumPy2 alias, dtype / shape, documented full-mixture API,
explicit RNG scoping and the fixed observation/data selectors are adapted.
Original author bytes under reference/ are never edited.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / 'reference/sp_u12b5a/paper_period'
_REFERENCE_LOCK = threading.RLock()
FROZEN_POLICY = dict(PPCA=True, lock_psis=False, rs_clip=0.0,
                     max_condition_number=1e6, maxiter=100, tol=1e-4,
                     verbose=False)


class _Numpy2Alias:
    """Only the removed spelling Inf is supplied; all operations are NumPy's."""
    Inf = np.inf

    def __getattr__(self, name):
        return getattr(np, name)


def _import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


@lru_cache(maxsize=1)
def reference_modules():
    """Load original files with their original absolute imports resolved."""
    with _REFERENCE_LOCK:
        metadata = json.loads((ROOT / 'outputs/sp_u12b5a/historical_reference.json').read_text(encoding='utf8'))
        for repo, files in [('cplx-mfa', ('MFA_cplx.py', 'utils.py')),
                            ('MFA_estimator', ('MFA_estimator.py',))]:
            for name in files:
                path = REFERENCE / repo / name
                assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata[repo]['files'][name], 'REFERENCE_BYTES_CHANGED'
        utilities = _import_file('_sp_u12b5a_author_utils', REFERENCE / 'cplx-mfa/utils.py')
        saved = {name: sys.modules.get(name) for name in ('utils', 'MFA_cplx')}
        try:
            sys.modules['utils'] = utilities
            core = _import_file('_sp_u12b5a_author_core', REFERENCE / 'cplx-mfa/MFA_cplx.py')
            core.np = _Numpy2Alias()
            package = types.ModuleType('MFA_cplx')
            package.MFA_cplx = core
            sys.modules['MFA_cplx'] = package
            estimator = _import_file('_sp_u12b5a_author_estimator', REFERENCE / 'MFA_estimator/MFA_estimator.py')
        finally:
            for name, original in saved.items():
                if original is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original
        return core, estimator, utilities


def make_reference_model(n_components, loading_rank):
    """The paper's isotropic-per-component case with original remaining defaults."""
    _, estimator, _ = reference_modules()
    return estimator.MfaEstimator(n_components=n_components,
                                  latent_dim=loading_rank, **FROZEN_POLICY)


@contextmanager
def recorded_initialization(seed):
    """Use the author's legacy global RNG, with a recorded seed and restoration."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed <= 2**32-1:
        raise ValueError('A recorded uint32 initialization seed is required')
    with _REFERENCE_LOCK:
        previous = np.random.get_state()
        try:
            np.random.seed(int(seed))
            yield
        finally:
            np.random.set_state(previous)


def fit_reference(model, clean_vectors, initialization_seed):
    """One original fit; no restart, rescaling, additional floor or fallback."""
    data = np.asarray(clean_vectors, dtype=np.complex128)
    if data.ndim != 2:
        raise ValueError('clean_vectors must be a two-dimensional offline array')
    with recorded_initialization(initialization_seed):
        model.fit(data)
    return model


class MFAReferenceAdapter:
    def __init__(self, model, observation_operator):
        self.model = model
        self.A = np.asarray(observation_operator, dtype=np.complex128)
        if self.A.ndim != 2:
            raise ValueError('observation_operator must be two-dimensional')

    def estimate(self, pilot_observations, noise_covariance):
        """Paper's full posterior mean through the original float1.0 API."""
        y = np.asarray(pilot_observations, dtype=np.complex128)
        if y.ndim == 1:
            y = y[None, :]
        if y.ndim != 2 or y.shape[1] != self.A.shape[0]:
            raise ValueError('pilot observations must follow the frozen selector order')
        Cn = np.asarray(noise_covariance, dtype=np.complex128)
        if Cn.shape != (self.A.shape[0], self.A.shape[0]):
            raise ValueError('noise covariance has incompatible shape')
        return self.model.estimate(y=y, Cn=Cn, A=self.A, n_summands_or_proba=1.0)


def tensor_to_vector(tensor):
    a = np.asarray(tensor)
    if a.shape[-4:] != (2, 2, 64, 16):
        raise ValueError('tensor must end in [Rx=2,Tx=2,K=64,T=16]')
    return a.reshape(a.shape[:-4] + (4096,), order='C')


def vector_to_tensor(vector):
    a = np.asarray(vector)
    if a.shape[-1] != 4096:
        raise ValueError('full vector must contain 4096 complex coordinates')
    return a.reshape(a.shape[:-1] + (2, 2, 64, 16), order='C')


class ProjectPilotInterface:
    """Fixed joint4032 model and existing owner-pilot/data geometry."""
    def __init__(self):
        from src.spatial_information import observation_geometry
        from src.link import PILOT_SYMBOLS
        g = observation_geometry()
        self.selected_full = g.selected.copy()
        self.data_full = np.flatnonzero(g.data)
        self.joint_full = np.union1d(self.selected_full, self.data_full)
        self.excluded_full = np.setdiff1d(np.arange(4096), self.joint_full)
        self.pilot_positions = np.searchsorted(self.joint_full, self.selected_full)
        self.data_positions = np.searchsorted(self.joint_full, self.data_full)
        self.A = np.zeros((64, 4032), dtype=np.complex128)
        self.A[np.arange(64), self.pilot_positions] = 1
        _, owner, _, _ = np.unravel_index(self.selected_full, (2, 2, 64, 16), order='C')
        self.owner_symbols = PILOT_SYMBOLS[owner].copy()
        assert len(self.joint_full) == 4032 and len(self.excluded_full) == 64
        assert np.allclose(abs(self.owner_symbols)**2, 1, atol=1e-12, rtol=0)

    def offline_pack(self, tensor):
        full = tensor_to_vector(tensor)
        return full[..., self.joint_full], full[..., self.excluded_full]

    def offline_unpack(self, joint, excluded_sidecar):
        joint, sidecar = np.asarray(joint), np.asarray(excluded_sidecar)
        if joint.shape[-1] != 4032 or sidecar.shape != joint.shape[:-1] + (64,):
            raise ValueError('Exact full reconstruction requires the 64-coordinate sidecar')
        full = np.empty(joint.shape[:-1] + (4096,), dtype=np.result_type(joint, sidecar))
        full[..., self.joint_full] = joint
        full[..., self.excluded_full] = sidecar
        return vector_to_tensor(full)

    def divide_owner_pilots(self, received_owner_pilot_symbols):
        received = np.asarray(received_owner_pilot_symbols, dtype=np.complex128)
        if received.shape[-1] != 64:
            raise ValueError('64 received owner-pilot entries in frozen C-order required')
        return received / self.owner_symbols

    def noise_covariance(self, noise_variance):
        if not np.isreal(noise_variance) or not np.isfinite(noise_variance) or noise_variance <= 0:
            raise ValueError('known positive source-referenced complex noise variance required')
        return float(noise_variance) * np.eye(64, dtype=np.complex128)


class ProjectMFAReceiver:
    """No target truth/covariance/label/history input is accepted online."""
    def __init__(self, model):
        self.interface = ProjectPilotInterface()
        self.adapter = MFAReferenceAdapter(model, self.interface.A)

    def estimate_data(self, received_owner_pilot_symbols, noise_variance):
        p = self.interface.divide_owner_pilots(received_owner_pilot_symbols)
        joint = self.adapter.estimate(p, self.interface.noise_covariance(noise_variance))
        return joint[:, self.interface.data_positions]
