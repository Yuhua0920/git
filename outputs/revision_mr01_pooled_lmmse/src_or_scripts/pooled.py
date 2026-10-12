"""Population-covariance pooling for SP-U17, using existing frozen-statistics APIs.

Each input provides mean_p[P], mean_d[D], rpp[P,P], rdp[D,P], samples=N
and train_trajectory_ids. Covariances must use their population denominator N.
For a = N_i / sum_j N_j, this module applies:

  mu_p = sum_i a_i mu_p_i,  mu_d = sum_i a_i mu_d_i
  R_pp = sum_i a_i [R_pp_i + (mu_p_i-mu_p)(mu_p_i-mu_p)^H]
  R_dp = sum_i a_i [R_dp_i + (mu_d_i-mu_d)(mu_p_i-mu_p)^H]

Trajectory IDs are provenance only. They neither determine weights nor imply
independent frame samples. No generator, cache reader or scientific run is used.
The caller establishes its existing project root on sys.path before importing.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np

from src.cdl_benchmark import FrozenStatistics


def _array(value, name, origin):
    array = np.asarray(value)
    if array.dtype.kind not in "biufc":
        raise ValueError(f"origin {origin} {name} must be numeric")
    array = np.asarray(array, dtype=np.complex128)
    if not np.isfinite(array).all():
        raise ValueError(f"origin {origin} {name} must contain finite values")
    return array


def pool_statistics(sources) -> FrozenStatistics:
    """Pool centered sufficient statistics using sample-count weights.

    Inputs may be FrozenStatistics or compatible objects with the same fields.
    All outputs are fresh, read-only complex128 arrays. Provenance IDs are
    (origin_position, original_trajectory_id), preserving origin collisions and
    order. Their number is intentionally unrelated to .samples.
    """
    sources = tuple(sources)
    if not sources:
        raise ValueError("at least one origin is required")
    validated = []
    dimensions = None
    total = 0
    ids = []
    for origin, source in enumerate(sources):
        count = source.samples
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or count <= 0:
            raise ValueError(f"origin {origin} samples must be a positive integer")
        count = int(count)
        mp, md, pp, dp = (
            _array(getattr(source, name), name, origin)
            for name in ("mean_p", "mean_d", "rpp", "rdp")
        )
        if mp.ndim != 1 or md.ndim != 1 or mp.size == 0 or md.size == 0:
            raise ValueError(f"origin {origin} means must be nonempty one-dimensional vectors")
        if pp.shape != (mp.size, mp.size) or dp.shape != (md.size, mp.size):
            raise ValueError(f"origin {origin} covariance dimensions must match its means")
        if dimensions is None:
            dimensions = (mp.size, md.size)
        elif dimensions != (mp.size, md.size):
            raise ValueError("all origins must have the same pilot and data dimensions")
        validated.append((count, mp, md, pp, dp))
        total += count
        ids.extend((origin, item) for item in source.train_trajectory_ids)
    weights = [count / total for count, *_ in validated]
    npilot, ndata = dimensions
    mean_p = np.zeros(npilot, dtype=np.complex128)
    mean_d = np.zeros(ndata, dtype=np.complex128)
    for weight, (_, mp, md, _, _) in zip(weights, validated):
        mean_p += weight * mp
        mean_d += weight * md
    rpp = np.zeros((npilot, npilot), dtype=np.complex128)
    rdp = np.zeros((ndata, npilot), dtype=np.complex128)
    for weight, (_, mp, md, pp, dp) in zip(weights, validated):
        delta_p, delta_d = mp - mean_p, md - mean_d
        rpp += weight * (pp + np.outer(delta_p, delta_p.conj()))
        rdp += weight * (dp + np.outer(delta_d, delta_p.conj()))
    arrays = (mean_p, mean_d, rpp, rdp)
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("pooled statistics are nonfinite after accumulation")
    for array in arrays:
        array.setflags(write=False)
    return FrozenStatistics(*arrays, tuple(ids), total)
