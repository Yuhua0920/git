"""Square MIMO ZF detection with a separate channel matrix for every RE."""

import numpy as np


def zf_detect(y_grid: np.ndarray, h_grid: np.ndarray) -> np.ndarray:
    """Solve H[:,:,k,t] X[:,k,t] = Y[:,k,t] without forming an inverse.

    H[r,q,k,t] is the channel from TX q to RX r. This interface supports
    square channels only and returns the actual transmitted symbol scale.
    Singular matrices are rejected; no pseudoinverse fallback is used.
    """
    received = np.asarray(y_grid, dtype=np.complex128)
    channel = np.asarray(h_grid, dtype=np.complex128)
    if received.ndim != 3 or min(received.shape) < 1:
        raise ValueError("y_grid must have shape (nr, num_subcarriers, num_symbols)")
    if channel.ndim != 4 or min(channel.shape) < 1:
        raise ValueError("h_grid must have shape (nr, nt, num_subcarriers, num_symbols)")
    nr, nt, num_subcarriers, num_symbols = channel.shape
    if received.shape != (nr, num_subcarriers, num_symbols):
        raise ValueError("y_grid and h_grid must have matching nr, frequency, and time axes")
    if nr != nt:
        raise ValueError("zf_detect currently requires a square channel (nr == nt)")
    if not np.all(np.isfinite(received)) or not np.all(np.isfinite(channel)):
        raise ValueError("y_grid and h_grid must contain finite values")

    detected = np.empty((nt, num_subcarriers, num_symbols), dtype=np.complex128)
    for k in range(num_subcarriers):
        for t in range(num_symbols):
            try:
                detected[:, k, t] = np.linalg.solve(channel[:, :, k, t], received[:, k, t])
            except np.linalg.LinAlgError as exc:
                raise ValueError(f"singular channel at RE (k={k}, t={t})") from exc
    if not np.all(np.isfinite(detected)):
        raise ValueError("ZF solve produced nonfinite symbols")
    return detected


class LinearMMSEDetectionError(ValueError):
    """A failed D1 solve, with the actual RE and factorization counters."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


def linear_mmse_detect(
    y: np.ndarray, h_est: np.ndarray, noise_variance: float,
    stream_symbol_power: float, *, return_diagnostics: bool = False,
):
    """D1 Linear MMSE data detector, compatible with ZF's grid conventions.

    y[RX,K,T], h_est[RX,TX,K,T] -> actual scaled x_hat[TX,K,T].
    sigma2 is complex noise variance per RX branch, Ps is data stream energy,
    and lambda=sigma2/Ps. Positive-noise solves use Cholesky of H^H H+lambda I.
    Zero noise uses the existing square ZF solve (including singular failure).
    No inverse, pseudoinverse, jitter, fallback or amplitude restoration.
    """
    from scipy.linalg import cho_factor, cho_solve

    for name, value, strictly_positive in (
        ("noise_variance", noise_variance, False),
        ("stream_symbol_power", stream_symbol_power, True),
    ):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isrealobj(value) or not np.isfinite(value)
                or value < 0 or (strictly_positive and value == 0)):
            raise ValueError(f"{name} must be finite and {'positive' if strictly_positive else 'nonnegative'}")
    regularization = float(noise_variance) / float(stream_symbol_power)
    if not np.isfinite(regularization):
        raise ValueError("noise_variance/stream_symbol_power must be finite")
    received = np.asarray(y, dtype=np.complex128)
    channel = np.asarray(h_est, dtype=np.complex128)
    if received.ndim != 3 or min(received.shape) < 1:
        raise ValueError("y must have shape (nr, num_subcarriers, num_symbols)")
    if channel.ndim != 4 or min(channel.shape) < 1:
        raise ValueError("h_est must have shape (nr, nt, num_subcarriers, num_symbols)")
    nr, nt, nk, ns = channel.shape
    if received.shape != (nr, nk, ns) or nr != nt:
        raise ValueError("y/h_est must match and use a square channel (nr == nt)")
    if not np.all(np.isfinite(received)) or not np.all(np.isfinite(channel)):
        raise ValueError("y/h_est must contain finite values")
    diagnostics = {"regularization": regularization, "cholesky_successes": 0,
        "cholesky_failures": 0, "zero_noise_zf_solves": 0, "solve_failures": 0,
        "explicit_inverse_used": False, "pseudo_inverse_used": False, "jitter_used": False}
    if noise_variance == 0:
        try:
            detected = zf_detect(received, channel)
        except ValueError as exc:
            diagnostics["solve_failures"] = 1
            raise LinearMMSEDetectionError(str(exc), diagnostics) from exc
        diagnostics["zero_noise_zf_solves"] = nk * ns
    else:
        detected = np.empty((nt, nk, ns), dtype=np.complex128)
        identity = np.eye(nt, dtype=np.complex128)
        for k in range(nk):
            for t in range(ns):
                h = channel[:, :, k, t]
                hermitian = h.conj().T
                a = hermitian @ h + regularization * identity
                b = hermitian @ received[:, k, t]
                try:
                    factor = cho_factor(a, lower=True, check_finite=True)
                except (np.linalg.LinAlgError, ValueError) as exc:
                    diagnostics["cholesky_failures"] += 1
                    diagnostics["failed_RE"] = {"k": k, "t": t}
                    raise LinearMMSEDetectionError(f"Cholesky failed at RE (k={k}, t={t})", diagnostics) from exc
                diagnostics["cholesky_successes"] += 1
                try:
                    detected[:, k, t] = cho_solve(factor, b, check_finite=True)
                except (np.linalg.LinAlgError, ValueError) as exc:
                    diagnostics["solve_failures"] += 1
                    diagnostics["failed_RE"] = {"k": k, "t": t}
                    raise LinearMMSEDetectionError(f"Linear MMSE solve failed at RE (k={k}, t={t})", diagnostics) from exc
    if not np.all(np.isfinite(detected)):
        diagnostics["solve_failures"] += 1
        raise LinearMMSEDetectionError("Linear MMSE solve produced nonfinite symbols", diagnostics)
    return (detected, diagnostics) if return_diagnostics else detected
