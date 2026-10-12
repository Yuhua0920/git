"""MR-02 diagnostic real-coordinate least squares and exact risk identities.

No deployment fallback, covariance fitting, channel generation, or outcome
selection is implemented here. Truth-assisted functions belong to diagnostics.
"""
from __future__ import annotations
import math
import numpy as np

RCOND = 1e-12
ATOL = 1e-10
RTOL = 1e-10
EPS = np.finfo(np.float64).eps


def _finite(a, label):
    a = np.asarray(a)
    if not np.isfinite(a).all():
        raise ValueError("NONFINITE_" + label)
    return a


def _real(a, label):
    a = _finite(a, label)
    if np.iscomplexobj(a):
        raise ValueError("REAL_" + label + "_REQUIRED")
    return np.asarray(a, dtype=np.float64)


def _design(X, r):
    X = np.asarray(_finite(X, "DESIGN"), dtype=np.complex128)
    r = np.asarray(_finite(r, "RESPONSE"), dtype=np.complex128)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0 or r.shape != (X.shape[0],):
        raise ValueError("DESIGN_RESPONSE_SHAPE")
    return X, r


def real_sufficient(X, r):
    """For real c: ||Xc-r||²=k-2cᵀb+cᵀHc, with complex X and r."""
    X, r = _design(X, r)
    H = np.asarray((X.conj().T @ X).real, dtype=np.float64)
    b = np.asarray((X.conj().T @ r).real, dtype=np.float64)
    k = float(np.vdot(r, r).real)
    return H, b, k


def quadratic_risk(H, b, k, c):
    """Scalar or leading-batch quadratic; signed terms are summed accurately."""
    H = _real(H, "GRAM")
    b = _real(b, "RHS")
    c = _real(c, "COORDINATE")
    if H.ndim != 2 or H.shape[0] != H.shape[1] or b.shape != (H.shape[0],) or c.shape[-1:] != b.shape:
        raise ValueError("QUADRATIC_SHAPE")
    if not np.isfinite(k):
        raise ValueError("NONFINITE_CONSTANT")
    if c.ndim == 1:
        # Combine all signed scalar terms rather than rounding partial groups.
        return math.fsum([float(k)] + [float(-2*c[i]*b[i]) for i in range(len(b))]
                         + [float(c[i]*H[i,j]*c[j]) for i in range(len(b)) for j in range(len(b))])
    return np.asarray([quadratic_risk(H,b,k,v) for v in c.reshape(-1,len(b))]).reshape(c.shape[:-1])


def _normal_residual(H, b, c):
    return float(np.linalg.norm(H @ c-b) / max(np.linalg.norm(H @ c), np.linalg.norm(b), np.finfo(float).tiny))


def gram_diagnostics(H, rcond=RCOND):
    """A real Gram has a separate roundoff-resolution floor after squaring.

    rcond² is the intended squared design cutoff. A Gram cannot reliably
    resolve eigenvalues below 64*eps times its scale, even if the direct design
    SVD can; both cutoffs are returned rather than silently calling them equal.
    """
    H = _real(H, "GRAM")
    if H.ndim != 2 or H.shape[0] != H.shape[1] or H.shape[0] == 0:
        raise ValueError("GRAM_SHAPE")
    if not 0 < rcond < 1:
        raise ValueError("RCOND_RANGE")
    scale = max(float(np.linalg.norm(H, ord=2)), np.finfo(float).tiny)
    if np.linalg.norm(H-H.T) > 64*EPS*scale:
        raise ValueError("NONSYMMETRIC_GRAM")
    eig = np.linalg.eigvalsh((H+H.T)/2)
    roundoff = 64*EPS*scale
    if eig[0] < -roundoff:
        raise ValueError("NON_PSD_GRAM")
    eig = np.maximum(eig, 0.)
    largest = float(eig[-1])
    cutoff = max(rcond*rcond*largest, roundoff) if largest else 0.
    rank = int(np.count_nonzero(eig > cutoff))
    full = rank == len(eig)
    return dict(rank=rank, eigenvalues=eig, min_normalized_eigenvalue=float(eig[0]/largest) if largest else 0.,
                minimum_normalized_eigenvalue=float(eig[0]/largest) if largest else 0.,
                condition=float(largest/eig[0]) if full else float('inf'), rcond=float(rcond),
                design_squared_cutoff=rcond*rcond*largest, gram_roundoff_floor=roundoff,
                rank_cutoff=cutoff, rank_rule='lambda > max(rcond^2*lambda_max,64*eps*||H||2)')


def fixed_reference(Hsum, bsum, rcond=RCOND):
    """Finite empirical fixed reference; diagnostic minnorm iff not identified."""
    H = _real(Hsum, "GRAM")
    b = _real(bsum, "RHS")
    if b.shape != (H.shape[0],):
        raise ValueError("REFERENCE_SHAPE")
    diag = gram_diagnostics(H, rcond)
    if diag['rank'] == len(b):
        coeff = np.linalg.solve(H, b)
        flag = 'IDENTIFIED_DIRECT_SOLVE'
    else:
        eig, V = np.linalg.eigh((H+H.T)/2)
        keep = eig > diag['rank_cutoff']
        # Nullspace RHS consistency matters: not every arbitrary normal pair
        # corresponds to a valid least-squares objective.
        projection = V[:,keep] @ (V[:,keep].T @ b)
        if np.linalg.norm(b-projection) > ATOL + RTOL*max(np.linalg.norm(b),1.):
            raise ValueError("INCONSISTENT_NORMAL_RHS")
        coeff = V[:,keep] @ ((V[:,keep].T @ b)/eig[keep]) if keep.any() else np.zeros_like(b)
        flag = 'NONUNIQUE_DIAGNOSTIC_MINIMUM_NORM'
    return dict(coeff=coeff, rank=diag['rank'], eigenvalues=diag['eigenvalues'],
                normal_residual=_normal_residual(H,b,coeff), solve_flag=flag, **{k:v for k,v in diag.items() if k not in ('rank','eigenvalues')})


def real_oracle(X, r, rcond=RCOND):
    """Direct real-stacked SVD oracle in this observed output space only."""
    X, r = _design(X,r)
    if not 0 < rcond < 1:
        raise ValueError("RCOND_RANGE")
    A = np.concatenate((X.real, X.imag), axis=0)
    y = np.concatenate((r.real, r.imag))
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    cutoff = float(rcond*s[0]) if len(s) else 0.
    keep = s > cutoff
    coeff = Vh[keep].T @ ((U[:,keep].T @ y)/s[keep]) if keep.any() else np.zeros(X.shape[1])
    error_vector = X @ coeff-r
    error = float(np.vdot(error_vector,error_vector).real)
    H,b,k = real_sufficient(X,r)
    formula = quadratic_risk(H,b,k,coeff)
    scale = max(k, float(np.linalg.norm(X@coeff)**2), error, 1.)
    if abs(error-formula) > ATOL + RTOL*scale:
        raise ValueError("ORACLE_DIRECT_QUADRATIC_MISMATCH")
    return dict(coeff=coeff, rank=int(np.count_nonzero(keep)), singular_values=s,
                error=error, direct_residual_norm=float(np.linalg.norm(error_vector)),
                normal_residual=_normal_residual(H,b,coeff), rcond=float(rcond),
                rank_cutoff=cutoff, retained_subspace_only=bool(np.any(~keep)),
                direct_quadratic_residual=float(error-formula))


def assert_oracle_bound(oracle, X, r, candidates, atol=ATOL, rtol=RTOL):
    """Do not clip an unfavorable result. Any unexplained violation raises."""
    X,r = _design(X,r)
    candidates = _real(candidates, "CANDIDATES")
    if candidates.ndim == 1:
        candidates = candidates[None,:]
    if candidates.ndim != 2 or candidates.shape[1] != X.shape[1]:
        raise ValueError("CANDIDATES_SHAPE")
    max_violation = -float('inf')
    for coeff in candidates:
        e = X@coeff-r
        error = float(np.vdot(e,e).real)
        violation = float(oracle['error']-error)
        scale = max(error, oracle['error'], float(np.vdot(r,r).real), float(np.linalg.norm(X@coeff)**2), 1.)
        if violation > atol + rtol*scale:
            raise ValueError("ORACLE_INEQUALITY_FAILURE")
        max_violation = max(max_violation,violation)
    return float(max_violation)


def risk_transfer(H, b, cp, cd):
    """Eval risk difference includes the empirical cd's nonzero eval gradient."""
    H = _real(H, "GRAM")
    b = _real(b, "RHS")
    cp = _real(cp, "COORDINATE")
    cd = _real(cd, "COORDINATE")
    if cp.shape != b.shape or cd.shape != b.shape or H.shape != (len(b),len(b)):
        raise ValueError("TRANSFER_SHAPE")
    delta = cp-cd
    distance = float(delta @ H @ delta)
    gradient = float(2*delta @ (H@cd-b))
    exactdiff = math.fsum([quadratic_risk(H,b,0.,cp), -quadratic_risk(H,b,0.,cd)])
    residual = math.fsum([exactdiff,-distance,-gradient])
    scale = max(abs(exactdiff),abs(distance),abs(gradient),1.)
    if abs(residual) > ATOL + RTOL*scale:
        raise ValueError("TRANSFER_IDENTITY_FAILURE")
    return dict(distance=distance, gradient=gradient, exactdiff=exactdiff, residual=residual,
                gradient_vector=2*(H@cd-b))


def compression_terms(eD, delta):
    eD = np.asarray(_finite(eD,"DENSE_ERROR"),dtype=np.complex128)
    delta = np.asarray(_finite(delta,"COMPRESSION_DELTA"),dtype=np.complex128)
    if eD.ndim != 1 or delta.shape != eD.shape:
        raise ValueError("COMPRESSION_SHAPE")
    A = float(np.vdot(delta,delta).real)
    B = float(2*np.vdot(eD,delta).real)
    dense_error = float(np.vdot(eD,eD).real)
    lr4_error = float(np.vdot(eD+delta,eD+delta).real)
    total = math.fsum([A,B])
    difference = lr4_error-dense_error
    residual = math.fsum([difference,-A,-B])
    scale = max(A,abs(B),dense_error,lr4_error,1.)
    if abs(residual) > ATOL + RTOL*scale:
        raise ValueError("COMPRESSION_IDENTITY_FAILURE")
    return dict(A=A,B=B,total=total,difference=difference,dense_error=dense_error,
                lr4_error=lr4_error,identity_residual=residual,identity_scale=scale)


def joint_gram(H_D, H_L, cross):
    """Gram of [X_D X_L], useful as an algebra check; no deployed new method."""
    H_D = _real(H_D,"GRAM")
    H_L = _real(H_L,"GRAM")
    cross = _real(cross,"CROSS_GRAM")
    if H_D.shape != H_L.shape or cross.shape != H_D.shape:
        raise ValueError("JOINT_GRAM_SHAPE")
    return np.block([[H_D,cross],[cross.T,H_L]])


def streaming_one_frame(g, target, X_D, X_L, online, S, b, pilot_k=None):
    """Calculate and check vectors now, retain only small exact sufficient data.

    Space axis is Dense then LR4. Fixed references are evaluated later using
    quadratic_risk and their oracle bounds checked from saved sufficient risks.
    """
    g = np.asarray(_finite(g,"BASE"),dtype=np.complex128)
    target = np.asarray(_finite(target,"TRUTH"),dtype=np.complex128)
    if g.ndim != 1 or target.shape != g.shape:
        raise ValueError("BASE_TRUTH_SHAPE")
    online = _real(online,"COORDINATE")
    if online.shape != (2,):
        raise ValueError("TWO_REAL_COORDINATES_REQUIRED")
    S = _real(S,"PILOT_GRAM")
    b = _real(b,"PILOT_RHS")
    if S.shape != (2,2) or b.shape != (2,):
        raise ValueError("PILOT_NORMAL_SHAPE")
    r = target-g
    Xs = [_design(X_D,r)[0],_design(X_L,r)[0]]
    if any(X.shape != (len(g),2) for X in Xs):
        raise ValueError("TWO_DATA_SPACE_SHAPE")
    Hs=[];bs=[];oracles=[];online_errors=[];quadratic_residual=[];bound=[]
    for X in Xs:
        H,bb,k = real_sufficient(X,r)
        e = g + X@online-target
        direct = float(np.vdot(e,e).real)
        formula = quadratic_risk(H,bb,k,online)
        # Coefficient terms can cancel; use contribution-scaled tolerance.
        scale = max(k,abs(float(2*online@bb)),abs(float(online@H@online)),direct,1.)
        if abs(direct-formula) > ATOL + RTOL*scale:
            raise ValueError("STREAMING_DIRECT_QUADRATIC_FAILURE")
        oracle = real_oracle(X,r)
        bound.append(assert_oracle_bound(oracle,X,r,np.stack((online,np.zeros(2),np.array([1.,0.]),np.array([0.,1.])))))
        Hs.append(H);bs.append(bb);oracles.append(oracle);online_errors.append(direct);quadratic_residual.append(direct-formula)
    cross = np.asarray((Xs[0].conj().T@Xs[1]).real,dtype=np.float64)
    comp = compression_terms(g+Xs[0]@online-target,(Xs[1]-Xs[0])@online)
    if pilot_k is not None:
        pilot_loss = quadratic_risk(S,b,float(pilot_k),online)
    else:
        pilot_loss = None
    return dict(data_gram=np.stack(Hs),data_rhs=np.stack(bs),data_constant=float(np.vdot(r,r).real),
                cross_gram=cross,oracle_coordinates=np.stack([v['coeff'] for v in oracles]),
                oracle_rank=np.array([v['rank'] for v in oracles],dtype=np.int64),
                oracle_singular_values=np.stack([v['singular_values'] for v in oracles]),
                oracle_error=np.array([v['error'] for v in oracles]),
                oracle_normal_residual=np.array([v['normal_residual'] for v in oracles]),
                oracle_bound_max_violation=np.array(bound), online_error=np.array(online_errors),
                source_error=float(np.vdot(r,r).real),pilot_loss=pilot_loss,
                compression_A=comp['A'],compression_B=comp['B'],compression_difference=comp['difference'],
                audit=dict(direct_quadratic_max_abs=float(np.max(np.abs(quadratic_residual))),
                           compression_identity_residual=comp['identity_residual'],
                           compression_identity_scale=comp['identity_scale']))
