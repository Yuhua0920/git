"""T16D oracle module representation mathematics; no predictor or channel calls."""
from dataclasses import dataclass
import numpy as np
from src.lowrank_filter import readonly

T16C_COMMIT = "d93416234cae60f86b4fae26c3be8a576ff0513d"
RANK_TOLERANCE = 1e-12
BUDGETS = (8, 16, 24)
EXPERIMENT_TYPE = "modular composition gate for profile- and mobility-induced low-rank LMMSE mismatch corrections"


def orthonormal_union(profile, mobility, per_module):
    """SVD orthonormalization of development-only D=[U_P,k,U_M,k]."""
    p, m = map(lambda a: np.asarray(a, np.complex128), (profile, mobility))
    if per_module not in (4, 8, 12) or p.ndim != 2 or p.shape != m.shape or p.shape[1] < per_module:
        raise ValueError("equal development basis shapes and declared module ranks required")
    if not np.isfinite(p).all() or not np.isfinite(m).all():
        raise ValueError("finite development bases required")
    d = np.concatenate((p[:, :per_module], m[:, :per_module]), axis=1)
    u, s, _ = np.linalg.svd(d, full_matrices=False)
    rank = int(np.count_nonzero(s > RANK_TOLERANCE*s[0])) if s[0] else 0
    if rank == 0:
        raise ValueError("empty union span")
    q = readonly(u[:, :rank])
    residual = float(np.linalg.norm(q.conj().T@q-np.eye(rank)))
    if residual > 1e-11:
        raise ValueError("union orthonormality failed")
    s.flags.writeable = False
    return q, dict(dimension_budget=2*per_module, per_module_vectors=per_module,
                   numerical_rank=rank, relative_rank_tolerance=RANK_TOLERANCE,
                   singular_values=s.tolist(), orthonormality_residual=residual,
                   dictionary_relative_projection_residual=float(np.linalg.norm(d-q@(q.conj().T@d))/np.linalg.norm(d)))


def frozen_module_unions(bases):
    if bases["CDL-B-30"].scope != "CDL-B-30" or bases["CDL-C-120"].scope != "CDL-C-120":
        raise ValueError("only frozen B30/C120 development module bases allowed")
    p, m = bases["CDL-B-30"].u32, bases["CDL-C-120"].u32
    return {budget: orthonormal_union(p, m, budget//2) for budget in BUDGETS}


def weighted_projection(delta, lower, q):
    delta, lower, q = map(lambda a: np.asarray(a, np.complex128), (delta, lower, q))
    if lower.shape != (delta.shape[1], delta.shape[1]) or q.shape[0] != delta.shape[0]:
        raise ValueError("weighted projection dimensions disagree")
    if not np.allclose(q.conj().T@q, np.eye(q.shape[1]), rtol=1e-11, atol=1e-11):
        raise ValueError("orthonormal output basis required")
    g = readonly(q.conj().T@delta)
    matrix = delta@lower
    energy = float(np.vdot(matrix, matrix).real)
    residual = (delta-q@g)@lower
    excess = float(np.vdot(residual, residual).real)
    return g, dict(capture=float(np.linalg.norm(q.conj().T@matrix)**2/energy) if energy else None,
                   weighted_residual=excess, correction_energy_per_frame=energy,
                   weighted_residual_fraction=excess/energy if energy else None,
                   units="summed data-domain error energy per frame")


@dataclass(frozen=True)
class ScalarComposition:
    coefficients: np.ndarray
    correction: np.ndarray
    residual: np.ndarray
    diagnostics: dict

    @classmethod
    def fit(cls, target_delta, module_deltas, lower):
        """Unconstrained REAL one/two-mode target-weighted normal equations."""
        if len(module_deltas) not in (1, 2):
            raise ValueError("one or two real modes required")
        target = np.asarray(target_delta, np.complex128)
        modules = [np.asarray(a, np.complex128) for a in module_deltas]
        if any(a.shape != target.shape for a in modules) or not all(np.isfinite(a).all() for a in [target, *modules, lower]):
            raise ValueError("finite matching operators required")
        b = target@lower
        a = [w@lower for w in modules]
        gram = np.array([[np.vdot(x, y).real for y in a] for x in a], float)
        rhs = np.array([np.vdot(x, b).real for x in a], float)
        singular = np.linalg.svd(gram, compute_uv=False)
        rank = int(np.count_nonzero(singular > RANK_TOLERANCE*singular[0])) if singular[0] else 0
        if rank != len(modules):
            raise ValueError("scalar normal equations rank deficient; no silent regularization")
        coefficients = np.linalg.solve(gram, rhs)
        correction = sum(c*w for c, w in zip(coefficients, modules))
        residual = target-correction
        weighted = residual@lower
        energy = float(np.vdot(b, b).real)
        excess = float(np.vdot(weighted, weighted).real)
        coefficients.flags.writeable = False
        return cls(coefficients, readonly(correction), readonly(residual),
                   dict(real_coefficients=True, normal_equation_gram=gram.tolist(), normal_equation_rhs=rhs.tolist(),
                        normal_equation_relative_residual=float(np.linalg.norm(gram@coefficients-rhs)/max(np.linalg.norm(rhs),np.finfo(float).tiny)),
                        gram_numerical_rank=rank, gram_condition_number=float(singular[0]/singular[-1]),
                        weighted_residual=excess, weighted_residual_fraction=excess/energy if energy else None,
                        correction_energy_per_frame=energy, oracle_target_operator_used=True,
                        deployable=False, units="summed data-domain error energy per frame"))


def interaction_spectrum(residual, lower):
    singular = np.linalg.svd(residual@lower, compute_uv=False)
    energy = singular**2
    total = float(energy.sum())
    p = energy/total if total else np.zeros_like(energy)
    effective = float(np.exp(-np.sum(p[p > 0]*np.log(p[p > 0])))) if total else 0.
    numerical = int(np.count_nonzero(singular > RANK_TOLERANCE*singular[0])) if singular[0] else 0
    return dict(singular_values=singular.tolist(), effective_rank=effective, numerical_rank=numerical,
                relative_rank_tolerance=RANK_TOLERANCE, weighted_residual_energy_per_frame=total,
                effective_rank_definition="exp(entropy of normalized squared singular values)")


def scientific_gate(union, scalar):
    focus = [r for r in union if r["snr_db"] in (10, 20)]
    primary = [r for r in scalar if r["snr_db"] in (10, 20)]
    expected = {(name, s, q) for name in ("CDL-C-75", "CDL-B-120") for s in (10, 20) for q in BUDGETS}
    if len(focus) != 12 or {(r["shift"],r["snr_db"],r["dimension_budget"]) for r in focus} != expected:
        raise ValueError("complete actual union validation required")
    if len(primary) != 4 or {(r["shift"],r["snr_db"]) for r in primary} != {(n,s) for n in ("CDL-C-75","CDL-B-120") for s in (10,20)}:
        raise ValueError("complete actual scalar validation required")
    if any(r.get("metric_type") != "actual_validation" or r["closure"] is None for r in [*focus, *primary]):
        raise ValueError("gates require actual finite-gap validation, never theoretical anchor")
    passes = lambda q: all(r["closure"] >= .8 for r in focus if r["dimension_budget"] == q)
    promising = all(r["closure"] >= .8 for r in primary)
    insufficient = any(r["closure"] < .5 for r in primary)
    return dict(modular_union_q8_strong=passes(8), modular_union_q16_strong=not passes(8) and passes(16),
                modular_union_weak=not passes(24), scalar_mode_composition_promising=promising,
                scalar_mode_composition_insufficient=insufficient, scalar_mode_composition_partial=not promising and not insufficient)
