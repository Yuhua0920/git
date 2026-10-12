"""T16F pilot-only candidate selection and exact two-variable simplex LS."""
import numpy as np

CANDIDATE_ORDER = ("W0", "WP", "WM")
SIMPLEX_CANDIDATE_ORDER = (
    "vertex_W0", "vertex_WP", "vertex_WM", "edge_alpha_zero",
    "edge_beta_zero", "edge_sum_one", "interior",
)


def objective_moments(cv, pilot_frames):
    """Preserve the frozen PilotCV.fit frame and reduction order exactly."""
    x = np.asarray(pilot_frames, dtype=np.complex128)
    if x.ndim != 2 or x.shape[1] != 64 or not 1 <= len(x) <= 5:
        raise ValueError("one to five causal pilot frames required")
    moments = [cv.frame_objective(p) for p in x]
    gram = sum(v[0] for v in moments).real
    rhs = sum(v[1] for v in moments).real
    baseline = sum(v[2] for v in moments)
    return _validated_moments(gram, rhs, baseline)


def _validated_moments(gram, rhs, baseline):
    """Audit PSD input; tolerance admits only floating Gram roundoff, no ridge."""
    g = np.asarray(gram)
    b = np.asarray(rhs)
    if np.iscomplexobj(g) or np.iscomplexobj(b):
        raise ValueError("real two-variable objective required")
    g = np.asarray(g, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = float(baseline)
    if g.shape != (2, 2) or b.shape != (2,) or not np.isfinite(g).all() or not np.isfinite(b).all() or not np.isfinite(c):
        raise ValueError("finite 2-by-2 Gram, two rhs values and baseline required")
    scale = max(float(np.max(np.abs(g))), np.finfo(float).tiny)
    tolerance = 32 * np.finfo(float).eps * scale
    if np.max(np.abs(g - g.T)) > tolerance:
        raise ValueError("symmetric real Gram required")
    if np.linalg.eigvalsh(g).min() < -tolerance:
        raise ValueError("positive semidefinite real Gram required")
    return g, b, c


def objective_value(coefficients, gram, rhs, baseline):
    a = np.asarray(coefficients, dtype=np.float64)
    if a.shape != (2,) or not np.isfinite(a).all():
        raise ValueError("two finite real coefficients required")
    value = float(baseline - 2 * a @ rhs + a @ gram @ a)
    if not np.isfinite(value):
        raise ValueError("finite objective required")
    return value


def geometry(coefficients):
    """Strict signs for outside fraction; exact positive-weight boundary class."""
    a = np.asarray(coefficients, dtype=np.float64)
    if a.shape != (2,) or not np.isfinite(a).all():
        raise ValueError("two finite real coefficients required")
    alpha, beta = map(float, a)
    weights = (1.0 - alpha - beta, alpha, beta)
    if not np.isfinite(weights).all():
        raise ValueError("finite candidate weights required")
    outside = tuple(v < 0 for v in weights)
    positives = sum(v > 0 for v in weights)
    kind = "outside" if any(outside) else {1: "vertex", 2: "edge", 3: "interior"}[positives]
    return dict(alpha=alpha, beta=beta, lambda0=weights[0], lambdaP=alpha,
        lambdaM=beta, outside_lambda0=outside[0], outside_alpha=outside[1],
        outside_beta=outside[2], outside_any=any(outside), active_kind=kind)


def hard_selection(gram, rhs, baseline):
    """Compare all three candidate CV losses; exact ties prefer W0, WP, WM."""
    g, b, c = _validated_moments(gram, rhs, baseline)
    candidates = (np.array([0., 0.]), np.array([1., 0.]), np.array([0., 1.]))
    losses = [objective_value(a, g, b, c) for a in candidates]
    selected = min(range(3), key=lambda j: losses[j])
    coefficients = candidates[selected].copy()
    return coefficients, dict(selected_candidate=CANDIDATE_ORDER[selected],
        selected_index=selected, candidate_losses=dict(zip(CANDIDATE_ORDER, losses)),
        fitted_cv_loss=losses[selected], source_cv_loss=c,
        tie_order=list(CANDIDATE_ORDER), geometry=geometry(coefficients))


def convex_mixing(gram, rhs, baseline):
    """Global convex quadratic optimum by vertices, all edges and interior."""
    g, b, c = _validated_moments(gram, rhs, baseline)
    candidates = [(SIMPLEX_CANDIDATE_ORDER[0], np.array([0., 0.])),
        (SIMPLEX_CANDIDATE_ORDER[1], np.array([1., 0.])),
        (SIMPLEX_CANDIDATE_ORDER[2], np.array([0., 1.]))]
    edges = (("edge_alpha_zero", np.array([0., 0.]), np.array([0., 1.])),
             ("edge_beta_zero", np.array([0., 0.]), np.array([1., 0.])),
             ("edge_sum_one", np.array([0., 1.]), np.array([1., -1.])))
    for name, origin, direction in edges:
        curvature = float(direction @ g @ direction)
        numerator = float(direction @ (b - g @ origin))
        if curvature > 0:
            parameter = min(1.0, max(0.0, numerator / curvature))
        else:
            # A flat edge has a linear objective; endpoints are already present.
            parameter = 1.0 if numerator > 0 else 0.0
        a = origin + parameter * direction
        if name == "edge_sum_one":
            a = np.array([parameter, 1.0 - parameter])
        candidates.append((name, a))
    interior_solved = False
    try:
        a = np.linalg.solve(g, b)
    except np.linalg.LinAlgError:
        a = None
    if a is not None and np.isfinite(a).all() and not geometry(a)["outside_any"]:
        candidates.append(("interior", a))
        interior_solved = True
    losses = [objective_value(a, g, b, c) for _, a in candidates]
    selected = min(range(len(candidates)), key=lambda j: losses[j])
    name, coefficients = candidates[selected]
    info = geometry(coefficients)
    if info["outside_any"]:
        raise ValueError("active-set solution must satisfy the exact simplex")
    return coefficients.copy(), dict(selected_active_set=name,
        active_kind=info["active_kind"], fitted_cv_loss=losses[selected],
        source_cv_loss=c, candidate_count=len(candidates),
        candidate_losses=[dict(active_set=n, coefficients=a.tolist(), objective=v)
            for (n, a), v in zip(candidates, losses)],
        tie_order=list(SIMPLEX_CANDIDATE_ORDER), interior_candidate_included=interior_solved,
        regularized=False, black_box_optimizer_used=False, geometry=info)


def gain(hard_db, convex_db, affine_db):
    values = np.asarray([hard_db, convex_db, affine_db], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("finite actual NMSE values required")
    return dict(affine_vs_hard_gain_db=float(values[0] - values[2]),
                affine_vs_convex_gain_db=float(values[1] - values[2]))


def scientific_gate(gain_by_condition, outside_fraction_by_condition):
    """The task's fixed five-frame gate; no inferred or post-hoc thresholds."""
    expected = {(n, s) for n in ("CDL-C-75", "CDL-B-120") for s in (10, 20)}
    if set(gain_by_condition) != expected:
        raise ValueError("all four primary five-frame conditions required")
    keys = ("affine_vs_hard_gain_db", "affine_vs_convex_gain_db")
    if any(not np.isfinite([row[k] for k in keys]).all() for row in gain_by_condition.values()):
        raise ValueError("finite primary gains required")
    composed = [("CDL-B-120", s) for s in (10, 20)]
    if not all(k in outside_fraction_by_condition for k in composed):
        raise ValueError("both B120 five-frame outside fractions required")
    fractions = [float(outside_fraction_by_condition[k]) for k in composed]
    if not np.isfinite(fractions).all() or any(v < 0 or v > 1 for v in fractions):
        raise ValueError("finite outside fractions in [0,1] required")
    convex = [gain_by_condition[k][keys[1]] for k in composed]
    c75 = [gain_by_condition[("CDL-C-75", s)][keys[1]] for s in (10, 20)]
    material = all(v >= 1.0 for v in convex) and all(v >= -.25 for v in c75)
    weak = all(v < .25 for v in convex)
    return dict(affine_composition_value_material=material,
        affine_composition_value_weak=weak,
        affine_composition_value_partial=not material and not weak,
        continuous_composition_beats_discrete_selection=all(gain_by_condition[k][keys[0]] >= 1.0 for k in composed),
        composed_shift_requires_extrapolation_empirically=all(v > .75 for v in fractions))
