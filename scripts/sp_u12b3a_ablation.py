"""Fixed P-only/M-only ablation arms; the Both fitter is reused unchanged.

No condition, truth, file access, arm selection, regularization or fallback API.
"""
from dataclasses import dataclass
import numpy as np
from src.pilot_cv import deployment
from src.lowrank_modules import factored_affine_predict

ARMS = ('P-only', 'M-only', 'Both')

class TechnicalSolveFailure(RuntimeError):
    pass

def solve_scalar(gram, rhs):
    """One real coordinate: b/S; exact zero is a technical failure."""
    if not np.isfinite([gram, rhs]).all():
        raise TechnicalSolveFailure('NONFINITE_SCALAR_NORMAL_EQUATION')
    if gram <= 0:
        raise TechnicalSolveFailure('ZERO_SCALAR_EXCITATION')
    coefficient = float(rhs / gram)
    if not np.isfinite(coefficient):
        raise TechnicalSolveFailure('NONFINITE_SCALAR_COORDINATE')
    return np.array([coefficient], dtype=np.float64)

@dataclass(frozen=True)
class SingleDirectionCV:
    cv: object
    direction: str

    def __post_init__(self):
        if self.direction not in ('P', 'M'):
            raise ValueError('EXPLICIT_SINGLE_DIRECTION_REQUIRED')

    def frame_objective(self, pilots):
        p = np.asarray(pilots, dtype=np.complex128)
        if p.shape != (64,) or not np.isfinite(p).all():
            raise ValueError('FINITE_64_PILOTS_REQUIRED')
        u = p - self.cv.source_mean_p
        kernels = (self.cv.profile_delta_k if self.direction == 'P'
                   else self.cv.mobility_delta_k)
        columns, responses = [], []
        for f, (observed, held) in enumerate(self.cv.folds):
            x = u[observed]
            columns.append((kernels[f] @ x)[:, None])
            responses.append(u[held] - self.cv.source_k[f] @ x)
        design = np.concatenate(columns)
        response = np.concatenate(responses)
        return design.conj().T @ design, design.conj().T @ response

    def fit(self, pilot_frames):
        x = np.asarray(pilot_frames, dtype=np.complex128)
        if x.ndim != 2 or x.shape[1] != 64 or not 1 <= len(x) <= 5:
            raise ValueError('ONE_TO_FIVE_CAUSAL_FRAMES_REQUIRED')
        moments = [self.frame_objective(p) for p in x]
        gram = sum(v[0] for v in moments).real
        rhs = sum(v[1] for v in moments).real
        coefficient = solve_scalar(float(gram[0, 0]), float(rhs[0]))
        residual = float(np.linalg.norm(gram @ coefficient - rhs))
        denominator = float(np.linalg.norm(rhs))
        relative = residual / denominator if denominator else residual
        return coefficient, dict(gram=gram.tolist(), rhs=rhs.tolist(),
            normal_equation_relative_residual=relative, window_frames=len(x),
            real_ls_size=1, clipped=False, regularized=False,
            fallback=False, fit_direction=self.direction)

def deployed_arms(pilots, arrays, snr, cv, profile_factor, mobility_factor):
    """Declared three-arm contrasts over a frame stream; no online model selection."""
    p = np.asarray(pilots, dtype=np.complex128)
    if p.ndim != 2 or p.shape[1] != 64 or not np.isfinite(p).all():
        raise ValueError('FINITE_PILOT_STREAM_REQUIRED')
    if profile_factor.rank != 4 or mobility_factor.rank != 4:
        raise ValueError('FROZEN_RANK_FOUR_REQUIRED')
    single = {m: SingleDirectionCV(cv, m) for m in ('P', 'M')}
    result = []
    for j in range(4, len(p)):
        window = p[j-4:j+1]
        args = (p[j], arrays['source_mean_p'], arrays['source_mean_d'],
                arrays[f'Wsource_{snr}'])
        u = p[j] - arrays['source_mean_p']
        base = arrays['source_mean_d'] + u @ arrays[f'Wsource_{snr}'].T
        arms = {}
        for arm, m, factor in (('P-only', 'P', profile_factor),
                               ('M-only', 'M', mobility_factor)):
            coefficient, diagnostics = single[m].fit(window)
            dense = base + coefficient[0] * (u @ arrays[f'DeltaW{m}_{snr}'].T)
            lowrank = base + coefficient[0] * factor.apply(u)
            arms[arm] = dict(coefficients=coefficient, dense_coefficients=coefficient,
                lr4_coefficients=coefficient, predictions={'Dense': dense, 'LR4': lowrank},
                diagnostics=diagnostics)
        # This is the original two-dimensional solve and original deployment order.
        coefficient, diagnostics = cv.fit(window, modules=('P', 'M'))
        arms['Both'] = dict(coefficients=coefficient, dense_coefficients=coefficient,
            lr4_coefficients=coefficient, predictions={
                'Dense': deployment(*args, arrays[f'DeltaWP_{snr}'],
                    arrays[f'DeltaWM_{snr}'], coefficient, ('P', 'M')),
                'LR4': factored_affine_predict(*args, coefficient, profile_factor, mobility_factor)},
            diagnostics=diagnostics)
        if any(not np.isfinite(v).all() for arm in arms.values()
               for v in arm['predictions'].values()):
            raise ValueError('NONFINITE_ABLATION_PREDICTION')
        result.append(dict(frame_id=j, window_start=j-4, arms=arms))
    return result

def coordinate_fields(arm, coefficient):
    if arm == 'P-only':
        return dict(coordinate_p=float(coefficient[0]), coordinate_m=None)
    if arm == 'M-only':
        return dict(coordinate_p=None, coordinate_m=float(coefficient[0]))
    if arm == 'Both':
        return dict(coordinate_p=float(coefficient[0]), coordinate_m=float(coefficient[1]))
    raise ValueError('UNDECLARED_ARM')
