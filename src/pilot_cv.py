"""T16E pilot-only analytical coordinates; no channel/oracle/shift inputs."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from scipy.linalg import cho_factor, cho_solve

T16D_COMMIT = "ee6cc6034e5f8c70edae7533c5a07f55bac65053"
EXPERIMENT_TYPE = "pilot-only analytical identifiability gate for modular LMMSE mismatch-correction coordinates under sparse pilots"


def frozen_folds(selected):
    selected = np.asarray(selected, dtype=np.int64)
    if selected.shape != (64,) or len(set(selected)) != 64:
        raise ValueError("64 unique actual pilot coefficients required")
    rx, tx, k, t = np.unravel_index(selected, (2, 2, 64, 16))
    labels = (k//16+t//4) % 4
    if any(len(set(k[(rx==r)&(tx==a)])) != 4 for r in range(2) for a in range(2)):
        raise ValueError("frozen dF16 geometry required")
    folds = []
    for f in range(4):
        hold = np.flatnonzero(labels == f)
        observed = np.flatnonzero(labels != f)
        if len(hold) != 16 or len(observed) != 48:
            raise ValueError("four balanced folds required")
        for r in range(2):
            for a in range(2):
                subset = hold[(rx[hold]==r)&(tx[hold]==a)]
                if len(subset)!=4 or len(set(k[subset]))!=4 or len(set(t[subset]))!=4:
                    raise ValueError("each fold must cover all links/frequencies/times")
        hold.flags.writeable = observed.flags.writeable = False
        folds.append((observed, hold))
    definition = dict(rule="fold=(physical_k//16+symbol_t//4)%4", order="actual C-order [rx,tx,k,t] pilot selection",
        selected=selected.tolist(), labels=labels.tolist(),
        folds=[dict(observed=o.tolist(), heldout=h.tolist()) for o,h in folds])
    definition["sha256"] = hashlib.sha256(json.dumps(definition, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return tuple(folds), definition


def cross_pilot_operators(rpp, sigma2, folds):
    """K_f=R_HO (R_OO+sigma2 I)^-1, solved by Cholesky adjoints."""
    rpp = np.asarray(rpp, dtype=np.complex128)
    if rpp.shape!=(64,64) or not np.isfinite(rpp).all() or sigma2<=0 or not np.isfinite(sigma2):
        raise ValueError("finite frozen pilot covariance/positive variance required")
    if not np.allclose(rpp,rpp.conj().T,atol=1e-11,rtol=1e-11):
        raise ValueError("Hermitian pilot covariance required")
    operators=[];diagnostics=[]
    for observed,hold in folds:
        if np.intersect1d(observed,hold).size or len(set(np.r_[observed,hold]))!=64:
            raise ValueError("heldout cannot be observed")
        covariance=rpp[np.ix_(observed,observed)]+sigma2*np.eye(len(observed))
        cross=rpp[np.ix_(hold,observed)]
        factor=cho_factor(covariance,lower=True)
        weight=cho_solve(factor,cross.conj().T).conj().T
        residual=float(np.linalg.norm(weight@covariance-cross)/max(np.linalg.norm(cross),np.finfo(float).tiny))
        if residual>1e-11:raise ValueError("cross-pilot solve residual failed")
        weight.flags.writeable=False;operators.append(weight)
        diagnostics.append(dict(shape=list(weight.shape),solve_relative_residual=residual))
    return tuple(operators),diagnostics


@dataclass(frozen=True)
class PilotCV:
    folds: tuple
    source_mean_p: np.ndarray
    source_k: tuple
    profile_delta_k: tuple
    mobility_delta_k: tuple

    def frame_objective(self,pilots):
        """All folds once; only source-centered received pilot-LS coefficients."""
        pilots=np.asarray(pilots,dtype=np.complex128)
        if pilots.shape!=(64,) or not np.isfinite(pilots).all():raise ValueError("finite current-frame pilots required")
        z=pilots-self.source_mean_p
        columns=[];responses=[]
        for f,(observed,hold) in enumerate(self.folds):
            x=z[observed];base=self.source_k[f]@x
            columns.append(np.column_stack((self.profile_delta_k[f]@x,self.mobility_delta_k[f]@x)))
            responses.append(z[hold]-base)
        design=np.concatenate(columns);response=np.concatenate(responses)
        return design.conj().T@design,design.conj().T@response,float(np.vdot(response,response).real)

    def fit(self,pilot_frames,*,modules):
        """Explicit arm is structural prior, never inferred from a shift label."""
        x=np.asarray(pilot_frames,dtype=np.complex128)
        if x.ndim!=2 or x.shape[1]!=64 or not 1<=len(x)<=5:raise ValueError("one to five causal frames required")
        if modules not in (("M",),("P","M")):raise ValueError("declared one/two-module arms only")
        moments=[self.frame_objective(p) for p in x]
        indices=[1] if modules==("M",) else [0,1]
        gram=sum(v[0] for v in moments).real[np.ix_(indices,indices)]
        rhs=sum(v[1] for v in moments).real[indices]
        coefficients=np.linalg.solve(gram,rhs)  # No clipping, inverse, pinv or regularization.
        if not np.isfinite(coefficients).all():raise ValueError("nonfinite unconstrained real LS")
        baseline=sum(v[2] for v in moments)
        residual=float(baseline-2*coefficients@rhs+coefficients@gram@coefficients)
        return coefficients,dict(gram=gram.tolist(),rhs=rhs.tolist(),condition_number=float(np.linalg.cond(gram)),
            normal_equation_relative_residual=float(np.linalg.norm(gram@coefficients-rhs)/max(np.linalg.norm(rhs),np.finfo(float).tiny)),
            source_cv_loss=baseline,fitted_cv_loss=residual,window_frames=len(x),real_ls_size=len(indices),
            clipped=False,regularized=False,independent_frames_assumed=False)


def causal_windows(pilots,window):
    x=np.asarray(pilots)
    if x.shape!=(10,64) or window not in (1,5):raise ValueError("ten frames; window1 or5 required")
    return [(j,x[j-window+1:j+1]) for j in range(window-1,10)]


def deployment(pilot,source_mean_p,source_mean_d,source_w,profile_delta,mobility_delta,coefficients,modules):
    """Source means only; frozen full data-domain modules, no oracle input."""
    x=np.asarray(pilot)-source_mean_p
    base=source_mean_d+x@source_w.T
    if modules==("M",):return base+coefficients[0]*(x@mobility_delta.T)
    if modules==("P","M"):return base+coefficients[0]*(x@profile_delta.T)+coefficients[1]*(x@mobility_delta.T)
    raise ValueError("declared one/two-module arms required")


def recovery(source_db,matched_db,oracle_db,estimate_db):
    gap=source_db-matched_db;capacity=source_db-oracle_db;gain=source_db-estimate_db
    return dict(full_gap_db=gap,recovery_db=gain,closure=gain/gap if gap!=0 else None,
                oracle_scalar_recovery_db=capacity,oracle_scalar_efficiency=gain/capacity if capacity!=0 else None)


def primary_prediction(cv,pilot_frames,source_mean_d,source_w,profile_delta,mobility_delta):
    """Uniform P+M for every sample; reporting labels cannot enter this API."""
    coefficients,diagnostics=cv.fit(pilot_frames,modules=("P","M"))
    prediction=deployment(pilot_frames[-1],cv.source_mean_p,source_mean_d,source_w,
                          profile_delta,mobility_delta,coefficients,("P","M"))
    return coefficients,prediction,diagnostics


def fitting_gate(rows,oracle_scalar_insufficient):
    focus=[r for r in rows if r['method']=='pilotCV-5' and r['snr_db'] in (10,20)]
    expected={(n,s) for n in ('CDL-C-75','CDL-B-120') for s in (10,20)}
    if len(focus)!=4 or {(r['shift'],r['snr_db']) for r in focus}!=expected:
        raise ValueError('all four primary actual-validation conditions required')
    if any(r.get('metric_type')!='actual_validation' or r['oracle_scalar_efficiency'] is None for r in focus):
        raise ValueError('defined actual efficiencies required')
    values=[r['oracle_scalar_efficiency'] for r in focus]
    viable=all(v>=.8 for v in values);insufficient=sum(v<.4 for v in values)>=2
    available=not oracle_scalar_insufficient
    return dict(analytical_pilot_only_module_fitting_viable=viable,
        analytical_pilot_only_module_fitting_insufficient=insufficient,
        analytical_pilot_only_module_fitting_partial=not viable and not insufficient,
        t16d_oracle_scalar_capacity_available=available,
        t16d_oracle_capacity_rule='not scalar_mode_composition_insufficient',
        learning_based_coordinate_inference_may_be_justified=insufficient and available)
