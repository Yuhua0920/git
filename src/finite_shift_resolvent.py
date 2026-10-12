"""Exact finite-shift Wiener bookkeeping; no inverse arrays or approximation."""
from dataclasses import dataclass
import math
import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky
from src.wiener_perturbation import matrix, hermitian, frozen, energy

NAMES = ('0', 'P', 'M', 'PM')
TERMS = ('C1', 'C2', 'C3', 'C4')


def relative(a, b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a), np.finfo(float).tiny))


@dataclass(frozen=True)
class ExactWienerMap:
    d: dict
    p: dict
    factors: dict
    w: dict
    sigma2: float

    @classmethod
    def build(cls, d, p, sigma2):
        if set(d) != set(NAMES) or set(p) != set(NAMES):
            raise ValueError('four fixed covariance conditions required')
        if not np.isfinite(sigma2) or sigma2 <= 0:
            raise ValueError('positive finite noise variance required')
        p = {k: frozen(hermitian(v)) for k,v in p.items()}
        shape = p['0'].shape
        if any(v.shape != shape for v in p.values()):
            raise ValueError('pilot dimensions differ')
        d = {k: frozen(matrix(v)) for k,v in d.items()}
        if any(v.shape != d['0'].shape or v.shape[1] != shape[0] for v in d.values()):
            raise ValueError('cross covariance dimensions differ')
        factors = {k: cho_factor(v+sigma2*np.eye(shape[0]), lower=True) for k,v in p.items()}
        w = {k: frozen(cho_solve(factors[k], v.conj().T).conj().T) for k,v in d.items()}
        return cls(d,p,factors,w,float(sigma2))

    def right_solve(self, value, name):
        value = matrix(value)
        if value.shape[1] != self.p[name].shape[0]:
            raise ValueError('right solve dimensions differ')
        return cho_solve(self.factors[name], value.conj().T).conj().T

    def decomposition(self):
        d0,dp,dm,dpm = (self.d[k] for k in NAMES)
        f_p,f_m,f_i = dp-d0,dm-d0,dpm-dp-dm+d0
        c = dict(C1=self.right_solve(f_i,'PM'),
                 C2=self.right_solve(f_p,'PM')-self.right_solve(f_p,'P'),
                 C3=self.right_solve(f_m,'PM')-self.right_solve(f_m,'M'),
                 C4=self.right_solve(d0,'PM')-self.right_solve(d0,'P')
                    -self.right_solve(d0,'M')+self.right_solve(d0,'0'))
        w0,wp,wm,wpm = (self.w[k] for k in NAMES)
        observed = wpm-(wp+wm-w0)
        summed = c['C1']+c['C2']+c['C3']+c['C4']
        identities = dict(four_term_relative_residual=relative(observed,summed),
            parenthesization_relative_residual=relative(observed,wpm-wp-wm+w0),
            four_term_residual_scaled_by_contribution_norms=float(np.linalg.norm(observed-summed)/sum(np.linalg.norm(v) for v in c.values())))
        return observed,c,identities

    def resolvent_audit(self, action):
        """Verify identities on a supplied right action; never solve identity RHS."""
        result=[]
        for a,b in (('PM','P'),('PM','M'),('P','0'),('M','0')):
            lhs=self.right_solve(action,a)-self.right_solve(action,b)
            rhs=-self.right_solve(self.right_solve(action,a)@(self.p[a]-self.p[b]),b)
            result.append(dict(from_condition=b,to_condition=a,
                               action_relative_residual=relative(lhs,rhs)))
        return result

    def lower(self):
        return cholesky(self.p['PM']+self.sigma2*np.eye(self.p['PM'].shape[0]),lower=True)

    def counterfactual(self, observed, lower):
        dadd=self.d['P']+self.d['M']-self.d['0']
        padd=self.p['P']+self.p['M']-self.p['0']
        s=padd+self.sigma2*np.eye(padd.shape[0])
        hermitian_residual=float(np.linalg.norm(s-s.conj().T)/max(np.linalg.norm(s),1e-300))
        hermitian(s)
        eigen=np.linalg.eigvalsh(s)
        floor=float(len(eigen)*np.finfo(float).eps*np.max(np.abs(eigen)))
        condition=float(np.max(np.abs(eigen))/np.min(np.abs(eigen))) if np.min(np.abs(eigen)) else None
        record=dict(hermitian_relative_residual=hermitian_residual,minimum_eigenvalue=float(eigen[0]),
            condition_number=condition,numerical_PD_floor=floor,
            additive_statistics_counterfactual_available=bool(eigen[0]>floor),regularization_used=False)
        if not record['additive_statistics_counterfactual_available']:
            return record
        factor=cho_factor(s,lower=True)
        wadd=cho_solve(factor,dadd.conj().T).conj().T
        a=self.w['PM']-wadd
        b=wadd-self.w['P']-self.w['M']+self.w['0']
        ea,eb,er=energy(a@lower),energy(b@lower),energy(observed@lower)
        cross=float(2*np.vdot(a@lower,b@lower).real)
        record.update(covariance_nonadditivity_energy=ea,wiener_nonseparability_energy=eb,
            pair_cross_energy=cross,interaction_energy=er,
            counterfactual_cancellation_index=(ea+eb)/er,
            decomposition_relative_residual=relative(observed,a+b),
            energy_identity_relative_residual=abs(ea+eb+cross-er)/er)
        return record


def weighted_bookkeeping(observed, correction, contributions, lower):
    er,ed=energy(observed@lower),energy(correction@lower)
    if er <= 0 or ed <= 0 or set(contributions)!=set(TERMS):
        raise ValueError('positive reference energies and four contributions required')
    mapped=[contributions[k]@lower for k in TERMS]
    gram=np.array([[np.vdot(a,b) for b in mapped] for a in mapped])
    energies=[float(gram[i,i].real) for i in range(4)]
    pair_energies=[float(2*gram[i,j].real) for i in range(4) for j in range(i+1,4)]
    cross=math.fsum(pair_energies)
    individual_total=math.fsum(energies)
    # Sum original diagonal and pair terms together: destructive cancellation
    # can erase the small remainder if each group is first rounded separately.
    reconstructed=math.fsum(energies+pair_energies)
    record=dict(interaction_energy=er,complete_correction_energy=ed,
        exact_affine_closure=1.-er/ed,interaction_norm_ratio=float(np.sqrt(er/ed)),
        cancellation_index=individual_total/er,cross_fraction=cross/er,
        energy_identity_relative_residual=abs(reconstructed-er)/er,
        reconstructed_Gram_energy=reconstructed,
        energy_identity_scaled_by_individual_energy=abs(reconstructed-er)/individual_total,
        dominant_contribution=TERMS[int(np.argmax(energies))],
        maximum_contribution_norm_ratio=float(np.sqrt(max(energies)/ed)),
        normalized_contribution_energies=[e/ed for e in energies],
        contribution_energy_shares=[e/individual_total for e in energies],
        weighted_gram_real=gram.real.tolist(),weighted_gram_imag=gram.imag.tolist(),
        cross_energy_matrix=(2*gram.real).tolist(),cross_energy_diagonal_definition='Xii=2*E(Ci); energy identity uses upper off-diagonal only')
    rows=[dict(term=k,weighted_energy=e,weighted_norm=float(np.sqrt(e)),
        relative_to_complete_correction=float(np.sqrt(e/ed)),relative_to_interaction=float(np.sqrt(e/er)),
        normalized_energy_relative_to_complete_correction=e/ed) for k,e in zip(TERMS,energies)]
    return record,rows


def decision(rows):
    selected={int(r['snr_db']):r for r in rows if int(r['snr_db']) in (10,20)}
    if set(selected)!={10,20}:raise ValueError('10/20 dB required')
    r=list(selected.values())
    values=[v[k] for v in r for k in ('exact_affine_closure','cancellation_index','maximum_contribution_norm_ratio')]
    if not np.isfinite(values).all():raise ValueError('finite scientific decision inputs required')
    closure=all(v['exact_affine_closure']>=.90 for v in r)
    structural=closure and all(v['cancellation_index']<=2. and v['maximum_contribution_norm_ratio']<=.50 for v in r)
    cancellation=closure and any(v['cancellation_index']>=10. or v['maximum_contribution_norm_ratio']>=1. for v in r)
    return dict(finite_shift_separability_supported=structural,
        affine_additivity_cancellation_dominated=cancellation,finite_shift_separability_partial=not structural and not cancellation)
