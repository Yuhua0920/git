"""T19D real operator-module-space diagnostics; no deployed estimator."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from src import operator_space_audit as old

DEVELOPMENT=('CDL-C-30-50','CDL-C-30-200')
HELDOUT=('CDL-C-90-50','CDL-C-90-200','CDL-B-30-50','CDL-B-30-200','CDL-B-90-50','CDL-B-90-200')
SNRS=(10,20)
REPLICATES=(1,2,3,4,5)
EXPERIMENT_TYPE='pure-delay residual operator-subspace dimensionality and held-out factor-transfer gate'
C12='A low-dimensional delay-residual operator subspace learned only from pure-delay C30 statistics transfers to held-out profile/mobility+delay contexts.'


def augment(z):
    z=old.complex_array(z)
    return np.concatenate((z.real.ravel(order='C'),z.imag.ravel(order='C')))


def unaugment(v,shape):
    v=np.asarray(v)
    size=int(np.prod(shape))
    if np.iscomplexobj(v) or v.shape!=(2*size,) or not np.isfinite(v).all():
        raise ValueError('finite real-augmented vector of declared shape required')
    return (v[:size]+1j*v[size:]).reshape(shape,order='C').astype(np.complex128)


def unit(v):
    v=np.asarray(v)
    if np.iscomplexobj(v) or not np.isfinite(v).all():raise ValueError('finite real vector required')
    norm=float(np.linalg.norm(v))
    if norm<=0:raise ValueError('zero residual:STOP; do not drop development rows')
    return np.asarray(v/norm,dtype=np.float64),norm


def sign_rule(v):
    v=np.asarray(v,dtype=np.float64).copy()
    pivot=int(np.argmax(np.abs(v)))
    sign=-1 if v[pivot]<0 else 1
    v*=sign
    return v,dict(pivot_index=pivot,pivot_value=float(v[pivot]),sign_multiplier=sign)


def recover(z,lower):
    # D L = Z: transpose, not conjugate transpose, for a right triangular solve.
    return solve_triangular(old.complex_array(lower).T,old.complex_array(z).T,lower=False).T


def learn(residuals,source_lower,*,excluded_pair=None):
    expected={(n,r) for n in DEVELOPMENT for r in REPLICATES if r!=excluded_pair}
    if excluded_pair is not None and excluded_pair not in REPLICATES:raise ValueError('declared LORO pair required')
    if set(residuals)!=expected:raise ValueError('pure-delay development only; exact pair exclusion required')
    keys=[(n,r) for n in DEVELOPMENT for r in REPLICATES if r!=excluded_pair]
    shape=next(iter(residuals.values())).shape
    normalized=[];norms=[]
    for key in keys:
        if residuals[key].shape!=shape:raise ValueError('common operator shape required')
        v,norm=unit(augment(residuals[key]@source_lower));normalized.append(v);norms.append(norm)
    stack=np.stack(normalized)
    if stack.dtype!=np.float64 or np.iscomplexobj(stack):raise ValueError('SVD must use real-augmented float64 stack')
    _,singular,vt=np.linalg.svd(stack,full_matrices=False)
    ratio=singular**2/float(np.sum(singular**2))
    directions=[];vectors=[];signs=[]
    for v in vt[:2]:
        v,sentinel=sign_rule(v)
        d=recover(unaugment(v,shape),source_lower)
        d/=np.sqrt(old.energy(d@source_lower))
        vector=augment(d@source_lower)
        if vector[sentinel['pivot_index']]<=0:raise ValueError('recovered basis sign drift')
        directions.append(d);vectors.append(vector);signs.append(sentinel)
    vectors=np.stack(vectors)
    if not np.allclose(vectors@vectors.T,np.eye(2),rtol=1e-11,atol=1e-11):raise ValueError('source-weighted basis orthonormality failed')
    return dict(directions=np.stack(directions),vectors=vectors,singular_values=singular,
                explained_energy_ratio=ratio,cumulative_energy=np.cumsum(ratio),input_norms=np.array(norms),
                construction_keys=keys,sign_sentinels=signs,excluded_pair=excluded_pair,
                centered=False,construction='equal-direction normalized real-augmented uncentered SVD')


def projection_capture(residual,lower,vectors,q):
    if q not in (1,2):raise ValueError('additional module-space dimension q=1/2 required')
    v,_=unit(augment(residual@lower))
    return float(np.sum((vectors[:q]@v)**2))


def four_real_ls(response,columns):
    """Four-column oracle diagnostic; same frozen normal-equation conventions.

    Historical two/three-column implementation remains unchanged. No ridge,
    explicit inverse, SVD fallback or coefficient clipping is introduced.
    """
    if len(columns)!=4:raise ValueError('exactly four columns required')
    response=old.complex_array(response).reshape(-1)
    design=np.stack([old.complex_array(c).reshape(-1) for c in columns],axis=1)
    if design.shape[0]!=response.size:raise ValueError('equal-length columns required')
    real=np.concatenate((design.real,design.imag),axis=0)
    rhs_vector=np.concatenate((response.real,response.imag))
    gram=real.T@real;rhs=real.T@rhs_vector;eig=np.linalg.eigvalsh(gram)
    if eig[0]<=1e-12*eig[-1]:raise ValueError('ill-conditioned real Gram; do not change basis or regularize')
    coefficients=cho_solve(cho_factor(gram,lower=True),rhs)
    residual=response-design@coefficients
    return dict(coefficients=coefficients,gram=gram,rhs=rhs,residual=residual,excess=old.energy(residual),
        normal_relative_residual=float(np.linalg.norm(gram@coefficients-rhs)/max(np.linalg.norm(rhs),1e-300)))


def joint_fit(source,target,lower,modules,directions,q):
    if len(modules)!=2 or q not in (0,1,2):raise ValueError('two frozen modules and q=0/1/2 required')
    columns=[m@lower for m in modules]+[d@lower for d in directions[:q]]
    response=(target-source)@lower
    return four_real_ls(response,columns) if q==2 else old.real_ls(response,columns)


def transfer_gate(rows):
    expected={(n,s) for n in HELDOUT for s in SNRS}
    if len(rows)!=12 or {(v['environment'],v['snr_db']) for v in rows}!=expected:
        raise ValueError('all twelve held-out factor conditions required; no development rows')
    for row in rows:
        if not np.isfinite([row[k] for k in ('capture_q0','capture_q1','capture_q2','increment_q1','increment_q2','marginal_q2')]).all():
            raise ValueError('finite held-out metrics required; no sample exclusion')
        if row['increment_q1']!=row['capture_q1']-row['capture_q0'] or row['increment_q2']!=row['capture_q2']-row['capture_q0'] or row['marginal_q2']!=row['capture_q2']-row['capture_q1']:
            raise ValueError('unclipped incremental formula drift')
    q1=sum(v['capture_q1']>=.70 and v['increment_q1']>=.10 for v in rows)
    q2=sum(v['capture_q2']>=.70 and v['increment_q2']>=.10 for v in rows)
    marginal=sum(v['marginal_q2']>=.05 for v in rows)
    weak_count=sum(v['increment_q2']<.05 for v in rows)
    single=q1>=9;two=not single and q2>=9 and marginal>=6
    weak=not single and not two and weak_count>=9
    return dict(single_delay_direction_transfer_supported=single,two_direction_delay_subspace_transfer_supported=two,
                delay_residual_subspace_transfer_weak=weak,delay_residual_subspace_transfer_partial=not single and not two and not weak,
                q1_promising_count=q1,q2_promising_count=q2,q2_marginal_at_least_005_count=marginal,q2_increment_below_005_count=weak_count,
                c12_claim_status=('SUPPORTED_WITH_ONE_ADDITIONAL_ORACLE_DIRECTION' if single else
                    'SUPPORTED_WITH_TWO_DIMENSION_ORACLE_SUBSPACE' if two else
                    'CONTRADICTED_BY_HELDOUT_FACTOR_TRANSFER' if weak else 'INCONCLUSIVE'))


def calibration():
    k=np.arange(30,dtype=np.float64).reshape(10,3)
    modules=[np.sin(k*a)+1j*np.cos(k*b) for a,b in ((.17,.29),(.43,.61),(.79,.97),(1.13,1.31))]
    lower=np.array([[1.,0,0],[.2+.3j,.7,0],[.1-.2j,.05+.1j,.9]],np.complex128)
    coefficients=np.array([.3,-.2,.7,-.4]);response=sum(c*m for c,m in zip(coefficients,modules))
    fit=four_real_ls(response@lower,[m@lower for m in modules])
    e=float(np.max(abs(fit['coefficients']-coefficients)))
    recovery=float(np.max(abs(recover(response@lower,lower)-response)))
    v=augment(response);mapping=float(np.max(abs(unaugment(v,response.shape)-response)))
    if max(e,recovery,mapping)>1e-11:raise ValueError('independent mathematical calibration failed')
    return dict(passed=True,four_column_max_absolute_coefficient_error=e,triangular_recovery_max_absolute_error=recovery,
                real_mapping_max_absolute_error=mapping,normal_relative_residual=fit['normal_relative_residual'])
