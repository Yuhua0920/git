"""Pre-CDL numeric tests; synthetic channels, guarded immutable model inputs."""
import inspect
from pathlib import Path
import numpy as np
import pytest
from scripts import sp_u2b_diagnostic as d
from scripts.run_sp_u2b import diagnose
from src.pilot_cv import PilotCV, causal_windows
from src.lowrank_modules import factored_affine_predict
from src.spatial_information import observation_geometry

d.install_guard()


@pytest.fixture(scope='module')
def fixture():
    arrays,cvs,factors,proof=d.frozen_method()
    index=np.arange(2*2*64*160).reshape(2,2,64,160)
    h=np.sin(index/19)+.6j*np.cos(index/37)
    pilots,hashes,power=d.pair_channel(h,'smoke',0)
    return arrays,cvs,factors,h,pilots,hashes,power


def test_clean_noisy_exact_pairing(fixture):
    _,_,_,_,pilots,hashes,_=fixture
    for j in range(10):
        same=hashes[3*j:3*j+3]
        for key in ('H_sha256','bits_sha256','symbols_sha256','transmitted_sha256','useful_sha256','unit_noise_sha256'):
            assert len({r[key] for r in same})==1
        d.close((pilots[1,j]-pilots[0,j])/np.sqrt(d.VARIANCES[10]),
                (pilots[2,j]-pilots[0,j])/np.sqrt(d.VARIANCES[20]))


def test_clean_zero_noise_reconstruction(fixture):
    from src.cdl_benchmark import frames
    _,_,_,h,pilots,_,power=fixture
    g=observation_geometry()
    d.close(pilots[0],np.stack([v.ravel()[g.selected] for v in frames(h)]))
    assert power['clean_reconstruction_error_max']<d.TOL


def test_explicit_G_production_equivalence(fixture):
    _,cvs,_,_,pilots,_,_=fixture
    for si,s in ((0,10),(1,10),(2,20)):
        G,z=d.explicit_design(cvs[s],pilots[si,:5])
        c,diag=cvs[s].fit(pilots[si,:5],modules=('P','M'))
        assert G.shape==(640,2) and z.shape==(640,)
        d.close(G.T@G,diag['gram']);d.close(G.T@z,diag['rhs'])
        d.close(np.linalg.lstsq(G,z,rcond=None)[0],c)


def test_design_gram_condition_convention(fixture):
    _,cvs,_,_,pilots,_,_=fixture
    G,_=d.explicit_design(cvs[10],pilots[1,:5])
    geo=d.geometry(G)
    assert geo['rank_real']==2
    assert geo['gram_condition']==pytest.approx(geo['kappa_G']**2,rel=d.TOL)


def test_delta_G_reconstruction(fixture):
    _,cvs,_,_,pilots,_,_=fixture
    cv=cvs[10]
    G0,z0=d.explicit_design(cv,pilots[0,:5]);G,z=d.explicit_design(cv,pilots[1,:5])
    delta_cv=PilotCV(cv.folds,np.zeros(64,complex),cv.source_k,cv.profile_delta_k,cv.mobility_delta_k)
    E,response=d.explicit_design(delta_cv,pilots[1,:5]-pilots[0,:5])
    d.close(E,G-G0);d.close(response,z-z0)


def test_coordinate_reproducibility(fixture):
    _,cvs,_,h,pilots,hashes,_=fixture
    p2,h2,_=d.pair_channel(h,'smoke',0)
    assert np.array_equal(p2,pilots) and h2==hashes
    for si in (0,1):
        c=cvs[10].fit(pilots[si,:5],modules=('P','M'))[0]
        assert np.array_equal(c,cvs[10].fit(p2[si,:5],modules=('P','M'))[0])


def test_rank_deficient_case_preserved(fixture):
    _,cvs,_,_,pilots,_,_=fixture
    cv=cvs[10]
    singular=PilotCV(cv.folds,cv.source_mean_p,cv.source_k,cv.profile_delta_k,cv.profile_delta_k)
    G,_=d.explicit_design(singular,pilots[0,:5])
    assert np.linalg.matrix_rank(G)==1
    with pytest.raises(np.linalg.LinAlgError):
        singular.fit(pilots[0,:5],modules=('P','M'))


def test_coordinate_output_propagation(fixture):
    arrays,_,factors,_,pilots,_,_=fixture
    p=pilots[1,4];u=p-arrays['source_mean_p']
    fp,fm=factors['P',10],factors['M',10]
    args=(p,arrays['source_mean_p'],arrays['source_mean_d'],arrays['Wsource_10'])
    c0=np.array([1.2,-.7]);c=np.array([-.6,3.4])
    X=np.column_stack([fp.apply(u),fm.apply(u)])
    measured=factored_affine_predict(*args,c,fp,fm)-factored_affine_predict(*args,c0,fp,fm)
    d.close(measured,X@(c-c0))
    d.close(np.vdot(measured,measured).real,(c-c0)@(X.conj().T@X).real@(c-c0))


def test_channel_oracle_direct_real_LS():
    index=np.arange(50)
    X=np.column_stack([np.sin(index/3)+1j*np.cos(index/5),np.cos(index/7)-.4j*np.sin(index/2)])
    base=np.sin(index/13).astype(complex)
    true=base+X@np.array([1.3,-.8])+.1*np.cos(index)
    c,M,err=d.channel_oracle(true,base,X)
    assert err<d.TOL
    d.close(M@c,(X.conj().T@(true-base)).real)
    c2=np.array([2.,3.])
    loss=lambda coeff:np.linalg.norm(true-base-X@coeff)**2
    d.close(loss(c2)-loss(c),(c2-c)@M@(c2-c))


def test_linear_MSE_penalty_bookkeeping_keeps_negative():
    noisy,clean,oracle=1.,2.,.5
    noise_penalty=noisy-clean;surrogate_penalty=clean-oracle
    assert noise_penalty<0 and noise_penalty+surrogate_penalty==noisy-oracle


@pytest.mark.parametrize('path',['outputs/t20b_formal_results.json','outputs/../outputs/t20b_x.csv',
                                 'outputs/t18a_generalization_nmse.csv'])
def test_no_old_outcome_access(path):
    with pytest.raises(PermissionError,match='firewall'):
        (d.ROOT/path).read_bytes()


def test_future_frame_causality(fixture):
    _,cvs,_,_,pilots,_,_=fixture
    changed=pilots[1].copy();changed[5:]+=1e9
    before=dict(causal_windows(pilots[1],5))[4]
    after=dict(causal_windows(changed,5))[4]
    assert np.array_equal(before,after)
    assert np.array_equal(cvs[10].fit(before,modules=('P','M'))[0],cvs[10].fit(after,modules=('P','M'))[0])


def test_production_outputs_unchanged_and_diagnostics_complete(fixture):
    arrays,cvs,factors,h,pilots,_,_=fixture
    cv=cvs[10]
    c=cv.fit(pilots[1,:5],modules=('P','M'))[0]
    args=(pilots[1,4],arrays['source_mean_p'],arrays['source_mean_d'],arrays['Wsource_10'])
    pred=factored_affine_predict(*args,c,factors['P',10],factors['M',10])
    rows,_=diagnose(pilots,h,arrays,cvs,factors,'smoke',0)
    first=rows[0]
    assert first['prediction_sha256']==d.array_hash(pred)
    assert np.array_equal(c,cv.fit(pilots[1,:5],modules=('P','M'))[0])
    assert len(rows)==12 and all(r['ndata']==3968 for r in rows)
    assert all(r['joint_identity_error']<d.TOL and not r['bound_failure'] for r in rows)
    assert all(r['decomposition_identity_error']<d.TOL for r in rows)
    assert not any('truth' in k or 'oracle' in k for k in inspect.signature(PilotCV.fit).parameters)
    # Serializer preflight exercises the complete scientific row, not a toy subset.
    import json
    json.dumps(rows,allow_nan=False)


def test_new_seed_role_values_unique_and_stages_disjoint():
    values=[]
    for stage,count in (('smoke',20),('development',100)):
        for i in range(count):
            for j in range(10):
                s=d.seeds(stage,i,j)
                values.extend(s[k] for k in s if j==0 or k not in ('sionna_seed','torch_seed','numpy_seed'))
    assert len(values)==len(set(values))==3960


def test_joint_bound_zero_change_policy(fixture):
    _,cvs,_,_,pilots,_,_=fixture
    G,z=d.explicit_design(cvs[10],pilots[0,:5]);c=cvs[10].fit(pilots[0,:5],modules=('P','M'))[0]
    result=d.joint_diagnostics(G,z,G,z,c,c)
    assert result['near_zero_coordinate_change'] and result['bound_actual_ratio'] is None
    assert result['joint_bound']==0 and not result['bound_failure']
