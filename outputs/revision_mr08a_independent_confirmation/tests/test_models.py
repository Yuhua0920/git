"""Seven-SNR frozen-method wrapper tests using synthetic pilots only."""
from pathlib import Path
import sys
import numpy as np
import pytest

from models import METHODS, SNRS, FRAMES, TenMethodBank, dependencies, prepare_assets, POWER

ROOT = Path(__file__).resolve().parents[3]
# Outside-project scratch tests resolve the real project explicitly; the final
# new project output gets ROOT through parents[3] without a C-drive dependency.
if not (ROOT / 'src/pilot_cv.py').exists():
    ROOT = Path(r'D:\Users\yuhua\Desktop\Project')
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope='module')
def bank():
    _, gmm, controls = dependencies(ROOT)
    from src.pilot_cv import PilotCV, frozen_folds
    from src.lowrank_modules import RankFactors
    from src.cdl_benchmark import observation_geometry
    from src.frozen_affine_validation import deployed
    rng = np.random.default_rng(829018)
    meanp = rng.normal(size=64) / 10 + 1j * rng.normal(size=64) / 10
    meand = np.zeros(3968, np.complex128); meand[:8] = .5 + .2j
    arrays = {'source_mean_p':meanp, 'source_mean_d':meand,
              'pool_mean_p':meanp + .1j, 'pool_mean_d':meand + .03j}
    folds, _ = frozen_folds(observation_geometry().selected)
    cv, factors, whole, mixtures = {}, {}, {}, {}
    for s in SNRS:
        W = np.zeros((3968, 64), np.complex128)
        W[:8, :8] = np.eye(8) * (1 + .1j)
        BP = np.zeros_like(W); BP[:8, 8:16] = np.eye(8) * (.2 + .3j)
        BM = np.zeros_like(W); BM[:8, 16:24] = np.eye(8) * (-.4 + .1j)
        arrays.update({f'Wsource_{s}':W, f'WP_{s}':W+BP, f'WM_{s}':W+BM,
                       f'DeltaWP_{s}':BP, f'DeltaWM_{s}':BM, f'Wpool_{s}':W*.73,
                       f'Direct_W_{s}':W*.63, f'Direct_mean_p_{s}':meanp-.2,
                       f'Direct_mean_d_{s}':meand+.1})
        K = np.zeros((4,16,48), np.complex128)
        KP = K.copy(); KM = K.copy()
        for f in range(4):
            KP[f, :, :16] = np.eye(16)*(.6+.2j)
            KM[f, :, 16:32] = np.eye(16)*(-.3+.7j)
        cv[s] = PilotCV(folds, meanp, tuple(K), tuple(KP), tuple(KM))
        for m, shift in [('P',8), ('M',16)]:
            A = np.zeros((3968,4), np.complex128); A[:4] = np.eye(4)*(.2+.1j)
            Q = np.zeros((4,64), np.complex128); Q[:,shift:shift+4] = np.eye(4)
            factors[m,s] = RankFactors(A,Q)
        fs=[]
        for q in range(3):
            A=np.zeros((3968,24),np.complex128); A[:24]=np.eye(24)*(q+1)
            Q=np.zeros((24,64),np.complex128); Q[:,:24]=np.eye(24)*.2
            fs.append(RankFactors(A,Q))
        whole[s]={'WHOLE_POOL_M2_R24':tuple(fs)}
        sigma=POWER/10**(s/10)
        components=[]
        for q in range(3):
            C=np.eye(64,dtype=np.complex128)*(1+q+sigma)
            L=np.eye(64,dtype=np.complex128)*np.sqrt(1+q+sigma)
            components.append(gmm.GaussianComponent(str(q),meanp+(q*.1),meand+q*.02,
                              C,L,W*(q+1)*.4,64*np.log(1+q+sigma),{}))
        mixtures[s]=gmm.PreparedGaussianMixture(tuple(components),np.full(3,1/3),
                       np.full(3,np.log(1/3)),sigma,{})
    return TenMethodBank(arrays,cv,factors,whole,mixtures,controls,deployed)


@pytest.fixture(scope='module')
def pilots():
    rng=np.random.default_rng(32801)
    return (rng.normal(size=(7,10,64))+1j*rng.normal(size=(7,10,64))).astype(np.complex128)


def test_inventory_shape_finiteness_and_posterior(bank,pilots):
    before=pilots.copy(); got=bank.predict(pilots)
    assert got['methods']==METHODS and len(METHODS)==10
    assert got['snrs']==SNRS and got['frame_ids']==FRAMES
    assert got['predictions'].shape==(7,10,6,3968)
    assert got['predictions'].dtype==np.complex128
    assert np.isfinite(got['predictions']).all()
    assert got['coordinates_tw5'].shape==got['coordinates_tw1'].shape==(7,6,2)
    assert got['gram_tw5'].shape==got['gram_tw1'].shape==(7,6,2,2)
    assert got['gmm_posterior_weights'].shape==(7,6,3)
    np.testing.assert_allclose(got['gmm_posterior_weights'].sum(axis=2),1,atol=1e-12,rtol=0)
    assert np.all(got['normal_residual_tw1']<=1e-11)
    assert np.all(got['normal_residual_tw5']<=1e-11)
    assert np.all(got['convex_coordinates']>=0)
    assert np.all(got['convex_coordinates'].sum(axis=2)<=1)
    np.testing.assert_array_equal(pilots,before)


def test_original_five_frame_outputs_and_current_frame_fit(bank,pilots):
    from src.lowrank_modules import factored_affine_predict
    got=bank.predict(pilots)
    for si,s in enumerate(SNRS):
        original=bank.deployed(pilots[si],bank.arrays,s,bank.cvs[s],bank.factors['P',s],bank.factors['M',s])
        mapping={'source':'SOURCE','hard':'HARD_TW5','convex':'CONVEX_TW5',
                 'dense-affine':'DENSE_TW5','lr4-affine':'LR4_TW5'}
        for ri,row in enumerate(original):
            j=FRAMES[ri]
            for old,new in mapping.items():
                np.testing.assert_array_equal(got['predictions'][si,METHODS.index(new),ri],row['predictions'][old])
            np.testing.assert_array_equal(got['coordinates_tw5'][si,ri],row['dense_coefficients'])
            assert row['dense_coefficients'] is row['lr4_coefficients']
            c1,_=bank.cvs[s].fit(pilots[si,j:j+1],modules=('P','M'))
            np.testing.assert_array_equal(got['coordinates_tw1'][si,ri],c1)
            pred=factored_affine_predict(pilots[si,j],bank.arrays['source_mean_p'],
                  bank.arrays['source_mean_d'],bank.arrays[f'Wsource_{s}'],c1,
                  bank.factors['P',s],bank.factors['M',s])
            np.testing.assert_array_equal(got['predictions'][si,METHODS.index('LR4_TW1'),ri],pred)


def test_natural_means_current_frame_gmm_and_whole_coords(bank,pilots):
    got=bank.predict(pilots)
    for si,s in enumerate(SNRS):
        mix=bank.gmm[s].predict(pilots[si,4:10])
        np.testing.assert_array_equal(got['predictions'][si,METHODS.index('GMM_PREPARED_3')],mix.estimate)
        for ri,j in enumerate(FRAMES):
            p=pilots[si,j];u=p-bank.arrays['source_mean_p'];c=got['coordinates_tw5'][si,ri]
            pool=bank.arrays['pool_mean_d']+(p-bank.arrays['pool_mean_p'])@bank.arrays[f'Wpool_{s}'].T
            direct=bank.arrays[f'Direct_mean_d_{s}']+bank.arrays[f'Direct_W_{s}']@(p-bank.arrays[f'Direct_mean_p_{s}'])
            whole=bank.controls.deployed('WHOLE_POOL_M2_R24',u,c,bank.arrays,bank.whole[s],s)
            np.testing.assert_allclose(got['predictions'][si,METHODS.index('POOLED_NATURAL'),ri],pool,atol=2e-13,rtol=2e-13)
            np.testing.assert_array_equal(got['predictions'][si,METHODS.index('PPCA_DIRECT_R32'),ri],direct)
            np.testing.assert_array_equal(got['predictions'][si,METHODS.index('WHOLE_POOL_M2_R24'),ri],whole)


def test_repeat_prediction_exact_determinism(bank,pilots):
    first=bank.predict(pilots);second=bank.predict(pilots)
    for k,v in first.items():
        if isinstance(v,np.ndarray):np.testing.assert_array_equal(v,second[k])


def test_singular_window_stops_with_identity_no_fallback(bank):
    p=np.broadcast_to(bank.arrays['source_mean_p'],(7,10,64)).copy()
    with pytest.raises(ValueError,match='TW5_SHARED_AFFINE_DENSE_LR4_WHOLE24') as ex:
        bank.predict(p)
    assert '"snr_db": 0' in str(ex.value)
    assert '"frame_id": 4' in str(ex.value)


@pytest.mark.parametrize('bad',[np.zeros((3,10,64),np.complex128),np.zeros((7,6,64),np.complex128),
                              np.zeros((7,10,64),np.float64),np.full((7,10,64),np.nan,np.complex128)])
def test_bad_pilot_shape_dtype_or_finite_rejected(bank,bad):
    with pytest.raises(ValueError,match='SEVEN_SNR_TEN_FRAME_COMPLEX128'):
        bank.predict(bad)


def test_prepare_cache_never_overwrites_existing(tmp_path):
    (tmp_path/'data').mkdir();p=tmp_path/'data/estimators_7snr.npz';p.write_bytes(b'original')
    with pytest.raises(FileExistsError,match='MODEL_CACHE_OR_REPORT_EXISTS'):
        prepare_assets(tmp_path/'absent_root',tmp_path)
    assert p.read_bytes()==b'original'
