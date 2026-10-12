import numpy as np
from src.pilot_cv import PilotCV,frozen_folds
from src.cdl_benchmark import observation_geometry
from benchmark_mr05 import fit_both

def test_timing_calls_original_explicit_two_direction_arm():
    rng=np.random.default_rng(170501)
    folds,_=frozen_folds(observation_geometry().selected)
    source=tuple(np.zeros((16,48),np.complex128) for _ in folds)
    p=tuple(rng.normal(size=(16,48)).astype(np.complex128) for _ in folds)
    m=tuple((rng.normal(size=(16,48))+1j*rng.normal(size=(16,48))) for _ in folds)
    cv=PilotCV(folds,np.zeros(64,np.complex128),source,p,m)
    window=rng.normal(size=(5,64))+1j*rng.normal(size=(5,64))
    actual,diag=fit_both(cv,window)
    expected,_=cv.fit(window,modules=('P','M'))
    assert np.array_equal(actual,expected)
    assert diag['real_ls_size']==2 and diag['window_frames']==5
