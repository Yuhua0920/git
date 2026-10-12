import json
import numpy as np
import pytest
from run_mr08a import make_pilots,archive,write,SNRS,METHODS

def fixture():
    k=np.arange(2*2*64*160,dtype=float).reshape(2,2,64,160)
    h=np.asarray(.5+(.2+.1j)*np.exp(1j*k*.01),dtype=np.complex128)
    comm=[]
    for j in range(10):
        symbols=np.full((2,64,16),(1+1j)/np.sqrt(2),np.complex128)
        bits=np.zeros((2,64,16,2),np.int64)
        z=np.asarray((np.arange(2*64*16).reshape(2,64,16)%7-3)*(.01+.02j)*(j+1),np.complex128)
        comm.append((bits,symbols,z))
    return h,comm

def test_all_snr_shared_input_immutable_shape_and_identity():
    h,comm=fixture();before=h.copy();original=[tuple(z.copy() for z in c) for c in comm]
    p,t,ids=make_pilots(h,comm)
    assert p.shape==(7,10,64) and p.dtype==np.complex128 and t.shape==(6,3968)
    assert len(ids)==70 and len(METHODS)==10 and SNRS==(0,5,10,15,20,25,30)
    assert np.array_equal(h,before)
    assert all(np.array_equal(a,b) for c,d in zip(comm,original) for a,b in zip(c,d))
    for j in range(10):
        rows=[r for r in ids if r['frame_id']==j]
        for key in ('truth_frame_SHA256','unit_noise_SHA256','transmitted_symbols_SHA256','bits_SHA256'):assert len({r[key] for r in rows})==1
    p2,t2,ids2=make_pilots(h,comm);assert np.array_equal(p,p2) and np.array_equal(t,t2) and ids==ids2

def test_same_unit_noise_scaled_source_snr_only():
    from scripts.sp_u12b2a_common import variance
    h,comm=fixture();p,_,_=make_pilots(h,comm)
    ref=(p[0]-p[-1])/(np.sqrt(variance(0))-np.sqrt(variance(30)))
    for si,s in enumerate(SNRS[:-1]):
        observed=(p[si]-p[-1])/(np.sqrt(variance(s))-np.sqrt(variance(30)))
        assert np.allclose(observed,ref,rtol=1e-10,atol=1e-10)

def test_wrong_input_dtype_rejected():
    h,comm=fixture()
    with pytest.raises(ValueError,match='H_INPUT_SHAPE'):make_pilots(h.astype(np.complex64),comm)
    with pytest.raises(ValueError,match='H_INPUT_SHAPE'):make_pilots(h[:,:,:,:159],comm)

def test_raw_write_is_exclusive(tmp_path):
    p=tmp_path/'raw.npz';jp=tmp_path/'raw.json';a=np.arange(420.).reshape(7,10,6)
    archive(p,frame_error=a,frame_truth=np.ones(6));write(jp,dict(complete=True))
    with np.load(p,allow_pickle=False) as z:assert np.array_equal(z['frame_error'],a)
    assert json.loads(jp.read_text())['complete']
    with pytest.raises(FileExistsError):archive(p,frame_error=a)
    with pytest.raises(FileExistsError):write(jp,dict(complete=False))
