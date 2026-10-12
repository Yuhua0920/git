"""Deterministic tiny tests only; no formal seeds/data are generated."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
import pytest
from statistics_mr08a import METHODS,SNRS,CONDITIONS,FAMILIES,TAIL,FAMILY_SIZE,exclusive_array,store_indices,aggregate,bootstrap_values,primary_differences,condition_statistics,adjusted_intervals,analyze_dataset

def tiny():
    fe=np.arange(1.,4*7*10*6+1).reshape(4,7,10,6);fq=np.arange(1.,25.).reshape(4,6)
    return fe,fq

def test_tail_and_three_distinct_families():
    assert FAMILY_SIZE==35 and TAIL==.05/(2*35)
    assert [f[0] for f in FAMILIES]==['A','B','C']
    assert FAMILIES==(('A','POOLED_NATURAL','LR4_TW5'),('B','GMM_PREPARED_3','LR4_TW1'),('C','LR4_TW5','WHOLE_POOL_M2_R24'))
    assert TAIL!=.05/(2*105)

def test_pooled_not_average_of_trajectory_ratios():
    fe=np.array([[[[1.]]],[[[100.]]]]);fq=np.array([[1.],[10.]])
    _,_,p=aggregate(fe,fq)
    assert p.item()==101/11
    assert p.item()!=np.mean([1.,10.])

def test_lossesless_indices_match_trusted_int32_stream(tmp_path):
    path=tmp_path/'indices.npy';meta=store_indices(path,1743,2,4,21,block=3)
    expected=np.random.default_rng(np.random.SeedSequence([1743,2])).integers(0,4,size=(21,4),dtype=np.int32)
    ids=np.load(path,mmap_mode='r')
    assert ids.dtype==np.uint16 and np.array_equal(ids,expected)
    assert meta['lossless_cast'] and meta['generation_dtype']=='int32'

def test_exclusive_write_does_not_overwrite(tmp_path):
    path=tmp_path/'preserved.npy';path.write_bytes(b'preserve-me')
    with pytest.raises(FileExistsError):exclusive_array(path,(2,2),np.float64)
    assert path.read_bytes()==b'preserve-me'

def test_count_route_equals_literal_pairing_and_SNR(tmp_path):
    fe,fq=tiny();te,tq,p=aggregate(fe,fq)
    ids=np.array([[0,1,2,3],[3,3,0,1],[2,2,2,2]],np.uint16)
    actual=bootstrap_values(te,tq,ids)
    for k,row in enumerate(ids):
        expected=te[row].sum(axis=0)/tq[row].sum()
        assert np.allclose(actual[k],expected,atol=1e-14,rtol=1e-14)
    assert actual.shape==(3,7,10)

def test_all_primary_signs():
    linear=np.ones((7,10));linear[:,METHODS.index('POOLED_NATURAL')]=4
    linear[:,METHODS.index('LR4_TW5')]=2;linear[:,METHODS.index('GMM_PREPARED_3')]=8
    linear[:,METHODS.index('LR4_TW1')]=4;linear[:,METHODS.index('WHOLE_POOL_M2_R24')]=1
    assert np.allclose(primary_differences(linear),np.full((7,3),10*np.log10(2)))

def test_memmap_full_statistics_tiny(tmp_path):
    fe,fq=tiny();ix=tmp_path/'indices.npy';store_indices(ix,33,0,4,23)
    result=condition_statistics(fe,fq,ix,tmp_path/'linear.npy',tmp_path/'differences.npy',block=4)
    draws=np.load(ix);te,tq,p=aggregate(fe,fq);expected=np.stack([te[d].sum(axis=0)/tq[d].sum() for d in draws])
    assert np.allclose(np.load(tmp_path/'linear.npy'),expected,rtol=1e-14,atol=1e-14)
    assert np.allclose(np.load(tmp_path/'differences.npy'),primary_differences(expected),rtol=1e-14,atol=1e-14)
    assert np.array_equal(adjusted_intervals(tmp_path/'differences.npy'),np.quantile(primary_differences(expected),[TAIL,1-TAIL],axis=0,method='linear'))

@pytest.mark.parametrize('kind',['negative_error','nan_error','zero_truth','nan_truth'])
def test_bad_energy_rejected(kind):
    fe,fq=tiny()
    if kind=='negative_error':fe.flat[0]=-1
    elif kind=='nan_error':fe.flat[0]=np.nan
    elif kind=='zero_truth':fq.flat[0]=0
    else:fq.flat[0]=np.nan
    with pytest.raises(ValueError):aggregate(fe,fq)

@pytest.mark.parametrize('ids',[np.array([[4,0,1,2]],np.uint16),np.array([[0.,1.,2.,3.]]),np.array([[0,1,2]],np.uint16)])
def test_wrong_draws_rejected(ids):
    fe,fq=tiny();te,tq,_=aggregate(fe,fq)
    with pytest.raises(ValueError):bootstrap_values(te,tq,ids)

def test_formal_counts_cannot_be_reduced(tmp_path):
    with pytest.raises(ValueError,match='FROZEN_FORMAL_COUNTS'):analyze_dataset(tmp_path,'formal',33,replicates=23,n=4)

def test_smoke_same_full_chain_no_inferential_intervals(tmp_path):
    (tmp_path/'reports').mkdir();raw=tmp_path/'data/smoke';raw.mkdir(parents=True);receipts=[]
    for ci,c in enumerate(CONDITIONS):
        for i in range(2):
            p=raw/f'{c}_{i}.npz';j=p.with_suffix('.json');fe=np.arange(1.,421.).reshape(7,10,6)+(ci+i);fq=np.arange(1.,7.)
            with p.open('xb') as f:np.savez_compressed(f,frame_error=fe,frame_truth=fq,method_names=np.array(METHODS),snr_db=np.array(SNRS),frame_ids=np.arange(4,10),failure_flags=np.zeros((7,10,6),bool))
            j.write_text(json.dumps(dict(mode='smoke',condition=c,trajectory_id=i,complete=True)),encoding='utf8')
            receipts.append(dict(condition=c,trajectory_id=i,path=p.relative_to(tmp_path).as_posix(),SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),JSON_SHA256=hashlib.sha256(j.read_bytes()).hexdigest()))
    (tmp_path/'reports/smoke_receipt.json').write_text(json.dumps(dict(PASS=True,failures=0,nonfinite=0,receipts=receipts)),encoding='utf8')
    result=analyze_dataset(tmp_path,'smoke',1743,replicates=23,n=2)
    assert result['PASS'] and result['NMSE_rows']==350 and result['primary_rows']==105 and result['smoke_uncertainty_disabled']
    with (tmp_path/'tables/smoke_primary_contrasts.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    assert all(r['CI_lower']=='' and r['CI_upper']=='' and r['classification']=='SMOKE_ONLY' and r['tail_probability']=='' for r in rows)
    assert [r['indices']['shape'] for r in result['records']]==[[23,2]]*5


@pytest.mark.parametrize('dtype',[np.float32,np.complex128])
def test_wrong_precision_or_complex_energy_rejected(dtype):
    fe,fq=tiny()
    with pytest.raises(ValueError,match='ENERGY_FLOAT64_REQUIRED'):aggregate(fe.astype(dtype),fq)
