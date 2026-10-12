"""Necessary synthetic integration of the diagnostic runner and frozen infer."""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path('D:/Users/yuhua/Desktop/Project')
sys.path[:0]=[str(ROOT),str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'src_or_scripts')]
import mechanism
import run_mr02 as runner
from scripts import sp_u12b2a_common as common
from src.cdl_benchmark import observation_geometry

@pytest.fixture(scope='module')
def synthetic():
    rng=np.random.default_rng(710026);g=observation_geometry()
    def z(shape):return .05*(rng.normal(size=shape)+1j*rng.normal(size=shape))
    op={'source_mean_p':z((64,)),'source_mean_d':z((3968,)),'pool_mean_p':z((64,)),'pool_mean_d':z((3968,))}
    for s in runner.SNRS:
        for key in ('Wsource','Wpool','DeltaWP','DeltaWM'):op[f'{key}_{s}']=z((3968,64))
        for key in ('KSource','DeltaKP','DeltaKM'):op[f'{key}_{s}']=z((4,16,48))
        for q in ('P','M'):op[f'A_{q}_{s}']=z((3968,4));op[f'Q_{q}_{s}']=z((4,64))
    cvs,factors=common.family(op,runner.SNRS)
    runner.c=common;runner.m=mechanism;runner.g=g;runner.op=op;runner.cvs=cvs;runner.factors=factors
    H=z((2,2,64,160));pilots=z((3,10,64))
    a,audit=runner.diagnostics(H,pilots)
    return H,pilots,a,audit

def test_actual_frozen_infer_shared_coordinates_and_fields(synthetic):
    H,p,a,audit=synthetic
    assert a['online'].shape==(3,6,2)
    assert a['data_H'].shape==(3,6,2,2,2)
    assert a['cross_gram'].shape==(3,6,2,2)
    assert a['pilot_loss'].shape==(3,6,9)
    assert audit['oracle_inequalities_checked']==180
    assert audit['max_compression_scaled_error']<1e-12
    assert audit['max_vector_quadratic_scaled_error']<1e-12

def test_fit_only_reference_then_evaluate_saved_frozen_online(synthetic):
    H,p,a,_=synthetic
    collection={k:np.tile(v,(5,2,*([1]*v.ndim))) for k,v in a.items()}
    refs,rows=runner.fit_arrays(collection)
    assert len(rows)==45
    assert all(r['rank']==2 for r in rows)
    old={'coordinates':a['online'],'normal_gram':a['pilot_S'],'normal_rhs':a['pilot_b'],
         'frame_truth':a['truth'],'frame_error':np.moveaxis(a['baseline_error'],1,2)}
    class Matched:
        def estimate(self,pilot):return runner.op['source_mean_d']+pilot@runner.op['Wsource_10'].T
    result,audit=runner.diagnostics(H,p,original=old,refs={k:v[0] for k,v in refs.items()},matched={s:Matched() for s in runner.SNRS})
    assert result['error'].shape==(3,6,13)
    assert audit['replayed_baseline_bitwise'] and audit['replayed_coordinates_bitwise']
    assert audit['oracle_inequalities_checked']==252
    assert np.all(result['error']>=0)
    for si in range(3):
        for ji in range(6):
            for ai in range(2):
                assert result['oracle_error'][si,ji,ai]<=result['error'][si,ji,4+ai]+1e-10

def test_identity_failure_is_not_skipped(synthetic):
    H,p,a,_=synthetic
    old={'coordinates':a['online'].copy(),'normal_gram':a['pilot_S'],'normal_rhs':a['pilot_b'],
         'frame_truth':a['truth'],'frame_error':np.moveaxis(a['baseline_error'],1,2)}
    old['coordinates'][0,0,0]+=1
    with pytest.raises(ValueError,match='MR01_REPLAY_COORDINATES'):
        runner.diagnostics(H,p,original=old)

def test_window_multiplicity_preserved():
    multiplicity=[sum(j-4<=ell<=j for j in range(4,10)) for ell in range(10)]
    assert multiplicity==[1,2,3,4,5,5,4,3,2,1]
    assert sum(multiplicity)==30
