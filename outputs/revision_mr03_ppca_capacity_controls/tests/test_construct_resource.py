import pytest
import construct_ppca as c

def test_real_construction_blas_contract():
    assert all(x['threads']==6 for x in c.blas_threads(6))
    with pytest.raises(ValueError,match='CONSTRUCTION_BLAS_THREAD_MISMATCH'):
        c.blas_threads(1)

def test_memory_query_reports_ram_and_commit():
    m=c.memory()
    assert 0<=m['available_RAM']<=m['total_RAM']
    assert 0<=m['available_commit']<=m['total_commit']
