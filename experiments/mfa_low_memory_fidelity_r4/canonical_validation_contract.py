"""Future-run assertion design using exact frozen canonical metadata labels.

This is a metadata-only companion to the symbolic counter in validation_contract.
It is NOT integrated into the immutable formal runner in R4.
"""
from itertools import product

def metadata_contract(cfg):
    assert cfg['canonical_origins']==['Source/CDL-C-30','CDL-B-30','CDL-C-120']
    assert cfg['trajectory_ids']['validation']==list(range(400,500))
    assert cfg['frames']['validation']==list(range(4,10))
    assert sorted(cfg['noise_variances'])==['10','20']
    grid=[(x['K'],x['rank']) for x in cfg['candidate_order']]
    assert grid==list(product((1,2,4),(4,8,16)))
    return set(product(grid,cfg['canonical_origins'],(10,20),range(400,500),range(4,10)))

def assert_exact_identity_coverage(tokens,cfg):
    expected=metadata_contract(cfg);seen=set();count=0
    for token in tokens:
        assert token in expected,'UNEXPECTED_CANDIDATE_ORIGIN_SNR_TRAJECTORY_FRAME'
        assert token not in seen,'DUPLICATE_PREDICTION_IDENTITY'
        seen.add(token);count+=1
    assert seen==expected and count==32400,'INCOMPLETE_VALIDATION_GRID'
    for candidate in [(c['K'],c['rank']) for c in cfg['candidate_order']]:
        assert sum(t[0]==candidate for t in seen)==3600
    return {'PASS':True,'scored_frame_predictions':count,'per_candidate':3600,'batch_calls':5400,
        'scope':'metadata-only assertion design; frozen formal runner unchanged'}
