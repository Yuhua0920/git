"""EXPERIMENTAL_NOT_FORMAL_BASELINE: metadata-only future count assertions."""
from itertools import product
ORIGINS=('Source_C30','Profile_B30','Mobility_C120')
SNRS=(10,20);IDS=tuple(range(400,500));FRAMES=tuple(range(4,10))
GRID=tuple(product((1,2,4),(4,8,16)))
PER_CANDIDATE=len(ORIGINS)*len(SNRS)*len(IDS)*len(FRAMES)
ALL_CANDIDATES=len(GRID)*PER_CANDIDATE

def expected_tokens():
    return set(product(GRID,ORIGINS,SNRS,IDS,FRAMES))

def assert_coverage(tokens):
    seen=set();duplicate=0
    for token in tokens:
        if token in seen:duplicate+=1
        seen.add(token)
    assert duplicate==0,'DUPLICATE_VALIDATION_PREDICTION'
    assert seen==expected_tokens(),'MISSING_OR_UNEXPECTED_VALIDATION_IDENTITY'
    assert len(seen)==32400
    return {'predictions':len(seen),'per_candidate':3600,'duplicates':0}
