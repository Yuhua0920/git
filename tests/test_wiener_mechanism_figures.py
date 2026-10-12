"""Small CSV-only plotting/QA before the operator scan; no physical mock data."""
import copy
import json
import numpy as np
import pytest
import matplotlib
from scripts import plot_wiener_mechanism as plot
from scripts.run_cdl_mismatch import csvsave


def fixture_csv(directory):
    # Explicit synthetic plotting fixture, confined to pytest temporary storage.
    closure=[]
    for snr in plot.SNRS:
        for k,(name,*_) in enumerate(plot.CURVES):
            residual=0. if k==0 else k*(snr/5.)**2
            closure.append(dict(snr_db=snr,prediction=name,relative_weighted_residual=residual,closure=1.-residual**2))
    spectra=[]
    for snr in plot.SNRS:
        for name in ('R_exact','R_pred_2'):
            s=np.arange(64,0,-1,dtype=float)**(1 if name=='R_exact' else 2)
            capture=np.cumsum(s*s)/np.sum(s*s)
            for mode,v in enumerate(capture,1):
                spectra.append(dict(snr_db=snr,operator=name,weighting='target_weighted',mode=mode,cumulative_capture=float(v)))
    csvsave(directory/plot.DATA[0],closure);csvsave(directory/plot.DATA[1],spectra)
    return closure,spectra


def test_all_frozen_models_and_all64_modes_mapped(tmp_path):
    c,s=fixture_csv(tmp_path)
    assert len(plot.residual_series(c))==5 and len(plot.spectrum_series(s))==6
    assert len(s)==384


@pytest.mark.parametrize('fault',['duplicate','missing','nonfinite','inconsistent_closure'])
def test_plot_integrity_guards(tmp_path,fault):
    c,_=fixture_csv(tmp_path)
    if fault=='duplicate':c.append(c[0])
    elif fault=='missing':c.pop()
    elif fault=='nonfinite':c[1]['relative_weighted_residual']=float('nan')
    else:c[1]['closure']=.999
    with pytest.raises(ValueError):plot.residual_series(c)


def test_actual_CSV_only_rebuild_QA_and_scoped_style(tmp_path):
    fixture_csv(tmp_path)
    before={k:copy.deepcopy(matplotlib.rcParams[k]) for k in plot.STYLE}
    report=plot.audit(tmp_path,tmp_path/'figures')
    assert report['source_returncode']==0 and all(report['CSV_only_rebuild'].values())
    assert all(v['text_returncode']==v['collision_returncode']==0 for v in report['rendered_checks'].values())
    assert all(matplotlib.rcParams[k]==v for k,v in before.items())
    for stem in plot.STEMS:
        assert json.loads((tmp_path/'figures'/f'{stem}.alignment.json').read_text(encoding='utf8'))
