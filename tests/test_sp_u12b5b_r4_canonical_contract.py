import sys
from pathlib import Path
import json,pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u12b5b_r4_common import firewall
firewall()
from experiments.mfa_low_memory_fidelity_r4.canonical_validation_contract import metadata_contract,assert_exact_identity_coverage

def config():return json.loads((ROOT/'configs/sp_u12b5b_mfa_selection.yaml').read_text(encoding='utf8'))

def test_canonical_labels_and_complete_grid():
    c=config();result=assert_exact_identity_coverage(metadata_contract(c),c)
    assert result['PASS'] and result['scored_frame_predictions']==32400

@pytest.mark.parametrize('defect',['half','duplicate','alias','wrong_frame'])
def test_canonical_grid_rejects_defect(defect):
    c=config();v=list(metadata_contract(c))
    if defect=='half':v=v[:16200]
    elif defect=='duplicate':v.append(v[0])
    elif defect=='alias':v[0]=v[0][:1]+('Source_C30',)+v[0][2:]
    else:v[0]=v[0][:-1]+(3,)
    with pytest.raises(AssertionError):assert_exact_identity_coverage(v,c)
