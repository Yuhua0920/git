import json
import numpy as np

def test_audit_energy_equality_is_json_serializable_after_explicit_bool():
    flag=True
    flag &= bool(1.0==np.float64(1.0))
    assert type(flag) is bool
    assert json.loads(json.dumps({'flag':flag}))=={'flag':True}
