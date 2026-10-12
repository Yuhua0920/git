"""Independent spectral Gram and observed poor-result correctness regression."""
import json

import numpy as np

from scripts.audit_robust_lmmse import ROOT,independent_spectral_gram
from src import robust_lmmse as hr
from src.lmmse_train import sha256_file


def test_independent_positive_spectral_Gram_matches_fixed_prior():
    analytic,_=hr.build_prior()
    np.testing.assert_allclose(independent_spectral_gram(),analytic,atol=1e-12,rtol=0)


def test_recorded_poor_result_independently_verified_and_no_rescan():
    audit=json.loads((ROOT/"outputs/t13b_robust_lmmse_independent_audit.json").read_text(encoding="utf8"))
    assert audit["all_checks_passed"] and all(audit["checks"].values())
    assert not audit["numerical_scan_rerun"] and not audit["new_seed_used"]
    assert audit["source_sha256"]==sha256_file(ROOT/"scripts/audit_robust_lmmse.py")
    assert audit["prior_mismatch_error_per_data_RE"]>100*audit["observation_noise_error_per_data_RE"]
