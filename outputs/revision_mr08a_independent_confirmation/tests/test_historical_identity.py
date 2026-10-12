"""Focused metadata regression; no simulator, inference or real seed allocation."""
import inspect
import pytest
import historical_identity as module

def test_whole_trajectory_hash_shape():
 assert module.h_hash("a"*64) == "a"*64
 for value in ("a"*63, "G"*64, "", 123, None):
  with pytest.raises(ValueError, match="WHOLE_TRAJECTORY"):
   module.h_hash(value)

def test_preparation_rejects_wrong_root_or_path_before_read(tmp_path):
 with pytest.raises(ValueError, match="EXACT_NEW_MR08"):
  module.prepare_seeds("relative", tmp_path)

def test_no_hardcoded_new_seed_values_or_new_rng_system():
 source = inspect.getsource(module.prepare_seeds)
 assert "module.allocate(" in source
 assert "np.random" not in source
 assert "random.seed" not in source
 assert "165001" in source
 assert module.SMOKE_NAMESPACE != module.FORMAL_NAMESPACE
 assert "H_trajectory_sha256" == module.CSV_H["outputs/t20b_formal_manifest.csv"][1]
