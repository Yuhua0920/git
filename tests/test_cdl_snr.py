"""T15A-FIX small SNR mathematics and original-output preservation."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from src.cdl_snr import (useful_power_per_rx,TrainReceivePowerAccumulator,
    ReceivePowerReference,TRAIN_IDS)

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("snr",[5,10,20])
def test_tdl_snr_consistency(snr):
    p=useful_power_per_rx(np.ones((2,2)))
    np.testing.assert_array_equal(p,[1,1])
    reference=ReceivePowerReference(tuple(p),TRAIN_IDS)
    assert reference.reference==1.
    assert reference.noise_variance(snr)==pytest.approx(10**(-snr/10),rel=1e-15)


def test_train_only_power_and_equal_trajectory_weights():
    accumulator=TrainReceivePowerAccumulator()
    h=np.ones((2,2,3,4),np.complex128);h[0]*=2;h[1]*=3
    for i in TRAIN_IDS: accumulator.update(i,h)
    ref=accumulator.freeze()
    assert ref.per_rx_signal_power==(4.,9.)
    assert ref.reference==6.5
    assert ref.train_trajectory_ids==TRAIN_IDS


@pytest.mark.parametrize("trajectory_id",[500,599,-1])
def test_no_validation_calibration(trajectory_id):
    a=TrainReceivePowerAccumulator()
    with pytest.raises(ValueError): a.update(trajectory_id,np.ones((2,2,1,1)))
    assert not a._ids and not a._link_power_sum.any()


def test_incomplete_or_duplicate_train_set_rejected():
    a=TrainReceivePowerAccumulator();h=np.ones((2,2,1,1))
    a.update(0,h)
    with pytest.raises(ValueError): a.freeze()
    with pytest.raises(ValueError): a.update(0,h)


@pytest.mark.parametrize("snr",[5,10,20])
def test_target_snr_identity_and_common_noise(snr):
    ref=ReceivePowerReference((4.,9.),TRAIN_IDS)
    sigma=ref.noise_variance(snr)
    assert np.ndim(sigma)==0
    assert 10*np.log10(ref.reference/sigma)==pytest.approx(snr,abs=2e-15)
    np.testing.assert_allclose(ref.branch_offsets_db(),10*np.log10(np.array([4.,9.])/6.5),atol=0,rtol=0)


def test_power_accumulation_does_not_normalize_or_mutate_h():
    a=TrainReceivePowerAccumulator();h=np.arange(1,17).reshape(2,2,2,2).astype(np.complex128)
    before=h.copy();a.update(0,h)
    assert np.array_equal(h,before)
    np.testing.assert_array_equal(a._link_power_sum,np.mean(abs(before)**2,axis=(2,3)))


def test_nominal_attempt_archive_preserves_all_files():
    archive=ROOT/"outputs/t15a_attempt01_nominal_tx_snr"
    metadata=json.loads((archive/"superseded_metadata.json").read_text(encoding="utf8"))
    assert metadata["status"]=="superseded_due_to_snr_semantics"
    assert metadata["original_JSON_preserved_byte_for_byte"]
    for path,info in metadata["file_hashes"].items():
        assert hashlib.sha256((archive/path).read_bytes()).hexdigest()==info["sha256"]
    for n in (100,250,500):
        path=f"outputs/t15a_cdl_statistics_{n}.npz"
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==metadata["file_hashes"][path]["sha256"]


def test_failed_historical_gate_blocks_corrected_filters_and_metrics():
    from scripts.fix_cdl_snr import corrected_run
    # Invalid remaining inputs prove the gate exits before touching data/filter.
    with pytest.raises(ValueError,match="historical attempt01 replay gate"):
        corrected_run({"all_checks_passed":False},None,None,None,None)


def test_historical_metric_comparator_is_exact(tmp_path):
    from scripts.fix_cdl_snr import compare_metric_rows
    p=tmp_path/"old.csv";p.write_text("value,label\n1.0,A\n",encoding="utf8")
    assert compare_metric_rows([dict(value=1.0,label="A")],p)["exact"]
    mismatch=compare_metric_rows([dict(value=np.nextafter(1.,2.),label="A")],p)
    assert not mismatch["exact"] and not mismatch["tolerance_used"]
    assert mismatch["maximum_numeric_errors"]["value"]["max_abs_error"]>0


def test_replay_hashes_detect_mutation_and_readonly_inputs():
    from scripts.fix_cdl_snr import freeze_replay_inputs,replay_input_hashes,assert_replay_unchanged
    h=np.ones((2,2,2,2),np.complex128)
    communication=[(np.zeros((2,2,2,2),np.uint8),np.ones((2,2,2),np.complex128),np.ones((2,2,2),np.complex128))]
    original=replay_input_hashes(h,communication)
    freeze_replay_inputs(h,communication)
    assert assert_replay_unchanged(h,communication,original)==original
    with pytest.raises(ValueError):h[0,0,0,0]=2
    changed=h.copy();changed[0,0,0,0]=2
    with pytest.raises(ValueError,match="changed"):assert_replay_unchanged(changed,communication,original)


def test_frozen_statistics_loader_never_refits():
    from scripts.fix_cdl_snr import load_frozen_statistics
    old=json.loads((ROOT/"outputs/t15a_attempt01_nominal_tx_snr/outputs/t15a_cdl_results.json").read_text(encoding="utf8"))
    stats,checks=load_frozen_statistics(old)
    for n in (100,250,500):
        assert checks[str(n)]["unchanged"]
        assert not stats[n].rpp.flags.writeable and not stats[n].mean_d.flags.writeable
        assert stats[n].train_trajectory_ids==tuple(range(n))


def test_corrected_artifacts_if_generated():
    r=json.loads((ROOT/"outputs/t15a_cdl_results.json").read_text(encoding="utf8"))
    if r.get("task")!="T15A-FIX":return
    assert r["historical_full_frame_hashes_available"] is False
    assert r["deterministic_cdl_replay_used"] and r["historical_six_h_hashes_matched"]
    assert r["historical_attempt01_metrics_reproduced"] and r["clean_channel_statistics_unchanged"]
    assert r["snr_semantics_repaired"] and r["attempt01_superseded"]
    assert r["empirical_lmmse_stable"] is None
    assert r["empirical_lmmse_stability_assessment"]=="descriptive_only"
    assert r["train_reference_definition"]["validation_used_for_calibration"] is False
    assert r["train_reference_definition"]["common_noise_variance"]
    assert r["replay_shared_input_identity"]["all_unchanged"]
    assert all(r["noise_independent_diagnostics_unchanged"].values())
    for variance in r["noise_variances"]:
        assert variance["new_sigma2"]==r["train_reference_signal_power"]/10**(variance["snr_db"]/10)
        assert variance["target_identity_db"]==pytest.approx(variance["snr_db"],abs=2e-15)
    assert all(a["reference_recalibrated_from_validation"] is False for a in r["validation_realized_snr_audit"])
    assert r["formal_test_used"] is False and r["formal_test_sealed"] is True


def test_no_generator_or_normalization_in_corrected_phase():
    import ast,inspect
    from scripts.fix_cdl_snr import corrected_run
    source=inspect.getsource(corrected_run)
    assert "generate(" not in source and "normalize" not in source
    original=ROOT/"outputs/t15a_attempt01_nominal_tx_snr/src/cdl_benchmark.py"
    assert original.read_bytes()==(ROOT/"src/cdl_benchmark.py").read_bytes()
    tree=ast.parse((ROOT/"src/cdl_benchmark.py").read_text(encoding="utf8"))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="cir_to_ofdm_channel"]
    assert len(calls)==1
    assert any(k.arg=="normalize" and isinstance(k.value,ast.Constant) and k.value.value is False for k in calls[0].keywords)
