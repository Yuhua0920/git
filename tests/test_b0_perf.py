"""Independent historical references and actual Windows-worker calibration."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from src import b0_development as p0, ml_data as md
from src.b0_perf import (T12B_COMMIT, build_batch, batch_pilot_ls, vectorized_r3,
                         EpochBatchSampler, make_loader, close_loader, raw_hash)
from src.b0_model import configure_reproducibility
from src.lmmse_train import formal_manifest_rows, formal_trajectory, sha256_file
from src.pilots import build_fdm_pilot_layout
from src.trajectories import prepare_trajectory

ROOT = Path(__file__).resolve().parents[1]
LAYOUT = build_fdm_pilot_layout(2, 64, 16, 16, 4)


@pytest.fixture(scope="module")
def truth():
    prepared = prepare_trajectory(); rows = formal_manifest_rows()
    return np.concatenate([formal_trajectory(prepared, rows[i]).h_frames for i in range(2)])


def test_accepted_T12B_commit_and_old_core_hashes():
    assert subprocess.check_output(["git", "show", "-s", "--format=%s", T12B_COMMIT], cwd=ROOT, text=True).strip() == "feat: add first full B0 dF16 development training"
    old = json.loads((ROOT/"outputs/t12b_b0_df16_dev_results.json").read_text(encoding="utf8"))
    for path, digest in old["provenance"]["source_sha256"].items():
        assert sha256_file(ROOT/path) == digest
    assert old["all_checks_passed"]


@pytest.mark.parametrize("epoch", [1, 2, 7, 30])
def test_full_complex128_and_model_input_exact_with_original_R3(truth, epoch):
    indices = np.arange(20)
    old = build_batch(truth, epoch, indices, "P0", include_complex=True)
    new = build_batch(truth, epoch, indices, include_complex=True)
    assert old["hashes"] == new["hashes"]
    for key in ("base_complex", "H_true", "unit_noise"):
        assert np.array_equal(old[key], new[key])
    assert torch.equal(old["base"], new["base"]) and torch.equal(old["target"], new["target"])
    for j in range(20):
        original = md.direct_pilot_ls(truth[j], LAYOUT, old["unit_noise"][j], int(old["SNR"][j]))
        batched = batch_pilot_ls(new["H_true"], new["unit_noise"], new["SNR"].numpy(), LAYOUT)[j]
        assert np.array_equal(original, batched, equal_nan=True)


def test_order_filters_global_permutation_without_shuffle():
    subset = p0.epoch_order(1)[:1024]
    for epoch in (1, 2):
        sampler = EpochBatchSampler(1024, epoch, subset)
        expected = p0.epoch_order(epoch)[np.isin(p0.epoch_order(epoch), subset)]
        assert np.array_equal(sampler.ordered_ids(), expected)
        assert all(len(key[1]) == 128 and key[0] == epoch for key in sampler)
        assert raw_hash(md.balanced_training_snr(p0.SEED, epoch)[expected]) == raw_hash(md.balanced_training_snr(p0.SEED, epoch)[sampler.ordered_ids()])


@pytest.mark.parametrize("workers,persistent", [(0, False), (2, False), (2, True)])
def test_actual_spawn_worker_epoch_noise_order_and_hashes(tmp_path, truth, workers, persistent):
    path = tmp_path/"h.npy"; np.save(path, truth)
    sampler = EpochBatchSampler(20, 1, np.arange(20))
    loader = make_loader(path, sampler, workers, persistent=persistent)
    try:
        for epoch in (1, 2):
            sampler.epoch = epoch
            batches = list(loader)
            expected = build_batch(truth, epoch, sampler.ordered_ids(), "P0")
            assert len(batches) == 1 and batches[0]["hashes"] == expected["hashes"]
    finally:
        close_loader(loader)


def test_no_epoch31_or_wrong_density_or_precision(truth):
    with pytest.raises(ValueError):
        build_batch(truth, 31, np.arange(20))
    a = build_batch(truth, 1, np.arange(20), include_complex=True)
    observations = np.full((20, 2, 2, 64, 16), np.nan+1j*np.nan, np.complex128)
    for q in range(2):
        observations[:, :, q, LAYOUT.pilot_mask[q]] = a["H_true"][:, :, q, LAYOUT.pilot_mask[q]]
    with pytest.raises(ValueError):
        vectorized_r3(observations.astype(np.complex64), LAYOUT)
    with pytest.raises(ValueError):
        vectorized_r3(observations, build_fdm_pilot_layout(2, 64, 16, 8, 4))


def test_GPU_original_update_pinned_nonblocking_and_stream_exact(truth):
    from src.b0_perf_runtime import EventProfile, profile_original_training, train_stream
    configure_reproducibility(p0.SEED)
    ids = np.resize(np.arange(20), 128)
    batch = build_batch(truth, 1, ids)
    checkpoint = ROOT/"outputs/t12b_b0_df16_dev_epoch30.pt"
    a, _ = p0.checkpoint_load(checkpoint, torch.device("cuda"))
    b, _ = p0.checkpoint_load(checkpoint, torch.device("cuda"))
    recorder = EventProfile(); optimizer = p0.optimizer_for(a)
    with profile_original_training(optimizer, recorder):
        p0.train_epoch(a, optimizer, batch["base"], batch["target"], batch["SNR"].numpy(), LAYOUT.data_mask, np.arange(128))
    for key in ("base", "target"):
        batch[key] = batch[key].pin_memory()
        copied = batch[key].to("cuda", non_blocking=True)
        assert raw_hash(copied) == raw_hash(batch[key])
    result = train_stream(b, p0.optimizer_for(b), [batch], LAYOUT.data_mask, non_blocking=True)
    assert result["loss_trace"] == recorder.losses
    assert p0.state_hash(a) == p0.state_hash(b)
    assert result["pilot_modification"] == 0


@pytest.fixture(scope="module")
def report():
    path = ROOT/"outputs/t12c_perf_results.json"
    if not path.exists():
        pytest.skip("full engineering benchmark not executed yet")
    return json.loads(path.read_text(encoding="utf8"))


def test_artifact_scientific_protocol_and_frozen_files(report):
    assert report["task"] == "T12C-PERF"
    assert not any(report[k] for k in ("scientific_protocol_changed", "training_performed", "formal_test_used", "AMP_used"))
    assert report["batch_size"] == 128 and report["calibration_optimizer_updates_only"]
    assert report["provenance"]["accepted_before"] == report["provenance"]["accepted_after"]
    for path, digest in report["provenance"]["accepted_after"].items():
        assert sha256_file(ROOT/path) == digest


def test_full_training_equivalence_and_update_replay(report):
    assert report["sample_hash_equivalence"]["all_checks_passed"]
    assert report["sample_hash_equivalence"]["full_training_frames"] == 20000
    assert report["R3_equivalence"]["complex128_max_error"] == 0
    assert report["R3_equivalence"]["float32_bitwise_equal"]
    replay = report["state_dict_replay"]
    assert replay["loss_trace_exact"] and replay["batch_IDs_exact"] and replay["final_state_exact"]
    assert replay["optimizer_steps_each_path"] == 16
    assert report["full_epoch_update_regression"]["loss_trace_exact"]
    assert report["full_epoch_update_regression"]["final_state_exact"]


def test_validation_checkpoint_prediction_metric_and_pinned_equivalence(report):
    val = report["checkpoint_inference_regression"]
    assert val["observations_each_path"] == 20000 and val["all_checks_passed"]
    assert val["checkpoint_file_unchanged"] and val["checkpoint_state_unchanged"]
    for row in report["validation_profiles"]:
        assert row["input_exact"] and row["predictions_exact"] and row["metrics_exact"]
    assert report["pinned_nonblocking_equivalence"]["all_checks_passed"]


def test_measured_candidate_coverage_and_memory_gate(report):
    assert {r["cpu_threads"] for r in report["thread_profiles"]} == {4, 8, 12, 14}
    assert {r["workers"] for r in report["worker_profiles"]} == {0, 2, 4, 8}
    assert {r["batch_size"] for r in report["future_batch_benchmark"]} == {128, 256, 512}
    assert report["selected_profile"]["equivalence_passed"]
    assert report["selected_profile"]["memory_safe"]
    for row in report["worker_profiles"]:
        if row["completed"]:
            assert row["sample_hashes_exact"] and row["ordered_IDs_exact"]
    assert report["speedup"] > 1
