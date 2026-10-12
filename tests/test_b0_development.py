"""T12B protocol gates, independent sentinels and full-run artifact regression."""

import csv
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src import b0_development as dev
from src import ml_data as md
from src.b0_model import ResidualCNN, configure_reproducibility, predict_channel
from src.lmmse_train import formal_manifest_rows, formal_trajectory, sha256_file
from src.nearest_validation import load_references
from src.pilots import build_fdm_pilot_layout
from src.trajectories import prepare_trajectory

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs"


def setup_module():
    configure_reproducibility(dev.SEED)


def test_T12A_is_standalone_accepted_commit_with_exact_files():
    subject = subprocess.check_output(["git", "show", "-s", "--format=%s", dev.T12A_COMMIT], cwd=ROOT, text=True).strip()
    assert subject == "feat: add calibrated B0 residual-CNN learning pipeline"
    files = subprocess.check_output(["git", "diff-tree", "--no-commit-id", "--name-only", "-r", dev.T12A_COMMIT], cwd=ROOT, text=True).splitlines()
    assert len(files) == 11 and "src/b0_model.py" in files and "outputs/t12a_b0_pipeline_results.json" in files


@pytest.mark.parametrize("epoch", list(range(1, 31)))
def test_balanced_20000_frame_schedule_and_deterministic_shuffle(epoch):
    snrs = md.balanced_training_snr(dev.SEED, epoch)
    order = dev.epoch_order(epoch)
    assert np.array_equal(np.unique(snrs, return_counts=True)[1], np.full(5, 4000))
    assert np.array_equal(np.sort(order), np.arange(20000))
    assert np.array_equal(order, dev.epoch_order(epoch))
    assert not np.array_equal(order, md.balanced_training_snr(dev.SEED, epoch))


@pytest.mark.parametrize("epoch", [0, 31, True, -1])
def test_training_epoch_outside_fixed_budget_rejected(epoch):
    with pytest.raises(ValueError):
        dev.epoch_order(epoch)


def test_training_noise_changes_only_epoch_and_validation_stays_frozen():
    a = md.training_unit_noise(dev.SEED, 1, 9, 2)
    assert not np.array_equal(a, md.training_unit_noise(dev.SEED, 2, 9, 2))
    assert np.array_equal(a, md.training_unit_noise(dev.SEED, 1, 9, 2))
    reference = load_references(OUT)["realizations"][2000]["frame_realizations"][0]
    assert dev.raw_array_sha256(md.validation_unit_noise(2000, 0)) == reference["unit_noise_sha256"]


def test_optimizer_is_exact_Adam_without_weight_decay_or_scheduler():
    opt = dev.optimizer_for(ResidualCNN())
    assert type(opt) is torch.optim.Adam
    assert opt.param_groups[0]["lr"] == 0.001
    assert opt.param_groups[0]["weight_decay"] == 0
    assert dev.BATCH_SIZE == 128 and dev.EPOCHS == 30 and dev.SEED == 13261001


def test_small_two_step_training_and_safe_checkpoint_reload(tmp_path):
    # Calibration only; no additional full development model or random experiment seed.
    model = ResidualCNN()
    base = torch.full((8, 8, 64, 16), 0.4)
    target = 0.7*base
    mask = build_fdm_pilot_layout(2, 64, 16, 16, 4).data_mask
    result = dev.train_epoch(model, dev.optimizer_for(model), base, target,
        np.array([0, 5, 10, 15, 20, 0, 5, 10]), mask, np.arange(8))
    assert result["maximum_pilot_modification"] == 0 and result["loss_weights_gradients_finite"]
    assert sum(r["sample_count"] for r in result["train_per_snr"]) == 8
    assert result["train_mse"] == pytest.approx(2*(0.4-0.28)**2, rel=1e-6)
    assert result["gradient_norm"] > 0
    for epoch in (3, 30):
        file = tmp_path/f"{epoch}.pt"
        saved = dev.checkpoint_save(file, model, epoch, result["train_mse"], {"parameter_count": 23144})
        rng = torch.random.get_rng_state().clone()
        loaded, metadata = dev.checkpoint_load(file, torch.device("cpu"))
        assert torch.equal(rng, torch.random.get_rng_state())
        assert saved["state_dict_tensor_sha256"] == dev.state_hash(model) == dev.state_hash(loaded)
        assert torch.equal(predict_channel(loaded, base, mask), predict_channel(model, base, mask))
        assert not metadata["counts_toward_formal_five_seed_results"]


def test_selection_ties_keep_earlier_and_never_use_baseline_or_BER():
    assert dev.select_best(0.3, 0.4)
    assert not dev.select_best(0.3, 0.3)
    assert not dev.select_best(0.4, 0.3)
    with pytest.raises(FloatingPointError):
        dev.select_best(float("nan"), 1)


@pytest.mark.parametrize("b0, expected", [(-10.10001, True), (-10.10, False), (-10.09, False), (1, False)])
def test_predeclared_Oracle_red_flag_threshold(b0, expected):
    assert dev.oracle_red_flag([{"b0_nmse_db": b0, "r5_oracle_nmse_db": -10}]) is expected


def test_fixed_T10D_full_link_and_R4_sentinel_before_expensive_training():
    layout = build_fdm_pilot_layout(2, 64, 16, 16, 4)
    h = formal_trajectory(prepare_trajectory(), formal_manifest_rows()[2000]).h_frames[0]
    z = md.validation_unit_noise(2000, 0)
    bits, y = dev.bits_and_y(h, 2000, 0, z, 10, layout, dev.replay_link(ROOT))
    ls, r3 = dev.link.estimate_ls_linear_channel(y, layout)
    b0 = md.tensor_to_complex(predict_channel(ResidualCNN(), md.complex_to_tensor(r3), layout.data_mask))[0]
    diagnostic = dev.fixed_sample(ROOT, h, r3, b0, ls, y, bits, layout)
    assert diagnostic["frozen_T11CB_sentinel_exact_match"]
    assert diagnostic["norm_B0_minus_R3_over_R3"] < 1e-7
    assert not diagnostic["R4_full_performance_scan"]
    assert md.full_link_pilot_sentinel(h, layout, z, 10)["machine_precision_match"]


def test_csv_only_plot_export_and_rendered_QA_gate(tmp_path):
    from scripts.run_b0_development import render_and_audit
    from scripts.run_lmmse_train_validation import save_csv
    from scripts.plot_b0_development import FIGURE_NAMES
    # Explicit test fixture, not an experimental observation; retained only in pytest tmp.
    log = [{"epoch": i, "train_mse": None if i == 0 else 1/(i+1), "val_mse_pooled": 1/(i+2)} for i in range(31)]
    frozen = load_references(OUT)["scan"]
    rows = []
    for snr in md.SNR_DB:
        old = next(r for r in frozen if int(r["dF"]) == 16 and int(r["snr_db"]) == snr)
        rows.append({"snr_db": snr, "r3_nmse_db": old["ls_nmse_db"], "b0_nmse_db": old["ls_nmse_db"],
            "r4_nmse_db": old["train_nmse_db"], "r5_oracle_nmse_db": old["oracle_nmse_db"],
            "r3_ber": old["ls_ber"], "b0_ber": old["ls_ber"], "r4_ber": old["train_ber"],
            "r5_ber": old["oracle_ber"], "perfect_ber": old["perfect_ber"]})
    save_csv(tmp_path/f"{dev.PREFIX}_training_log.csv", log)
    save_csv(tmp_path/f"{dev.PREFIX}_validation.csv", rows)
    with patch.object(dev, "full_link_validation", side_effect=AssertionError("CSV only")), \
         patch.object(md, "direct_pilot_ls", side_effect=AssertionError("CSV only")):
        records = render_and_audit(tmp_path, tmp_path/"figures")
    assert len(records) == 3
    for name, record in zip(FIGURE_NAMES, records):
        assert record["no_data_exclusions"]
        for ext in ("png", "pdf", "svg"):
            assert (tmp_path/"figures"/f"{name}.{ext}").is_file()


@pytest.fixture(scope="module")
def report():
    file = OUT/f"{dev.PREFIX}_results.json"
    if not file.exists():
        pytest.skip("full development run has not executed yet")
    return json.loads(file.read_text(encoding="utf8"))


def test_full_run_identity_sealed_sets_and_frozen_sources(report):
    assert report["numeric_checks_passed"] and report["development_model"]
    assert report["development_seed"] == 13261001 and report["density"] == 16
    assert not report["counts_toward_formal_five_seed_results"]
    assert report["train_protocol"]["trajectory_ids"] == list(range(2000))
    assert report["validation_protocol"]["trajectory_ids"] == list(range(2000, 2400))
    assert report["formal_test_sealed"] and not report["formal_test_used"] and not report["OOD_generated"]
    assert not report["normalization_used"] and not report["mask_or_SNR_input"]
    assert report["validation_protocol"]["full_link_performance_paths_added"] == ["best B0+ZF"]
    assert report["provenance"]["accepted_before"] == report["provenance"]["accepted_after"] == dev.accepted_files(ROOT)
    for path, digest in report["provenance"]["source_sha256"].items():
        assert sha256_file(ROOT/path) == digest


def test_complete_30_epoch_budget_and_pooled_checkpoint_selection(report):
    assert len(report["training_curve"]) == len(report["epoch_records"]) == 31
    assert report["batch_size"] == 128 and report["optimizer"] == "Adam" and report["lr"] == 0.001
    assert report["max_epochs"] == 30 and report["weight_decay"] == 0
    assert not any(report[f] for f in ("early_stopping", "scheduler", "gradient_clipping", "EMA"))
    values = [r["val_mse_pooled"] for r in report["training_curve"]]
    assert report["best_epoch"] == int(np.argmin(values))
    for epoch in report["epoch_records"][1:]:
        train = epoch["training"]
        assert train["optimizer_steps"] == 157
        assert all(r["sample_count"] == 4000 for r in train["train_per_snr"])
        assert train["loss_weights_gradients_finite"] and train["maximum_pilot_modification"] == 0
        assert epoch["validation"]["validation_observations"] == 20000
    assert report["epoch0_reference"]["bitwise_quantized_R3_match"]
    assert all(abs(r["nmse_db_difference"]) <= 1e-4 for r in report["epoch0_reference"]["records"])


@pytest.mark.parametrize("checkpoint", ["best", "epoch30"])
def test_full_run_checkpoints_reload_with_hash_and_metadata(report, checkpoint):
    record = report["checkpoints"][checkpoint]
    file = OUT/f"{dev.PREFIX}_{checkpoint}.pt"
    assert sha256_file(file) == record["file_sha256"]
    model, meta = dev.checkpoint_load(file, torch.device("cpu"))
    assert dev.state_hash(model) == record["state_dict_tensor_sha256"]
    assert meta["epoch"] == (report["best_epoch"] if checkpoint == "best" else 30)


def test_exact_replay_rows_and_unmodified_baseline_metrics(report):
    refs = load_references(OUT)
    with (OUT/f"{dev.PREFIX}_validation_trajectory_metrics.csv").open(encoding="utf8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2000
    assert {(int(r["trajectory_id"]), int(r["snr_db"])) for r in rows} == {(i, s) for i in range(2000, 2400) for s in md.SNR_DB}
    for row in rows:
        old = refs["trajectory_index"][int(row["trajectory_id"]), 16, int(row["snr_db"])]
        assert int(row["frames_aggregated"]) == 10
        assert int(row["total_bits"]) == int(old["total_data_bits"])
        assert float(row["r3_nmse_error_energy"]) == float(old["ls_nmse_error_energy"])
        assert float(row["r4_nmse_error_energy"]) == float(old["train_nmse_error_energy"])
        assert int(row["r3_bit_errors"]) == int(old["ls_bit_errors"])
        assert int(row["r4_bit_errors"]) == int(old["train_bit_errors"])
        assert all(row[key] == "True" for key in ("same_H", "same_Y", "same_bits"))
    assert len(report["validation_aggregate"]) == 5
    assert report["exact_T10D_replay"] == {"H_long_hashes_matched": 400, "H_frame_hashes_matched": 4000,
        "bits_hashes_matched": 4000, "unit_noise_hashes_matched": 4000,
        "Y_exact_committed_T10D_link_matched": 20000, "all_checks_passed": True}
    for row in report["validation_aggregate"]:
        old = next(r for r in refs["scan"] if int(r["dF"]) == 16 and int(r["snr_db"]) == row["snr_db"])
        for new, prior in (("r3_nmse_db", "ls_nmse_db"), ("r4_nmse_db", "train_nmse_db"), ("r5_oracle_nmse_db", "oracle_nmse_db"),
                           ("r3_ber", "ls_ber"), ("r4_ber", "train_ber"), ("r5_ber", "oracle_ber"), ("perfect_ber", "perfect_ber")):
            assert row[new] == float(old[prior])
        group = [r for r in rows if int(r["snr_db"]) == row["snr_db"]]
        energy = sum(float(r["b0_nmse_error_energy"]) for r in group)
        truth = sum(float(r["nmse_truth_energy"]) for r in group)
        bits = sum(int(r["total_bits"]) for r in group)
        assert energy/truth == row["b0_nmse_linear"]
        assert sum(int(r["b0_bit_errors"]) for r in group)/bits == row["b0_ber"]
        assert row["r3_r4_gap_closed_fraction"] == row["r3_to_b0_gain_db"]/row["r3_to_r4_gain_db"]
    assert report["oracle_red_flag"] is dev.oracle_red_flag(report["validation_aggregate"])
