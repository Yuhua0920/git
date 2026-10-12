"""T12D gates, immutable originals, bridge math and complete Adam checkpoints."""
import json
import subprocess
from pathlib import Path
import numpy as np
import pytest
import torch
from src import b0_development as old, b0_dev60 as dev, b0_perf as perf, ml_data as md
from src.b0_model import ResidualCNN, configure_reproducibility
from src.lmmse_train import formal_manifest_rows, formal_trajectory, sha256_file
from src.trajectories import prepare_trajectory

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs"


def test_T12C_commit_and_selected_profile_exact():
    assert subprocess.check_output(["git", "show", "-s", "--format=%s", dev.T12C_COMMIT], cwd=ROOT, text=True).strip() == "feat: optimize B0 training pipeline without protocol changes"
    report = json.loads((OUT/"t12c_perf_results.json").read_text(encoding="utf8"))
    profile = dev.selected_profile(report)
    assert all(profile[k] == v for k, v in dev.PROFILE.items())
    for path, digest in report["provenance"]["source_sha256"].items():
        assert sha256_file(ROOT/path) == digest
    wrong = {**report, "selected_profile": {**report["selected_profile"], "workers": 4}}
    with pytest.raises(ValueError):
        dev.selected_profile(wrong)


@pytest.mark.parametrize("epoch", list(range(1, 31)))
def test_first30_schedule_order_and_SNR_hashes_original(epoch):
    before = json.loads((OUT/"t12b_b0_df16_dev_results.json").read_text(encoding="utf8"))["epoch_records"][epoch]
    assert np.array_equal(dev.epoch_order(epoch), old.epoch_order(epoch))
    assert perf.raw_hash(dev.epoch_order(epoch)) == before["shuffle_order_sha256"]
    assert perf.raw_hash(md.balanced_training_snr(old.SEED, epoch)) == before["SNR_assignment_sha256"]


@pytest.mark.parametrize("epoch", [31, 45, 60])
def test_extended_epoch_policy_uses_same_domains(epoch):
    expected = np.random.default_rng(np.random.SeedSequence([12004, 13261001, epoch])).permutation(20000)
    assert np.array_equal(dev.epoch_order(epoch), expected)
    assert np.array_equal(np.unique(md.balanced_training_snr(old.SEED, epoch), return_counts=True)[1], np.full(5, 4000))


@pytest.mark.parametrize("epoch", [0, 61, True, -1])
def test_budget_rejects_invalid_epochs(epoch):
    with pytest.raises(ValueError):
        dev.epoch_order(epoch)


@pytest.fixture(scope="module")
def truth():
    prepared = prepare_trajectory(); rows = formal_manifest_rows()
    return np.concatenate([formal_trajectory(prepared, rows[i]).h_frames for i in range(2)])


@pytest.mark.parametrize("epoch", [1, 2, 7, 30, 31, 60])
def test_batch_math_exact_frozen_original(truth, epoch):
    ids = np.arange(20); batch = dev.training_batch(truth, epoch, ids)
    if epoch <= 30:
        assert batch["hashes"] == perf.build_batch(truth, epoch, ids, "P0")["hashes"]
    for j in range(20):
        z = md.training_unit_noise(old.SEED, epoch, j//10, j%10)
        reference = md.complex_to_tensor(md.r3_from_pilot_ls(md.direct_pilot_ls(truth[j], dev.LAYOUT, z, int(batch["SNR"][j])), dev.LAYOUT))[0]
        assert torch.equal(batch["base"][j], reference)


def test_actual_persistent_worker_transition_keeps_explicit_epochs(tmp_path, truth):
    path = tmp_path/"h.npy"; np.save(path, truth)
    sampler = dev.Dev60Sampler(20, 30, np.arange(20)); loader = dev.training_loader(path, sampler)
    try:
        for epoch in (30, 31):
            sampler.epoch = epoch
            actual = list(loader)
            reference = dev.training_batch(truth, epoch, sampler.ordered_ids())
            assert len(actual) == 1 and actual[0]["hashes"] == reference["hashes"]
    finally:
        perf.close_loader(loader)


def test_bounded_hash_and_direct_H_cache_exact_original(tmp_path, truth):
    hashes = dev.prepare_H_cache(tmp_path/"direct.npy", prepare_trajectory(), formal_manifest_rows()[:2])
    array = np.load(tmp_path/"direct.npy", mmap_mode="r")
    assert np.array_equal(array, truth) and len(hashes) == 2
    tensor = md.complex_to_tensor(truth)
    assert dev.tensor_stream_hash(tensor) == perf.raw_hash(tensor)
    assert dev.tensor_stream_hash(tensor.to("cuda")) == perf.raw_hash(tensor)
    grouped = torch.stack((tensor, tensor))
    assert dev.tensor_stream_hash(grouped) == perf.raw_hash(grouped)


def test_original_initialization_and_T12B_optimizer_audit():
    configure_reproducibility(old.SEED)
    report = json.loads((OUT/"t12b_b0_df16_dev_results.json").read_text(encoding="utf8"))
    model = ResidualCNN()
    assert old.state_hash(model) == report["epoch0_state_dict_sha256"]
    assert sum(p.numel() for p in model.parameters()) == 23144
    payload = torch.load(OUT/"t12b_b0_df16_dev_epoch30.pt", weights_only=True, map_location="cpu")
    assert "state_dict" in payload and "optimizer_state_dict" not in payload
    assert not any("optim" in k.lower() for k in payload)


def test_bridge_original_training_exact_and_full_checkpoint_resume(tmp_path, truth):
    configure_reproducibility(old.SEED)
    a = ResidualCNN().to("cuda"); b = ResidualCNN().to("cuda"); b.load_state_dict(a.state_dict())
    oa, ob = old.optimizer_for(a), old.optimizer_for(b)
    ids = np.resize(np.arange(20), 128); batch = dev.training_batch(truth, 1, ids)
    direct = old.train_epoch(a, oa, batch["base"], batch["target"], batch["SNR"].numpy(), dev.LAYOUT.data_mask, np.arange(128))
    # The original trainer receives canonical IDs from the index bridge.
    adapted, _, _ = dev.train_epoch(b, ob, [batch], ids, 1)
    assert direct == adapted
    assert old.state_hash(a) == old.state_hash(b)
    assert dev.structured_hash(oa.state_dict()) == dev.structured_hash(ob.state_dict())
    original_id = id(ob); before = dev.optimizer_steps(ob)
    file = tmp_path/"complete.pt"; protocol = sha256_file(ROOT/"docs/b0_dev60_protocol.md")
    record = dev.checkpoint_save(file, b, ob, 1, .25, protocol, {"parameter_count": 23144})
    loaded, restored, payload = dev.checkpoint_load(file, torch.device("cuda"), protocol)
    assert record["model_and_optimizer_exact_reload"] and payload["optimizer_total_steps"] == before
    for field in ("model_state_dict", "optimizer_state_dict", "epoch", "training_seed", "density", "optimizer_config", "protocol_hash"):
        assert field in payload
    assert dev.structured_hash(ob.state_dict()) == dev.structured_hash(restored.state_dict())
    # A calibration update proves both original live Adam and saved/resumed Adam retain moments.
    dev.train_epoch(b, ob, [batch], ids, 1); dev.train_epoch(loaded, restored, [batch], ids, 1)
    assert id(ob) == original_id and dev.optimizer_steps(ob) == before+1
    assert old.state_hash(b) == old.state_hash(loaded)
    assert dev.structured_hash(ob.state_dict()) == dev.structured_hash(restored.state_dict())
    with pytest.raises(ValueError):
        dev.checkpoint_load(file, torch.device("cpu"), "wrong-protocol")


def test_gate_refuses_missing_replay_or_wrong_epoch30_model():
    with pytest.raises(ValueError):
        dev.require_extension_gate([], ResidualCNN())
    oldreport = json.loads((OUT/"t12b_b0_df16_dev_results.json").read_text(encoding="utf8"))
    item = oldreport["epoch_records"][1]
    source = {"ordered_ID_hash": item["shuffle_order_sha256"]}
    assert dev.replay_epoch(1, item["training"], item["validation"], source, oldreport)["validation_all_fields_exact"]
    wrong = {**item["validation"], "pooled_mse": item["validation"]["pooled_mse"]+.01}
    with pytest.raises(ValueError):
        dev.replay_epoch(1, item["training"], wrong, source, oldreport)


def test_csv_only_three_figures_and_QA_without_training(tmp_path):
    from unittest.mock import patch
    from scripts.run_b0_dev60 import render_and_audit
    from scripts.run_lmmse_train_validation import save_csv
    from scripts.plot_b0_dev60 import FIGURE_NAMES
    # Explicit unit-test fixture in pytest temp; not scientific observations.
    log = [{"epoch": i, "train_mse": None if i == 0 else 1/(i+1), "val_mse_pooled": 1/(i+2)} for i in range(61)]
    original = json.loads((OUT/"t12b_b0_df16_dev_results.json").read_text(encoding="utf8"))
    rows = []
    for row in original["validation_aggregate"]:
        rows.append({**row, "b0_epoch30_nmse_db": row["b0_nmse_db"], "b0_best60_nmse_db": row["b0_nmse_db"], "epoch30_to_best60_gain_db": 0.0})
    save_csv(tmp_path/f"{dev.PREFIX}_training_log.csv", log)
    save_csv(tmp_path/f"{dev.PREFIX}_validation.csv", rows)
    with patch.object(old, "train_epoch", side_effect=AssertionError("CSV-only")), patch.object(old, "full_link_validation", side_effect=AssertionError("CSV-only")):
        records = render_and_audit(tmp_path, tmp_path/"figures")
    assert len(records) == 3
    for name, record in zip(FIGURE_NAMES, records):
        assert record["no_data_exclusions"]
        for ext in ("png", "pdf", "svg"):
            assert (tmp_path/"figures"/f"{name}.{ext}").is_file()


@pytest.mark.parametrize("best,gain,high,under,meaningful,persists", [
    (35, .03, .5, False, True, False),
    (36, .02, .5, True, True, False),
    (36, .019, .199, False, False, True),
    (60, .03, .2, True, False, False),
])
def test_predeclared_diagnostic_threshold_boundaries(best, gain, high, under, meaningful, persists):
    # Deterministic mathematical fixture, no generated communication observations.
    values = np.full(61, 1.1); values[30] = 1.; values[best] = 1.-gain
    log = [{"epoch": i, "val_mse_pooled": float(v), "best_so_far": i == best} for i, v in enumerate(values)]
    records = [{"validation": {"per_snr": [{"snr_db": s, "nmse_db": 0.} for s in md.SNR_DB]}} for _ in range(61)]
    records[best]["validation"]["per_snr"][-1]["nmse_db"] = -high
    result = dev.convergence_diagnosis(log, records)
    assert result["under_training_evidence"] == under
    assert (result["high_snr_undertraining_contribution"] == "meaningful") == meaningful
    assert result["high_snr_limitation_persists"] == persists


def test_actual_log_schema_includes_epoch0_wait_and_extended_phase(tmp_path):
    import csv
    from scripts.run_b0_dev60 import log_row
    from scripts.run_lmmse_train_validation import save_csv
    model = ResidualCNN()
    validation = {"pooled_mse": .1, "per_snr": [{"snr_db": s, "nmse_db": -s} for s in md.SNR_DB]}
    train = {"train_mse": .2, "gradient_norm": 1., **old.model_norms(model)}
    first = log_row(0, None, validation, (1, 0, 1, 2), True, model)
    next_row = log_row(1, train, validation, (1, 1, 1, 2), True, model, .5)
    extended = log_row(31, train, validation, (1, 1, 1, 2), True, model, .5)
    assert list(first) == list(next_row) == list(extended)
    save_csv(tmp_path/"log.csv", [first, next_row, extended])
    with (tmp_path/"log.csv").open(encoding="utf8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 3 and float(rows[0]["DataLoader_waiting_seconds"]) == 0
    assert rows[-1]["source_run_phase"] == "extended_31_60"


@pytest.fixture(scope="module")
def report():
    path = OUT/f"{dev.PREFIX}_results.json"
    if not path.exists():
        pytest.skip("single60-epoch run has not executed")
    return json.loads(path.read_text(encoding="utf8"))


def test_complete_single60_run_and_gate(report):
    assert report["development_seed"] == 13261001 and report["density"] == 16
    assert not report["counts_toward_formal_five_seed_results"]
    assert report["epoch0_30_replay_verified"]["all_checks_passed"]
    assert report["epoch0_30_replay_verified"]["epoch30_model_tensor_sha256"] == dev.EPOCH30_HASH
    assert [r["epoch"] for r in report["training_curve"]] == list(range(61))
    assert report["optimizer_continuity"]["no_reset"]
    assert report["optimizer_continuity"]["final_total_steps"] == 9420
    assert len(set(r["optimizer_instance_id"] for r in report["epoch_records"][1:])) == 1
    assert report["epoch_records"][31]["optimizer_steps_before"] == 4710
    assert report["provenance"]["accepted_before"] == report["provenance"]["accepted_after"]
    for path, digest in report["provenance"]["accepted_after"].items():
        assert sha256_file(ROOT/path) == digest
    assert report["formal_test_sealed"] and not report["formal_test_used"] and not report["OOD_generated"]


@pytest.mark.parametrize("which", ["best60", "epoch60"])
def test_best_and_final_full_checkpoint_reloads(report, which):
    record = report["checkpoints"][which]
    path = OUT/f"{dev.PREFIX}_{which}.pt"
    assert sha256_file(path) == record["file_sha256"]
    model, optimizer, payload = dev.checkpoint_load(path, torch.device("cpu"), report["provenance"]["protocol_sha256"])
    assert old.state_hash(model) == record["model_tensor_sha256"]
    assert dev.structured_hash(optimizer.state_dict()) == record["optimizer_tensor_sha256"]
    assert payload["optimizer_total_steps"] == record["epoch"]*157


def test_global_selection_and_descriptive_thresholds(report):
    log = report["training_curve"]; best = int(np.argmin([r["val_mse_pooled"] for r in log]))
    assert report["best_epoch"] == best
    diagnosis = dev.convergence_diagnosis(log, report["epoch_records"])
    for k, value in diagnosis.items():
        assert report[k] == value
    assert report["exact_T10D_replay"]["all_checks_passed"]
    assert len(report["paired_trajectory_diagnostics"]) == 20
    assert all(r["trajectory_units"] == 400 and r["p_value"] is None for r in report["paired_trajectory_diagnostics"])
    assert len(report["validation_results"]) == len(report["residual_diagnostics"]) == 5
    assert [(p["k"], p["t"]) for p in report["fixed_sample"]["points"]] == [(1, 2), (1, 15)]
