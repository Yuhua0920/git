"""T12E protocol boundaries, frozen math, real small link and final artifacts."""
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src import b0_high_snr as probe, b0_dev60 as dev, b0_development as old, b0_perf as perf, ml_data as md
from src.b0_model import ResidualCNN, configure_reproducibility, predict_channel
from src.lmmse_train import formal_manifest_rows, formal_trajectory, raw_array_sha256, sha256_file
from src.nearest_validation import load_references, verify_frame
from src.trajectories import prepare_trajectory

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs"


def test_T12D_committed_and_architecture_profile_optimizer_frozen():
    title = subprocess.check_output(["git", "show", "-s", "--format=%s", probe.T12D_COMMIT], cwd=ROOT, text=True).strip()
    assert title == "feat: extend B0 development training for convergence diagnosis"
    report = json.loads((OUT/f"{dev.PREFIX}_results.json").read_text(encoding="utf8"))
    for path, digest in report["provenance"]["source_sha256"].items():
        assert sha256_file(ROOT/path) == digest
    configure_reproducibility(old.SEED, 4); model = ResidualCNN()
    assert sum(p.numel() for p in model.parameters()) == 23144
    assert list(old.optimizer_for(model).param_groups)[0]["lr"] == .001
    assert old.optimizer_for(model).param_groups[0]["weight_decay"] == 0
    assert old.SEED == 13261001 and probe.LAYOUT.offsets_by_tx == (0, 8)
    assert np.array_equal(np.flatnonzero(probe.LAYOUT.pilot_mask[0,:,0]), np.arange(0,64,16))
    assert probe.LAYOUT.pilot_union.mean() == 1/32
    profile = dev.selected_profile(json.loads((OUT/"t12c_perf_results.json").read_text(encoding="utf8")))
    assert all(profile[k] == v for k, v in dev.PROFILE.items())


@pytest.mark.parametrize("epoch", [1, 2, 12, 30, 57, 60])
def test_all20dB_and_same_original_shuffle_and_exposures(epoch):
    assert np.array_equal(probe.training_snrs(epoch), np.full(20000, 20))
    assert np.array_equal(dev.epoch_order(epoch), np.random.default_rng(np.random.SeedSequence([12004, 13261001, epoch])).permutation(20000))
    if epoch == 12:
        assert epoch*20000 == 60*4000 == 240000 and 57*4000 == 228000


@pytest.mark.parametrize("epoch", [0, 61, True, -1])
def test_invalid_epoch_rejected(epoch):
    with pytest.raises(ValueError):
        probe.training_snrs(epoch)


@pytest.fixture(scope="module")
def truth():
    prepared = prepare_trajectory()
    return np.concatenate([formal_trajectory(prepared, r).h_frames for r in formal_manifest_rows()[:2]])


@pytest.mark.parametrize("epoch", [1, 12, 30, 60])
def test_only_noise_scale_SNR_changes_and_direct_math_exact(truth, epoch):
    ids = np.arange(20); probe_batch = probe.training_batch(truth, epoch, ids)
    mixed_batch = dev.training_batch(truth, epoch, ids)
    for field in ("H_true", "unit_noise", "target", "sample_IDs"):
        assert probe_batch["hashes"][field] == mixed_batch["hashes"][field]
    assert torch.equal(probe_batch["target"], mixed_batch["target"])
    for j in ids:
        noise = md.training_unit_noise(old.SEED, epoch, int(j)//10, int(j)%10)
        expected = md.complex_to_tensor(md.r3_from_pilot_ls(md.direct_pilot_ls(truth[j], probe.LAYOUT, noise, 20), probe.LAYOUT))[0]
        assert torch.equal(probe_batch["base"][j], expected)
        independent = np.random.default_rng(np.random.SeedSequence([12002,13261001,epoch,int(j)//10,int(j)%10]))
        from src.numerics import complex_gaussian_noise
        assert np.array_equal(noise, complex_gaussian_noise((2,64,16),1.,independent))


def test_original_train_math_same_Adam_and_complete_checkpoint(tmp_path, truth):
    configure_reproducibility(old.SEED, 4)
    a, b = ResidualCNN().to("cuda"), ResidualCNN().to("cuda"); b.load_state_dict(a.state_dict())
    oa, ob = old.optimizer_for(a), old.optimizer_for(b)
    ids = np.resize(np.arange(20),128); batch = probe.training_batch(truth, 12, ids)
    direct = old.train_epoch(a, oa, batch["base"], batch["target"], batch["SNR"].numpy(), probe.LAYOUT.data_mask, np.arange(128))
    actual, source, _ = probe.train_epoch(b, ob, [batch], ids, 12)
    assert direct == actual and source["all_training_examples_20db"]
    assert old.state_hash(a) == old.state_hash(b) and dev.structured_hash(oa.state_dict()) == dev.structured_hash(ob.state_dict())
    original_id = id(ob); path = tmp_path/"full.pt"; protocol = sha256_file(ROOT/"docs/b0_20db_protocol.md")
    record = probe.checkpoint_save(path, b, ob, 12, .1, protocol, {"parameters": 23144})
    loaded, restored, payload = probe.checkpoint_load(path, torch.device("cuda"), protocol)
    assert record["model_and_optimizer_exact_reload"]
    for field in ("model_state_dict","optimizer_state_dict","torch_rng_state","cuda_rng_states","numpy_rng_state","epoch","training_seed","protocol_hash"):
        assert field in payload
    assert payload["selection_metric"] == probe.SELECTION_METRIC
    probe.train_epoch(b, ob, [batch], ids, 12); probe.train_epoch(loaded, restored, [batch], ids, 12)
    assert id(ob) == original_id and dev.optimizer_steps(ob) == 2
    assert old.state_hash(b) == old.state_hash(loaded)
    assert dev.structured_hash(ob.state_dict()) == dev.structured_hash(restored.state_dict())


def test_persistent_workers_receive_epoch12_then13(tmp_path, truth):
    path = tmp_path/"h.npy"; np.save(path, truth)
    sampler = dev.Dev60Sampler(20,12,np.arange(20)); loader = probe.training_loader(path,sampler)
    try:
        for epoch in (12,13):
            sampler.epoch=epoch; actual=list(loader)
            assert len(actual)==1 and actual[0]["hashes"] == probe.training_batch(truth,epoch,sampler.ordered_ids())["hashes"]
    finally:
        perf.close_loader(loader)


def test_selection_reads_only20dB_MSE_and_earlier_ties():
    log = [{"val_mse_20db": 3., "cross_0db": -999., "BER": 0.},
           {"val_mse_20db": 2., "cross_0db": 999., "BER": 1.},
           {"val_mse_20db": 2., "cross_0db": -999., "BER": 0.}]
    assert probe.best_epoch(log) == 1
    for row in log:
        row["cross_0db"] *= -1; row["BER"] = .3
    assert probe.best_epoch(log) == 1
    with pytest.raises(FloatingPointError):
        probe.best_epoch([{"val_mse_20db":float("nan")}])


@pytest.mark.parametrize("gain,strong,moderate,persists", [(1.,True,False,False),(.999,False,True,False),(.2,False,True,False),(.199,False,False,True),(-1.,False,False,True)])
def test_predeclared_specialization_thresholds(gain,strong,moderate,persists):
    assert probe.specialization_flags(gain) == {"strong_specialization_gain":strong,"moderate_specialization_gain":moderate,"high_snr_limitation_persists_under_dedicated_training":persists}


@pytest.mark.parametrize("gain,flag",[(.1001,True),(.1,False),(.0999,False)])
def test_original_Oracle_threshold(gain,flag):
    assert old.oracle_red_flag([{"b0_nmse_db":-gain,"r5_oracle_nmse_db":0.}]) == flag


def test_20dB_validation_only_reads20dB_and_zero_init(truth):
    configure_reproducibility(old.SEED,4); model=ResidualCNN().to("cuda")
    batch=probe.training_batch(truth,1,np.arange(20))
    # Other SNR slots are poison. They must not be touched by the selector path.
    base=torch.full((5,20,8,64,16),float("nan")); base[4]=batch["base"]
    val=probe.evaluate_20db(model,{"base":base,"h":truth},epoch_zero=True)
    energies=md.complex_data_error_energies(batch["base"],truth,probe.LAYOUT.data_mask)
    assert val["error_energy"]==energies["error_energy"] and val["snr_db"]==20


def test_region_masks_hand_counts_nonperiodic_distance_and_no_exclusion():
    h=np.ones((2,2,64,16),dtype=np.complex128)
    diag=probe.RegionDiagnostics(); diag.add(h,2*h,3*h)
    rows=diag.rows(); distances=probe.distance_by_tx()
    assert distances[0,63]==15 and distances[1,0]==8 and distances.max()==15
    assert distances[0,0]==0 and distances[1,8]==0
    expected={"interior_1_11":4*(11*64-2*8),"trailing_13_15":4*3*64,"remaining_0_12":4*(2*64-2*8)}
    for method,multiplier in (("R3",1.),("B0_20db_only",4.)):
        for group in ("time","frequency_distance"):
            selected=[r for r in rows if r["method"]==method and r["group"]==group]
            assert sum(r["complex_coefficient_count"] for r in selected)==3968
            assert sum(r["error_energy"] for r in selected)==3968*multiplier
            assert sum(r["truth_energy"] for r in selected)==3968
        for row in rows:
            if row["method"]==method and row["group"]=="time":
                assert row["complex_coefficient_count"]==expected[row["region_or_distance"]]
    assert any(r["region_or_distance"]==0 and not r["requested_group"] for r in rows)


def test_actual_T10D_20dB_fixed_link_and_diagnostic_instrumentation():
    prepared=prepare_trajectory(); refs=load_references(OUT); historical=old.replay_link(ROOT)
    row=refs["manifest"][0]; draw=formal_trajectory(prepared,row); h=draw.h_frames[0]
    assert raw_array_sha256(draw.h_long)==refs["realizations"][2000]["H_long_sha256"]
    z=md.validation_unit_noise(2000,0); bits,y=old.bits_and_y(h,2000,0,z,20,probe.LAYOUT,historical)
    verify_frame(refs["realizations"][2000]["frame_realizations"][0],0,h,bits,z)
    ls,r3=old.link.estimate_ls_linear_channel(y,probe.LAYOUT)
    direct=md.direct_pilot_ls(h,probe.LAYOUT,z,20)
    chosen=np.broadcast_to(probe.LAYOUT.pilot_mask[None],ls.shape)
    assert np.allclose(ls[chosen],direct[chosen],rtol=0,atol=1e-14)
    configure_reproducibility(old.SEED,4); model=ResidualCNN().to("cuda")
    with torch.inference_mode():
        predicted=md.tensor_to_complex(predict_channel(model,md.complex_to_tensor(r3,"cuda"),probe.LAYOUT.data_mask))[0]
    collector=probe.FullLinkDiagnostics(ROOT,model)
    # Exercise the real five-SNR ordering with one actual frame, includingR4 fixed sample.
    for snr in md.SNR_DB:
        _,yy=old.bits_and_y(h,2000,0,z,snr,probe.LAYOUT,historical)
        collector.estimate(yy,probe.LAYOUT)
    for _ in md.SNR_DB:
        result=collector.energy(predicted,h,probe.LAYOUT.data_mask)
        assert result==old.channel_error_energies(predicted,h,probe.LAYOUT.data_mask)
    assert not collector.pending
    assert collector.fixed_sample["snr_db"]==20
    assert collector.fixed_sample["Y_sha256"]==perf.raw_hash(y)
    assert [(p["k"],p["t"]) for p in collector.fixed_sample["points"]]==[(1,2),(1,15)]
    assert collector.fixed_sample["R4_evaluated_frames"]==1


def test_actual_csv_schema_zero_train_exposure_and_final(tmp_path):
    from scripts.run_b0_high_snr import log_row
    from scripts.run_lmmse_train_validation import save_csv
    model=ResidualCNN(); val={"mse":.1,"nmse_db":-10.}; train={"train_mse":.2,"gradient_norm":1.}
    rows=[log_row(0,None,val,model,True,1.),log_row(12,train,val,model,True,1.),log_row(60,train,val,model,False,1.)]
    assert list(rows[0])==list(rows[1])==list(rows[2])
    save_csv(tmp_path/"log.csv",rows)
    assert rows[1]["cumulative_20db_exposures"]==240000 and rows[1]["exposure_matched"]


def test_three_csv_only_figures_and_all_QA_before_training(tmp_path):
    from scripts.run_b0_high_snr import render_and_audit
    from scripts.run_lmmse_train_validation import save_csv
    from scripts.plot_b0_high_snr import FIGURE_NAMES
    # Unit fixture only, isolated in pytest temporary directory.
    log=[{"epoch":i,"train_mse_20db":None if i==0 else 1/(i+2),"val_mse_20db":1/(i+3)} for i in range(61)]
    names=["R3","mixed_B0_best57","B0_20db_epoch12","B0_20db_best","R4","R5"]
    primary=[{"method":name,"nmse_db":-10-i} for i,name in enumerate(names)]
    mixed=json.loads((OUT/f"{dev.PREFIX}_results.json").read_text(encoding="utf8"))
    cross=[{"checkpoint":"global_best","snr_db":r["snr_db"],"r3_nmse_db":r["r3_nmse_db"],"mixed_B0_best57_nmse_db":r["b0_nmse_db"],
        "b0_20db_only_nmse_db":r["b0_nmse_db"],"r4_nmse_db":r["r4_nmse_db"],"r5_nmse_db":r["r5_oracle_nmse_db"]} for r in mixed["validation_results"]]
    for suffix,rows in (("_training_log",log),("_primary_comparison",primary),("_validation",cross)):
        save_csv(tmp_path/f"{probe.PREFIX}{suffix}.csv",rows)
    with patch.object(old,"train_epoch",side_effect=AssertionError("CSV only")),patch.object(old,"full_link_validation",side_effect=AssertionError("CSV only")):
        records=render_and_audit(tmp_path,tmp_path/"figures")
    assert len(records)==3 and all(r["collision_audit"]["verdict"]=="PASS" for r in records)
    assert all((tmp_path/"figures"/f"{name}.{ext}").is_file() for name in FIGURE_NAMES for ext in ("png","pdf","svg"))


@pytest.fixture(scope="module")
def report():
    path=OUT/f"{probe.PREFIX}_results.json"
    if not path.exists():
        pytest.skip("fixed60-epoch capacity probe has not executed")
    return json.loads(path.read_text(encoding="utf8"))


def test_complete_probe_and_frozen_protocol(report):
    assert report["training_snr"]==[20] and report["development_seed"]==13261001 and report["density"]==16
    assert report["diagnostic_model"] and not report["formal_seed"] and not report["counts_toward_formal_five_seed_results"]
    assert report["optimizer_continuity"]["final_steps"]==9420 and report["optimizer_continuity"]["no_reset"]
    assert len(set(r["optimizer_instance_id"] for r in report["epoch_records"][1:]))==1
    assert [r["epoch"] for r in report["training_curve"]]==list(range(61))
    assert report["best_epoch"]==probe.best_epoch(report["training_curve"])
    assert all([r["sample_count"] for r in e["training"]["train_per_snr"]]==[0,0,0,0,20000] for e in report["epoch_records"][1:])
    assert report["provenance"]["accepted_before"]==report["provenance"]["accepted_after"]
    assert all(sha256_file(ROOT/p)==h for p,h in report["provenance"]["accepted_after"].items())
    assert report["formal_test_sealed"] and not report["formal_test_used"] and not report["OOD_generated"]


@pytest.mark.parametrize("which",["best","epoch12","epoch60"])
def test_full_probe_checkpoints_reload(report,which):
    record=report["checkpoints"][which]; path=OUT/f"{probe.PREFIX}_{which}.pt"
    assert sha256_file(path)==record["file_sha256"]
    model,optimizer,payload=probe.checkpoint_load(path,torch.device("cpu"),report["provenance"]["protocol_sha256"])
    assert old.state_hash(model)==record["model_tensor_sha256"]
    assert dev.structured_hash(optimizer.state_dict())==record["optimizer_tensor_sha256"]
    assert dev.optimizer_steps(optimizer)==record["epoch"]*157


def test_final_replay_exposure_cross_diagnostics_and_regions(report):
    assert report["exact_T10D_replay"]["all_checks_passed"]
    assert report["epoch12_comparison"]["probe_20db_exposures"]==240000
    assert report["epoch12_comparison"]["mixed_epoch60_20db_exposures"]==240000
    assert report["epoch12_comparison"]["mixed_best57_20db_exposures"]==228000
    cross=report["checkpoint_cross_snr_diagnostics"]
    assert {r["checkpoint"] for r in cross}=={"epoch0","epoch12","epoch30","epoch60","global_best"}
    for checkpoint in {r["checkpoint"] for r in cross}:
        assert [r["snr_db"] for r in cross if r["checkpoint"]==checkpoint]==list(md.SNR_DB)
    assert len(report["cross_snr_results"])==5 and report["region_totals_verified"]
    assert report["fixed_sample"]["snr_db"]==20
    from scripts.plot_b0_high_snr import read_csv
    assert len(read_csv(OUT/f"{probe.PREFIX}_primary_trajectory_metrics.csv"))==400
    assert len(read_csv(OUT/f"{probe.PREFIX}_validation_trajectory_metrics.csv"))==2000
    assert report["oracle_red_flag"]==old.oracle_red_flag([{"b0_nmse_db":r["b0_20db_only_nmse_db"],"r5_oracle_nmse_db":r["r5_nmse_db"]} for r in report["cross_snr_results"]])
    for k,v in probe.specialization_flags(report["dedicated_gain_vs_mixed"]).items():
        assert report[k]==v
