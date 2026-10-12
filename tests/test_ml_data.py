"""Split isolation, direct observation and deterministic B0 data schedules."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src import ml_data as md
from src.lmmse_train import raw_array_sha256
from src.pilots import build_fdm_pilot_layout
from src.trajectories import prepare_trajectory

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def manifest():
    return md.frozen_manifest(ROOT/"outputs/t10c_train_validation_manifest.csv")


@pytest.mark.parametrize("epoch", [0,1,7])
def test_epoch_has_exactly_20000_examples_4000_per_SNR(manifest,epoch):
    schedule = md.training_plan(manifest,md.TRAINING_SEED,epoch)
    assert len(schedule) == 20000
    assert {r["trajectory_id"] for r in schedule} == set(range(2000))
    assert {r["split"] for r in schedule} == {"train"}
    labels = np.array([r["snr_db"] for r in schedule])
    assert np.array_equal(np.unique(labels,return_counts=True)[1],np.full(5,4000))


def test_SNR_epoch_permutation_is_seeded_deterministic_and_changes():
    a = md.balanced_training_snr(md.TRAINING_SEED,0)
    assert np.array_equal(a,md.balanced_training_snr(md.TRAINING_SEED,0))
    assert not np.array_equal(a,md.balanced_training_snr(md.TRAINING_SEED,1))
    assert not np.array_equal(a,md.balanced_training_snr(md.TRAINING_SEED+1,0))


def test_validation_plan_is_fixed_and_disjoint_without_channel_generation(manifest):
    with patch("src.trajectories.PreparedTrajectory.generate",side_effect=AssertionError("metadata only")):
        plan = md.validation_plan(manifest)
    assert len(plan) == 20000
    assert {r["trajectory_id"] for r in plan} == set(range(2000,2400))
    assert {r["split"] for r in plan} == {"validation_id"}
    assert not {r["trajectory_id"] for r in plan} & set(range(2000))
    assert len({(r["trajectory_id"],r["frame_index"],r["snr_db"]) for r in plan}) == 20000


def test_manifest_corruption_or_resplitting_is_rejected(manifest,tmp_path):
    file = tmp_path/"manifest.csv"; file.write_text("changed",encoding="utf8")
    with pytest.raises(ValueError,match="SHA256"):
        md.frozen_manifest(file)
    altered = [dict(r) for r in manifest]; altered[0]["split"] = "validation_id"
    with pytest.raises(ValueError):
        md.training_plan(altered,md.TRAINING_SEED,0)


def test_training_noise_is_density_independent_and_seed_epoch_frame_dependent():
    hashes = [raw_array_sha256(md.training_unit_noise(md.TRAINING_SEED,0,17,3)) for df in (2,4,8,16)]
    assert len(set(hashes)) == 1
    a = hashes[0]
    for seed,epoch,tid,frame in ((md.TRAINING_SEED+1,0,17,3),(md.TRAINING_SEED,1,17,3),
            (md.TRAINING_SEED,0,18,3),(md.TRAINING_SEED,0,17,4)):
        assert raw_array_sha256(md.training_unit_noise(seed,epoch,tid,frame)) != a


def test_validation_noise_is_fixed_across_training_seed_density_and_SNR():
    expected = json.loads((ROOT/"outputs/t10d_lmmse_train_validation_results.json").read_text(encoding="utf8"))
    digest = expected["realization_hashes"][0]["frame_realizations"][0]["unit_noise_sha256"]
    for seed in (1,2):
        for df in (2,4,8,16):
            for snr in md.SNR_DB:
                assert raw_array_sha256(md.validation_unit_noise(2000,0)) == digest


@pytest.mark.parametrize("function,args", [
    (md.training_unit_noise,(1,0,2000,0)), (md.training_unit_noise,(1,0,2400,0)),
    (md.validation_unit_noise,(1999,0)), (md.validation_unit_noise,(2400,0)),
    (md.calibration_unit_noise,(1,4,0)), (md.training_unit_noise,(1,0,0,10)),
    (md.balanced_training_snr,(True,0)), (md.balanced_training_snr,(1,-1)),
])
def test_wrong_namespace_or_seed_frame_is_rejected(function,args):
    with pytest.raises(ValueError):
        function(*args)


@pytest.mark.parametrize("density", [2,4,8,16])
def test_direct_pilot_equals_actual_full_link_and_R3_is_existing_interpolation(density):
    model = prepare_trajectory(); h = model.generate(0).h_frames[0]
    layout = build_fdm_pilot_layout(2,64,16,density,4)
    z = md.calibration_unit_noise(md.TRAINING_SEED,0,0)
    check = md.full_link_pilot_sentinel(h,layout,z,10)
    assert check["machine_precision_match"] and check["max_absolute_error"] <= check["tolerance"]
    from src.link import estimate_ls_linear_channel,apply_frequency_mimo_channel,PILOT_SYMBOLS
    from src.pilots import map_pilot_symbols
    y = apply_frequency_mimo_channel(h,map_pilot_symbols(layout,PILOT_SYMBOLS)) + np.sqrt(0.1)*z
    _,reference = estimate_ls_linear_channel(y,layout)
    actual = md.r3_from_pilot_ls(md.direct_pilot_ls(h,layout,z,10),layout)
    assert np.max(np.abs(actual-reference)) < 5e-15
    assert actual.dtype == np.complex128


def test_unobserved_truth_cannot_leak_into_R3():
    h = prepare_trajectory().generate(1).h_frames[0].copy()
    layout = build_fdm_pilot_layout(2,64,16,16,4)
    z = md.calibration_unit_noise(md.TRAINING_SEED,1,0)
    a = md.direct_pilot_ls(h,layout,z,10)
    observed = np.broadcast_to(layout.pilot_mask[None],h.shape)
    h[~observed] += 1000 + 500j
    b = md.direct_pilot_ls(h,layout,z,10)
    assert np.array_equal(a,b,equal_nan=True)
    assert np.array_equal(md.r3_from_pilot_ls(a,layout),md.r3_from_pilot_ls(b,layout))


def test_noise_uses_ensemble_variance_and_does_not_scale_with_channel_power():
    h = prepare_trajectory().generate(2).h_frames[0]
    layout = build_fdm_pilot_layout(2,64,16,16,4)
    z = md.calibration_unit_noise(md.TRAINING_SEED,2,0)
    selected = np.broadcast_to(layout.pilot_mask[None],h.shape)
    for snr in md.SNR_DB:
        a = md.direct_pilot_ls(h,layout,z,snr)
        b = md.direct_pilot_ls(10*h,layout,z,snr)
        assert np.max(np.abs((a-h)[selected]-(b-10*h)[selected])) < 1e-14


def test_eight_channel_mapping_round_trip_quantization_and_accumulation():
    pytest.importorskip("torch")
    h = np.zeros((2,2,64,16),dtype=np.complex128)
    grid = np.arange(1024).reshape(64,16)/1024
    for rx in range(2):
        for tx in range(2):
            h[rx,tx] = 100*rx+10*tx+grid + 1j*(100*rx+10*tx+2+grid)
    tensor = md.complex_to_tensor(h)
    assert tuple(tensor.shape) == (1,8,64,16)
    for rx in range(2):
        for tx in range(2):
            assert np.array_equal(tensor[0,4*rx+2*tx].numpy(),h[rx,tx].real)
            assert np.array_equal(tensor[0,4*rx+2*tx+1].numpy(),h[rx,tx].imag)
    assert np.array_equal(md.tensor_to_complex(tensor)[0],h)
    random = np.random.default_rng(20260930).normal(size=h.shape).astype(np.complex128)*(1+1j)
    diag = md.quantization_diagnostics(random)
    assert 0 < diag["max_complex_absolute_error"] < 5e-7
    assert 0 < diag["RMS_complex_error"] < 1e-7
    assert not diag["normalization_used"]
    restored = md.tensor_to_complex(md.complex_to_tensor(random))
    assert restored.dtype == np.complex128
    mask = build_fdm_pilot_layout(2,64,16,16,4).data_mask
    energy = md.complex_data_error_energies(md.complex_to_tensor(random),random,mask)
    expected = np.sum(np.abs(restored[0][...,mask[0]]-random[...,mask[0]])**2,dtype=np.float64)
    assert energy["error_energy"] == expected


def test_tensor_mapping_does_not_normalize_amplitude():
    pytest.importorskip("torch")
    h = np.full((2,2,64,16),20+7j,dtype=np.complex128)
    a = md.complex_to_tensor(h); b = md.complex_to_tensor(2*h)
    assert np.array_equal(b.numpy(),2*a.numpy())


def test_ml_dependency_is_optional_and_communication_dependencies_unchanged():
    import tomllib
    config = tomllib.loads((ROOT/"pyproject.toml").read_text(encoding="utf8"))
    ml_requirements = [line.strip() for line in (ROOT/"requirements-ml.txt").read_text(encoding="utf8").splitlines()
                       if line.strip() and not line.startswith("#")]
    assert ml_requirements == ["torch==2.11.0"]
    assert "torch" not in str(config["project"]["dependencies"])
    import subprocess,hashlib
    old = subprocess.check_output(["git","show","47aec1dff7d515cc01fc6bcacfc48d56c11fae13:pyproject.toml"],cwd=ROOT)
    assert hashlib.sha256(old).hexdigest() == md.sha256_file(ROOT/"pyproject.toml")
    assert config["project"]["dependencies"] == ["numpy>=1.26","scipy>=1.12","matplotlib>=3.8","PyYAML>=6.0"]


def test_saved_pipeline_contract_sources_and_frozen_history():
    report = json.loads((ROOT/"outputs/t12a_b0_pipeline_results.json").read_text(encoding="utf8"))
    assert report["all_checks_passed"]
    assert report["parameter_count"] == 23144 and report["mac_per_frame"] == 23592960
    assert report["model_density_strategy"] == "four independent fixed-density models"
    assert report["tensor_mapping"] == list(md.TENSOR_MAPPING)
    assert all(report["zero_init_checks"].values())
    assert report["gradient_checks"]["all_checks_passed"]
    assert report["same_seed_smoke_reproducibility"]["all_checks_passed"]
    assert report["tiny_overfit"]["required_decrease_passed"]
    assert report["visited_channel_namespaces"] == {
        "calibration_ids":[0,1,2,3],"formal_train_benchmark_ids":list(range(50)),
        "validation_channel_ids":[],"formal_test_ids":[],"OOD_ids":[]}
    assert report["training_snr_schedule"]["examples_per_epoch"] == 20000
    assert set(report["training_snr_schedule"]["counts"].values()) == {4000}
    assert report["validation_manifest"]["evaluation_examples_per_density"] == 20000
    assert report["validation_noise_policy"]["T10D_unit_noise_sentinel_exact_match"]
    assert report["formal_test_sealed"]
    for field in ("formal_training_performed","formal_test_used","OOD_generated", "normalization_used",
                  "validation_model_selection_performed","hyperparameter_search_performed"):
        assert not report[field]
    assert report["num_filtered_samples"] == 0 and report["numerical_failures"] == []
    provenance = report["provenance"]
    assert provenance["historical_file_hash_before"] == provenance["historical_file_hash_after"]
    for group in ("historical_file_hash_after","source_sha256"):
        for path,digest in provenance[group].items():
            assert md.sha256_file(ROOT/path) == digest
    compute = report["runtime_benchmark"]["model_compute"]
    assert compute["optimizer_steps"] == 0 and compute["model_parameters_unchanged"]
    assert compute["batch_sizes"] == [32,64,128]
    for row in compute["records"]:
        assert row["mean_forward_seconds"] > 0 and row["mean_backward_seconds"] > 0
        if report["cuda_available"]:
            assert row["GPU_peak_allocated_bytes"] > 0
    assert all(value is None for key,value in report["final_training_hyperparameters"].items()
               if key != "requires_researcher_confirmation_before_T12B")
