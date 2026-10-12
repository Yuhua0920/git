"""T12A/B0-PRE only: calibration smoke and TRAIN-only runtime measurements."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn

from src.b0_model import (ResidualCNN, configure_reproducibility, data_complex_mse,
                          predict_channel, training_step)
from src.config import load_config
from src.lmmse_train import formal_trajectory, raw_array_sha256, sha256_file
from src import ml_data as md
from src.pilots import build_fdm_pilot_layout, count_layout_resources
from src.trajectories import prepare_trajectory


ACCEPTED_COMMIT = "47aec1dff7d515cc01fc6bcacfc48d56c11fae13"
SMOKE_STEPS = 100
SMOKE_REQUIRED_LOSS_RATIO = 0.5
SMOKE_LR = 1e-3
BENCHMARK_TRAJECTORIES = 50
BENCHMARK_BATCHES = (32, 64, 128)
BENCHMARK_REPEATS = 5
RESULT_PATH = ROOT/"outputs/t12a_b0_pipeline_results.json"


def json_save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                   allow_nan=False)+"\n", encoding="utf8")


def state_hash(model):
    digest = hashlib.sha256()
    for name, array in model.state_dict().items():
        digest.update(name.encode("utf8"))
        digest.update(array.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def frozen_files():
    paths = subprocess.check_output(["git", "ls-files", "outputs", "src", "scripts", "tests", "configs"],
                                    cwd=ROOT, text=True).splitlines()
    return {p: sha256_file(ROOT/p) for p in paths}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def process_memory():
    """Current-process working set and process-lifetime peak, not per-batch RSS."""
    if os.name == "nt":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
                (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)
        info = Counters(); info.cb = ctypes.sizeof(info)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(info), info.cb):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return {"RSS_estimate_bytes": int(info.WorkingSetSize),
                "process_lifetime_peak_RSS_bytes": int(info.PeakWorkingSetSize),
                "method": "Windows WorkingSetSize / PeakWorkingSetSize"}
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"RSS_estimate_bytes": None, "process_lifetime_peak_RSS_bytes": int(peak*1024),
            "method": "resource ru_maxrss; no current RSS measurement"}


def architecture_report(model):
    convs = [layer for layer in model.modules() if isinstance(layer, nn.Conv2d)]
    actual_parameters = sum(p.numel() for p in model.parameters())
    actual_mac = sum(c.weight.numel()*64*16 for c in convs)
    independent_parameters = (8*32*9+32) + 2*(32*32*9+32) + (32*8*9+8)
    independent_mac = (8*32*9+32*32*9+32*32*9+32*8*9)*64*16
    if actual_parameters != independent_parameters or actual_parameters != 23144 or actual_mac != independent_mac or actual_mac != 23592960:
        raise ValueError("B0 parameter/MAC calibration failed")
    return {"layers": [{"in_channels": c.in_channels, "out_channels": c.out_channels,
                        "kernel": list(c.kernel_size), "stride": list(c.stride),
                        "padding": list(c.padding), "bias": c.bias is not None} for c in convs],
            "activations": ["ReLU", "ReLU", "ReLU", "final linear"],
            "residual_semantics": "H_pred=H_base+Delta_H on data RE; H_pred=H_base on pilot-union RE",
            "input": ["B",8,64,16], "output": ["B",8,64,16],
            "parameter_count": actual_parameters, "independent_parameter_count": independent_parameters,
            "mac_per_frame": actual_mac, "independent_mac_per_frame": independent_mac,
            "MAC_scope": "dense convolution multiply-accumulates only; padding included; not FLOP",
            "normalization": None, "mask_or_SNR_network_input": False}


def calibration_data(prepared, training_seed=md.TRAINING_SEED):
    layout = build_fdm_pilot_layout(2,64,16,16,4)
    inputs, targets, records, sentinels = [], [], [], []
    for tid in range(4):
        trajectory = prepared.generate(tid)
        for frame in (0,1):
            h = trajectory.h_frames[frame]
            z = md.calibration_unit_noise(training_seed,tid,frame)
            if (tid,frame) == (0,0):
                # Establish the complete physical observation check before fast data generation.
                for df in md.DENSITIES:
                    sentinel_layout = build_fdm_pilot_layout(2,64,16,df,4)
                    sentinels.append({"dF": df, **md.full_link_pilot_sentinel(h,sentinel_layout,z,10)})
            ls = md.direct_pilot_ls(h,layout,z,10)
            base = md.r3_from_pilot_ls(ls,layout)
            inputs.append(base); targets.append(h.copy())
            records.append({"namespace": "calibration", "trajectory_id": tid, "frame_index": frame,
                "channel_seed": trajectory.channel_seed, "noise_SeedSequence_entropy":
                [md.CALIBRATION_NOISE_DOMAIN,training_seed,tid,frame], "SNR_dB": 10, "dF":16,
                "H_true_sha256": raw_array_sha256(h), "unit_noise_sha256": raw_array_sha256(z),
                "H_base_sha256": raw_array_sha256(base)})
    return np.stack(inputs), np.stack(targets), layout, records, sentinels


def run_smoke(h_base, h_true, layout, device, steps=SMOKE_STEPS):
    settings = configure_reproducibility(md.TRAINING_SEED)
    model = ResidualCNN().to(device)
    base = md.complex_to_tensor(h_base,device)
    target = md.complex_to_tensor(h_true,device)
    mask = torch.from_numpy(layout.data_mask).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=SMOKE_LR)
    before = predict_channel(model,base,mask)
    zero_residual = bool(torch.count_nonzero(model(base)).item() == 0)
    exact = before.detach().cpu().numpy().tobytes() == base.cpu().numpy().tobytes()
    if not exact or not zero_residual:
        raise ValueError("B0 initialization must be exact quantized R3")
    initial_loss = float(data_complex_mse(before,target,mask).detach().cpu())
    gradient_records, losses = [], []
    sync(device); started = perf_counter()
    for step in range(steps):
        if step < 2:
            optimizer.zero_grad(set_to_none=True)
            loss = data_complex_mse(predict_channel(model,base,mask),target,mask)
            loss.backward()
            norms = [float(model.layers[index].weight.grad.norm().detach().cpu()) for index in (0,2,4,6)]
            gradient_records.append({"backward_step": step+1, "conv_weight_gradient_norms": norms})
            losses.append(float(loss.detach().cpu())); optimizer.step()
        else:
            losses.append(training_step(model,optimizer,base,target,mask))
    with torch.no_grad():
        prediction = predict_channel(model,base,mask)
        final_loss = float(data_complex_mse(prediction,target,mask).cpu())
    sync(device); seconds = perf_counter()-started
    first = gradient_records[0]["conv_weight_gradient_norms"]
    second = gradient_records[1]["conv_weight_gradient_norms"]
    gradients_passed = first[-1] > 0 and all(g == 0 for g in first[:-1]) and all(g > 0 for g in second)
    union = torch.from_numpy(layout.pilot_union).to(device)
    pilot_preserved = torch.equal(prediction[...,union],base[...,union])
    return {"namespace": "calibration only", "optimizer": "Adam", "learning_rate": SMOKE_LR,
            "steps": steps, "batch_size": len(base), "initial_loss": initial_loss,
            "final_loss": final_loss, "final_to_initial_loss_ratio": final_loss/initial_loss,
            "required_loss_ratio": SMOKE_REQUIRED_LOSS_RATIO, "loss_decreased": final_loss < initial_loss,
            "required_decrease_passed": final_loss <= SMOKE_REQUIRED_LOSS_RATIO*initial_loss,
            "loss_trace": losses, "final_model_state_sha256": state_hash(model), "seconds": seconds,
            "zero_init_checks": {"residual_exact_zero": zero_residual, "prediction_bitwise_equals_quantized_R3":exact,
                "difference_from_original_H_base_is_quantization_only": np.array_equal(md.tensor_to_complex(before),md.tensor_to_complex(base))},
            "gradient_checks": {"steps": gradient_records, "all_checks_passed": gradients_passed},
            "pilot_preservation": {"all_union_pilot_values_exact_preserved": bool(pilot_preserved)},
            "reproducibility_settings": settings, "performance_claim": False,
            "all_checks_passed": exact and zero_residual and gradients_passed and pilot_preserved
                                 and final_loss <= SMOKE_REQUIRED_LOSS_RATIO*initial_loss}


def train_generation_benchmark(prepared, manifest):
    plan = md.training_plan(manifest,md.TRAINING_SEED,0)
    selected = manifest[:BENCHMARK_TRAJECTORIES]
    if len(selected) != 50 or any(row["split"] != "train" for row in selected):
        raise ValueError("runtime benchmark requires exactly first 50 TRAIN trajectories")
    layout = build_fdm_pilot_layout(2,64,16,16,4)
    inputs, targets, records = [], [], []
    phases = {name:0.0 for name in ("channel_generation", "unit_noise_generation", "Pilot_LS", "R3_interpolation", "tensor_conversion")}
    rss_before = process_memory(); started = perf_counter()
    for row in selected:
        stamp = perf_counter(); trajectory = formal_trajectory(prepared,row)
        phases["channel_generation"] += perf_counter()-stamp
        tid = row["trajectory_id"]
        for frame,h in enumerate(trajectory.h_frames):
            snr = plan[10*tid+frame]["snr_db"]
            stamp = perf_counter(); z = md.training_unit_noise(md.TRAINING_SEED,0,tid,frame)
            phases["unit_noise_generation"] += perf_counter()-stamp
            stamp = perf_counter(); ls = md.direct_pilot_ls(h,layout,z,snr)
            phases["Pilot_LS"] += perf_counter()-stamp
            stamp = perf_counter(); base = md.r3_from_pilot_ls(ls,layout)
            phases["R3_interpolation"] += perf_counter()-stamp
            inputs.append(base); targets.append(h.copy())
        records.append({"trajectory_id":tid, "split":"train", "channel_seed":row["channel_seed"],
                        "H_long_sha256":raw_array_sha256(trajectory.h_long)})
    base_array, true_array = np.stack(inputs), np.stack(targets)
    stamp = perf_counter(); base_tensor = md.complex_to_tensor(base_array); target_tensor = md.complex_to_tensor(true_array)
    phases["tensor_conversion"] += perf_counter()-stamp
    total = perf_counter()-started
    return base_tensor,target_tensor,layout,{"trajectory_ids":list(range(50)), "frames":500,
        "density":16, "epoch":0, "training_seed":md.TRAINING_SEED,
        "dataset_generation_seconds":total, "frames_per_second":500/total,
        "phase_seconds":phases, "channel_records":records,
        "complex128_H_base_and_H_true_bytes":base_array.nbytes+true_array.nbytes,
        "float32_input_and_target_bytes":base_tensor.numel()*4+target_tensor.numel()*4,
        "RSS_before":rss_before, "RSS_after":process_memory(), "formal_model_trained":False}


def compute_benchmark(base_tensor, target_tensor, layout, device):
    configure_reproducibility(md.TRAINING_SEED)
    model = ResidualCNN().to(device); before = state_hash(model)
    mask = torch.from_numpy(layout.data_mask).to(device); rows = []
    for batch_size in BENCHMARK_BATCHES:
        base = base_tensor[:batch_size].to(device); target = target_tensor[:batch_size].to(device)
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            data_complex_mse(predict_channel(model,base,mask),target,mask).backward()
        model.zero_grad(set_to_none=True); sync(device)
        rss_before = process_memory()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            allocation_baseline = torch.cuda.memory_allocated(device)
        forwards, backwards = [], []
        for _ in range(BENCHMARK_REPEATS):
            model.zero_grad(set_to_none=True)
            sync(device); stamp = perf_counter(); prediction = predict_channel(model,base,mask)
            sync(device); forwards.append(perf_counter()-stamp)
            loss = data_complex_mse(prediction,target,mask)
            sync(device); stamp = perf_counter(); loss.backward()
            sync(device); backwards.append(perf_counter()-stamp)
            del prediction,loss
        forward_mean, backward_mean = float(np.mean(forwards)),float(np.mean(backwards))
        rows.append({"batch_size":batch_size, "measured_repeats":BENCHMARK_REPEATS, "warmup_repeats":2,
            "forward_seconds":forwards, "backward_seconds":backwards,
            "mean_forward_seconds":forward_mean, "mean_backward_seconds":backward_mean,
            "forward_backward_examples_per_second":batch_size/(forward_mean+backward_mean),
            "GPU_peak_allocated_bytes":torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "GPU_peak_reserved_bytes":torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
            "GPU_allocation_baseline_bytes":allocation_baseline if device.type == "cuda" else None,
            "RSS_before":rss_before, "RSS_after":process_memory(), "optimizer_steps":0})
        del base,target
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    after = state_hash(model)
    if before != after:
        raise ValueError("throughput benchmark must not update formal model parameters")
    return {"device":str(device), "batch_sizes":list(BENCHMARK_BATCHES), "records":rows,
            "model_state_before_sha256":before, "model_state_after_sha256":after,
            "model_parameters_unchanged":True, "optimizer_steps":0,
            "timing_scope":"forward includes external gating; backward excludes optimizer and loss construction",
            "GPU_timing_synchronized":device.type == "cuda",
            "RSS_scope":"process working set / lifetime peak; includes framework, allocator and library residency"}


def main():
    started = perf_counter()
    subject = subprocess.check_output(["git","show","-s","--format=%s",ACCEPTED_COMMIT],cwd=ROOT,text=True).strip()
    if subject != "feat: add limited ZF versus linear-MMSE detector sensitivity validation":
        raise ValueError("accepted T11C-B commit required")
    old = json.loads((ROOT/"outputs/t11c_b_detector_sensitivity_results.json").read_text(encoding="utf8"))
    if not old["all_checks_passed"] or not old["formal_test_sealed"]:
        raise ValueError("accepted T11C-B results required")
    before = frozen_files()
    config = load_config(ROOT/"configs/base.yaml")
    if config["noise"]["snr_db"] != list(md.SNR_DB) or config["pilot"]["spacing_frequency_options"] != list(md.DENSITIES):
        raise ValueError("B0 protocol must match the frozen five SNRs and four densities")
    manifest = md.frozen_manifest(ROOT/"outputs/t10c_train_validation_manifest.csv")
    settings = configure_reproducibility(md.TRAINING_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prepared = prepare_trajectory()
    base,truth,layout,calibration_records,pilot_sentinels = calibration_data(prepared)
    architecture = architecture_report(ResidualCNN())
    print(f"B0 params={architecture['parameter_count']}; MAC/frame={architecture['mac_per_frame']}; device={device}",flush=True)
    smoke = run_smoke(base,truth,layout,device)
    repeated = run_smoke(base,truth,layout,device)
    reproducible = (smoke["loss_trace"] == repeated["loss_trace"]
                    and smoke["final_loss"] == repeated["final_loss"]
                    and smoke["final_model_state_sha256"] == repeated["final_model_state_sha256"])
    print(f"Calibration smoke loss {smoke['initial_loss']:.9g} -> {smoke['final_loss']:.9g}; repeated exact={reproducible}",flush=True)
    benchmark_base,benchmark_target,benchmark_layout,generation = train_generation_benchmark(prepared,manifest)
    compute = compute_benchmark(benchmark_base,benchmark_target,benchmark_layout,device)
    train_schedule = md.training_plan(manifest,md.TRAINING_SEED,0)
    labels,counts = np.unique([r["snr_db"] for r in train_schedule],return_counts=True)
    fixed_validation = md.validation_plan(manifest)
    val_z = md.validation_unit_noise(2000,0)
    frozen_z_sha256 = old["realization_hashes"][0]["frame_realizations"][0]["unit_noise_sha256"]
    fixed_noise_matches = raw_array_sha256(val_z) == frozen_z_sha256
    after = frozen_files()
    if before != after:
        raise ValueError("historical core/config/output file changed during B0 calibration")
    success = smoke["all_checks_passed"] and repeated["all_checks_passed"] and reproducible and fixed_noise_matches
    prop = torch.cuda.get_device_properties(device) if device.type == "cuda" else None
    install_path = ROOT/"outputs/t12a_torch_install_report.json"
    report = {"task":"T12A / B0-PRE",
        "experiment_type":"B0 residual-CNN architecture and supervised-learning protocol calibration",
        "torch_version":torch.__version__, "torch_build_git_version":torch.version.git_version,
        "torch_build_config":torch.__config__.show(), "device":str(device),
        "cuda_available":torch.cuda.is_available(), "torch_cuda_build_version":torch.version.cuda,
        "gpu_name":prop.name if prop else None, "GPU_total_memory_bytes":prop.total_memory if prop else None,
        "GPU_compute_capability":[prop.major,prop.minor] if prop else None,
        "cpu_threads":torch.get_num_threads(), "python_version":platform.python_version(),
        "platform":platform.platform(), "python_executable":sys.executable,
        "ML_optional_dependency":{"file":"requirements-ml.txt","requirement":"torch==2.11.0",
            "installed_build":torch.__version__,"pyproject_historical_content_unchanged":True,
            "global_Python_modified":False,"torchvision_or_torchaudio_installed":False},
        "model_definition":architecture,
        "parameter_count":architecture["parameter_count"], "mac_per_frame":architecture["mac_per_frame"],
        "B0_role":"diagnostic learning baseline, not proposed innovation", "model_density_strategy":"four independent fixed-density models",
        "density_options":list(md.DENSITIES), "tensor_mapping":list(md.TENSOR_MAPPING),
        "tensor_shape":["B",8,64,16], "model_dtype":"float32", "simulation_dtype":"complex128 / float64",
        "quantization_diagnostics":{"H_base":md.quantization_diagnostics(base),"H_true":md.quantization_diagnostics(truth)},
        "train_manifest":{"source":"outputs/t10c_train_validation_manifest.csv", "sha256":md.MANIFEST_SHA256,
            "split":"train", "trajectory_id_range":[0,1999],"independent_trajectories":2000,"channel_frames":20000},
        "validation_manifest":{"source":"outputs/t10c_train_validation_manifest.csv", "sha256":md.MANIFEST_SHA256,
            "split":"validation_id", "trajectory_id_range":[2000,2399],"independent_trajectories":400,"channel_frames":4000,
            "evaluation_examples_per_density":len(fixed_validation)},
        "training_snr_schedule":{"SNR_dB":list(md.SNR_DB),"examples_per_epoch":len(train_schedule),
            "counts":{str(int(k)):int(v) for k,v in zip(labels,counts)},
            "policy":"SeedSequence([12001,training_seed,epoch]); permutation of 4000 repeats of each SNR",
            "expanded_to_five_examples_per_frame":False},
        "training_noise_policy":{"shape":[2,64,16],"complex_variance":1.0,
            "SeedSequence_entropy":"[12002,training_seed,epoch,trajectory_id,frame_index]",
            "depends_on_density":False,"Python_hash_used":False},
        "validation_noise_policy":{"seed":"90261001+100*trajectory_id+frame_index (frozen T10D)",
            "depends_on_training_seed":False,"depends_on_density":False,"same_unit_noise_across_SNR":True,
            "T10D_unit_noise_sentinel_exact_match":fixed_noise_matches,"unit_noise_sha256":raw_array_sha256(val_z)},
        "received_noise_variance":"10^(-SNR_dB/10) per RX complex sample; N=sqrt(sigma2)*Z",
        "loss_definition":"data-RE mean(error_Re^2+error_Im^2) over B*Nr*Nt*actual data REs; no per-frame energy division",
        "pilot_preservation":smoke["pilot_preservation"],"normalization_used":False,
        "zero_init_checks":smoke["zero_init_checks"],"gradient_checks":smoke["gradient_checks"],
        "tiny_overfit":smoke,"calibration_data_records":calibration_records,"direct_pilot_full_link_sentinels":pilot_sentinels,
        "same_seed_smoke_reproducibility":{"loss_trace_exact_match":smoke["loss_trace"] == repeated["loss_trace"],
            "final_state_exact_match":smoke["final_model_state_sha256"] == repeated["final_model_state_sha256"],
            "second_final_loss":repeated["final_loss"], "second_run_seconds":repeated["seconds"],
            "scope":"this installed environment/device only", "all_checks_passed":reproducible},
        "runtime_benchmark":{"TRAIN_dataset_generation":generation,"model_compute":compute},
        "memory_benchmark":{"dataset_RSS":generation["RSS_after"],"model_batches":compute["records"],
            "CPU_RSS_is_estimate":True,"GPU_memory_method":"torch.cuda.max_memory_allocated/reserved" if prop else None},
        "reproducibility_settings":settings,"future_five_seed_protocol":{"num_training_seeds":5,
            "seed_values_frozen":False,"num_densities":4,"planned_models":20,"formal_models_trained_this_task":0,
            "calibration_smoke_runs":2},
        "final_training_hyperparameters":{"epochs":None,"patience":None,"batch_size":None,"learning_rate":None,
            "optimizer":None,"learning_rate_schedule":None,
            "requires_researcher_confirmation_before_T12B":True},
        "formal_training_performed":False,"formal_test_used":False,"formal_test_sealed":True,"OOD_generated":False,
        "validation_model_selection_performed":False,"hyperparameter_search_performed":False,
        "visited_channel_namespaces":{"calibration_ids":list(range(4)),"formal_train_benchmark_ids":list(range(50)),
            "validation_channel_ids":[],"formal_test_ids":[],"OOD_ids":[]},
        "num_filtered_samples":0,"numerical_failures":[],
        "provenance":{"accepted_T11C_B_commit":ACCEPTED_COMMIT,"T11C_B_committed_cleanly_at_task_start":True,
            "historical_file_hash_before":before,"historical_file_hash_after":after,
            "source_sha256":{p:sha256_file(ROOT/p) for p in ("src/b0_model.py","src/ml_data.py", "scripts/validate_b0_pipeline.py",
                "tests/test_b0_model.py","tests/test_ml_data.py","docs/ml_protocol.md","pyproject.toml","requirements-ml.txt")},
            "torch_install_report_sha256":sha256_file(install_path)},
        "total_command_seconds":perf_counter()-started,"all_checks_passed":bool(success)}
    json_save(RESULT_PATH,report)
    print(json.dumps({"result":str(RESULT_PATH),"all_checks_passed":bool(success),"total_seconds":report["total_command_seconds"]},indent=2))
    if not success:
        raise RuntimeError("B0 pipeline calibration failed; results retained without changing seeds or data")
    return report


if __name__ == "__main__":
    main()
