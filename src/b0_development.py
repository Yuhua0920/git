"""Fixed T12B development orchestration; accepted numerical cores are reused."""

import hashlib
import json
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from src import ml_data as md
from src import link
from src.b0_model import ResidualCNN, data_complex_mse, predict_channel
from src.lmmse_train import formal_trajectory, raw_array_sha256, sha256_file
from src.lmmse_train_validation import batched_data_zf, bits_seed
from src.metrics import channel_error_energies, count_bit_errors
from src.nearest_validation import load_references, verify_frame
from src.numerics import noise_variance_from_snr
from src.stochastic_scan import correct_bit_rate_proxy, frame_duration_seconds
from src.tdl_awgn_scan import scale_unit_noise
from src.tdl_noiseless_scan import complex_record

SEED = 13261001
BATCH_SIZE = 128
EPOCHS = 30
SHUFFLE_DOMAIN = 12004
T12A_COMMIT = "acf6bd0eabf7d86b3335f33757070aa858e9d5b3"
T10D_COMMIT = "6129743ca9940ad41a5d9de2f70005c9160822c2"
PREFIX = "t12b_b0_df16_dev"


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def epoch_order(epoch):
    if isinstance(epoch, bool) or not isinstance(epoch, int) or not 1 <= epoch <= EPOCHS:
        raise ValueError("training epoch must be 1..30")
    return np.random.default_rng(np.random.SeedSequence([SHUFFLE_DOMAIN, SEED, epoch])).permutation(20000)


def optimizer_for(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)


def checkpoint_save(path, model, epoch, val_mse, architecture):
    payload = {"state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
               "architecture": architecture, "development_seed": SEED, "density": 16,
               "development_model": True, "counts_toward_formal_five_seed_results": False,
               "epoch": epoch, "validation_mse_pooled": float(val_mse),
               "selection_metric": "pooled validation data-RE complex MSE"}
    torch.save(payload, path)
    loaded, meta = checkpoint_load(path, next(model.parameters()).device)
    if state_hash(loaded) != state_hash(model):
        raise ValueError("checkpoint state did not reload identically")
    return {"path": str(path), "file_sha256": sha256_file(path),
            "state_dict_tensor_sha256": state_hash(loaded), "epoch": meta["epoch"],
            "validation_mse_pooled": meta["validation_mse_pooled"],
            "weights_only_load": True, "identical_reload": True}


def checkpoint_load(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload["development_seed"] != SEED or payload["density"] != 16
            or payload["counts_toward_formal_five_seed_results"]
            or not 0 <= payload["epoch"] <= EPOCHS):
        raise ValueError("wrong development checkpoint identity")
    # Preserve RNG: checkpoint reads must not alter future training randomness.
    with torch.random.fork_rng(devices=[]):
        model = ResidualCNN()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    return model, {k: v for k, v in payload.items() if k != "state_dict"}


def accepted_files(root):
    paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", T12A_COMMIT,
        "outputs", "src", "scripts", "tests", "configs", "docs"], cwd=root, text=True).splitlines()
    return {p: sha256_file(Path(root)/p) for p in paths if p != "docs/stage_status.md"}


def replay_link(root):
    namespace = {}
    source = subprocess.check_output(["git", "show", f"{T10D_COMMIT}:src/link.py"], cwd=root)
    exec(compile(source, "accepted_T10D_link", "exec"), namespace)
    return namespace


def bits_and_y(h, tid, frame, z, snr, layout, historical_link):
    bits, symbols = link.generate_master_qpsk(bits_seeds=tuple(bits_seed(tid, frame, q) for q in range(2)))
    x = link.assemble_pilot_data_frame(symbols, layout)
    noise = scale_unit_noise(z, noise_variance_from_snr(snr))
    y = link.add_received_noise(link.apply_frequency_mimo_channel(h, x), noise)
    old_x = historical_link["assemble_pilot_data_frame"](symbols, layout)
    old_y = historical_link["add_received_noise"](historical_link["apply_frequency_mimo_channel"](h, old_x), noise)
    if not np.array_equal(x, old_x) or not np.array_equal(y, old_y):
        raise ValueError("Y/X not exact accepted T10D replay")
    return bits, y


def cache_truth(prepared, manifest, split, progress=None):
    rows = [r for r in manifest if r["split"] == split]
    expected = list(range(2000)) if split == "train" else list(range(2000, 2400))
    if [r["trajectory_id"] for r in rows] != expected:
        raise ValueError("only frozen train/validation split allowed")
    h = np.empty((len(rows)*10, 2, 2, 64, 16), dtype=np.complex128)
    digests = []
    for index, row in enumerate(rows):
        draw = formal_trajectory(prepared, row)
        h[10*index:10*index+10] = draw.h_frames
        digests.append({"trajectory_id": row["trajectory_id"], "H_long_sha256": raw_array_sha256(draw.h_long)})
        if progress and (index+1) % 100 == 0:
            progress(f"{split} continuous trajectories cached {index+1}/{len(rows)}")
    return h, digests


def generate_training_inputs(h, layout, epoch):
    if h.shape != (20000, 2, 2, 64, 16) or h.dtype != np.complex128:
        raise ValueError("exact 20000 original complex128 training frames required")
    snrs = md.balanced_training_snr(SEED, epoch)
    base = torch.empty((20000, 8, 64, 16), dtype=torch.float32)
    # This array is new each epoch. Only noise-independent H is shared across epochs.
    for index in range(20000):
        tid, frame = divmod(index, 10)
        z = md.training_unit_noise(SEED, epoch, tid, frame)
        ls = md.direct_pilot_ls(h[index], layout, z, int(snrs[index]))
        base[index] = md.complex_to_tensor(md.r3_from_pilot_ls(ls, layout))[0]
    return base, snrs


def prepare_validation(h, digests, refs, layout, historical_link, progress=None):
    if h.shape != (4000, 2, 2, 64, 16):
        raise ValueError("exact 4000 validation frames required")
    base = torch.empty((5, 4000, 8, 64, 16), dtype=torch.float32)
    received = np.empty((5, 4000, 2, 64, 16), dtype=np.complex128)
    bits_cache = []
    for index in range(4000):
        tid, frame = 2000+index//10, index % 10
        record = refs["realizations"][tid]
        if frame == 0 and digests[index//10]["H_long_sha256"] != record["H_long_sha256"]:
            raise ValueError("validation long H hash mismatch")
        z = md.validation_unit_noise(tid, frame)
        for si, snr in enumerate(md.SNR_DB):
            bits, y = bits_and_y(h[index], tid, frame, z, snr, layout, historical_link)
            _, r3 = link.estimate_ls_linear_channel(y, layout)
            base[si, index] = md.complex_to_tensor(r3)[0]
            received[si, index] = y
            if si == 0:
                verify_frame(record["frame_realizations"][frame], frame, h[index], bits, z)
                bits_cache.append(bits)
        if progress and (index+1) % 500 == 0:
            progress(f"fixed validation observations {index+1}/4000 x five SNR")
    return {"base": base, "h": h, "received": received, "bits": bits_cache,
            "prepared_observations": 20000, "fixed_across_epochs": True}


def model_norms(model):
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError("nonfinite model weights")
    return {f"conv{i+1}_weight_norm": float(model.layers[2*i].weight.detach().double().norm().cpu())
            for i in range(4)}


def train_epoch(model, optimizer, base, target, snrs, mask, order):
    """One pass with finite checks; gradient norm is mean across optimizer steps."""
    device = next(model.parameters()).device
    model.train()
    energy, counts = {s: 0.0 for s in md.SNR_DB}, {s: 0 for s in md.SNR_DB}
    gradients = []
    maximum_pilot_modification = 0.0
    selected = torch.as_tensor(mask[0], dtype=torch.bool, device=device)
    ndata = int(np.count_nonzero(mask[0]))
    for start in range(0, len(order), BATCH_SIZE):
        idx = order[start:start+BATCH_SIZE]
        b, t = base[idx].to(device), target[idx].to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = predict_channel(model, b, mask)
        loss = data_complex_mse(pred, t, mask)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite training loss")
        loss.backward()
        norm2 = torch.zeros((), dtype=torch.float64, device=device)
        for p in model.parameters():
            if p.grad is None or not bool(torch.isfinite(p.grad).all()):
                raise FloatingPointError("missing or nonfinite gradient")
            norm2 += p.grad.detach().double().square().sum()
        gradients.append(float(norm2.sqrt().cpu()))
        maximum_pilot_modification = max(maximum_pilot_modification,
            float((pred[..., ~selected]-b[..., ~selected]).abs().max().detach().cpu()))
        # Actual float32 training predictions/targets; float64 sufficient-statistic accumulation.
        errors = (pred.detach().double()-t.double()).reshape(-1, 4, 2, 64, 16)
        sample_energy = errors.square().sum(dim=2)[..., selected].sum(dim=(1, 2)).cpu().numpy()
        for snr in md.SNR_DB:
            chosen = snrs[idx] == snr
            energy[snr] += float(np.sum(sample_energy[chosen], dtype=np.float64))
            counts[snr] += int(np.count_nonzero(chosen))
        optimizer.step()
        model_norms(model)
    if maximum_pilot_modification != 0:
        raise ValueError("pilot union modified during training")
    return {"train_mse": sum(energy.values())/(len(order)*4*ndata),
            "train_per_snr": [{"snr_db": s, "sample_count": counts[s], "error_energy": energy[s]} for s in md.SNR_DB],
            "gradient_norm": float(np.mean(gradients)), "gradient_norm_max": max(gradients),
            "optimizer_steps": len(gradients), "loss_weights_gradients_finite": True,
            "maximum_pilot_modification": maximum_pilot_modification, **model_norms(model)}


def evaluate_validation(model, cache, layout, epoch_zero=False):
    device = next(model.parameters()).device
    model.eval()
    stats = []
    maximum_pilot_modification = 0.0
    zero_identity = True
    with torch.inference_mode():
        for si, snr in enumerate(md.SNR_DB):
            error, truth, count, bit_errors, total_bits = 0.0, 0.0, 0, 0, 0
            for start in range(0, 4000, BATCH_SIZE):
                end = min(start+BATCH_SIZE, 4000)
                b = cache["base"][si, start:end].to(device)
                pred = predict_channel(model, b, layout.data_mask)
                if epoch_zero:
                    zero_identity = zero_identity and torch.equal(pred, b)
                union = torch.as_tensor(layout.pilot_union, device=device)
                maximum_pilot_modification = max(maximum_pilot_modification,
                    float((pred[..., union]-b[..., union]).abs().max().cpu()))
                energies = md.complex_data_error_energies(pred, cache["h"][start:end], layout.data_mask)
                error += energies["error_energy"]; truth += energies["reference_energy"]
                count += energies["num_complex_observations"]
                if epoch_zero:
                    complex_pred = md.tensor_to_complex(pred)
                    for j, hh in enumerate(complex_pred):
                        index = start+j
                        detected = batched_data_zf(cache["received"][si, index], hh, layout.data_mask)
                        errors, bits = count_bit_errors(link.demodulate_data_bits(np.sqrt(2)*detected, layout.data_mask),
                            link.extract_reference_data_bits(cache["bits"][index], layout.data_mask))
                        bit_errors += errors; total_bits += bits
            stats.append({"snr_db": snr, "error_energy": error, "truth_energy": truth,
                "num_complex_coefficients": count, "mse": error/count,
                "nmse_linear": error/truth, "nmse_db": float(10*np.log10(error/truth)),
                **({"bit_errors": bit_errors, "total_bits": total_bits, "ber": bit_errors/total_bits} if epoch_zero else {})})
    if maximum_pilot_modification != 0 or (epoch_zero and not zero_identity):
        raise ValueError("pilot preservation / epoch0 identity failed")
    return {"pooled_mse": sum(r["error_energy"] for r in stats)/sum(r["num_complex_coefficients"] for r in stats),
            "per_snr": stats, "maximum_pilot_modification": maximum_pilot_modification,
            "epoch0_matches_quantized_R3_bitwise": zero_identity if epoch_zero else None,
            "validation_observations": 20000}


def initial_reference(val, refs):
    records = []
    for row in val["per_snr"]:
        old = next(r for r in refs["scan"] if int(r["dF"]) == 16 and int(r["snr_db"]) == row["snr_db"])
        delta = row["nmse_db"]-float(old["ls_nmse_db"])
        records.append({**row, "frozen_r3_nmse_db": float(old["ls_nmse_db"]),
            "nmse_db_difference": delta, "frozen_r3_ber": float(old["ls_ber"]),
            "ber_difference": row["ber"]-float(old["ls_ber"]),
            "bit_error_difference": row["bit_errors"]-int(old["ls_bit_errors"]),
            "nmse_tolerance_passed": abs(delta) <= 1e-4})
    if not val["epoch0_matches_quantized_R3_bitwise"] or not all(r["nmse_tolerance_passed"] for r in records):
        raise ValueError("epoch0 R3 NMSE calibration failed; stop before optimizer updates")
    return {"records": records, "NMSE_tolerance_db": 1e-4, "bitwise_quantized_R3_match": True,
            "BER_not_required_integer_equal": True, "all_checks_passed": True}


def select_best(metric, best_metric):
    if not np.isfinite(metric) or metric < 0:
        raise FloatingPointError("invalid checkpoint selection metric")
    return metric < best_metric


def budget_diagnosis(log):
    values = np.array([r["val_mse_pooled"] for r in log], dtype=np.float64)
    best = int(np.argmin(values))
    last = values[26:31]
    improvement = float((last[0]-last[-1])/last[0])
    overall = float((values[0]-values[best])/values[0])
    worsening = float(values[-1]/values[best]-1)
    plateau = float(np.ptp(last)/np.mean(last))
    if overall < 0.01:
        category, explanation = "D", "learning-baseline limitation / optimization issue"
    elif best < 25 and worsening > 0.05:
        category, explanation = "C", "overfitting indication"
    elif best >= 28 and np.all(np.diff(last) < 0) and improvement > 0.005:
        category, explanation = "B", "budget may be insufficient"
    elif best < 25 and plateau <= 0.01 and worsening <= 0.02:
        category, explanation = "A", "current budget likely sufficient"
    else:
        category, explanation = "mixed/unclassified", "does not meet any predeclared A-D rule"
    return {"best_epoch": best, "best_val_mse": float(values[best]), "epoch30_val_mse": float(values[-1]),
            "last_5_epoch_relative_improvement": improvement, "last_5_epoch_range_over_mean": plateau,
            "epoch0_to_best_relative_improvement": overall, "final_relative_worsening_from_best": worsening,
            "category": category, "description": explanation, "extra_training_performed": False,
            "thresholds_predeclared_in": "docs/b0_development_protocol.md"}


def oracle_red_flag(rows):
    return any(r["b0_nmse_db"] < r["r5_oracle_nmse_db"]-0.10 for r in rows)


def aggregate(rows, refs):
    result, paired = [], []
    for snr in md.SNR_DB:
        group = [r for r in rows if r["snr_db"] == snr]
        old = next(r for r in refs["scan"] if int(r["dF"]) == 16 and int(r["snr_db"]) == snr)
        error = sum(r["b0_nmse_error_energy"] for r in group)
        truth = sum(r["nmse_truth_energy"] for r in group)
        errors, bits = sum(r["b0_bit_errors"] for r in group), sum(r["total_bits"] for r in group)
        db = float(10*np.log10(error/truth))
        r3, r4 = float(old["ls_nmse_db"]), float(old["train_nmse_db"])
        b0_ber = errors/bits
        row = {"snr_db": snr, "pilot_ratio": 1/32, "num_trajectories": 400,
            "frames_aggregated": 4000, "total_bits": bits, "b0_nmse_error_energy": error,
            "nmse_truth_energy": truth, "b0_bit_errors": errors, "b0_nmse_linear": error/truth,
            "r3_nmse_db": r3, "b0_nmse_db": db, "r4_nmse_db": r4,
            "r5_oracle_nmse_db": float(old["oracle_nmse_db"]),
            "r3_to_b0_gain_db": r3-db, "r3_to_r4_gain_db": r3-r4,
            "r3_r4_gap_closed_fraction": (r3-db)/(r3-r4), "b0_minus_r4_db": db-r4,
            "b0_minus_oracle_db": db-float(old["oracle_nmse_db"]),
            "r3_ber": float(old["ls_ber"]), "b0_ber": b0_ber,
            "r4_ber": float(old["train_ber"]), "r5_ber": float(old["oracle_ber"]),
            "perfect_ber": float(old["perfect_ber"]), "r3_proxy": float(old["ls_proxy_bps"]),
            "b0_proxy": correct_bit_rate_proxy(bits/4000, b0_ber, frame_duration_seconds())[1],
            "r4_proxy": float(old["train_proxy_bps"]),
            "ber_gap_closed_fraction": (float(old["ls_ber"])-b0_ber)/(float(old["ls_ber"])-float(old["train_ber"]))}
        result.append(row)
        ed = np.array([r["b0_nmse_error_energy"]-r["r3_nmse_error_energy"] for r in group])
        bd = np.array([r["b0_bit_errors"]-r["r3_bit_errors"] for r in group])
        def quartiles(v):
            q = np.quantile(v, [0.25, 0.5, 0.75])
            return {"Q25": float(q[0]), "median": float(q[1]), "Q75": float(q[2])}
        paired.append({"snr_db": snr, "num_trajectory_units": len(group),
            "B0_minus_R3_error_energy": quartiles(ed), "B0_minus_R3_bit_errors": quartiles(bd),
            "fraction_lower_B0_bit_errors": float(np.mean(bd < 0)), "significance_claim": False})
    return result, paired


def fixed_sample(root, h, r3, b0, ls, y, bits, layout):
    from src.lmmse_train import load_frozen_covariance, prepare_train_lmmse
    from src.lmmse import estimate_mimo_from_pilot_ls
    frozen = load_frozen_covariance(Path(root)/"outputs/t10c_lmmse_train_covariance.npz", md.MANIFEST_SHA256)
    estimators = tuple(prepare_train_lmmse(frozen, layout.pilot_mask[q], 0.1) for q in range(2))
    r4 = estimate_mimo_from_pilot_ls(ls, estimators)
    historical = json.loads((Path(root)/"outputs/t11c_b_detector_sensitivity_results.json").read_text(encoding="utf8"))
    reference = historical["validation_t10d_replay_regression"]
    if raw_array_sha256(y) != reference["Y_sha256"]:
        raise ValueError("fixed Y sentinel changed")
    for name, channel in (("R3", r3), ("R4", r4)):
        recorded = next(r for r in reference["records"] if r["estimator"] == name)
        if raw_array_sha256(channel) != recorded["H_est_sha256"]:
            raise ValueError(f"fixed {name} historical channel changed")
    points = []
    for k, t in ((1, 2), (1, 15)):
        points.append({"k": k, "t": t, "H_true": complex_record(h[:, :, k, t]),
            **{f"H_{name}": complex_record(channel[:, :, k, t]) for name, channel in (("R3", r3), ("B0", b0), ("R4", r4))},
            **{f"absolute_error_{name}": np.abs(channel[:, :, k, t]-h[:, :, k, t]).tolist()
               for name, channel in (("R3", r3), ("B0", b0), ("R4", r4))}})
    return {"trajectory_id": 2000, "frame_index": 0, "snr_db": 10, "dF": 16, "points": points,
            "norm_B0_minus_R3_over_R3": float(np.linalg.norm(b0-r3)/np.linalg.norm(r3)),
            "norm_B0_minus_R4_over_R4": float(np.linalg.norm(b0-r4)/np.linalg.norm(r4)),
            "H_R3_sha256": raw_array_sha256(r3), "H_R4_sha256": raw_array_sha256(r4),
            "Y_sha256": raw_array_sha256(y), "frozen_T11CB_sentinel_exact_match": True,
            "R4_evaluations": 1, "R4_full_performance_scan": False}


def full_link_validation(root, prepared, refs, model, layout, historical_link, progress=None):
    device = next(model.parameters()).device
    model.eval()
    rows, realizations, residuals = [], [], {s: [0.0, 0.0, 0j, 0] for s in md.SNR_DB}
    diagnostic = None
    maximum_pilot_modification = 0.0
    for row in refs["manifest"]:
        tid = row["trajectory_id"]
        draw = formal_trajectory(prepared, row)
        reference = refs["realizations"][tid]
        if raw_array_sha256(draw.h_long) != reference["H_long_sha256"]:
            raise ValueError("full-link H_long replay mismatch")
        sums = {s: [0.0, 0.0, 0, 0] for s in md.SNR_DB}
        bases, details, frames = [], [], []
        for frame, h in enumerate(draw.h_frames):
            z = md.validation_unit_noise(tid, frame)
            for snr in md.SNR_DB:
                bits, y = bits_and_y(h, tid, frame, z, snr, layout, historical_link)
                if snr == 0:
                    frames.append(verify_frame(reference["frame_realizations"][frame], frame, h, bits, z))
                ls, r3 = link.estimate_ls_linear_channel(y, layout)
                bases.append(md.complex_to_tensor(r3)[0])
                details.append((frame, snr, h, y, bits, ls, r3))
        batch = torch.stack(bases).to(device)
        with torch.inference_mode():
            prediction = predict_channel(model, batch, layout.data_mask)
            union = torch.as_tensor(layout.pilot_union, device=device)
            maximum_pilot_modification = max(maximum_pilot_modification,
                float((prediction[..., union]-batch[..., union]).abs().max().cpu()))
            channels = md.tensor_to_complex(prediction)
            quantized = md.tensor_to_complex(batch)
        for index, (frame, snr, h, y, bits, ls, r3) in enumerate(details):
            b0 = channels[index]
            error, truth = channel_error_energies(b0, h, layout.data_mask)
            detected = batched_data_zf(y, b0, layout.data_mask)
            errors, count = count_bit_errors(link.demodulate_data_bits(np.sqrt(2)*detected, layout.data_mask),
                link.extract_reference_data_bits(bits, layout.data_mask))
            sums[snr][0] += error; sums[snr][1] += truth; sums[snr][2] += errors; sums[snr][3] += count
            selected = layout.data_mask[0]
            delta = (b0-quantized[index])[..., selected]
            base = quantized[index][..., selected]
            residuals[snr][0] += float(np.sum(np.abs(delta)**2, dtype=np.float64))
            residuals[snr][1] += float(np.sum(np.abs(base)**2, dtype=np.float64))
            residuals[snr][2] += np.sum(delta, dtype=np.complex128)
            residuals[snr][3] += delta.size
            if (tid, frame, snr) == (2000, 0, 10):
                diagnostic = fixed_sample(root, h, r3, b0, ls, y, bits, layout)
        for snr in md.SNR_DB:
            error, truth, errors, bits = sums[snr]
            old = refs["trajectory_index"][tid, 16, snr]
            if (bits != int(old["total_data_bits"]) or not np.isclose(truth, float(old["nmse_truth_energy"]), rtol=1e-13)):
                raise ValueError("actual data mask/bit count/truth energy drift")
            rows.append({"trajectory_id": tid, "snr_db": snr, "noise_variance": noise_variance_from_snr(snr),
                "frames_aggregated": 10, "total_bits": bits, "b0_nmse_error_energy": error,
                "nmse_truth_energy": truth, "b0_bit_errors": errors,
                "r3_nmse_error_energy": float(old["ls_nmse_error_energy"]), "r3_bit_errors": int(old["ls_bit_errors"]),
                "r4_nmse_error_energy": float(old["train_nmse_error_energy"]), "r4_bit_errors": int(old["train_bit_errors"]),
                "same_H": True, "same_Y": True, "same_bits": True})
        realizations.append({"trajectory_id": tid, "H_long_sha256": raw_array_sha256(draw.h_long),
            "frame_realizations": frames, "Y_exact_accepted_T10D_replays": 50})
        if progress and (tid+1-2000) % 25 == 0:
            progress(f"best frozen B0+ZF full link {tid+1-2000}/400 trajectories")
    if maximum_pilot_modification != 0:
        raise ValueError("full-link pilot values modified")
    residual_report = []
    for snr, (de, be, mean, count) in residuals.items():
        residual_report.append({"snr_db": snr, "RMS_residual": float(np.sqrt(de/count)),
            "RMS_H_base": float(np.sqrt(be/count)), "RMS_ratio": float(np.sqrt(de/be)),
            "residual_mean": complex_record(np.asarray(mean/count)), "num_complex_data_coefficients": count})
    aggregates, paired = aggregate(rows, refs)
    return {"trajectory_rows": rows, "aggregates": aggregates, "paired": paired,
            "residual": residual_report, "fixed_sample": diagnostic, "realizations": realizations,
            "replay": {"H_long_hashes_matched": 400, "H_frame_hashes_matched": 4000,
                "bits_hashes_matched": 4000, "unit_noise_hashes_matched": 4000,
                "Y_exact_committed_T10D_link_matched": 20000, "all_checks_passed": True},
            "maximum_pilot_modification": maximum_pilot_modification}
