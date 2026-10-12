"""Instrumentation and disposable calibration only; no development continuation."""

from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter
from unittest.mock import patch
import hashlib

import numpy as np
import torch

from src import b0_development as p0, ml_data as md
from src.b0_perf import raw_hash


class EventProfile:
    def __init__(self):
        self.events = defaultdict(list)
        self.losses = []

    def call(self, stage, fn, *args, **kwargs):
        first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        first.record(); result = fn(*args, **kwargs); last.record()
        self.events[stage].append((first, last))
        return result

    def seconds(self):
        torch.cuda.synchronize()
        return {name: sum(a.elapsed_time(b) for a, b in pairs)/1000 for name, pairs in self.events.items()}


@contextmanager
def profile_original_training(optimizer, recorder):
    forward, loss, backward, transfer, step = p0.predict_channel, p0.data_complex_mse, torch.Tensor.backward, torch.Tensor.to, optimizer.step
    def recorded_loss(*args, **kwargs):
        value = recorder.call("loss", loss, *args, **kwargs)
        recorder.losses.append(float(value.detach().cpu()))
        return value
    def recorded_to(tensor, *args, **kwargs):
        if tensor.device.type == "cpu" and tensor.ndim == 4 and tensor.shape[1:] == (8, 64, 16):
            return recorder.call("H2D", transfer, tensor, *args, **kwargs)
        return transfer(tensor, *args, **kwargs)
    with patch.object(p0, "predict_channel", lambda *a, **k: recorder.call("forward", forward, *a, **k)), \
         patch.object(p0, "data_complex_mse", recorded_loss), \
         patch.object(torch.Tensor, "backward", lambda *a, **k: recorder.call("backward", backward, *a, **k)), \
         patch.object(torch.Tensor, "to", recorded_to), \
         patch.object(optimizer, "step", lambda *a, **k: recorder.call("optimizer_step", step, *a, **k)):
        yield


def train_stream(model, optimizer, batches, mask, *, non_blocking=False, resources=None):
    """P0 operations and monitors, with a FIFO source of exactly the same batches.

    Runtime calibration weights are discarded by the caller. No epoch/checkpoint writer.
    """
    device = next(model.parameters()).device
    model.train()
    selected = torch.as_tensor(mask[0], dtype=torch.bool, device=device)
    recorder = EventProfile()
    energy, counts = {s: 0.0 for s in md.SNR_DB}, {s: 0 for s in md.SNR_DB}
    gradients, ordered_ids, ordered_snrs, hashes = [], [], [], []
    waiting = 0.0; maximum_pilot = 0.0; num_samples = 0
    stage_seconds = defaultdict(float)
    iterator = iter(batches)
    if resources is not None:
        resources.children = [w.pid for w in getattr(iterator, "_workers", [])]
    started = perf_counter()
    while True:
        stamp = perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        waiting += perf_counter()-stamp
        if resources is not None and resources.rows[-1]["available_RAM_bytes"] < 1.5*2**30:
            raise MemoryError("candidate rejected: available RAM below predeclared 1.5GiB")
        for k, v in batch["timings"].items():
            stage_seconds[k] += v
        ids, snrs = batch["IDs"].numpy(), batch["SNR"].numpy()
        ordered_ids.extend(ids.tolist()); ordered_snrs.extend(snrs.tolist()); hashes.append(batch["hashes"])
        num_samples += len(ids)
        b = recorder.call("H2D", batch["base"].to, device, non_blocking=non_blocking)
        t = recorder.call("H2D", batch["target"].to, device, non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)
        pred = recorder.call("forward", p0.predict_channel, model, b, mask)
        loss = recorder.call("loss", p0.data_complex_mse, pred, t, mask)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite calibration loss")
        recorder.losses.append(float(loss.detach().cpu()))
        recorder.call("backward", loss.backward)
        norm2 = torch.zeros((), dtype=torch.float64, device=device)
        for p in model.parameters():
            if p.grad is None or not bool(torch.isfinite(p.grad).all()):
                raise FloatingPointError("missing or nonfinite calibration gradient")
            norm2 += p.grad.detach().double().square().sum()
        gradients.append(float(norm2.sqrt().cpu()))
        maximum_pilot = max(maximum_pilot, float((pred[..., ~selected]-b[..., ~selected]).abs().max().detach().cpu()))
        errors = (pred.detach().double()-t.double()).reshape(-1, 4, 2, 64, 16)
        sample_energy = errors.square().sum(dim=2)[..., selected].sum(dim=(1, 2)).cpu().numpy()
        for snr in md.SNR_DB:
            chosen = snrs == snr
            energy[snr] += float(np.sum(sample_energy[chosen], dtype=np.float64))
            counts[snr] += int(np.count_nonzero(chosen))
        recorder.call("optimizer_step", optimizer.step)
        p0.model_norms(model)
    torch.cuda.synchronize()
    wall = perf_counter()-started
    if maximum_pilot:
        raise ValueError("pilot union modified")
    # The same final finite/norm check as P0.
    p0.model_norms(model)
    return {"wall_seconds": wall, "samples": num_samples, "samples_per_second": num_samples/wall,
            "GPU_stage_seconds": recorder.seconds(), "CPU_worker_stage_seconds_sum": dict(stage_seconds),
            "DataLoader_waiting_seconds": waiting, "loss_trace": recorder.losses,
            "batch_hashes": hashes, "ordered_ID_hash": raw_hash(np.asarray(ordered_ids, dtype=np.int64)),
            "SNR_sequence_hash": raw_hash(np.asarray(ordered_snrs, dtype=np.int64)),
            "state_dict_hash": p0.state_hash(model), "optimizer_steps": len(recorder.losses),
            "pilot_modification": maximum_pilot}


class TransferredSlice:
    def __init__(self, tensor, recorder, non_blocking):
        self.tensor, self.recorder, self.non_blocking = tensor, recorder, non_blocking
    def to(self, device):
        return self.recorder.call("H2D", self.tensor.to, device, non_blocking=self.non_blocking)


class ValidationTransport:
    def __init__(self, tensor, recorder, non_blocking=False):
        self.tensor, self.recorder, self.non_blocking = tensor, recorder, non_blocking
    def __getitem__(self, key):
        return TransferredSlice(self.tensor[key], self.recorder, self.non_blocking)


def validation_run(model, cache, layout, tensor=None, non_blocking=False):
    """Call the complete original metric path; change only tensor residence/transfer."""
    recorder, digest = EventProfile(), hashlib.sha256()
    predictions = []
    forward = p0.predict_channel
    def recorded_forward(*args, **kwargs):
        pred = recorder.call("validation_forward", forward, *args, **kwargs)
        cpu = pred.detach().cpu()
        digest.update(cpu.numpy().tobytes())
        predictions.append(raw_hash(cpu))
        return pred
    source = cache["base"] if tensor is None else tensor
    transport = {**cache, "base": ValidationTransport(source, recorder, non_blocking)}
    torch.cuda.synchronize(); stamp = perf_counter()
    with patch.object(p0, "predict_channel", recorded_forward):
        metrics = p0.evaluate_validation(model, transport, layout)
    torch.cuda.synchronize()
    return {"wall_seconds": perf_counter()-stamp, "samples": 20000, "metrics": metrics,
            "prediction_tensor_SHA256": digest.hexdigest(), "prediction_batch_hashes": predictions,
            "stages": recorder.seconds(), "input_tensor_SHA256": raw_hash(source)}
