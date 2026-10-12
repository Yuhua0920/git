"""Lossless fixed-density batching and ordered Windows DataLoader for PERF-Popt."""

import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from time import perf_counter

from src import b0_development as p0
from src import ml_data as md
from src.estimators import frequency_linear_interpolate_ls, time_linear_interpolate_frequency_estimates
from src.pilots import build_fdm_pilot_layout, map_pilot_symbols
from src.link import PILOT_SYMBOLS
from src.numerics import noise_variance_from_snr

T12B_COMMIT = "3cc98f9153070abe039e1e077e3301af4447b9fe"


def raw_hash(array):
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    return hashlib.sha256(np.asarray(array).tobytes(order="C")).hexdigest()


def _linear_axis(values, anchors, size, axis):
    """Use the same two real np.interp operations, without Python trace loops."""
    x = np.arange(size)
    low = np.clip(np.searchsorted(anchors, x, side="right")-1, 0, len(anchors)-2)
    high = low+1
    shape = [1]*values.ndim; shape[axis] = size
    width = (anchors[high]-anchors[low]).reshape(shape)
    distance = (x-anchors[low]).reshape(shape)
    left, right = np.take(values, low, axis=axis), np.take(values, high, axis=axis)
    real = (right.real-left.real)/width*distance+left.real
    imag = (right.imag-left.imag)/width*distance+left.imag
    out = real+1j*imag
    out = np.where((x<anchors[0]).reshape(shape), np.take(values, [0], axis=axis), out)
    out = np.where((x>anchors[-1]).reshape(shape), np.take(values, [-1], axis=axis), out)
    key = [slice(None)]*values.ndim; key[axis] = anchors
    out[tuple(key)] = values
    return out


def vectorized_r3(pilot_ls, layout, timings=None):
    observations = np.asarray(pilot_ls)
    if observations.ndim != 5 or observations.shape[1:] != (2, 2, 64, 16) or observations.dtype != np.complex128:
        raise ValueError("complex128 [B,2,2,64,16] required")
    if layout.offsets_by_tx != (0, 8):
        raise ValueError("T12C-PERF only dF16")
    marked = np.broadcast_to(layout.pilot_mask[None, None], observations.shape)
    if (not np.all(np.isfinite(observations[marked])) or
            not np.all(np.isnan(observations[~marked].real)) or not np.all(np.isnan(observations[~marked].imag))):
        raise ValueError("finite marked Pilot-LS and NaN+jNaN elsewhere required")
    times = np.flatnonzero(np.any(layout.pilot_mask, axis=(0, 1)))
    stamp = perf_counter(); frequency = []
    for q in range(2):
        anchors = np.flatnonzero(layout.pilot_mask[q, :, times[0]])
        compact = np.take(np.take(observations[:, :, q], anchors, axis=-2), times, axis=-1)
        frequency.append(_linear_axis(compact, anchors, 64, -2))
    frequency = np.stack(frequency, axis=2)
    if timings is not None:
        timings["R3_frequency_interpolation"] = perf_counter()-stamp
    stamp = perf_counter(); out = _linear_axis(frequency, times, 16, -1)
    if timings is not None:
        timings["R3_time_interpolation"] = perf_counter()-stamp
    return out


def batch_pilot_ls(h, noise, snrs, layout):
    if h.ndim != 5 or h.shape[1:] != (2, 2, 64, 16) or h.dtype != np.complex128 or noise.shape != (len(h), 2, 64, 16) or noise.dtype != np.complex128:
        raise ValueError("original complex128 H/noise batch required")
    if len(snrs) != len(h) or any(s not in md.SNR_DB for s in snrs):
        raise ValueError("original five-SNR assignment required")
    sigma = np.sqrt(np.array([noise_variance_from_snr(int(s)) for s in snrs], dtype=np.float64))
    n = sigma[:, None, None, None]*noise
    known = map_pilot_symbols(layout, PILOT_SYMBOLS)
    ls = np.full(h.shape, np.nan+1j*np.nan, dtype=np.complex128)
    for q in range(2):
        mask = layout.pilot_mask[q]
        ls[:, :, q, mask] = h[:, :, q, mask]+n[:, :, mask]/known[q, mask][None, None]
    return ls


def build_batch(h_cache, epoch, indices, mode="vectorized", verify=True, include_complex=False):
    if epoch not in (1, 2, 7, 30):
        raise ValueError("only historical calibration epochs permitted")
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or not len(indices) or np.any(indices < 0) or np.any(indices >= 20000):
        raise ValueError("train canonical frame IDs 0..19999 required")
    layout = build_fdm_pilot_layout(2, 64, 16, 16, 4)
    snrs = md.balanced_training_snr(p0.SEED, epoch)[indices]
    times = {}
    stamp = perf_counter(); h = np.asarray(h_cache[indices], dtype=np.complex128)
    times["H_true_cache_access"] = perf_counter()-stamp
    stamp = perf_counter()
    noise = np.stack([md.training_unit_noise(p0.SEED, epoch, int(i)//10, int(i)%10) for i in indices])
    times["unit_noise_generation"] = perf_counter()-stamp
    stamp = perf_counter()
    if mode == "vectorized":
        ls = batch_pilot_ls(h, noise, snrs, layout)
        times["Pilot_LS"] = perf_counter()-stamp
        base_complex = vectorized_r3(ls, layout, times)
    elif mode == "P0":
        ls = [md.direct_pilot_ls(h[j], layout, noise[j], int(snrs[j])) for j in range(len(indices))]
        times["Pilot_LS"] = perf_counter()-stamp
        stamp = perf_counter(); frequency = [frequency_linear_interpolate_ls(a, layout.pilot_mask)[0] for a in ls]
        times["R3_frequency_interpolation"] = perf_counter()-stamp
        stamp = perf_counter()
        base_complex = np.stack([time_linear_interpolate_frequency_estimates(a, [0, 4, 8, 12]) for a in frequency])
        times["R3_time_interpolation"] = perf_counter()-stamp
    else:
        raise ValueError("unknown pipeline mode")
    stamp = perf_counter(); base, target = md.complex_to_tensor(base_complex), md.complex_to_tensor(h)
    times["complex_to_float32"] = perf_counter()-stamp
    hashes = {"H_true": raw_hash(h), "unit_noise": raw_hash(noise), "H_base": raw_hash(base),
              "target": raw_hash(target), "sample_IDs": raw_hash(indices), "SNR": raw_hash(snrs)} if verify else {}
    result = {"base": base, "target": target, "IDs": torch.from_numpy(indices.copy()),
              "SNR": torch.from_numpy(snrs.copy()), "hashes": hashes, "timings": times}
    if include_complex:
        result.update({"base_complex": base_complex, "H_true": h, "unit_noise": noise})
    return result


def identity_collate(value):
    return value


def worker_init(worker_id):
    # Randomness comes exclusively from explicit sample coordinates, never worker_id.
    torch.set_num_threads(1)


class EpochBatchSampler(Sampler):
    def __init__(self, count=20000, epoch=1, fixed_subset=None):
        self.count, self.epoch, self.fixed_subset = count, epoch, fixed_subset

    def ordered_ids(self):
        order = p0.epoch_order(self.epoch)
        if self.fixed_subset is not None:
            order = order[np.isin(order, self.fixed_subset)]
        return order[:self.count]

    def __iter__(self):
        order = self.ordered_ids()
        for start in range(0, len(order), 128):
            yield (self.epoch, order[start:start+128].tolist())

    def __len__(self):
        return (self.count+127)//128


class BatchDataset(Dataset):
    def __init__(self, cache_path, mode="vectorized", verify=True):
        self.cache_path, self.mode, self.verify = str(cache_path), mode, verify
        self.h = None

    def __len__(self):
        return 20000

    def __getitem__(self, key):
        epoch, indices = key
        if self.h is None:
            self.h = np.load(self.cache_path, mmap_mode="r", allow_pickle=False)
        return build_batch(self.h, epoch, indices, self.mode, self.verify)

    def __getstate__(self):
        state = self.__dict__.copy(); state["h"] = None
        return state


def make_loader(cache_path, sampler, workers=0, prefetch=2, persistent=True, pin=False, mode="vectorized"):
    kwargs = {"dataset": BatchDataset(cache_path, mode), "sampler": sampler, "batch_size": None,
              "collate_fn": identity_collate, "num_workers": workers, "pin_memory": pin, "in_order": True}
    if workers:
        kwargs.update({"multiprocessing_context": "spawn", "worker_init_fn": worker_init,
                       "prefetch_factor": prefetch, "persistent_workers": persistent})
    return DataLoader(**kwargs)


def close_loader(loader):
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()
        loader._iterator = None
