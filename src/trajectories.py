"""Continuous multi-frame TDL-A trajectories; explicit reusable Jakes factor.

One trajectory-level seed draws all 160 symbols. Frames are views of that
process. The existing Doppler/PDP/fractional-delay implementations are reused.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from src.doppler import (
    CARRIER_FREQUENCY_HZ, SPEED_KMH, correlated_complex_gaussian,
    covariance_eigenfactor, jakes_covariance, mobility_parameters,
    ofdm_time_grid_seconds,
)
from src.tdl import (
    centered_frequency_grid_hz, fractional_delay_frequency_response,
    random_tdl_a_taps, tdl_a_profile,
)


CHANNEL_SEED_BASE = 40261001
FRAMES_PER_TRAJECTORY = 10
SYMBOLS_PER_FRAME = 16
SPLIT_NAMES = ("train", "validation_id", "ood_dev", "test_id", "test_ood")


def _integer(name, value, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def frame_metadata(trajectory_id, frames_per_trajectory=10, symbols_per_frame=16):
    """Global start/stop are inclusive symbol indices, not Python slice stops."""
    _integer("trajectory_id", trajectory_id)
    _integer("frames_per_trajectory", frames_per_trajectory, 1)
    _integer("symbols_per_frame", symbols_per_frame, 1)
    return [{"trajectory_id": int(trajectory_id), "frame_index": f,
             "global_symbol_start": f * symbols_per_frame,
             "global_symbol_stop": (f + 1) * symbols_per_frame - 1}
            for f in range(frames_per_trajectory)]


def slice_channel_frames(h_long, symbols_per_frame=16):
    """Return a shared-memory [frame,RX,TX,K,symbol] view, without copying."""
    _integer("symbols_per_frame", symbols_per_frame, 1)
    h = np.asarray(h_long)
    if (h.ndim != 4 or h.shape[:3] != (2, 2, 64) or h.shape[-1] < 1
            or h.shape[-1] % symbols_per_frame or h.dtype != np.complex128):
        raise ValueError("expected complex128 H[2,2,64,N] with whole frames")
    frames = h.reshape(2, 2, 64, h.shape[-1] // symbols_per_frame, symbols_per_frame).transpose(3, 0, 1, 2, 4)
    if not np.shares_memory(frames, h):
        raise ValueError("frame slicing must preserve shared memory")
    return frames


@dataclass(frozen=True)
class ChannelTrajectory:
    trajectory_id: int
    channel_seed: int
    taps: np.ndarray
    h_long: np.ndarray
    h_frames: np.ndarray
    metadata: list[dict]


@dataclass(frozen=True)
class PreparedTrajectory:
    time_grid_s: np.ndarray
    covariance: np.ndarray
    factor: np.ndarray
    factorization_summary: dict
    mobility: dict
    delays_s: np.ndarray
    powers: np.ndarray
    frequencies_hz: np.ndarray
    phase_matrix: np.ndarray
    frames_per_trajectory: int
    symbols_per_frame: int
    symbol_duration: float
    preparation_timing: dict

    def generate_taps(self, channel_seed):
        _integer("channel_seed", channel_seed)
        rng = np.random.default_rng(channel_seed)
        if self.mobility["maximum_doppler_hz"] == 0:
            # Reuse T08A's seeded draw and copy exactly, including every frame.
            return np.repeat(random_tdl_a_taps(rng)[..., None], self.time_grid_s.size, axis=-1)
        processes = correlated_complex_gaussian(rng, self.factor, (2, 2, 23))
        return processes * np.sqrt(self.powers)[None, None, :, None]

    def frequency_response(self, taps):
        taps = np.asarray(taps)
        if taps.shape != (2, 2, 23, self.time_grid_s.size) or taps.dtype != np.complex128:
            raise ValueError("taps must be complex128 [2,2,23,Nsymbols]")
        if self.mobility["maximum_doppler_hz"] == 0:
            response = taps[..., 0] @ self.phase_matrix
            return np.repeat(response[..., None], self.time_grid_s.size, axis=-1)
        response = np.moveaxis(taps, -2, -1) @ self.phase_matrix
        return np.moveaxis(response, -2, -1)

    def generate(self, trajectory_id):
        _integer("trajectory_id", trajectory_id)
        seed = CHANNEL_SEED_BASE + int(trajectory_id)
        taps = self.generate_taps(seed)
        h_long = self.frequency_response(taps)
        taps.setflags(write=False)
        h_long.setflags(write=False)
        frames = slice_channel_frames(h_long, self.symbols_per_frame)
        return ChannelTrajectory(int(trajectory_id), seed, taps, h_long, frames,
            frame_metadata(trajectory_id, self.frames_per_trajectory, self.symbols_per_frame))


def prepare_trajectory(frames_per_trajectory=10, symbols_per_frame=16, speed_kmh=SPEED_KMH):
    """Prepare covariance and full eigenfactor once; generate() never rebuilds."""
    _integer("frames_per_trajectory", frames_per_trajectory, 1)
    _integer("symbols_per_frame", symbols_per_frame, 1)
    started = perf_counter()
    mobility = mobility_parameters(speed_kmh, CARRIER_FREQUENCY_HZ)
    times = ofdm_time_grid_seconds(num_symbols=frames_per_trajectory * symbols_per_frame)
    covariance = jakes_covariance(times, mobility["maximum_doppler_hz"])
    covariance_seconds = perf_counter() - started
    factor_start = perf_counter()
    factor, summary = covariance_eigenfactor(covariance)
    factor_seconds = perf_counter() - factor_start
    phase_start = perf_counter()
    delays, powers = tdl_a_profile()
    frequencies = centered_frequency_grid_hz(64, 30000)
    # Obtain the already validated T08A phase matrix through its public API.
    phase = fractional_delay_frequency_response(np.eye(23, dtype=np.complex128), delays, frequencies)
    for array in (times, covariance, factor, delays, powers, frequencies, phase):
        array.setflags(write=False)
    timing = {"covariance_preparation_seconds": covariance_seconds,
              "factorization_seconds": factor_seconds,
              "profile_phase_preparation_seconds": perf_counter() - phase_start,
              "total_preparation_seconds": perf_counter() - started}
    return PreparedTrajectory(times, covariance, factor, summary, mobility, delays, powers,
        frequencies, phase, frames_per_trajectory, symbols_per_frame, 80 / 1920000, timing)


def validate_trajectory_splits(split_to_ids):
    """Validate an explicit manifest before any frame/noise/SNR variants exist.

No random splitting or formal ID ranges are invented here. Return the sole
trajectory-ID-to-split lookup shared by every future variant.
"""
    assignments = {}
    for split, ids in split_to_ids.items():
        if split not in SPLIT_NAMES:
            raise ValueError(f"unknown dataset split: {split}")
        for tid in ids:
            _integer("trajectory_id", tid)
            if tid in assignments:
                raise ValueError(f"trajectory {tid} appears more than once in split manifest")
            assignments[int(tid)] = split
    return assignments


def validate_variant_splits(records, assignments):
    """Reject every variant whose split differs from its trajectory manifest."""
    for record in records:
        tid = record["trajectory_id"]
        _integer("trajectory_id", tid)
        if tid not in assignments or record["split"] != assignments[tid]:
            raise ValueError(f"variant of trajectory {tid} violates trajectory-level split")
    return True
