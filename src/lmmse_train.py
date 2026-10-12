"""Manifest-controlled training second moments and frozen Train-LMMSE.

No Oracle covariance builder or validation observations enter accumulation.
The Cholesky/estimation mathematics delegates to the frozen T10A core.
"""

import csv
import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from src.lmmse import pilot_flat_indices, prepare_oracle_lmmse
from src.trajectories import ChannelTrajectory, frame_metadata, slice_channel_frames


FORMAL_CHANNEL_SEED_BASE = 60261001
NUM_TRAIN_TRAJECTORIES = 2000
NUM_VALIDATION_TRAJECTORIES = 400
FRAMES_PER_TRAJECTORY = 10
OBSERVATIONS_PER_TRAJECTORY = 40
MEAN_POLICY = "known zero-mean model"
FLATTEN_ORDER = "C; flat_index=k*16+t"
BATCH_SIZE = 25
PSD_TOLERANCE = 1e-12
SYMMETRY_TOLERANCE = 1e-12


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_array_sha256(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def formal_manifest_rows():
    return [{"split": "train" if tid < 2000 else "validation_id", "trajectory_id": tid,
        "channel_seed": FORMAL_CHANNEL_SEED_BASE + tid, "frames_per_trajectory": 10,
        "symbols_per_frame": 16, "symbols_per_trajectory": 160,
        "channel_model": "TDL-A", "delay_spread_ns": 100, "speed_kmh": 30}
        for tid in range(2400)]


def ensure_manifest(path):
    """Write once; existing manifests must exactly match the frozen specification."""
    path = Path(path)
    expected = formal_manifest_rows()
    if not path.exists():
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(expected[0]))
            writer.writeheader(); writer.writerows(expected)
    return read_manifest(path)


def read_manifest(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = formal_manifest_rows()
    if len(rows) != 2400 or any(dict(row) != {k:str(v) for k,v in spec.items()} for row,spec in zip(rows,expected)):
        raise ValueError("formal train/validation manifest differs from frozen 2400-row specification")
    return expected


def formal_trajectory(model, row):
    """Use the existing 160-symbol kernels with the explicit formal channel seed."""
    tid = row["trajectory_id"]
    if (isinstance(tid, bool) or not isinstance(tid, int) or tid < 0 or tid >= 2400
            or row["channel_seed"] != FORMAL_CHANNEL_SEED_BASE + tid
            or row["split"] != ("train" if tid < 2000 else "validation_id")
            or model.time_grid_s.size != 160 or model.frames_per_trajectory != 10):
        raise ValueError("formal trajectory requires its exact manifest ID/split/seed and 160-symbol model")
    taps = model.generate_taps(row["channel_seed"])
    h_long = model.frequency_response(taps)
    taps.setflags(write=False); h_long.setflags(write=False)
    frames = slice_channel_frames(h_long)
    return ChannelTrajectory(tid, row["channel_seed"], taps, h_long, frames, frame_metadata(tid))


def pack_link_frames(frames):
    if frames.shape != (10,2,2,64,16) or frames.dtype != np.complex128:
        raise ValueError("expected complex128 frames[10,2,2,64,16]")
    # Observation order [frame,RX,TX]; within each row [K,T] C order.
    return np.ascontiguousarray(frames).reshape(40,1024,order="C")


class SecondMomentAccumulator:
    def __init__(self, dimension=1024):
        self.second_moment_sum = np.zeros((dimension,dimension),dtype=np.complex128)
        self.mean_sum = np.zeros(dimension,dtype=np.complex128)
        self.power_sum = 0.0
        self.sample_count = 0

    def add(self, batch):
        x = np.asarray(batch)
        if (x.ndim != 2 or x.shape[1] != self.mean_sum.size or not x.shape[0]
                or x.dtype != np.complex128 or not np.all(np.isfinite(x))):
            raise ValueError("finite complex128 observations[B,dimension] required")
        self.second_moment_sum += x.T @ x.conj()
        self.mean_sum += np.sum(x,axis=0)
        self.power_sum += float(np.sum(np.abs(x)**2))
        self.sample_count += x.shape[0]

    def finish(self):
        if self.sample_count == 0:
            raise ValueError("cannot finalize an empty accumulator")
        raw = self.second_moment_sum / self.sample_count
        pre_error = float(np.max(np.abs(raw-raw.conj().T)))
        if pre_error > SYMMETRY_TOLERANCE:
            raise ValueError(f"substantive Hermitian error before symmetrization: {pre_error}")
        covariance = (raw+raw.conj().T)/2
        correction = float(np.max(np.abs(covariance-raw)))
        mean = self.mean_sum/self.sample_count
        independent_power = self.power_sum/(self.sample_count*self.mean_sum.size)
        diagonal_power = float(np.diag(covariance).real.mean())
        if not np.isclose(independent_power,diagonal_power,rtol=1e-12,atol=1e-15):
            raise ValueError("independent power sum disagrees with second-moment diagonal")
        return covariance, mean, {"sample_count":self.sample_count,
            "pre_symmetry_max_error":pre_error,"max_symmetrization_correction":correction,
            "post_symmetry_max_error":float(np.max(np.abs(covariance-covariance.conj().T))),
            "power_sum":self.power_sum,"independent_mean_channel_power":independent_power,
            "mean_policy":MEAN_POLICY,"sample_mean_subtracted":False,"denominator":self.sample_count,
            "mean_vector_rms_magnitude":float(np.sqrt(np.mean(np.abs(mean)**2))),
            "max_absolute_mean_entry":float(np.max(np.abs(mean))),
            "mean_power_relative_to_channel_power":float(np.mean(np.abs(mean)**2)/independent_power)}


def estimate_training_covariance(rows, model, batch_size=BATCH_SIZE, generator=formal_trajectory, progress=None):
    """Consume train manifest rows only; validation is rejected before generation."""
    if isinstance(batch_size,bool) or not isinstance(batch_size,int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if (not rows or any(r["split"] != "train" or not 0 <= r["trajectory_id"] < 2000 for r in rows)
            or len({r["trajectory_id"] for r in rows}) != len(rows)):
        raise ValueError("only unique train trajectory IDs 0..1999 may enter covariance accumulation")
    accumulator = SecondMomentAccumulator()
    buffer = np.empty((min(batch_size,len(rows))*40,1024),dtype=np.complex128)
    timing = {"trajectory_generation_seconds":0.0,"frame_extraction_seconds":0.0,
              "batch_packing_seconds":0.0,"covariance_GEMM_accumulation_seconds":0.0}
    visited, batch_counts = [], []
    started = perf_counter()
    for start in range(0,len(rows),batch_size):
        group = rows[start:start+batch_size]
        for index,row in enumerate(group):
            stamp = perf_counter(); draw = generator(model,row)
            timing["trajectory_generation_seconds"] += perf_counter()-stamp
            if draw.trajectory_id != row["trajectory_id"] or draw.channel_seed != row["channel_seed"]:
                raise ValueError("generator returned a different trajectory ID/seed than the train manifest")
            stamp = perf_counter(); frames = slice_channel_frames(draw.h_long)
            if not np.shares_memory(frames,draw.h_long):
                raise ValueError("formal frame extraction must use the continuous trajectory view")
            timing["frame_extraction_seconds"] += perf_counter()-stamp
            stamp = perf_counter(); buffer[40*index:40*(index+1)] = pack_link_frames(frames)
            timing["batch_packing_seconds"] += perf_counter()-stamp
            visited.append(row["trajectory_id"])
        stamp = perf_counter(); accumulator.add(buffer[:len(group)*40])
        timing["covariance_GEMM_accumulation_seconds"] += perf_counter()-stamp
        batch_counts.append(len(group)*40)
        if progress:
            progress(len(visited))
    covariance, mean, summary = accumulator.finish()
    if summary["sample_count"] != len(rows)*40:
        raise ValueError("link-frame covariance observation count mismatch")
    summary.update({"num_independent_trajectory_units":len(rows),"num_link_frame_observations":len(rows)*40,
        "visited_train_trajectory_ids":visited,"validation_trajectories_accessed":0,
        "batch_size":batch_size,"batch_observation_counts":batch_counts,"flatten_order":FLATTEN_ORDER,
        "runtime":{**timing,"total_accumulation_seconds":perf_counter()-started},
        "timing_definition":"generation includes the generator's initial view; extraction measures the explicit pack-input view separately"})
    return covariance,mean,summary


def covariance_statistics(covariance):
    """One final eigvalsh diagnosis; never alter the covariance or eigenvalues."""
    if covariance.shape != (1024,1024) or covariance.dtype != np.complex128:
        raise ValueError("expected complex128 covariance[1024,1024]")
    diagonal = np.diag(covariance)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[0] < -PSD_TOLERANCE:
        raise ValueError(f"substantive negative training covariance eigenvalue: {eigenvalues[0]}")
    return {"shape":[1024,1024],"dtype":"complex128","all_finite":bool(np.all(np.isfinite(covariance))),
        "hermitian_max_error":float(np.max(np.abs(covariance-covariance.conj().T))),
        "diagonal_imaginary_max":float(np.max(np.abs(diagonal.imag))),
        "diag_mean":float(diagonal.real.mean()),"diag_std":float(diagonal.real.std()),
        "diag_min":float(diagonal.real.min()),"diag_max":float(diagonal.real.max()),
        "min_eigenvalue":float(eigenvalues[0]),"max_eigenvalue":float(eigenvalues[-1]),
        "numerical_negative_count":int(np.count_nonzero(eigenvalues<0)),"PSD_tolerance":PSD_TOLERANCE,
        "eigenvalue_clipping_used":False,"rank_truncation_used":False}


@dataclass(frozen=True)
class FrozenTrainCovariance:
    R_train: np.ndarray
    empirical_mean: np.ndarray
    manifest_sha256: str
    file_sha256: str
    raw_array_sha256: str


def save_frozen_covariance(path,covariance,mean,manifest_sha256):
    """Deterministic NPZ timestamps/order; no pickle, compression or value edits."""
    arrays = {"R_train":covariance,"empirical_mean":mean,"mean_policy":np.array(MEAN_POLICY),
        "num_independent_trajectories":np.array(2000),"num_link_frame_observations":np.array(80000),
        "flatten_order":np.array(FLATTEN_ORDER),"manifest_sha256":np.array(manifest_sha256),
        "R_train_raw_array_sha256":np.array(raw_array_sha256(covariance))}
    with zipfile.ZipFile(path,"w",compression=zipfile.ZIP_STORED) as archive:
        for name,value in arrays.items():
            output = io.BytesIO(); np.save(output,value,allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy",date_time=(1980,1,1,0,0,0))
            archive.writestr(info,output.getvalue())
    return load_frozen_covariance(path,manifest_sha256)


def load_frozen_covariance(path,expected_manifest_sha256):
    with np.load(path,allow_pickle=False) as data:
        r,mean = data["R_train"].copy(),data["empirical_mean"].copy()
        manifest_sha = str(data["manifest_sha256"].item())
        raw_sha = str(data["R_train_raw_array_sha256"].item())
        if (manifest_sha != expected_manifest_sha256 or str(data["mean_policy"].item()) != MEAN_POLICY
                or int(data["num_independent_trajectories"]) != 2000
                or int(data["num_link_frame_observations"]) != 80000
                or str(data["flatten_order"].item()) != FLATTEN_ORDER
                or r.shape != (1024,1024) or r.dtype != np.complex128
                or mean.shape != (1024,) or not np.all(np.isfinite(r))
                or not np.allclose(r,r.conj().T,rtol=0,atol=1e-12)
                or raw_array_sha256(r) != raw_sha):
            raise ValueError("frozen Train covariance metadata/hash mismatch")
    r.setflags(write=False); mean.setflags(write=False)
    return FrozenTrainCovariance(r,mean,manifest_sha,sha256_file(path),raw_sha)


def prepare_train_lmmse(frozen,pilot_mask,noise_variance):
    if not isinstance(frozen,FrozenTrainCovariance):
        raise ValueError("Train estimator requires a loaded frozen covariance artifact")
    # Existing function is a generic covariance/Cholesky core despite its name.
    return prepare_oracle_lmmse(frozen.R_train,pilot_flat_indices(pilot_mask),noise_variance,(64,16))
