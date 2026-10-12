"""Oracle shared correction output subspaces; no coefficient predictor."""
from dataclasses import dataclass
import numpy as np
from src.lowrank_filter import readonly, closure

T16A_COMMIT = "b22c488fe0f86874cf2bfa1efee5113a03e9a0d2"
RANKS = (1, 2, 4, 8, 12, 16, 24, 32)
VALIDATION_RANKS = (4, 8, 16, 24)
SOURCE_MEAN_RANKS = (8, 16)
EXPERIMENT_TYPE = "shared output-subspace gate for low-rank source-to-target LMMSE filter corrections across realistic CDL shifts"


@dataclass(frozen=True)
class SharedOutputBasis:
    """Equal-condition, energy-normalized left singular output basis."""
    conditions: tuple
    normalized: np.ndarray
    left: np.ndarray
    singular: np.ndarray
    correction_energies: tuple

    @classmethod
    def build(cls, matrices):
        if not matrices: raise ValueError("at least one condition required")
        conditions = tuple(matrices)
        values = [np.asarray(matrices[c], np.complex128) for c in conditions]
        shape = values[0].shape
        if len(shape) != 2 or any(a.shape != shape or not np.isfinite(a).all() for a in values):
            raise ValueError("finite equal-shaped correction matrices required")
        norms = [float(np.linalg.norm(a)) for a in values]
        if any(n == 0 for n in norms): raise ValueError("zero correction energy cannot be normalized")
        z = np.concatenate([a/n for a, n in zip(values, norms)], axis=1)/np.sqrt(len(values))
        u, s, _ = np.linalg.svd(z, full_matrices=False)
        s.flags.writeable = False
        return cls(conditions, readonly(z), readonly(u), s, tuple(n*n for n in norms))

    @classmethod
    def for_shift(cls, matrices, shift):
        # Held-out matrices are never accessed, normalized or concatenated.
        subset = {c: m for c, m in matrices.items() if c[0] == shift}
        return cls.build(subset)

    def output(self, rank):
        if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or not 1 <= rank <= self.left.shape[1]:
            raise ValueError("invalid shared rank")
        return self.left[:, :rank]

    def coefficient_map(self, delta, rank):
        return readonly(self.output(rank).conj().T @ delta)

    def estimate(self, pilot, source_weight, mean_p, mean_d, rank, coefficient_map):
        centered = np.asarray(pilot, np.complex128)-mean_p
        return mean_d+centered@source_weight.T+(centered@coefficient_map.T)@self.output(rank).T

    def energy(self, matrix, rank):
        u = self.output(rank); projected = u.conj().T@matrix
        total = float(np.vdot(matrix, matrix).real)
        capture = float(np.vdot(projected, projected).real/total)
        # Independent direct residual avoids cancellation near capture=1.
        residual = matrix-u@projected
        excess = float(np.vdot(residual, residual).real)
        return dict(capture=capture, residual_fraction=1-capture,
                    weighted_excess_MSE_per_frame=excess, correction_energy_per_frame=total)

    def spectrum(self):
        p = self.singular**2; total = float(p.sum()); p = p/total
        return np.cumsum(self.singular**2)/total, float(np.exp(-np.sum(p[p > 0]*np.log(p[p > 0]))))


def covariance_mse(weight, rpp, rdp, trace_rdd, sigma2):
    """Expected data-vector energy from complete joint covariance (sentinel)."""
    c = rpp+sigma2*np.eye(rpp.shape[0])
    return float(trace_rdd-2*np.vdot(rdp, weight).real+np.vdot(weight, weight@c).real)


def anchored_nmse(excess_per_frame, matched_error_total, truth_energy_total, frame_count):
    """Descriptive frozen-validation anchor, NEVER used for decisions."""
    if frame_count <= 0 or truth_energy_total <= 0 or excess_per_frame < 0:
        raise ValueError("invalid anchor units/energies")
    matched_per_frame = matched_error_total/frame_count
    truth_per_frame = truth_energy_total/frame_count
    linear = (matched_per_frame+excess_per_frame)/truth_per_frame
    return dict(anchor_source="frozen_matched_validation", anchor_matched_error_per_frame=matched_per_frame,
                anchor_truth_energy_per_frame=truth_per_frame, anchored_predicted_nmse_linear=float(linear),
                anchored_predicted_nmse_db=float(10*np.log10(linear)),
                units="summed data-domain coefficient error energy per frame", anchor_used_for_decision=False)


def validation_qthreshold(rows, threshold):
    if any(r["q"] not in VALIDATION_RANKS or r.get("metric_type") != "actual_validation" for r in rows):
        raise ValueError("q thresholds require actual declared validation ranks, never anchor")
    passing = [r["q"] for r in rows if r["shared_closure"] >= threshold]
    return min(passing) if passing else None


def decision_gate(rows, cross_rows):
    if any(r.get("metric_type") != "actual_validation" for r in [*rows, *cross_rows]):
        raise ValueError("decision gates cannot use anchored predictions")
    focus = [r for r in rows if r["snr_db"] in (10, 20)]
    cross = [r for r in cross_rows if r["snr_db"] in (10, 20)]
    if len(focus) != 16 or len(cross) != 16: raise ValueError("four primary conditions x four actual ranks required")
    def all_pass(values, q, threshold):
        return all(r["shared_closure"] >= threshold for r in values if r["q"] == q)
    strong = any(all_pass(focus, q, .8) for q in (4, 8))
    moderate = not strong and all_pass(focus, 16, .8)
    weak = not all_pass(focus, 24, .8)
    reusable = any(all_pass(cross, q, .7) for q in (4, 8, 16))
    return dict(shared_output_basis_strong=bool(strong), shared_output_basis_moderate=bool(moderate),
                shared_output_basis_weak=bool(weak), cross_shift_basis_reusable=bool(reusable))


def complexity_memory(rank):
    stored = 3968*rank; base = 3968*64; coefficient_map = 64*rank
    return dict(q=rank, correction_reconstruction_complex_MAC_per_frame=stored,
        reconstruction_overhead_ratio=stored/base, oracle_linear_coefficient_map_extra_MAC=coefficient_map,
        oracle_total_extra_MAC=stored+coefficient_map, predictor_complexity_estimated=False,
        basis_complex_coefficients=stored, basis_bytes_complex64=stored*8, basis_bytes_complex128=stored*16,
        quantization_assumed=False)
