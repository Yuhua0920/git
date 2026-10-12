"""Current-frame controls using frozen fitting and MR05 deployment maps.

The online entry accepts received pilots and operating SNR only. Clean H is
accepted solely by make_received_pilots, the separate simulation input stage.
No channel generator, truth/oracle, label, rank selection, regularizer, or file
IO is present. Callers authenticate the supplied operator/factor banks.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from src.pilot_cv import PilotCV

SNRS = (10, 20, 30)
BACKGROUND_METHODS = (
    "SOURCE", "POOLED_NATURAL", "DENSE_TW5", "LR4_TW5",
    "WHOLE_POOL_M2_R23", "WHOLE_POOL_M2_R24",
)
METHODS = BACKGROUND_METHODS + ("DENSE_TW1", "LR4_TW1")
ORIGIN_ONLY_CONDITIONS = ("ID", "Profile", "Mobility")
V_NAMESPACE = "SP_U17_MR03_CAPACITY_VALIDATION_V1"
SMOKE_NAMESPACE = "SP_U17_MR03_CAPACITY_SMOKE_V1"
RTOL = 1e-10
ATOL = 1e-10
SOLVE_TOL = 1e-11


def _real_coordinates(value):
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or raw.dtype.kind not in "biuf":
        raise ValueError("REAL_COORDINATES_REQUIRED")
    result = np.asarray(raw, dtype=np.float64)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError("TWO_FINITE_COORDINATES_REQUIRED")
    return result


def _pilot(value, shape):
    result = np.asarray(value, dtype=np.complex128)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError("FINITE_PILOT_SHAPE_REQUIRED")
    return result


def _diagnostics(diag, cv, window_frames, *, reused=False):
    complex_rows = window_frames * sum(len(hold) for _, hold in cv.folds)
    result = dict(diag)
    result.update(window_frames=window_frames,
                  complex_rows=complex_rows, real_rows=2 * complex_rows,
                  coordinates_reused=reused)
    if diag.get("gram") is not None:
        if np.iscomplexobj(diag["gram"]) or np.iscomplexobj(diag["rhs"]):
            raise ValueError("REAL_FIT_DIAGNOSTICS_REQUIRED")
        result["gram"] = np.asarray(diag["gram"], dtype=np.float64)
        result["rhs"] = np.asarray(diag["rhs"], dtype=np.float64)
        if result["gram"].shape != (2, 2) or result["rhs"].shape != (2,):
            raise ValueError("FIT_DIAGNOSTIC_SHAPE")
        residual = float(diag["normal_equation_relative_residual"])
        if (not np.isfinite(result["gram"]).all()
                or not np.isfinite(result["rhs"]).all()
                or not np.isfinite(residual) or residual > SOLVE_TOL):
            raise ValueError("FIT_DIAGNOSTIC_FINITE_RESIDUAL")
        result["normal_equation_relative_residual"] = residual
    return result


class FrameControls:
    """Frozen eight-arm deployment with a separately identified TW1 fit.

    bank is indexed bank[snr][MR05 method]. The deployer defaults to the
    authenticated MR05 controls.deployed function; a caller may supply exactly
    that function explicitly to avoid module-name ambiguity. All existing
    background outputs share TW5 coordinates; only the final two arms use TW1.
    """

    def __init__(self, op, cvs, bank, *, deployer=None):
        if deployer is None:
            from controls import deployed
            deployer = deployed
        self.op, self.cvs, self.bank, self.deployer = op, cvs, bank, deployer
        if set(cvs) != set(SNRS) or set(bank) != set(SNRS):
            raise ValueError("THREE_SNR_BANK_REQUIRED")
        for snr in SNRS:
            cv = cvs[snr]
            if not isinstance(cv, PilotCV) or len(cv.folds) != 4:
                raise ValueError("FROZEN_FOUR_FOLD_PILOTCV_REQUIRED")
            if not np.array_equal(cv.source_mean_p, op["source_mean_p"]):
                raise ValueError("SOURCE_CENTERING_DRIFT")
            if any(len(obs) != 48 or len(hold) != 16 for obs, hold in cv.folds):
                raise ValueError("FROZEN_FOLD_SIZE_REQUIRED")

    def predict_tw1(self, current_pilots, snr):
        """Return Dense/LR4 TW1 using one actual frozen PilotCV.fit call."""
        if snr not in SNRS:
            raise ValueError("DECLARED_SNR_REQUIRED")
        pilot = _pilot(current_pilots, (64,))
        cv = self.cvs[snr]
        # Deliberately no exception recovery: singular windows remain failures.
        coeff, diag = cv.fit((pilot,), modules=("P", "M"))
        coeff = _real_coordinates(coeff)
        diag = _diagnostics(diag, cv, 1)
        u = pilot - self.op["source_mean_p"]
        predictions = {
            "DENSE_TW1": self.deployer("DENSE_TW5", u, coeff,
                                         self.op, self.bank[snr], snr),
            "LR4_TW1": self.deployer("CORR_ORIGIN_R4", u, coeff,
                                       self.op, self.bank[snr], snr),
        }
        return dict(predictions=predictions, coordinates=coeff, diagnostics=diag)

    def predict_frame(self, pilots_window, snr, saved_tw5=None,
                      include_background=True):
        """Current-inclusive five-frame input, with optional exact TW5 reuse.

        D_eval should supply saved_tw5 and use include_background=False except
        for predeclared vector replays. V_new supplies no saved coordinates,
        independently executes the original five-frame fit, and obtains all
        six background outputs. Neither entry accepts truth or a condition.

        saved_tw5 can be the saved coefficient vector or a mapping containing
        coordinates plus gram/rhs/normal_equation_relative_residual. If only
        coefficients are supplied, missing historic diagnostics remain None;
        they are never invented or obtained by silently refitting.
        """
        if snr not in SNRS:
            raise ValueError("DECLARED_SNR_REQUIRED")
        window = _pilot(pilots_window, (5, 64))
        cv = self.cvs[snr]
        if saved_tw5 is None:
            if not include_background:
                raise ValueError("SAVED_TW5_REQUIRED_FOR_REUSE")
            c5, d5 = cv.fit(window, modules=("P", "M"))
            c5 = _real_coordinates(c5)
            d5 = _diagnostics(d5, cv, 5)
        else:
            if isinstance(saved_tw5, Mapping):
                c5 = _real_coordinates(saved_tw5["coordinates"])
                d5 = dict(saved_tw5)
                d5.pop("coordinates")
                d5.setdefault("gram", None)
                d5.setdefault("rhs", None)
                d5.setdefault("normal_equation_relative_residual", None)
            else:
                c5 = _real_coordinates(saved_tw5)
                d5 = dict(gram=None, rhs=None,
                          normal_equation_relative_residual=None)
            d5 = _diagnostics(d5, cv, 5, reused=True)
        tw1 = self.predict_tw1(window[-1], snr)
        predictions = {}
        if include_background:
            u = window[-1] - self.op["source_mean_p"]
            mapping = dict(SOURCE="SOURCE", POOLED_NATURAL="POOLED_NATURAL",
                           DENSE_TW5="DENSE_TW5", LR4_TW5="CORR_ORIGIN_R4",
                           WHOLE_POOL_M2_R23="WHOLE_POOL_M2_R23",
                           WHOLE_POOL_M2_R24="WHOLE_POOL_M2_R24")
            predictions.update({name: self.deployer(old, u, c5, self.op,
                                                    self.bank[snr], snr)
                                for name, old in mapping.items()})
        predictions.update(tw1["predictions"])
        return dict(predictions=predictions,
                    tw1_coordinates=tw1["coordinates"], tw5_coordinates=c5,
                    tw1_diagnostics=tw1["diagnostics"], tw5_diagnostics=d5,
                    tw1_gram=tw1["diagnostics"]["gram"],
                    tw1_rhs=tw1["diagnostics"]["rhs"],
                    tw1_normal_residual=tw1["diagnostics"]["normal_equation_relative_residual"],
                    tw5_gram=d5["gram"], tw5_rhs=d5["rhs"],
                    tw5_normal_residual=d5["normal_equation_relative_residual"],
                    background_outputs_computed=6 if include_background else 0,
                    tw5_new_fit=saved_tw5 is None,
                    tw1_new_fit=True)


def make_received_pilots(h, communication, snrs=SNRS):
    """OFFLINE input simulation, matching MR01 source-referenced pilot LS.

    This stage has clean H because it simulates received inputs. Its output
    pilots, not H or condition labels, enters FrameControls. No generator/RNG
    is invoked; frozen bits/symbols/unit-noise arrays must already be supplied.
    """
    from scripts.sp_u12b2a_common import variance
    from src.cdl_benchmark import frames, observation_geometry, array_hash
    from src.link import (assemble_pilot_data_frame, apply_frequency_mimo_channel,
                          estimate_ls_linear_channel)
    if tuple(snrs) != SNRS or len(communication) != 10:
        raise ValueError("THREE_SNR_TEN_FRAME_INPUT_REQUIRED")
    h = np.asarray(h)
    if h.shape != (2, 2, 64, 160) or h.dtype != np.complex128 or not np.isfinite(h).all():
        raise ValueError("CLEAN_H_INPUT_SHAPE_DTYPE_FINITE")
    geometry, hf = observation_geometry(), frames(h)
    pilots = np.empty((3, 10, 64), np.complex128)
    identities = []
    for frame_id, (_, symbols, unit_noise) in enumerate(communication):
        unit_noise = np.asarray(unit_noise)
        if unit_noise.shape != (2, 64, 16) or unit_noise.dtype != np.complex128 or not np.isfinite(unit_noise).all():
            raise ValueError("UNIT_NOISE_INPUT_SHAPE_DTYPE_FINITE")
        clean = apply_frequency_mimo_channel(
            hf[frame_id], assemble_pilot_data_frame(symbols, geometry.layout))
        for si, snr in enumerate(SNRS):
            received = clean + np.sqrt(variance(snr)) * unit_noise
            sparse, _ = estimate_ls_linear_channel(received, geometry.layout)
            pilots[si, frame_id] = sparse.ravel(order="C")[geometry.selected]
            identities.append(dict(snr_db=snr, frame_id=frame_id,
                                   H_frame_sha256=array_hash(hf[frame_id]),
                                   unit_noise_sha256=array_hash(unit_noise),
                                   pilot_sha256=array_hash(pilots[si, frame_id]),
                                   received_sha256=array_hash(received)))
    if not np.isfinite(pilots).all():
        raise ValueError("NONFINITE_RECEIVED_PILOTS")
    pilots.flags.writeable = False
    return pilots, identities
