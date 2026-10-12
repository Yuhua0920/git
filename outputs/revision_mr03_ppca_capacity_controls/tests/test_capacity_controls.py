"""New interface tests only; deterministic synthetic data, no corpus access."""
import inspect
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(os.environ.get("MR03_PROJECT_ROOT", Path.cwd())).resolve()
sys.path[:0] = [str(ROOT),
               str(ROOT / "outputs/revision_mr05_compression_controls/src_or_scripts")]
from src.cdl_benchmark import observation_geometry, frames
from src.pilot_cv import PilotCV, frozen_folds
from src.lowrank_modules import RankFactors
from src.link import PILOT_SYMBOLS
from scripts import sp_u12b2a_common as common
from scripts.run_sp_u12b2a import allocate, consumed, seed_index
from controls import deployed
from capacity_controls import (FrameControls, METHODS, BACKGROUND_METHODS,
                               SNRS, RTOL, ATOL, V_NAMESPACE, SMOKE_NAMESPACE,
                               ORIGIN_ONLY_CONDITIONS, make_received_pilots)


def fixture():
    rng = np.random.default_rng(1703001)
    make = lambda shape: (rng.normal(size=shape) + 1j*rng.normal(size=shape)) * .1
    mp, md = make((64,)), make((32,))
    op = dict(source_mean_p=mp, source_mean_d=md,
              pool_mean_p=make((64,)), pool_mean_d=make((32,)))
    folds, _ = frozen_folds(observation_geometry().selected)
    cvs, bank = {}, {}
    for s in SNRS:
        for key in ("Wsource", "Wpool", "DeltaWP", "DeltaWM"):
            op[f"{key}_{s}"] = make((32, 64))
        cvs[s] = PilotCV(folds, mp,
                        tuple(make((16, 48)) for _ in range(4)),
                        tuple(make((16, 48)) for _ in range(4)),
                        tuple(make((16, 48)) for _ in range(4)))
        bank[s] = {
            "CORR_ORIGIN_R4": tuple(RankFactors(make((32, 4)), make((4, 64))) for _ in range(2)),
            "WHOLE_POOL_M2_R23": tuple(RankFactors(make((32, 23)), make((23, 64))) for _ in range(3)),
            "WHOLE_POOL_M2_R24": tuple(RankFactors(make((32, 24)), make((24, 64))) for _ in range(3)),
        }
    window = make((5, 64))
    engine = FrameControls(op, cvs, bank, deployer=deployed)
    return engine, window


def direct_fit(cv, pilots):
    columns, residual = [], []
    for pilot in pilots:
        z = pilot - cv.source_mean_p
        for index, (obs, hold) in enumerate(cv.folds):
            x = z[obs]
            columns.append(np.column_stack((cv.profile_delta_k[index]@x,
                                            cv.mobility_delta_k[index]@x)))
            residual.append(z[hold]-cv.source_k[index]@x)
    X, r = np.concatenate(columns), np.concatenate(residual)
    G, b = np.vstack((X.real, X.imag)), np.r_[r.real, r.imag]
    return np.linalg.lstsq(G, b, rcond=None)[0], G, b


def test_tw1_matches_independent_real_stack_and_actual_row_count():
    engine, window = fixture()
    answer = engine.predict_tw1(window[-1], 10)
    coeff, G, target = direct_fit(engine.cvs[10], (window[-1],))
    assert G.shape == (128, 2)
    np.testing.assert_allclose(answer["coordinates"], coeff, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(answer["diagnostics"]["gram"], G.T@G, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(answer["diagnostics"]["rhs"], G.T@target, rtol=RTOL, atol=ATOL)
    assert answer["diagnostics"]["window_frames"] == 1
    assert answer["diagnostics"]["complex_rows"] == 64
    assert answer["diagnostics"]["real_rows"] == 128
    assert not answer["diagnostics"]["clipped"]
    assert not answer["diagnostics"]["regularized"]


def test_new_validation_background_has_one_tw5_one_tw1_fit(monkeypatch):
    engine, window = fixture()
    lengths = []
    original = PilotCV.fit
    def spy(self, pilot_frames, *, modules):
        lengths.append((len(pilot_frames), modules))
        return original(self, pilot_frames, modules=modules)
    monkeypatch.setattr(PilotCV, "fit", spy)
    answer = engine.predict_frame(window, 20)
    assert lengths == [(5, ("P", "M")), (1, ("P", "M"))]
    assert tuple(answer["predictions"]) == METHODS
    assert answer["tw5_new_fit"] and answer["tw1_new_fit"]
    assert answer["background_outputs_computed"] == 6
    assert answer["tw5_diagnostics"]["real_rows"] == 640
    coeff5, G, target = direct_fit(engine.cvs[20], window)
    np.testing.assert_allclose(answer["tw5_coordinates"], coeff5, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(answer["tw5_gram"], G.T@G, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(answer["tw5_rhs"], G.T@target, rtol=RTOL, atol=ATOL)


def test_development_reuses_saved_tw5_and_does_not_recompute_background(monkeypatch):
    engine, window = fixture()
    c5 = np.array([-0.7, 2.1])
    saved = c5.copy()
    lengths, deploy_names = [], []
    original = PilotCV.fit
    def spy(self, pilot_frames, *, modules):
        lengths.append(len(pilot_frames))
        return original(self, pilot_frames, modules=modules)
    def deploy_spy(method, *args):
        deploy_names.append(method)
        return deployed(method, *args)
    engine.deployer = deploy_spy
    monkeypatch.setattr(PilotCV, "fit", spy)
    answer = engine.predict_frame(window, 10, saved_tw5=c5,
                                  include_background=False)
    assert lengths == [1]
    assert deploy_names == ["DENSE_TW5", "CORR_ORIGIN_R4"]
    assert tuple(answer["predictions"]) == ("DENSE_TW1", "LR4_TW1")
    assert np.array_equal(c5, saved)
    assert np.array_equal(answer["tw5_coordinates"], saved)
    assert not answer["tw5_new_fit"]
    assert answer["tw5_gram"] is None and answer["tw5_rhs"] is None
    assert answer["tw5_normal_residual"] is None


def test_background_six_share_unrestricted_tw5_coordinates_and_means():
    engine, window = fixture()
    coeff = np.array([-0.7, 2.1])
    answer = engine.predict_frame(window, 30, saved_tw5=coeff)
    u, op, bank = window[-1]-engine.op["source_mean_p"], engine.op, engine.bank[30]
    mapping = dict(SOURCE="SOURCE", POOLED_NATURAL="POOLED_NATURAL",
                   DENSE_TW5="DENSE_TW5", LR4_TW5="CORR_ORIGIN_R4",
                   WHOLE_POOL_M2_R23="WHOLE_POOL_M2_R23",
                   WHOLE_POOL_M2_R24="WHOLE_POOL_M2_R24")
    for name in BACKGROUND_METHODS:
        np.testing.assert_array_equal(answer["predictions"][name],
                                      deployed(mapping[name], u, coeff, op, bank, 30))
    S, P, M = bank["WHOLE_POOL_M2_R23"]
    expected = op["source_mean_d"] + (1-coeff.sum())*S.apply(u) + coeff[0]*P.apply(u) + coeff[1]*M.apply(u)
    np.testing.assert_allclose(answer["predictions"]["WHOLE_POOL_M2_R23"], expected, rtol=RTOL, atol=ATOL)


def test_saved_full_diagnostics_are_returned_without_refit(monkeypatch):
    engine, window = fixture()
    coeff, diag = engine.cvs[10].fit(window, modules=("P", "M"))
    saved = dict(coordinates=coeff, **diag)
    answer = engine.predict_frame(window, 10, saved_tw5=saved,
                                  include_background=False)
    np.testing.assert_array_equal(answer["tw5_gram"], np.asarray(diag["gram"]))
    np.testing.assert_array_equal(answer["tw5_rhs"], np.asarray(diag["rhs"]))
    assert answer["tw5_normal_residual"] == diag["normal_equation_relative_residual"]
    assert answer["tw5_diagnostics"]["coordinates_reused"]


def test_tw1_ignores_earlier_frames_but_five_frame_fit_can_change():
    engine, window = fixture()
    a = engine.predict_frame(window, 10)
    changed = window.copy()
    changed[:4] = changed[:4]*3 + 2j
    b = engine.predict_frame(changed, 10)
    np.testing.assert_array_equal(a["tw1_coordinates"], b["tw1_coordinates"])
    for name in ("DENSE_TW1", "LR4_TW1"):
        np.testing.assert_array_equal(a["predictions"][name], b["predictions"][name])
    assert np.linalg.norm(a["tw5_coordinates"]-b["tw5_coordinates"]) > 1e-3


def test_unidentifiable_tw1_raises_without_fallback():
    engine, window = fixture()
    cv = engine.cvs[10]
    zero = tuple(np.zeros((16, 48), np.complex128) for _ in range(4))
    engine.cvs[10] = PilotCV(cv.folds, cv.source_mean_p, cv.source_k, zero, zero)
    with pytest.raises(np.linalg.LinAlgError):
        engine.predict_frame(window, 10, saved_tw5=np.ones(2),
                             include_background=False)


@pytest.mark.parametrize("field", ["gram", "rhs"])
def test_complex_saved_normal_quantities_are_not_silently_cast_to_real(field):
    engine, window = fixture()
    saved = dict(coordinates=np.ones(2), gram=np.eye(2), rhs=np.ones(2),
                 normal_equation_relative_residual=0.0)
    saved[field] = saved[field].astype(np.complex128)
    with pytest.raises(ValueError, match="REAL_FIT_DIAGNOSTICS"):
        engine.predict_frame(window, 10, saved_tw5=saved,
                             include_background=False)


@pytest.mark.parametrize("shape", [(10, 64), (1, 64), (5, 63)])
def test_wrong_window_shape_rejected(shape):
    engine, _ = fixture()
    with pytest.raises(ValueError, match="FINITE_PILOT_SHAPE"):
        engine.predict_frame(np.ones(shape), 10)


def test_no_truth_or_condition_online_and_invalid_coordinates_rejected():
    engine, window = fixture()
    assert set(inspect.signature(FrameControls.predict_frame).parameters) == {
        "self", "pilots_window", "snr", "saved_tw5", "include_background"}
    with pytest.raises(TypeError):
        engine.predict_frame(window, 10, truth=np.ones(32))
    with pytest.raises(TypeError):
        engine.predict_frame(window, 10, condition="ID")
    with pytest.raises(ValueError, match="REAL_COORDINATES"):
        engine.predict_frame(window, 10, saved_tw5=np.ones(2, np.complex128))
    with pytest.raises(ValueError, match="SAVED_TW5_REQUIRED"):
        engine.predict_frame(window, 10, include_background=False)
    with pytest.raises(ValueError, match="DECLARED_SNR"):
        engine.predict_frame(window, 15)


def test_offline_received_pilots_owner_order_and_noise_added_once():
    indices = np.arange(2*2*64*160).reshape(2, 2, 64, 160)
    h = (indices*.001 + 1j*(.2+indices*.0003)).astype(np.complex128)
    symbols = np.ones((2, 64, 16), np.complex128)
    bits = np.zeros((2, 64, 16, 2), np.uint8)
    q = np.arange(2*64*16).reshape(2, 64, 16)
    noise = (.05+q*.0002 + 1j*(.02-q*.0001)).astype(np.complex128)
    comm = [(bits, symbols, noise+j*.0001) for j in range(10)]
    before_h, before_noise = h.copy(), [x[2].copy() for x in comm]
    pilots, identities = make_received_pilots(h, comm)
    geometry, hf = observation_geometry(), frames(h)
    rx, tx, k, t = np.unravel_index(geometry.selected, (2, 2, 64, 16), order="C")
    for si, snr in enumerate(SNRS):
        for j in range(10):
            expected = hf[j].ravel()[geometry.selected] + np.sqrt(common.variance(snr))*comm[j][2][rx, k, t]/PILOT_SYMBOLS[tx]
            np.testing.assert_allclose(pilots[si, j], expected, rtol=RTOL, atol=ATOL)
    assert pilots.shape == (3, 10, 64) and not pilots.flags.writeable
    assert len(identities) == 30
    assert len({(r["snr_db"], r["frame_id"]) for r in identities}) == 30
    for j in range(10):
        assert len({r["H_frame_sha256"] for r in identities if r["frame_id"] == j}) == 1
        assert len({r["unit_noise_sha256"] for r in identities if r["frame_id"] == j}) == 1
        np.testing.assert_array_equal(comm[j][2], before_noise[j])
    np.testing.assert_array_equal(h, before_h)


def test_joint_layout_and_origin_only_mapping_are_explicit():
    from src.baselines.mfa_reference_adapter import ProjectPilotInterface
    interface = ProjectPilotInterface()
    assert len(interface.joint_full) == 4032 and len(interface.excluded_full) == 64
    assert not np.array_equal(interface.pilot_positions, np.arange(64))
    np.testing.assert_array_equal(interface.joint_full[interface.pilot_positions], interface.selected_full)
    np.testing.assert_array_equal(interface.joint_full[interface.data_positions], interface.data_full)
    _, tx, _, _ = np.unravel_index(interface.selected_full, (2, 2, 64, 16), order="C")
    np.testing.assert_array_equal(interface.owner_symbols, PILOT_SYMBOLS[tx])
    assert ORIGIN_ONLY_CONDITIONS == ("ID", "Profile", "Mobility")
    assert [common.CONDITIONS[x] for x in ORIGIN_ONLY_CONDITIONS] == [
        "CDL-C-30-100", "CDL-B-30-100", "CDL-C-120-100"]


def test_synthetic_seed_allocation_role_scope_namespace_and_adjacent_rejection():
    used = set()
    a = allocate(SMOKE_NAMESPACE, 2, 0, {1703001}, used, True)
    first = {r["seed"] for r in a["rows"]}
    forbidden = first | {v-1 for v in first} | {v+1 for v in first}
    b = allocate(V_NAMESPACE, 2, 0, forbidden, used, True)
    second = {r["seed"] for r in b["rows"]}
    assert len(first) == len(second) == 5*2*33+1
    assert not first.intersection(second) and not second.intersection(forbidden)
    assert len({(r["condition"], r["group"], r["trajectory_id"], r["frame_id"], r["role"]) for r in b["rows"]}) == len(b["rows"])
    index = seed_index(b)
    for condition in common.CONDITIONS:
        channel = []
        for j in range(10):
            roles = consumed(index, condition, "EVAL", 0, j)
            assert set(roles) == set(common.ROLES)
            channel.append(tuple(roles[r] for r in common.ROLES[:3]))
        assert len(set(channel)) == 1
