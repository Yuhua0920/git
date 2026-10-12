"""Read saved MR05 development energies and reuse authenticated MR01 draws.

No prediction, coordinate fitting, channel generation or random sampling.
The saved bootstrap score arrays are newly materialized numerical summaries
of the old draws; MR01 did not persist bootstrap score arrays.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import csv
import hashlib
import json
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np

CONDITIONS = ("ID", "Profile", "Mobility", "Composed", "Boundary")
SNRS = (10, 20, 30)
METHODS = ("SOURCE", "POOLED_NATURAL", "DENSE_TW5", "CORR_ORIGIN_R4",
           "CORR_I_R4", "CORR_ORIGIN_M2_R4", "CORR_POOL_M2_R4",
           "WHOLE_POOL_M2_R23", "WHOLE_POOL_M2_R24")
OLD_METHODS = ("Source", "Pooled", "Dense", "LR4")
OLD_INDEX = (0, 1, 4, 5)
CONTROLS = METHODS[3:]
NEW = METHODS[4:]
RTOL = 1e-10
ATOL = 1e-10
INTERVAL = "Disclosed development; unadjusted exploratory 95% paired trajectory percentile"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def assert_close(a, b, label, scale=None):
    a, b = np.asarray(a), np.asarray(b)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("NONFINITE:" + label)
    error = np.abs(a - b)
    den = np.maximum(np.abs(a), np.abs(b))
    if scale is not None:
        den = np.maximum(den, np.asarray(scale))
    if np.any(error > ATOL + RTOL * den):
        raise ValueError("NUMERICAL_MISMATCH:" + label)
    return dict(max_abs=float(np.max(error)), scaled=float(np.max(error / (1 + den))))


def real_finite(a, shape, label, *, nonnegative=False, positive=False):
    a = np.asarray(a)
    if a.shape != shape or a.dtype != np.float64 or not np.isfinite(a).all():
        raise ValueError("FLOAT64_AXES:" + label)
    if nonnegative and np.any(a < 0):
        raise ValueError("NEGATIVE:" + label)
    if positive and np.any(a <= 0):
        raise ValueError("NONPOSITIVE:" + label)
    return a


def validate_draws(draw, n=100, resamples=2000):
    draw = np.asarray(draw)
    if draw.shape != (resamples, n) or draw.dtype != np.int64:
        raise ValueError("BOOTSTRAP_INDEX_AXES")
    if np.any(draw < 0) or np.any(draw >= n):
        raise ValueError("BOOTSTRAP_INDEX_RANGE")
    return draw


def paired_bootstrap(error, truth, draw):
    """Trajectory sums; all SNRs and methods use each same stored index row."""
    error, truth = np.asarray(error), np.asarray(truth)
    if error.ndim != 3 or truth.shape != (error.shape[0],):
        raise ValueError("POOLED_AXES")
    if error.dtype != np.float64 or truth.dtype != np.float64:
        raise ValueError("POOLED_FLOAT64")
    if not np.isfinite(error).all() or not np.isfinite(truth).all():
        raise ValueError("POOLED_NONFINITE")
    if np.any(error < 0) or np.any(truth <= 0):
        raise ValueError("POOLED_ENERGY")
    draw = validate_draws(draw, error.shape[0], np.asarray(draw).shape[0])
    point = error.sum(axis=0) / truth.sum()
    boot = error[draw].sum(axis=1) / truth[draw].sum(axis=1)[:, None, None]
    if np.any(point <= 0) or np.any(boot <= 0):
        raise ValueError("NONPOSITIVE_POOLED_NMSE")
    return point, 10 * np.log10(point), boot, 10 * np.log10(boot)


def contrast_definitions():
    rows = []
    rows += [("original_minus_control", METHODS[3], m) for m in NEW]
    rows += [("pooled_minus_control", METHODS[1], m) for m in CONTROLS]
    rows += [("control_minus_dense", m, METHODS[2]) for m in CONTROLS]
    rows += [("whole_minus_corrpool", m, METHODS[6]) for m in METHODS[7:]]
    if len(rows) != 19 or len(set(rows)) != 19:
        raise ValueError("CONTRAST_DESIGN")
    return tuple(rows)


def normalized_decomposition(error, truth, A, B, dense_index=2):
    """Return signed linear quantities, normalized by pooled truth energy."""
    error, truth, A, B = map(np.asarray, (error, truth, A, B))
    if any(x.dtype != np.float64 for x in (error, truth, A, B)):
        raise ValueError("DECOMPOSITION_FLOAT64")
    if error.ndim != 3 or error.shape != A.shape or error.shape != B.shape:
        raise ValueError("DECOMPOSITION_AXES")
    if truth.shape != (error.shape[0],) or np.any(truth <= 0):
        raise ValueError("DECOMPOSITION_TRUTH")
    if any(not np.isfinite(x).all() for x in (error, truth, A, B)) or np.any(A < 0):
        raise ValueError("DECOMPOSITION_NONFINITE_OR_NEGATIVE_A")
    delta = error - error[:, :, dense_index:dense_index + 1]
    scale = A + abs(B) + error + error[:, :, dense_index:dense_index + 1]
    check = assert_close(delta, A + B, "TRAJECTORY_SIGNED_IDENTITY", scale)
    Q = truth.sum()
    av, bv = A.sum(0) / Q, B.sum(0) / Q
    summed, diff = (A + B).sum(0) / Q, delta.sum(0) / Q
    aggregate = assert_close(summed, diff, "POOLED_SIGNED_IDENTITY",
                             av + abs(bv) + error.sum(0) / Q + error[:, :, dense_index].sum(0)[:, None] / Q)
    return av, bv, summed, diff, check, aggregate


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_npz(path):
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:
            raise ValueError("NPZ_CRC:" + str(path))
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def validate_block(block, old, old_summary, ci, i):
    E = real_finite(block["frame_error"], (3, 9, 6), "FRAME_ERROR", nonnegative=True)
    Q = real_finite(block["frame_truth"], (6,), "FRAME_TRUTH", positive=True)
    A = real_finite(block["A"], (3, 9, 6), "FRAME_A", nonnegative=True)
    B = real_finite(block["B"], (3, 9, 6), "FRAME_B")
    c = real_finite(block["coordinates"], (3, 6, 2), "ORIGINAL_COORDINATES")
    if not np.array_equal(block["snr_db"], np.array(SNRS)):
        raise ValueError("SNR_AXIS")
    if not np.array_equal(block["frame_ids"], np.arange(4, 10)):
        raise ValueError("SCORING_FRAME_AXIS")
    if tuple(block["method_names"]) != METHODS:
        raise ValueError("METHOD_AXIS")
    if not np.array_equal(c, old["coordinates"]) or not np.array_equal(c, old_summary["online"][ci, i]):
        raise ValueError("ORIGINAL_COORDINATE_DRIFT")
    if block["coordinate_sha256"].shape != () or str(block["coordinate_sha256"]) != array_hash(old["coordinates"]):
        raise ValueError("COORDINATE_HASH")
    if block["input_pilot_sha256"].shape != () or str(block["input_pilot_sha256"]) != array_hash(old["pilots"]):
        raise ValueError("PILOT_HASH")
    if not np.array_equal(Q, old["frame_truth"]) or not np.array_equal(Q, old_summary["truth"][ci, i]):
        raise ValueError("ORIGINAL_TRUTH_DRIFT")
    for mi, oi in enumerate(OLD_INDEX):
        if not np.array_equal(E[:, mi], old["frame_error"][:, oi]):
            raise ValueError("SAVED_ORIGINAL_ENERGY_DRIFT")
        if not np.array_equal(E[:, mi], old_summary["error"][ci, i][..., oi]):
            raise ValueError("MR02_BACKGROUND_ENERGY_DRIFT")
    delta = E - E[:, 2:3]
    scale = A + abs(B) + E + E[:, 2:3]
    check = assert_close(delta, A + B, "FRAME_SIGNED_IDENTITY", scale)
    origin_scale = A[:, 3] + abs(B[:, 3]) + E[:, 3] + E[:, 2]
    assert_close(A[:, 3], old_summary["compression_A"][ci, i], "ORIGINAL_A_REPLAY", origin_scale)
    assert_close(B[:, 3], old_summary["compression_B"][ci, i], "ORIGINAL_B_REPLAY", origin_scale)
    if i == 0:
        vector = block.get("vector_example")
        hashes = block.get("vector_full_sha256")
        if vector is None or vector.shape != (3, 9, 16) or vector.dtype != np.complex128 or not np.isfinite(vector).all():
            raise ValueError("VECTOR_EXAMPLE_AXES")
        if hashes is None or hashes.shape != (3, 9) or any(len(str(h)) != 64 or any(x not in "0123456789abcdef" for x in str(h)) for h in hashes.ravel()):
            raise ValueError("VECTOR_HASH_DECLARATION")
        if int(block["vector_example_frame"]) != 4 or not np.array_equal(block["vector_example_data_indices"], np.arange(16)):
            raise ValueError("VECTOR_EXAMPLE_LOCATION")
    return E, Q, A, B, c, check


def collect(root, out, mr1, mr2, runner):
    receipt_path = out / "reports/development_receipt.json"
    report = read(receipt_path)
    if report.get("PASS") is not True or report.get("physical_trajectories") != 500 or report.get("scored_positions_per_method") != 9000:
        raise ValueError("DEVELOPMENT_RECEIPT_NOT_COMPLETE")
    if report.get("all_raw_saved_before_aggregate") is not True:
        raise ValueError("RAW_PERSISTENCE_ORDER")
    expected_counts = {"reused_energy_positions": 36000, "new_control_positions": 45000,
                       "check_replay_positions": 36000}
    if any(report.get(k) != v for k, v in expected_counts.items()):
        raise ValueError("REUSE_NEW_CHECK_COUNTS")
    expected = {(c, i) for c in CONDITIONS for i in range(100)}
    receipts = report.get("receipts", [])
    by = {}
    for r in receipts:
        if type(r.get("trajectory_id")) is not int:
            raise ValueError("RECEIPT_ID_TYPE")
        key = (r.get("condition"), r["trajectory_id"])
        if key in by:
            raise ValueError("DUPLICATE_TRAJECTORY_RECEIPT")
        by[key] = r
    if len(receipts) != 500 or set(by) != expected:
        raise ValueError("TRAJECTORY_GRID")
    if len(list((out / "data/development").glob("*.npz"))) != 500 or len(list((out / "data/development").glob("*.json"))) != 500:
        raise ValueError("RAW_FILE_GRID")
    preserved = read(out / "reports/preservation_before.json")["hashes"]
    summary_path = mr2 / "data/eval_summary.npz"
    if sha(summary_path) != preserved[summary_path.relative_to(root).as_posix()]:
        raise ValueError("MR02_SUMMARY_HASH")
    summary = load_npz(summary_path)
    if summary["error"].shape != (5, 100, 3, 6, 13) or summary["online"].shape != (5, 100, 3, 6, 2):
        raise ValueError("MR02_SUMMARY_AXES")
    frame_E = np.empty((5, 100, 3, 9, 6), np.float64)
    frame_Q = np.empty((5, 100, 6), np.float64)
    frame_A = np.empty_like(frame_E)
    frame_B = np.empty_like(frame_E)
    coords = np.empty((5, 100, 3, 6, 2), np.float64)
    rows, max_abs, scaled = [], 0.0, 0.0
    for ci, condition in enumerate(CONDITIONS):
        for i in range(100):
            r = by[condition, i]
            rel = f"data/development/{condition}_{i:04d}.npz"
            if r.get("path") != rel:
                raise ValueError("RAW_PATH_IDENTITY")
            path = out / rel
            jp = path.with_suffix(".json")
            if path.stat().st_size != r.get("bytes") or r.get("bytes", 0) <= 0 or sha(path) != r.get("sha256") or sha(jp) != r.get("json_sha256"):
                raise ValueError("RAW_RECEIPT_HASH")
            meta = read(jp)
            identity = f"MR05:development:{condition}:{i}"
            raw_receipt = {k: r[k] for k in ("path", "sha256", "bytes")}
            if (meta.get("identity") != identity or meta.get("condition") != condition or
                    meta.get("trajectory_id") != i or meta.get("complete") is not True or
                    meta.get("raw") != raw_receipt or meta.get("method_axis") != list(METHODS) or
                    meta.get("original_coordinates_reused") is not True):
                raise ValueError("RAW_JSON_IDENTITY")
            old, binding = runner.source_input(condition, i)
            if meta.get("input") != binding:
                raise ValueError("ORIGINAL_INPUT_BINDING")
            block = load_npz(path)
            e, q, a, b, c, check = validate_block(block, old, summary, ci, i)
            frame_E[ci, i], frame_Q[ci, i] = e, q
            frame_A[ci, i], frame_B[ci, i], coords[ci, i] = a, b, c
            max_abs = max(max_abs, check["max_abs"])
            scaled = max(scaled, check["scaled"])
            rows.append(dict(condition=condition, trajectory_id=i, identity=identity,
                             original_identity=binding["original_identity"], path=rel,
                             sha256=r["sha256"], json_sha256=r["json_sha256"],
                             original_npz_sha256=binding["original_npz_sha256"],
                             original_json_sha256=binding["original_json_sha256"],
                             pilot_sha256=str(block["input_pilot_sha256"]),
                             coordinate_sha256=str(block["coordinate_sha256"])))
    return dict(frame_error=frame_E, frame_truth=frame_Q, frame_A=frame_A,
                frame_B=frame_B, coordinates=coords), rows, dict(
                    PASS=True, receipt_sha256=sha(receipt_path), physical_trajectories=500,
                    condition_SNR_cells=15, scored_positions_per_method=9000,
                    identity_rows=500, npz_CRC_checked=500, json_and_npz_SHA256_checked=500,
                    original_input_bindings_checked=500, original_coordinates_bitwise=True,
                    original_background_energies_bitwise=True,
                    frame_signed_identity_max_abs=max_abs, frame_signed_identity_scaled=scaled,
                    **expected_counts)


def unique_table(path, identity_keys):
    answer = {}
    for r in csv_rows(path):
        key = tuple(int(r[k]) if k == "snr_db" else r[k] for k in identity_keys)
        if key in answer:
            raise ValueError("OLD_TABLE_DUPLICATE:" + str(path))
        answer[key] = r
    return answer


def oriented_old_contrast(original, reverse=False):
    estimate, lower, upper = (float(original[k]) for k in ("estimate_db", "CI_lower", "CI_upper"))
    if not np.isfinite([estimate, lower, upper]).all() or lower > upper:
        raise ValueError("OLD_CONTRAST_INTERVAL")
    return (-estimate, -upper, -lower) if reverse else (estimate, lower, upper)


def calculate(frame, draws, mr1):
    E, Q = frame["frame_error"].sum(-1), frame["frame_truth"].sum(-1)
    A, B = frame["frame_A"].sum(-1), frame["frame_B"].sum(-1)
    old_abs = unique_table(mr1 / "tables/nmse_all_methods.csv", ("condition", "snr_db", "method"))
    old_pair = unique_table(mr1 / "tables/paired_contrasts.csv", ("condition", "snr_db", "first_method", "second_method"))
    definitions = contrast_definitions()
    nmse, contrasts, decomposed, trajectory = [], [], [], []
    bootstrap_linear = np.empty((5, 2000, 3, 9), np.float64)
    bootstrap_db = np.empty_like(bootstrap_linear)
    bootstrap_difference = np.empty((5, 2000, 3, 19), np.float64)
    old_checks, max_old_error, max_identity_error = [], 0.0, 0.0
    for ci, condition in enumerate(CONDITIONS):
        draw = validate_draws(draws[condition])
        point, db, boot, bdb = paired_bootstrap(E[ci], Q[ci], draw)
        bootstrap_linear[ci], bootstrap_db[ci] = boot, bdb
        limits = np.quantile(bdb, [.025, .975], axis=0, method="linear")
        av, bv, ab, diff, check, agg = normalized_decomposition(E[ci], Q[ci], A[ci], B[ci])
        max_identity_error = max(max_identity_error, check["scaled"], agg["scaled"])
        for si, snr in enumerate(SNRS):
            for mi, method in enumerate(METHODS):
                lo, hi = float(limits[0, si, mi]), float(limits[1, si, mi])
                source = "new control using original stored draws"
                if mi < 4:
                    original = old_abs[condition, snr, OLD_METHODS[mi]]
                    values = (E[ci, :, si, mi].sum(), Q[ci].sum(), point[si, mi], db[si, mi], lo, hi)
                    keys = ("error_energy", "truth_energy", "nmse_linear", "nmse_db", "CI_lower", "CI_upper")
                    verified = assert_close(values, [float(original[k]) for k in keys], "OLD_ABSOLUTE_TABLE")
                    max_old_error = max(max_old_error, verified["max_abs"])
                    lo, hi = float(original["CI_lower"]), float(original["CI_upper"])
                    source = "MR01 authenticated energies; original published development CI reused after numeric recheck"
                nmse.append(dict(condition=condition, snr_db=snr, method=method,
                    error_energy=float(E[ci, :, si, mi].sum()), truth_energy=float(Q[ci].sum()),
                    nmse_linear=float(point[si, mi]), nmse_db=float(db[si, mi]),
                    CI_lower=lo, CI_upper=hi, trajectory_count=100, scored_frames=600,
                    bootstrap=2000, interval=INTERVAL, source=source))
                summed = float(ab[si, mi]); difference = float(diff[si, mi])
                den = float(av[si, mi] + abs(bv[si, mi]) + point[si, mi] + point[si, 2])
                decomposed.append(dict(condition=condition, snr_db=snr, method=method,
                    A_over_truth=float(av[si, mi]), B_over_truth=float(bv[si, mi]),
                    sum_over_truth=summed, delta_nmse_linear=difference,
                    identity_abs_error=abs(summed - difference),
                    identity_scaled_error=abs(summed - difference) / (1 + den),
                    reference="DENSE_TW5", decomposition="signed linear; no componentwise dB",
                    trajectory_count=100, scored_frames=600))
            for ki, (name, first, second) in enumerate(definitions):
                fi, se = METHODS.index(first), METHODS.index(second)
                bd = bdb[:, si, fi] - bdb[:, si, se]
                bootstrap_difference[ci, :, si, ki] = bd
                lo, hi = (float(v) for v in np.quantile(bd, [.025, .975], method="linear"))
                estimate = float(db[si, fi] - db[si, se])
                source = "new comparison; original saved draws, no new random sampling"
                old_key = None
                reverse = False
                if name == "pooled_minus_control" and second == "CORR_ORIGIN_R4":
                    old_key = (condition, snr, "Pooled", "LR4")
                elif name == "control_minus_dense" and first == "CORR_ORIGIN_R4":
                    old_key = (condition, snr, "Dense", "LR4")
                    reverse = True
                if old_key is not None:
                    original = old_pair[old_key]
                    if int(original["bootstrap_resamples"]) != 2000 or int(original["trajectory_count"]) != 100:
                        raise ValueError("OLD_CONTRAST_DESIGN")
                    old_estimate, old_lo, old_hi = oriented_old_contrast(original, reverse)
                    checked = assert_close([estimate, lo, hi],
                        [old_estimate, old_lo, old_hi], "OLD_PAIRED_TABLE")
                    max_old_error = max(max_old_error, checked["max_abs"])
                    lo, hi = old_lo, old_hi
                    source = "original MR01 CI reused after same-draw independent numeric recheck" + ("; opposite original Dense-LR4 direction reversed" if reverse else "")
                    old_checks.append(dict(condition=condition, snr_db=snr, first=first,
                                           second=second, **checked, original_row=original,
                                           original_direction_reversed=reverse,
                                           reoriented_estimate_db=old_estimate,
                                           reoriented_CI_lower=old_lo, reoriented_CI_upper=old_hi))
                contrasts.append(dict(condition=condition, snr_db=snr, contrast=name,
                    first=first, second=second, estimate_db=estimate, CI_lower=lo, CI_upper=hi,
                    trajectory_count=100, bootstrap=2000, interval=INTERVAL,
                    sign="first minus second pooled NMSE(dB); positive favors second",
                    first_method_worse_trajectory_fraction=float(np.mean(E[ci, :, si, fi] > E[ci, :, si, se])),
                    interval_direction="FIRST_HIGHER" if lo > 0 else "FIRST_LOWER" if hi < 0 else "CROSSES_OR_TOUCHES_ZERO",
                    source=source))
        for i in range(100):
            for si, snr in enumerate(SNRS):
                for mi, method in enumerate(METHODS):
                    trajectory.append(dict(condition=condition, trajectory_id=i, snr_db=snr,
                        method=method, error_energy=float(E[ci, i, si, mi]), truth_energy=float(Q[ci, i]),
                        A=float(A[ci, i, si, mi]), B=float(B[ci, i, si, mi]),
                        delta_error_energy=float(E[ci, i, si, mi] - E[ci, i, si, 2]),
                        scored_frames=6, original_identity=f"MR01:development:{condition}:{i}"))
    if len(nmse) != 135 or len(contrasts) != 285 or len(old_checks) != 30:
        raise ValueError("ANALYSIS_ROW_COUNT")
    arrays = dict(Error=E, truth=Q, A=A, B=B, **frame,
                  condition_names=np.array(CONDITIONS), snr_db=np.array(SNRS), method_names=np.array(METHODS))
    boots = dict(nmse_linear=bootstrap_linear, nmse_db=bootstrap_db,
                 contrast_db=bootstrap_difference, condition_names=np.array(CONDITIONS),
                 snr_db=np.array(SNRS), method_names=np.array(METHODS),
                 contrast_names=np.array([r[0] for r in definitions]),
                 contrast_first=np.array([r[1] for r in definitions]),
                 contrast_second=np.array([r[2] for r in definitions]))
    check = dict(PASS=True, original_absolute_rows_rechecked=60,
                 original_paired_rows_rechecked_and_CI_reused=30,
                 original_table_max_abs_error=max_old_error,
                 signed_identity_scaled_error=max_identity_error,
                 reused_CI_details=old_checks,
                 bootstrap_scores_newly_materialized=True, new_random_draws=0,
                 old_bootstrap_score_arrays_existed=False,
                 old_background_bootstrap_role="numeric recheck and necessary paired-control reference, not a new experiment",
                 independent_unit="physical trajectory", frames_methods_and_SNR="paired repeated measurements",
                 multiplicity="unadjusted exploratory 95%; no confirmatory family claim",
                 practical_reference_db=.5, practical_reference_is_gate=False)
    return nmse, contrasts, decomposed, trajectory, arrays, boots, check


def exclusive_npz(path, arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def exclusive_json(path, obj):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)


def exclusive_csv(path, rows):
    with Path(path).open("x", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(root=None):
    started = time.monotonic()
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[3]
    out = root / "outputs/revision_mr05_compression_controls"
    mr1 = root / "outputs/revision_mr01_pooled_lmmse"
    mr2 = root / "outputs/revision_mr02_coordinate_mechanism"
    sys.path[:0] = [str(root), str(Path(__file__).parent)]
    import run_mr05 as runner
    runner.ROOT, runner.OUT, runner.MR1, runner.MR2 = root, out, mr1, mr2
    runner.prior.source_metadata.cache_clear()
    runner.verify_sources()
    config = read(out / "config.json")
    if (tuple(config["methods"]) != METHODS or tuple(config["conditions"]) != CONDITIONS or
            tuple(config["snrs"]) != SNRS or config["scored_frames"] != list(range(4, 10)) or
            config["physical_trajectories"] != 500 or config["bootstrap"]["resamples"] != 2000 or
            config["bootstrap"]["new_draws"] != 0 or config["bootstrap"]["adjusted"] is not False):
        raise ValueError("ANALYSIS_PROTOCOL")
    factor = read(out / "reports/factor_construction.json")
    if factor.get("PASS") is not True:
        raise ValueError("FACTOR_GATE")
    # The runner's current load_bank verifies its authenticated factor receipt.
    runner.load_bank()
    targets = [out / rel for rel in ("data/analysis_arrays.npz", "data/bootstrap_statistics.npz",
        "tables/nmse.csv", "tables/contrasts.csv", "tables/decomposition.csv",
        "tables/trajectory_energies.csv", "tables/analysis_input_manifest.csv", "reports/analysis_integrity.json")]
    if any(p.exists() for p in targets):
        raise ValueError("ANALYSIS_TARGET_ALREADY_EXISTS")
    frame, identities, integrity = collect(root, out, mr1, mr2, runner)
    old_draws = load_npz(mr1 / "data/exploratory_bootstrap_indices.npz")
    draws = load_npz(out / "data/bootstrap_indices.npz")
    if set(draws) != set(CONDITIONS) or set(old_draws) != set(CONDITIONS):
        raise ValueError("BOOTSTRAP_CONDITIONS")
    for c in CONDITIONS:
        validate_draws(draws[c])
        if not np.array_equal(draws[c], old_draws[c]):
            raise ValueError("SAVED_BOOTSTRAP_IDENTITY")
    nmse, contrasts, decomp, trajectory, arrays, boots, stats = calculate(frame, draws, mr1)
    runner.verify_sources()
    out_files = []
    exclusive_npz(targets[0], arrays)
    exclusive_npz(targets[1], boots)
    for target, rows in zip(targets[2:7], (nmse, contrasts, decomp, trajectory, identities)):
        exclusive_csv(target, rows)
    for path in targets[:-1]:
        out_files.append(dict(path=path.relative_to(out).as_posix(), bytes=path.stat().st_size, sha256=sha(path)))
    answer = dict(PASS=True, **{k: v for k, v in integrity.items() if k != "PASS"},
                  statistical_checks=stats, nmse_rows=len(nmse), contrast_rows=len(contrasts),
                  decomposition_rows=len(decomp), bootstrap=2000,
                  bootstrap_indices_sha256=sha(mr1 / "data/exploratory_bootstrap_indices.npz"),
                  new_draws=0, method_parameters_changed=False, old_outputs_written=False,
                  disclosed_development=True, final_design_selected=False,
                  old_formal_results_read=0, output_files=out_files,
                  elapsed_seconds=time.monotonic() - started)
    exclusive_json(targets[-1], answer)
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    try:
        report = analyze(args.root)
    except Exception as error:
        root = args.root.resolve() if args.root is not None else Path(__file__).resolve().parents[3]
        directory = root / "outputs/revision_mr05_compression_controls/reports"
        if directory.is_dir():
            exclusive_json(directory / f"analysis_failure_{time.time_ns()}.json",
                           dict(status="STOP", error=repr(error), traceback=traceback.format_exc(),
                                old_files_written=False, new_draws=0))
        raise
    print(json.dumps({k: report[k] for k in ("PASS", "physical_trajectories", "nmse_rows", "contrast_rows", "bootstrap", "new_draws", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
