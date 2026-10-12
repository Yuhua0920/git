"""Validate T04A FDM pilot, data, and mute masks using actual mask counts."""

import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, validate_config  # noqa: E402
from src.pilots import build_fdm_pilot_layout, count_layout_resources  # noqa: E402


EXPECTED_COUNTS = {
    2: {"per_tx": [128, 128], "union": 256, "ratio": 1 / 4, "data": 1536},
    4: {"per_tx": [64, 64], "union": 128, "ratio": 1 / 8, "data": 1792},
    8: {"per_tx": [32, 32], "union": 64, "ratio": 1 / 16, "data": 1920},
    16: {"per_tx": [16, 16], "union": 32, "ratio": 1 / 32, "data": 1984},
}
CSV_COLUMNS = (
    "spacing_frequency",
    "offset_tx0",
    "offset_tx1",
    "pilot_count_tx0",
    "pilot_count_tx1",
    "pilot_union_count",
    "pilot_union_ratio",
    "total_re",
    "two_stream_data_symbols",
    "mute_count_tx0",
    "mute_count_tx1",
    "all_checks_passed",
)


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    validate_config(config)
    system = config["system"]
    pilot = config["pilot"]
    fixed_system = {"nt": 2, "nr": 2, "active_carriers": 64, "num_symbols": 16}
    for key, value in fixed_system.items():
        if system[key] != value:
            raise ValueError(f"T04A requires {key}={value}; got {system[key]}")
    if system["nfft"] != system["active_carriers"]:
        raise ValueError("T04A requires active_carriers == nfft")
    if pilot["spacing_time"] != 4:
        raise ValueError("T04A requires spacing_time=4")
    if pilot["spacing_frequency_options"] != [2, 4, 8, 16]:
        raise ValueError("T04A requires spacing_frequency_options=[2,4,8,16]")

    nt = system["nt"]
    K = system["active_carriers"]
    T = system["num_symbols"]
    dT = pilot["spacing_time"]
    layouts = []
    csv_rows = []
    diagnostics = []
    for dF in pilot["spacing_frequency_options"]:
        layout = build_fdm_pilot_layout(nt, K, T, dF, dT)
        counts = count_layout_resources(layout)
        no_overlap = bool(np.all(np.sum(layout.pilot_mask, axis=0) <= 1))
        partition_valid = bool(
            np.all(
                layout.pilot_mask.astype(np.int8)
                + layout.data_mask.astype(np.int8)
                + layout.mute_mask.astype(np.int8)
                == 1
            )
        )
        non_owner_muted = bool(
            np.array_equal(
                layout.mute_mask,
                layout.pilot_union[None, :, :] & ~layout.pilot_mask,
            )
        )
        no_data_on_union = not bool(np.any(layout.data_mask[:, layout.pilot_union]))
        both_tx_data_outside_union = bool(
            np.all(layout.data_mask[:, ~layout.pilot_union])
        )
        complement_valid = bool(
            np.array_equal(layout.data_mask, np.broadcast_to(~layout.pilot_union, layout.data_mask.shape))
        )
        owner_valid = bool(np.all(layout.pilot_owner[~layout.pilot_union] == -1))
        for q in range(nt):
            owner_valid = owner_valid and bool(np.all(layout.pilot_owner[layout.pilot_mask[q]] == q))

        expected = EXPECTED_COUNTS[dF]
        counts_passed = bool(
            counts["total_re"] == 1024
            and counts["pilot_count_by_tx"] == expected["per_tx"]
            and counts["union_count"] == expected["union"]
            and counts["union_ratio"] == expected["ratio"]
            and counts["two_stream_data_symbols"] == expected["data"]
            and counts["mute_count_by_tx"] == expected["per_tx"]
        )
        pilot_symbols_valid = layout.pilot_symbol_indices == (0, 4, 8, 12)
        offsets_valid = layout.offsets_by_tx == (0, dF // nt)
        checks_passed = bool(
            no_overlap
            and partition_valid
            and non_owner_muted
            and no_data_on_union
            and both_tx_data_outside_union
            and complement_valid
            and owner_valid
            and counts_passed
            and pilot_symbols_valid
            and offsets_valid
        )
        layouts.append(
            {
                "dF": dF,
                "offsets": list(layout.offsets_by_tx),
                "pilot_count_by_tx": counts["pilot_count_by_tx"],
                "union_count": counts["union_count"],
                "union_ratio": counts["union_ratio"],
                "data_count_by_tx": counts["data_count_by_tx"],
                "two_stream_data_symbols": counts["two_stream_data_symbols"],
                "mute_count_by_tx": counts["mute_count_by_tx"],
                "total_re": counts["total_re"],
                "no_overlap": no_overlap,
                "partition_valid": partition_valid,
                "non_owner_muted": non_owner_muted,
                "no_data_on_union": no_data_on_union,
                "both_tx_data_outside_union": both_tx_data_outside_union,
                "complement_valid": complement_valid,
                "owner_valid": owner_valid,
                "counts_passed": counts_passed,
                "pilot_symbols_valid": pilot_symbols_valid,
                "offsets_valid": offsets_valid,
                "all_checks_passed": checks_passed,
            }
        )
        csv_rows.append(
            {
                "spacing_frequency": dF,
                "offset_tx0": layout.offsets_by_tx[0],
                "offset_tx1": layout.offsets_by_tx[1],
                "pilot_count_tx0": counts["pilot_count_by_tx"][0],
                "pilot_count_tx1": counts["pilot_count_by_tx"][1],
                "pilot_union_count": counts["union_count"],
                "pilot_union_ratio": counts["union_ratio"],
                "total_re": counts["total_re"],
                "two_stream_data_symbols": counts["two_stream_data_symbols"],
                "mute_count_tx0": counts["mute_count_by_tx"][0],
                "mute_count_tx1": counts["mute_count_by_tx"][1],
                "all_checks_passed": checks_passed,
            }
        )
        if dF == 4:
            for k, t in ((0, 0), (1, 0), (2, 0), (0, 1)):
                diagnostics.append(
                    {
                        "k": k,
                        "t": t,
                        "pilot": layout.pilot_mask[:, k, t].tolist(),
                        "mute": layout.mute_mask[:, k, t].tolist(),
                        "data": layout.data_mask[:, k, t].tolist(),
                        "owner": int(layout.pilot_owner[k, t]),
                    }
                )

    all_checks_passed = all(item["all_checks_passed"] for item in layouts)
    report = {
        "task": "T04A",
        "nt": nt,
        "K": K,
        "T": T,
        "spacing_time": dT,
        "pilot_symbol_indices": list(build_fdm_pilot_layout(nt, K, T, 4, dT).pilot_symbol_indices),
        "layouts": layouts,
        "df4_diagnostics": diagnostics,
        "all_checks_passed": all_checks_passed,
    }

    print("T04A 2x2 FDM pilot validation")
    print(f"K: {K}")
    print(f"T: {T}")
    print(f"Nt: {nt}")
    print(f"dT: {dT}")
    print(f"pilot symbols: {report['pilot_symbol_indices']}")
    print("dF  offsets  pilot/TX0  pilot/TX1  union  ratio     data symbols")
    for item in layouts:
        print(
            f"{item['dF']:>2}  {str(item['offsets']):>7}  "
            f"{item['pilot_count_by_tx'][0]:>9}  {item['pilot_count_by_tx'][1]:>9}  "
            f"{item['union_count']:>5}  {item['union_ratio']:>8.6f}  "
            f"{item['two_stream_data_symbols']:>12}"
        )
    print("dF=4 coordinate diagnostics:")
    for item in diagnostics:
        print(
            f"(k={item['k']},t={item['t']}): pilot={item['pilot']} "
            f"mute={item['mute']} data={item['data']} owner={item['owner']}"
        )
    print(f"No overlap: {'PASS' if all(item['no_overlap'] for item in layouts) else 'FAIL'}")
    print(f"Mask partition: {'PASS' if all(item['partition_valid'] for item in layouts) else 'FAIL'}")
    print(f"Counts: {'PASS' if all(item['counts_passed'] for item in layouts) else 'FAIL'}")
    print(f"All checks: {'PASS' if all_checks_passed else 'FAIL'}")

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "t04a_pilot_layout.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(csv_rows)
    json_path = output_dir / "t04a_pilot_layout_results.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"CSV saved: {csv_path}")
    print(f"JSON saved: {json_path}")
    if not all_checks_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
