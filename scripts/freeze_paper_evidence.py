"""Capture or audit existing tracked bytes only; no numerical experiment imports."""
import argparse
from pathlib import Path
import subprocess

from paper_support import ROOT, T20B_COMMIT, accepted_results, json_read, json_write, sha256


def capture(root=ROOT):
    root = Path(root)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != T20B_COMMIT:
        raise ValueError("Evidence capture requires the accepted T20B completion snapshot")
    result = accepted_results(root)
    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    paths = sorted(p.decode("utf8") for p in raw.split(b"\0") if p)
    hashes = {p: sha256(root / p) for p in paths}
    manifest = {"task": "T20C", "t20b_completion_commit": head,
                "historical_sha256": hashes, "historical_files": len(hashes),
                "formal_result_sha256": sha256(root / "outputs/t20b_formal_results.json"),
                "scientific_sources": result["source_sha256"],
                "formal_artifact_count": len(result["formal_artifact_sha256"]),
                "formal_data_use": "report_plot_summarize_only",
                "new_scientific_data_generated": False}
    json_write(root / "outputs/t20c_paper_evidence_manifest.json", manifest)
    return manifest


def audit(root=ROOT):
    root = Path(root)
    manifest = json_read(root / "outputs/t20c_paper_evidence_manifest.json")
    mismatches = [p for p, expected in manifest["historical_sha256"].items()
                  if not (root / p).is_file() or sha256(root / p) != expected]
    result = accepted_results(root)
    formal_bad = [p for p, expected in result["formal_artifact_sha256"].items()
                  if not (root / p).is_file() or sha256(root / p) != expected]
    report = {"task": "T20C", "historical_files_verified": manifest["historical_files"],
              "historical_content_mismatches": mismatches,
              "t20b_formal_artifacts_verified": len(result["formal_artifact_sha256"]),
              "t20b_formal_artifact_mismatches": formal_bad,
              "scientific_code_unchanged": not any(p.startswith("src/") for p in mismatches),
              "all_checks_passed": not mismatches and not formal_bad}
    json_write(root / "outputs/t20c_paper_integrity.json", report)
    if not report["all_checks_passed"]:
        raise ValueError(f"Historical evidence mismatch: {mismatches + formal_bad}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    report = capture() if args.capture else audit()
    print({k: v for k, v in report.items() if "sha256" not in k and k != "scientific_sources"})
