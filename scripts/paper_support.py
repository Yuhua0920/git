"""Read-only evidence access and deterministic paper artifact helpers."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T20B_COMMIT = "8b67a62a5f039239703c0775a94c1d1f0bc3f040"
T20B_RESULT_HASH = "6555b9f95be0c76fa1abdc55a2b408e99ae7c8908d7ad702ed8d3d70bc3b4a57"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def json_write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
                          encoding="utf8", newline="\n")


def csv_read(path):
    with Path(path).open(encoding="utf8", newline="") as stream:
        return list(csv.DictReader(stream))


def accepted_results(root=ROOT):
    path = Path(root) / "outputs/t20b_formal_results.json"
    if sha256(path) != T20B_RESULT_HASH:
        raise ValueError("Frozen T20B final result SHA256 mismatch")
    result = json_read(path)
    if not result["all_checks_passed"]:
        raise ValueError("T20B final acceptance is not PASS")
    return result


def require_sources(root, paths):
    """Validate frozen source bytes before reporting, without loading any channel."""
    root = Path(root)
    manifest = json_read(root / "outputs/t20c_paper_evidence_manifest.json")
    for path in paths:
        if sha256(root / path) != manifest["historical_sha256"][path]:
            raise ValueError(f"Frozen paper source changed: {path}")
    return {path: manifest["historical_sha256"][path] for path in paths}
