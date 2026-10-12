"""Editorial derivative of frozen tables: rename prose, preserve all numeric cells."""
import argparse
import hashlib
import re
import subprocess
from pathlib import Path
from paper_support import ROOT, json_read, json_write, require_sources, sha256


def snapshot(name, root=ROOT):
    commit = json_read(Path(root) / "outputs/t20e_before_manifest.json")["t20d_commit"]
    return subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=root)


def render(root=ROOT, output=None):
    root = Path(root)
    output = Path(output) if output else root / "paper/tables"
    output.mkdir(parents=True, exist_ok=True)
    historical = json_read(root / "paper/tables/provenance.json")
    sources = require_sources(root, historical["source_sha256"])
    products, row_audits = {}, {}
    for name, digest in historical["generated_table_sha256"].items():
        raw = snapshot("paper/tables/" + name, root)
        assert hashlib.sha256(raw).hexdigest() == digest
        old = raw.decode("utf8")
        text = old.replace("Formal", "Independent held-out").replace("formal", "independent held-out")
        text = text.replace("CV; causal", "Cross-pilot partitions; causal").replace("4 folds", "4 partitions")
        text = text.replace("C30/100", "CDL-C / 30 km/h / 100 ns").replace("B30/100", "CDL-B / 30 km/h / 100 ns").replace("C120/100", "CDL-C / 120 km/h / 100 ns")
        text = text.replace("Pilot CV", "Cross-pilot fitting").replace("rank4", "rank-4")
        text = text.replace("TX offsets", "Transmit-antenna offsets")
        text = text.replace("pooled data-domain NMSE (dB)", "pooled data-domain normalized mean-square error (NMSE), dB")
        text = text.replace("pooled uncoded data BER, ZF detector", "pooled uncoded data bit error rate (BER), zero-forcing (ZF) detector")
        text = text.replace("Endpoint &", "Comparison &").replace(" & Pass", " & Criterion met")
        text = text.replace("Preregistered primary endpoint cells", "Predefined primary comparison criteria")
        if name == "04_endpoints.tex":
            text = re.sub(r"^([ABC]) &", lambda m: {"A":"Source benefit", "B":"Affine benefit", "C":"Compression"}[m[1]]+" &", text, flags=re.M)
            text = text.replace(" & True", " & Yes").replace(" & False", " & No")
            start = text.index("A: Source minus LR4;")
            end = text.index(r"\endgroup", start)
            text = text[:start] + "Source benefit: Source minus LR4; affine benefit: Convex minus LR4; compression: LR4 minus Dense. One-sided Bonferroni bounds over ten cells, quantiles 0.005/0.995. The delay boundary is reported separately.\n" + text[end:]
        if name == "05_complexity.tex":
            text = text.replace("Counts exclude scalar products/solver, ZF, memory traffic and host/GPU transfer. No runtime speedup, real-time or hardware-efficiency inference.", "Counts cover matrix-vector arithmetic and correction storage; scalar products/solver, detection, memory traffic and device transfers are excluded.")
        # Numeric rows in the two performance tables must remain byte-identical.
        if name in ("02_nmse.tex", "03_ber.tex"):
            rows = lambda s: [line for line in s.splitlines() if re.match(r"^[BC]-", line)]
            assert rows(old) == rows(text)
            row_audits[name] = dict(numeric_rows=len(rows(text)), exact_row_match=True)
        (output / name).write_text(text, encoding="utf8", newline="\n")
        products[name] = sha256(output / name)
    report = dict(historical_snapshot_commit=json_read(root / "outputs/t20e_before_manifest.json")["t20d_commit"],
                  historical_table_provenance_unchanged=True, source_sha256=sources,
                  generated_table_sha256=products, performance_row_audits=row_audits,
                  raw_metric_cells_unchanged=len(historical["cells"]) == 160,
                  numeric_metrics_recomputed=False, editorial_only=True)
    json_write(output / "t20e_provenance.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    print(render(output=p.parse_args().output))
