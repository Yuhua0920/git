"""Remove printed drafting instructions without drafting off-scope sections."""
import subprocess
from paper_support import ROOT, json_read, json_write, sha256


def prepare():
    commit = json_read(ROOT / "outputs/t20e_before_manifest.json")["t20d_commit"]
    paths = ["paper/main.tex", "paper/appendix/evidence_slots.tex"] + [
        f"paper/sections/{name}.tex" for name in (
            "01_introduction", "02_related_work", "05_experimental_setup",
            "06_results", "07_discussion", "08_conclusion")]
    changes = {}
    for name in paths:
        old = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT).decode("utf8")
        text = "\n".join("% " + line if r"\TODO{" in line or line.startswith("TODO:") else line for line in old.splitlines()) + "\n"
        text = text.replace("System, CDL and frozen development/formal partitions", "System, CDL and development/evaluation partitions")
        text = text.replace("Formal Results", "Independent Held-Out Evaluation")
        text = text.replace("Correction fidelity, statistical endpoints and arithmetic cost", "Correction fidelity, statistical comparisons and arithmetic cost")
        text = text.replace("Formal pooled data NMSE", "Independent held-out pooled data NMSE")
        text = text.replace("Formal uncoded data BER", "Independent held-out uncoded data BER")
        text = text.replace("Point estimates and paired trajectory 95\\% intervals use the same saved data as the original formal figure; only display typography is revised.", "Point estimates and paired trajectory 95\\% intervals describe the same received realizations across methods.")
        text = text.replace(" A/B/C all pass within their respective registered scopes.", "")
        # Editorial bit-count notation avoids reusing the new fitting-residual e.
        if name.endswith("05_experimental_setup.tex"):
            text = text.replace(r"\sum_{i,j}e_{ij}", r"\sum_{i,j}b^{\mathrm{err}}_{ij}")
            text = text.replace(r"\sum_{i,j}n_{ij}", r"\sum_{i,j}b^{\mathrm{tot}}_{ij}")
        (ROOT / name).write_text(text, encoding="utf8", newline="\n")
        changes[name] = dict(current_sha256=sha256(ROOT/name), new_section_prose_drafted=False)
    json_write(ROOT / "outputs/t20e_scope_cleanup.json", changes)
    return changes


if __name__ == "__main__":
    print(prepare())
