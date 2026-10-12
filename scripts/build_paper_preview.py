"""Flatten the scaffold for the standalone editor; external figures become placeholders."""
import re
from pathlib import Path

from paper_support import ROOT, json_read, json_write, sha256


def bib_entries(text):
    """Parse the project's braced BibTeX fields with nested TeX groups."""
    entries = {}
    cursor = 0
    while match := re.search(r"@(\w+)\s*\{\s*([^,]+),", text[cursor:]):
        kind, key = match.group(1), match.group(2).strip()
        pos = cursor + match.end()
        fields = {"entry_type": kind}
        while True:
            pos += len(text[pos:]) - len(text[pos:].lstrip(" \r\n\t,"))
            if text[pos] == "}":
                pos += 1
                break
            field = re.match(r"(\w+)\s*=\s*\{", text[pos:])
            if not field:
                raise ValueError(f"Invalid braced BibTeX field near {text[pos:pos+60]}")
            name = field.group(1).lower()
            pos += field.end()
            start, depth = pos, 1
            while depth and pos < len(text):
                depth += (text[pos] == "{") - (text[pos] == "}")
                pos += 1
            if depth:
                raise ValueError("Unbalanced BibTeX braces")
            fields[name] = text[start:pos-1]
        if key in entries:
            raise ValueError(f"Duplicate citation key: {key}")
        entries[key] = fields
        cursor = pos
    return entries


def flatten(path, paper):
    text = path.read_text(encoding="utf8")
    def child(match):
        target = paper / (match.group(1) + ".tex")
        if paper.resolve() not in target.resolve().parents:
            raise ValueError("Input outside paper tree")
        return flatten(target, paper)
    return re.sub(r"\\input\{([^}]+)\}", child, text)


def build(root=ROOT):
    paper = Path(root) / "paper"
    text = flatten(paper / "main.tex", paper)
    entries = bib_entries((paper / "bibliography/references.bib").read_text(encoding="utf8"))
    bibliography = [r"\begin{thebibliography}{99}"]
    for key, fields in entries.items():
        venue = fields.get("journal", fields.get("booktitle", fields.get("howpublished", fields.get("institution"))))
        bibliography += [r"\bibitem{" + key + "} " + fields["author"] + ". " + fields["title"] + ". " + venue + ", " + fields["year"] + ".",
                         r"\url{" + fields.get("url", fields["verified_source"]) + "}"]
    bibliography.append(r"\end{thebibliography}")
    text = re.sub(r"\\includegraphics\[[^\]]+\]\{figures/([^}]+)\}",
        lambda m: r"\fbox{\parbox[c][35mm][c]{.9\linewidth}{\centering Figure placeholder: " +
            m.group(1).replace("_", r"\_") + "}}", text)
    text = re.sub(r"\\bibliographystyle\{[^}]+\}\s*\\bibliography\{[^}]+\}",
                  lambda _: "\n".join(bibliography), text)
    preview = paper / "preview.tex"
    preview.write_text("% Generated standalone preview; figures are placeholders, bibliography flattened.\n" + text,
                       encoding="utf8", newline="\n")
    report = {"standalone_preview_sha256": sha256(preview), "figure_placeholders": True,
              "external_BibTeX_toolchain_validated": False,
              "source_sha256": {p.relative_to(paper).as_posix(): sha256(p)
                                 for p in sorted(paper.rglob("*.tex")) if p != preview},
              "bibliography_sha256": sha256(paper / "bibliography/references.bib")}
    json_write(paper / "preview_provenance.json", report)
    return report


if __name__ == "__main__":
    print(build())
