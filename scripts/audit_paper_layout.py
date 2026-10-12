"""Read-only T20D typography audit; QA views never replace frozen figures."""
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

from paper_support import ROOT, json_write, sha256

# Machado et al. full-severity approximation, linear RGB; diagnostic only.
# No clinical/physical color-vision guarantee is inferred from these previews.
CVD = np.array([[.367322, .860646, -.227968], [.280085, .672501, .047413],
                [-.011820, .042940, .968881]])


def linear_rgb(rgb):
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)


def srgb(rgb):
    rgb = np.clip(rgb, 0, 1)
    return np.where(rgb <= .0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - .055)


def scale_pt(value, original_width_mm, target_width_mm):
    return value * target_width_mm / original_width_mm


def table_rows(text):
    # Body begins after repeated longtable header. Excludes caption/header/footer.
    body = text.split(r"\endhead", 1)[1].split(r"\bottomrule", 1)[0]
    return [line for line in body.splitlines() if " & " in line and line.rstrip().endswith(r"\\")]


def audit(root=ROOT):
    root = Path(root)
    output = root / "outputs/t20d_layout_qa"
    output.mkdir(parents=True, exist_ok=True)
    figures = {}
    for name in ("formal_nmse", "formal_ber", "formal_endpoints"):
        path = root / "paper/figures" / (name + ".pdf")
        doc = pymupdf.open(path)
        page = doc[0]
        fonts = [span["size"] for block in page.get_text("dict")["blocks"]
                 for line in block.get("lines", []) for span in line["spans"] if span["text"].strip()]
        width, height = page.rect.width * 25.4 / 72, page.rect.height * 25.4 / 72
        record = {"input_pdf_sha256": sha256(path), "width_mm": width, "height_mm": height,
                  "min_font_pt_original": min(fonts),
                  "min_font_pt_at_174mm": scale_pt(min(fonts), width, 174),
                  "min_font_pt_at_85mm": scale_pt(min(fonts), width, 85),
                  "full_width_planned": True}
        # Render immutable vector artwork at 2x point resolution (144dpi).
        # Final-width typography is measured separately; no plot/data calls.
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
        rgb = np.asarray(Image.frombytes("RGB", (pix.width, pix.height), pix.samples), dtype=float) / 255
        gray = np.sum(linear_rgb(rgb) * [.2126, .7152, .0722], axis=-1)
        gray_rgb = srgb(np.repeat(gray[..., None], 3, axis=-1))
        cvd_rgb = srgb(linear_rgb(rgb) @ CVD.T)
        for kind, view in (("grayscale", gray_rgb), ("deuteranopia_approx", cvd_rgb)):
            target = output / (name + "_" + kind + ".png")
            Image.fromarray(np.round(view * 255).astype(np.uint8)).save(target)
            record[kind + "_qa_path"] = target.relative_to(root).as_posix()
        doc.close()
        figures[name] = record
    tables = {}
    for name in ("02_nmse", "03_ber"):
        path = root / "paper/tables" / (name + ".tex")
        rows = table_rows(path.read_text(encoding="utf8"))
        columns = max(row.count(" & ") + 1 for row in rows)
        tables[name] = {"sha256": sha256(path), "body_rows": len(rows), "columns": columns,
                        "individual_data_fields": len(rows) * columns,
                        "recommended_placement": "full appendix; compact main derivative planned only"}
    report = {"task": "T20D", "figure_numeric_geometry_changed": False,
              "immutable_pdf_render_only": True, "figures": figures, "tables": tables,
              "cvd_model": "Machado2009 severity1 deuteranomaly approximation; diagnostic only",
              "cvd_source": "https://raw.githubusercontent.com/colour-science/colour/develop/colour/blindness/datasets/machado2010.py",
              "original_author_resource_access": "UFRGS project page retrieval timed out; matrix checked against official Colour library dataset documentation",
              "accessibility_warning": "NMSE/BER series use shared circle/solid style; color-only identity does not survive grayscale reliably",
              "endpoint_role": "main_text", "no_new_scientific_data": True}
    json_write(root / "outputs/t20d_layout_audit.json", report)
    return report


if __name__ == "__main__":
    r = audit()
    print({"figures": r["figures"], "tables": r["tables"], "warning": r["accessibility_warning"]})
