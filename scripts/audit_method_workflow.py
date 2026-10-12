"""Run schematic QA, preserving tool diagnostics apart from valid JSON reports."""
import json
import subprocess
import sys
import pymupdf
from paper_support import ROOT, json_read, json_write, sha256
from build_method_workflow import QA


def audit():
    commands={
        "source_preflight":[str(QA/"validate_figure.py"),"scripts/build_method_workflow.py","--backend","python","--json"],
        "font_audit":[str(QA/"audit_pdf_text.py"),"paper/figures/method_workflow.pdf","--min-pt","5","--json"],
        "collision_audit":[str(QA/"audit_figure_collisions.py"),"paper/figures/method_workflow.pdf","--json",
                           "--json-out","outputs/t20e_workflow_collision_audit.json",
                           "--overlay-pdf","outputs/t20e_workflow_collision_overlay.pdf"]}
    results={};diagnostics={}
    for key,args in commands.items():
        p=subprocess.run([sys.executable]+args,cwd=ROOT,capture_output=True,text=True)
        results[key]=json.loads(p.stdout)
        diagnostics[key]=dict(returncode=p.returncode,stderr=p.stderr,command=[sys.executable]+args)
        json_write(ROOT/f"outputs/t20e_workflow_{key}.json",results[key])
        if p.returncode:raise RuntimeError(key+":"+p.stdout+p.stderr)
    doc=pymupdf.open(ROOT/"paper/figures/method_workflow.pdf")
    doc[0].get_pixmap(matrix=pymupdf.Matrix(2,2)).save(ROOT/"outputs/t20e_workflow_rendered_pdf.png")
    width_mm=doc[0].rect.width*25.4/72
    minimum=results["font_audit"]["minimum_found_pt"]
    scaled=minimum*174/width_mm
    assert scaled>=5
    provenance=json_read(ROOT/"paper/figures/method_workflow_provenance.json")
    for name,digest in provenance["generated_sha256"].items():assert sha256(ROOT/"paper/figures"/name)==digest
    report=dict(tool_diagnostics=diagnostics, source_ready=results["source_preflight"]["summary"]["ready"],
        exported_minimum_glyph_pt=minimum, minimum_glyph_at_manuscript_174mm_pt=scaled,
        final_width_174mm_font_floor_passed=True, collision_failures=results["collision_audit"]["summary"]["fail"],
        collision_warnings=results["collision_audit"]["summary"]["warn"],
        single_schematic_canvas_alignment="not applicable to quantitative multi-panel alignment",
        no_tiff_submission_asset_requested=True, vector_pdf_and_svg_delivered=True,
        source_qa_warning_disposition="PNG is a preview; TIFF is not requested for this vector schematic.",
        diagram_contains_no_new_scientific_data=True, product_hashes_exact=True,
        all_automatic_checks_passed=results["source_preflight"]["summary"]["ready"] and results["collision_audit"]["summary"]["fail"]==0)
    json_write(ROOT/"outputs/t20e_workflow_audit.json",report)
    print(report)
    return report


if __name__=="__main__":audit()
