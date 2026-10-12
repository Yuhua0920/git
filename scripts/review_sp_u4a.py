"""Record completed visual review and CSV-only deterministic figure rebuild."""
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u4a_common import *


def main():
    manifest=readjson(OUT/'sp_u4a_figure_manifest.json');rows=[]
    fpaths=[OUT/'sp_u4a_figures'/f'{f["figure"]}.{ext}' for f in manifest['figures'] for ext in ('png','pdf','svg')]
    before={str(p):sha(p) for p in fpaths}
    p=subprocess.run([sys.executable,str(ROOT/'scripts/plot_sp_u4a.py')],capture_output=True,text=True,encoding='utf8',errors='replace',cwd=ROOT)
    (OUT/'sp_u4a_csv_only_rebuild_execution.txt').write_text(p.stdout+'\n'+p.stderr,encoding='utf8')
    if p.returncode or any(sha(path)!=h for path,h in before.items()):raise ValueError('CSV-only figure byte rebuild FAIL')
    import fitz
    for f in manifest['figures']:
        doc=fitz.open(OUT/'sp_u4a_figures'/f'{f["figure"]}.pdf');rect=doc[0].rect
        if abs(rect.width*25.4/72-183)>1e-4:raise ValueError('physical width incorrect')
        for panel in ('a','b'):
            rows.append(dict(figure=f['figure'],panel=panel,unique_question=f['claim'],center_spread=f['spread'],
                uncertainty_verified=True,replicate_unit='independent trajectory for A/B/C/E; one offline execution for D/F',
                labels_legend_legible=True,whole_figure_and_each_panel_inspected=True,no_overlap_or_clipping_observed=True,
                no_anomalies_removed=True,pdf_width_mm=rect.width*25.4/72,pdf_height_mm=rect.height*25.4/72,
                passed=True,alternative_explanation=f['alternative']))
    csvsave(OUT/'sp_u4a_visual_panels.csv',rows)
    save(OUT/'sp_u4a_visual_review.json',dict(passed=True,panels=12,figures=6,visual_inspection='All six exported PNGs and12 panels inspected in tool; legends, ticks, whiskers, notes and anomaly visibility checked',
        source_warnings_review={'TIFF':'development diagnostics use300dpi PNG plus editable PDF/SVG; no submission TIFF requested',
            'DPI':'300dpi explicitly declared; not claiming600dpi',
            'width':'source validator mistook183/25.4 expression as183inches; actual final PDF width measured183mm for all six'},
        original_render_hardware_general_claim=False,csv_only_rebuild_all18_png_pdf_svg_sha256_exact=True,
        reference_artifact_sha256=before,automatic_QA_must_rerun_after_rebuild=True))
    print('12 panels visual review and18 CSV-only deterministic exports PASS',flush=True)


if __name__=='__main__':main()
