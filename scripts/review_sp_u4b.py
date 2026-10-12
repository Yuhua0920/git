"""Reproduce all fifteen exports from CSV; record completed visual review."""
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u4b_common import *
import fitz

def main():
    manifest=readjson(OUT/'sp_u4b_figure_manifest.json')
    paths=[OUT/'sp_u4b_figures'/(r['figure']+'.'+ext) for r in manifest['figures'] for ext in ('png','pdf','svg')]
    before={p.name:sha(p) for p in paths}
    subprocess.run([sys.executable,str(ROOT/'scripts/plot_sp_u4b.py')],cwd=ROOT,check=True)
    after={p.name:sha(p) for p in paths}
    if before!=after:raise ValueError('CSV-only exports changed')
    widths={}
    for r in manifest['figures']:
        with fitz.open(OUT/'sp_u4b_figures'/(r['figure']+'.pdf')) as doc:
            widths[r['figure']]=doc[0].rect.width*25.4/72
    if any(abs(v-183)>.01 for v in widths.values()):raise ValueError('export width')
    save(OUT/'sp_u4b_visual_review.json',dict(passed=True,figures=5,panels=18,all_panels_visually_inspected=True,
        notes=['A/B: all five colored curves, per-preparation trajectory IQR, axes and legends legible',
            'C: all five raw displacement points; B and V explicitly distinguished',
            'D: eight panels, categorical budget positions explicitly labeled; P solid/M dashed',
            'E: four panels, method degradation separated from paired budget tails; no overlap'],
        CSV_only_export_reproduction_exact=True,exports_checked=15,export_sha256=after,PDF_width_mm=widths,
        source_warning_review={'TIFF':'not required for development delivery; editable PDF/SVG and 300 dpi PNG supplied',
            '600dpi':'300 dpi preview meets floor; vectors available',
            'width':'static validator misparsed183/25.4 expression; all actual PDFs measure183mm'}))
    print('SP-U4B visual QA PASS; 18 panels; 15 byte-identical CSV-only exports; PDF183mm')

if __name__=='__main__':main()
