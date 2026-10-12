"""Final-size QA of existing paper figures; PDF rendering only, no scientific calls."""
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys

import numpy as np
import pymupdf
from PIL import Image
from paper_support import ROOT, json_read, json_write, sha256
from audit_paper_layout import linear_rgb, srgb, CVD

SKILL=Path('C:/Users/yuhua/.codex/skills/nature-figure/scripts')
sys.path.insert(0,str(SKILL))
from audit_pdf_text import audit_pdf
from validate_figure import validate_source, summarize


def audit(root=ROOT):
    root=Path(root);out=root/'outputs/t20f_figure_qa';out.mkdir(exist_ok=True)
    source=(root/'scripts/build_paper_figures.py').read_text(encoding='utf8')+'\n'+(root/'scripts/plot_formal_test.py').read_text(encoding='utf8')
    findings=validate_source(source,'python')
    preflight=dict(summary=summarize(findings,False),findings=[asdict(r) for r in findings],
        delegated_plot_source_included=True,uncertainty_and_log_guard_from_frozen_plot=True)
    json_write(root/'outputs/t20f_figure_source_preflight.json',preflight)
    figures={}
    for name in ('formal_nmse','formal_ber','formal_endpoints'):
        path=root/'paper/figures'/(name+'.pdf')
        doc=pymupdf.open(path);page=doc[0]
        width=page.rect.width*25.4/72;scale=174/width
        glyph=audit_pdf(path.read_bytes(),minimum_pt=6/scale)
        json_write(out/(name+'_font.json'),glyph)
        spans=[s['size'] for b in page.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans'] if s['text'].strip()]
        pix=page.get_pixmap(matrix=pymupdf.Matrix(scale*180/72,scale*180/72),alpha=False)
        rgb=np.asarray(Image.frombytes('RGB',(pix.width,pix.height),pix.samples),dtype=float)/255
        gray=np.sum(linear_rgb(rgb)*[.2126,.7152,.0722],axis=-1)
        views={'color':rgb,'grayscale':srgb(np.repeat(gray[...,None],3,axis=-1)),
               'deuteranopia_approx':srgb(linear_rgb(rgb)@CVD.T)}
        for kind,values in views.items():
            Image.fromarray(np.round(values*255).astype(np.uint8)).save(out/(name+'_'+kind+'.png'))
        doc.close()
        collision_path=out/(name+'_collision.json')
        command=[sys.executable,str(SKILL/'audit_figure_collisions.py'),str(path),
                 '--json-out',str(collision_path),'--overlay-pdf',str(out/(name+'_collision_overlay.pdf'))]
        executed=subprocess.run(command,capture_output=True,text=True)
        (out/(name+'_collision_execution.txt')).write_text(executed.stdout+executed.stderr,encoding='utf8')
        collision=json_read(collision_path) if collision_path.is_file() else {}
        alignment=json_read(root/'paper/figures'/(name+'.alignment.json'))
        figures[name]=dict(pdf_sha256=sha256(path),intrinsic_width_mm=width,target_width_mm=174,
            min_glyph_pt_at_174mm=glyph.get('minimum_found_pt',0)*scale,
            min_text_span_pt_at_174mm=min(spans)*scale,
            all_glyphs_at_least_6pt=glyph.get('auditable',False) and glyph.get('below_minimum_count',1)==0,
            collision_exit_code=executed.returncode,collision_report=collision,alignment_report=alignment,
            qa_paths={kind:(out/(name+'_'+kind+'.png')).relative_to(root).as_posix() for kind in views})
    report=dict(figures=figures,source_preflight=preflight,quantitative_grid=True,backend='python',
        numeric_data_unchanged=True,manual_final_size_review_required=True,
        panel_roles={'nmse':['source-matched control','prepared profile shift','prepared mobility shift','held-out composition','delay boundary'],
                     'ber':['detection consequences in the same five conditions'],
                     'differences':['source benefit','affine over convex','compression fidelity']},
        all_automatic_checks_passed=preflight['summary']['ready'] and
            all(v['all_glyphs_at_least_6pt'] and v['collision_exit_code']==0 for v in figures.values()))
    json_write(root/'outputs/t20f_figure_layout_audit.json',report);return report


if __name__=='__main__':
    r=audit()
    print({k:{'min_glyph_pt_at_174mm':v['min_glyph_pt_at_174mm'],'collision_exit':v['collision_exit_code']} for k,v in r['figures'].items()})
    if not r['all_automatic_checks_passed']:raise SystemExit(1)
