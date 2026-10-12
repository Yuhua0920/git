"""Use the figure skill's PDF geometry checks on final MR02 plot exports."""
import argparse,json,sys,hashlib
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--skill',required=True);a=p.parse_args()
    out=Path(a.out);sys.path.insert(0,str(Path(a.skill)/'scripts'))
    from audit_pdf_text import audit_pdf as text_audit
    from audit_figure_collisions import audit_pdf as collision_audit,write_overlay_pdf
    records=[]
    for name in ('mr02_attainable_risk','mr02_coordinate_cases','mr02_compression_terms'):
        pdf=out/'figures'/f'{name}.pdf';glyph=text_audit(pdf.read_bytes(),minimum_pt=5)
        collision=collision_audit(pdf);alignment=json.loads((out/'figures'/f'{name}.alignment.json').read_text(encoding='utf8'))
        for suffix,result in (('glyph',glyph),('collision',collision)):
            (out/'reports'/f'{name}_{suffix}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        if collision['findings']:write_overlay_pdf(pdf,out/'reports'/f'{name}_collision_overlay.pdf',collision['findings'])
        records.append(dict(figure=name,sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),glyph_auditable=glyph['auditable'],glyph_min_pt=glyph['minimum_found_pt'],glyph_below_min=glyph['below_minimum_count'],collision=collision['verdict'],collision_summary=collision['summary'],alignment=alignment))
        print(name,glyph['minimum_found_pt'],collision['verdict'],collision['summary'])
    (out/'reports/automated_visual_qa.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    if any(not r['glyph_auditable'] or r['glyph_below_min'] or r['collision']=='FIX BEFORE DELIVERY' for r in records):raise ValueError('Visual QA failure')
if __name__=='__main__':main()
