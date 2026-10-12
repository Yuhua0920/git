"""Skill-mandated source/PDF/alignment/collision QA, no scientific recomputation."""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import sp_u2d_diagnostic as d
d.install_guard()
SKILL=Path('C:/Users/yuhua/.codex/skills/nature-figure/scripts')


def main():
    records=[]
    for letter in 'ABCDE':
        path=d.OUT/f'sp_u2d_figures/fig_{letter}.pdf'
        for name,args in [('text',['audit_pdf_text.py',str(path),'--min-pt','5']),
                          ('collision',['audit_figure_collisions.py',str(path),'--json-out',str(path.with_suffix('.collision.json'))])]:
            cmd=[sys.executable,str(SKILL/args[0]),*args[1:]]
            result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
            log=d.OUT/f'sp_u2d_qa_{letter}_{name}.txt';assert not log.exists()
            log.write_text(result.stdout+result.stderr,encoding='utf8')
            records.append(dict(figure=letter,audit=name,command=cmd,exit_code=result.returncode,log_sha256=d.sha(log)))
            print(letter,name,'exit',result.returncode,flush=True)
    d.save(d.OUT/'sp_u2d_figure_qa_execution.json',dict(records=records,all_exit_zero=all(a['exit_code']==0 for a in records)))
    if any(a['exit_code'] for a in records):raise RuntimeError('figure QA requires repair')


if __name__=='__main__':main()
