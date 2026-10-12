"""Run the skill's source, PDF glyph, alignment and collision audits."""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import sp_u2c_diagnostic as d
d.install_guard()
SKILL=Path('C:/Users/yuhua/.codex/skills/nature-figure/scripts')


def main():
    records=[]
    for letter in 'ABCDEF':
        path=d.OUT/f'sp_u2c_figures/fig_{letter}.pdf'
        for name,args in [('text',['audit_pdf_text.py',str(path),'--min-pt','5']),
                          ('collision',['audit_figure_collisions.py',str(path),'--json-out',str(path.with_suffix('.collision.json'))])]:
            command=[sys.executable,str(SKILL/args[0]),*args[1:]]
            result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
            log=d.OUT/f'sp_u2c_qa_{letter}_{name}.txt'
            assert not log.exists()
            log.write_text(result.stdout+result.stderr,encoding='utf8')
            records.append(dict(figure=letter,audit=name,exit_code=result.returncode,command=command,log_sha256=d.sha(log)))
            print(letter,name,'exit',result.returncode,flush=True)
    d.save(d.OUT/'sp_u2c_figure_qa_execution.json',dict(records=records,all_exit_zero=all(a['exit_code']==0 for a in records)))
    if any(a['exit_code'] for a in records):raise RuntimeError('rendered figure QA needs repair')


if __name__=='__main__':main()
