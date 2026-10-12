"""T17B CSV-only exact interaction size and signed energy decomposition."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parents[1]
SKILL=Path('C:/Users/yuhua/.codex/skills/nature-figure/scripts')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.pytest_cache/matplotlib-t17b'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0,str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment
SNRS=(5,10,20);TERMS=('C1','C2','C3','C4','R_exact')
DATA=('t17b_exact_interaction_terms.csv','t17b_cross_energy_matrix.csv')
STEMS=('t17b_contribution_norms','t17b_signed_energy')
STYLE={'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
    'font.size':10,'legend.fontsize':9,'svg.fonttype':'none','pdf.fonttype':42,
    'svg.hashsalt':'T17B-fixed','axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))


def number(row,key):
    value=float(row[key])
    if not math.isfinite(value):raise ValueError('nonfinite plotted value')
    return value


def series(terms,cross):
    values={};signed={};indices=[]
    if len(terms)!=15 or len(cross)!=48:raise ValueError('all15 terms and48 Gram entries required')
    for s in SNRS:
        group=[r for r in terms if int(r['snr_db'])==s]
        if sorted(r['term'] for r in group)!=sorted(TERMS):raise ValueError('all five fixed terms required')
        g={r['term']:r for r in group}
        for k in TERMS:
            val=number(g[k],'relative_to_complete_correction')
            if val<0:raise ValueError('norm ratio cannot be negative')
            if not math.isclose(val**2,number(g[k],'normalized_energy_relative_to_complete_correction'),rel_tol=2e-13,abs_tol=2e-15):raise ValueError('norm/energy mismatch')
            values[s,k]=val
        entries=[r for r in cross if int(r['snr_db'])==s]
        if {(r['term_i'],r['term_j']) for r in entries}!={(a,b) for a in TERMS[:4] for b in TERMS[:4]}:raise ValueError('complete Gram required')
        # Every pair is stored; only upper off-diagonal pairs count once.
        xc=sum(number(r,'normalized_cross_energy') for r in entries if TERMS.index(r['term_i'])<TERMS.index(r['term_j']))
        energies=[number(g[k],'normalized_energy_relative_to_complete_correction') for k in TERMS]
        if not math.isclose(sum(energies[:4])+xc,energies[4],rel_tol=1e-10,abs_tol=1e-12):raise ValueError('signed energy accounting mismatch')
        signed[s]=energies[:4]+[xc,energies[4]]
        indices.append(sum(energies[:4])/energies[4])
    return values,signed,indices


def export(fig,stem,directory):
    require_matplotlib_panel_alignment(fig,json_out=str(directory/f'{stem}.alignment.json'),strict=True)
    fig.savefig(directory/f'{stem}.png',dpi=300)
    fig.savefig(directory/f'{stem}.pdf',metadata={'CreationDate':None,'ModDate':None})
    path=directory/f'{stem}.svg';fig.savefig(path,metadata={'Date':None})
    path.write_text('\n'.join(l.rstrip() for l in path.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8')
    plt.close(fig)


def plot(input_dir,output_dir):
    source,directory=Path(input_dir),Path(output_dir)
    norms,signed,indices=series(rows(source/DATA[0]),rows(source/DATA[1]))
    directory.mkdir(parents=True,exist_ok=True)
    with matplotlib.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(7.2,4.8));fig.subplots_adjust(left=.14,right=.97,bottom=.27,top=.73)
        for k,color,marker,style in zip(TERMS,('#0072B2','#D55E00','#009E73','#CC79A7','0.2'),('o','s','^','D','x'),('-','--','-.',':','-')):
            ax.plot(SNRS,[norms[s,k] for s in SNRS],label=k.replace('_',' '),color=color,marker=marker,linestyle=style,linewidth=1.5)
        ax.set_xticks(SNRS);ax.set_xlabel('Frozen source deployment SNR label (dB)')
        ax.set_ylabel('Norm / complete B120 correction norm')
        ax.set_yscale('symlog',linthresh=.1);ax.set_ylim(bottom=0)
        ax.grid(axis='y',color='0.9');ax.set_axisbelow(True)
        fig.suptitle('Exact finite-shift interaction contributions',fontsize=12,y=.98)
        fig.legend(*ax.get_legend_handles_labels(),ncol=5,loc='upper center',bbox_to_anchor=(.55,.91))
        fig.text(.14,.105,'B120-weighted norms; all four exact contributions and final residual',fontsize=9)
        fig.text(.14,.045,'Symlog: linear below 0.1; frozen deterministic operators; no CI',fontsize=9)
        export(fig,STEMS[0],directory)

        fig,ax=plt.subplots(figsize=(7.2,4.8));fig.subplots_adjust(left=.14,right=.97,bottom=.27,top=.73)
        x=np.arange(6)
        for offset,s,color in zip((-.24,0,.24),SNRS,('#0072B2','#D55E00','0.35')):
            ax.bar(x+offset,signed[s],width=.22,color=color,label=f'{s} dB',linewidth=0)
        ax.axhline(0,color='0.4',linewidth=.7)
        ax.set_xticks(x,['C1','C2','C3','C4','Sum cross','R exact'])
        ax.set_ylabel('Signed energy / complete correction energy')
        ax.set_yscale('symlog',linthresh=.1)
        ax.grid(axis='y',color='0.9');ax.set_axisbelow(True)
        fig.suptitle('Individual energies and destructive cross contributions',fontsize=12,y=.98)
        fig.legend(*ax.get_legend_handles_labels(),ncol=3,loc='upper center',bbox_to_anchor=(.55,.91))
        fig.text(.14,.105,'C1+C2+C3+C4 energies + signed Sum cross = R exact energy',fontsize=9)
        fig.text(.14,.045,'Cancellation indices (5/10/20 dB): '+ ' / '.join(f'{v:.3g}' for v in indices),fontsize=9)
        export(fig,STEMS[1],directory)
    return dict(input_rows={DATA[0]:15,DATA[1]:48},plotted_rows={DATA[0]:15,DATA[1]:48},
        accounting='Upper cross entries counted once; full 4x4 retained; symlog linthresh0.1; no clipping')


def audit(input_dir,output_dir):
    start=time.perf_counter();directory=Path(output_dir);selection=plot(input_dir,directory)
    source=subprocess.run([sys.executable,str(SKILL/'validate_figure.py'),str(Path(__file__)),'--json'],capture_output=True,text=True)
    checks={}
    for stem in STEMS:
        text=subprocess.run([sys.executable,str(SKILL/'audit_pdf_text.py'),str(directory/f'{stem}.pdf'),'--min-pt','5','--json'],capture_output=True,text=True)
        collision=subprocess.run([sys.executable,str(SKILL/'audit_figure_collisions.py'),str(directory/f'{stem}.pdf'),
            '--json-out',str(directory/f'{stem}.collision-audit.json')],capture_output=True,text=True)
        checks[stem]=dict(text_returncode=text.returncode,collision_returncode=collision.returncode,
            text_output=text.stdout+text.stderr,collision_output=collision.stdout+collision.stderr)
    with tempfile.TemporaryDirectory(dir=ROOT/'.pytest_cache/pytest_workspace_temp') as temp:
        temp=Path(temp)
        for name in DATA:shutil.copy2(Path(input_dir)/name,temp/name)
        plot(temp,temp/'rebuilt')
        rebuilt={f'{stem}.{ext}':sha(directory/f'{stem}.{ext}')==sha(temp/'rebuilt'/f'{stem}.{ext}') for stem in STEMS for ext in ('png','pdf','svg')}
    report=dict(source_returncode=source.returncode,source_output=source.stdout+source.stderr,rendered_checks=checks,
        CSV_only_rebuild=rebuilt,source_data_selection=selection,source_data_hashes={n:sha(Path(input_dir)/n) for n in DATA},
        figure_contract=dict(backend='python',archetype='quantitative decomposition',question='Is small exact interaction caused by large cancelling contributions?',
            panel_roles=['Contribution size','Signed energy accounting'],uncertainty='none: frozen deterministic operators',
            exports=['PNG300dpi','editable PDF','editable SVG'],adaptation='style-only inheritance; new CSV mappings and guards'),
        runtime_plotting_and_QA_seconds=time.perf_counter()-start,visual_review='pending',all_checks_passed=False,
        reviewed_warnings=['No TIFF: research exports; PNG300dpi plus editable vectors, no journal submission requested.'])
    (directory/'t17b_figure_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    if source.returncode or not all(rebuilt.values()) or any(v['text_returncode'] or v['collision_returncode'] for v in checks.values()):raise ValueError('figure QA failed; numerical CSVs retained')
    print('T17B figure QA and CSV-only byte rebuild PASS; actual visual review pending')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-dir',default=str(ROOT/'outputs'));p.add_argument('--output-dir',default=str(ROOT/'outputs'));p.add_argument('--audit',action='store_true');a=p.parse_args()
    audit(a.input_dir,a.output_dir) if a.audit else plot(a.input_dir,a.output_dir)
