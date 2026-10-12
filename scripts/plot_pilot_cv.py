"""T16E CSV-only matched-subset NMSE and pilot-only oracle-efficiency figures."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
SKILL=Path('C:/Users/yuhua/.codex/skills/nature-figure/scripts')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.pytest_cache/matplotlib-t16e'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment
from scripts.run_cdl_mismatch import sha
STYLE={'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':10,
    'legend.fontsize':9,'svg.fonttype':'none','svg.hashsalt':'T16E-fixed','pdf.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False}
STEMS=('t16e_pilot_cv_nmse','t16e_pilot_cv_efficiency')
METHODS=(('source','Source','0.5','s','--'),('pilotCV-1','CV1','#56B4E9','o',':'),
    ('pilotCV-5','CV5','#0072B2','o','-'),('oracleScalar','Oracle scalar','#009E73','^','-.'),
    ('matched','Matched','#D55E00','D','--'))


def plot(input_dir,output_dir):
    with (Path(input_dir)/'t16e_pilot_cv_scan.csv').open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
    directory=Path(output_dir);directory.mkdir(parents=True,exist_ok=True)
    with matplotlib.rc_context(STYLE):
        for index,stem in enumerate(STEMS):
            fig,axes=plt.subplots(1,2,figsize=(7.2,4.8),sharey=True)
            fig.subplots_adjust(left=.12,right=.97,bottom=.27,top=.73,wspace=.22)
            for ax,name,title,letter in zip(axes,('CDL-C-75','CDL-B-120'),('CDL-C / 75 km/h','CDL-B / 120 km/h'),('a','b')):
                for method,label,color,marker,style in (METHODS if index==0 else METHODS[1:3]):
                    v=sorted((r for r in rows if r['shift']==name and r['subset']=='last6' and r['method']==method),key=lambda r:int(r['snr_db']))
                    if [int(r['snr_db']) for r in v]!=[5,10,20]:raise ValueError('all three SNRs and same-frame subset required')
                    key='nmse_db' if index==0 else 'oracle_scalar_efficiency'
                    if any(r[key]=='' for r in v):raise ValueError('undefined plotted ratio')
                    ax.plot([5,10,20],[float(r[key]) for r in v],label=label,color=color,marker=marker,linestyle=style,
                        markersize=4,linewidth=1.8 if method=='pilotCV-5' else 1.2)
                if index:
                    ax.axhline(.8,color='0.5',linestyle='--',linewidth=.7)
                    ax.axhline(.4,color='0.65',linestyle=':',linewidth=.7)
                ax.set_xticks([5,10,20]);ax.set_xlim(4,21);ax.set_xlabel('Source deployment SNR label (dB)')
                ax.set_title(title,fontsize=11,pad=12)
                ax.annotate(letter,xy=(0,1),xycoords='axes fraction',xytext=(-20,12),textcoords='offset points',fontsize=11,fontweight='bold')
                ax.grid(axis='y',color='0.9',linewidth=.6);ax.set_axisbelow(True)
            axes[0].set_ylabel('Actual validation NMSE (dB)' if index==0 else 'Oracle scalar recovery efficiency')
            fig.suptitle('Uniform two-module pilot-only deployment' if index==0 else 'Pilot-only fitting versus oracle capacity',fontsize=12,y=.98)
            handles,labels=axes[0].get_legend_handles_labels()
            fig.legend(handles,labels,ncol=5 if index==0 else 2,loc='upper center',bbox_to_anchor=(.55,.91),columnspacing=1.1)
            fig.text(.12,.105,'100 original trajectories/condition; paired frame4..9; dF=16',fontsize=9)
            footer=('CV1/CV5: source means; oracle/matched: frozen target means; no CI' if index==0 else
                'Unclipped pooled dB recovery ratio; 5-frame composite loss; no CI')
            fig.text(.12,.045,footer,fontsize=9)
            require_matplotlib_panel_alignment(fig,json_out=str(directory/f'{stem}.alignment.json'),strict=True)
            fig.savefig(directory/f'{stem}.png',dpi=300)
            fig.savefig(directory/f'{stem}.pdf',metadata={'CreationDate':None,'ModDate':None})
            fig.savefig(directory/f'{stem}.svg',metadata={'Date':None});plt.close(fig)
    return STEMS


def audit(input_dir,output_dir):
    directory=Path(output_dir);plot(input_dir,directory);checks={}
    source=subprocess.run([sys.executable,str(SKILL/'validate_figure.py'),str(Path(__file__)),'--json'],capture_output=True,text=True)
    for stem in STEMS:
        text=subprocess.run([sys.executable,str(SKILL/'audit_pdf_text.py'),str(directory/f'{stem}.pdf'),'--min-pt','5','--json'],capture_output=True,text=True)
        collision=subprocess.run([sys.executable,str(SKILL/'audit_figure_collisions.py'),str(directory/f'{stem}.pdf'),'--json-out',str(directory/f'{stem}.collision-audit.json')],capture_output=True,text=True)
        checks[stem]=dict(text_returncode=text.returncode,collision_returncode=collision.returncode,
            text_output=text.stdout+text.stderr,collision_output=collision.stdout+collision.stderr)
    with tempfile.TemporaryDirectory(dir=ROOT/'.pytest_cache') as d:
        temp=Path(d);shutil.copy2(Path(input_dir)/'t16e_pilot_cv_scan.csv',temp)
        plot(temp,temp/'rebuilt')
        rebuilt={f'{stem}.{ext}':sha(directory/f'{stem}.{ext}')==sha(temp/'rebuilt'/f'{stem}.{ext}') for stem in STEMS for ext in ('png','pdf','svg')}
    r=dict(source_returncode=source.returncode,source_output=source.stdout+source.stderr,rendered_checks=checks,
        CSV_only_rebuild=rebuilt,visual_review='pending',all_checks_passed=False,
        figure_contract=dict(backend='python',archetype='quantitative comparison',
            scientific_question='Can label-blind pilot-only fitting realize frozen oracle modular capacity?',
            panel_roles=['C75 mobility interpolation','B120 composed-shift stress test'],
            source_data='t16e_pilot_cv_scan.csv',subset='all methods frame4..9; all10 CV1 preserved in CSV',
            uncertainty='100 independent trajectories; correlated frames; descriptive std saved, no CI',
            diagnostic_excluded=True,exports=['PNG300dpi','editable PDF','editable SVG'],
            adaptation='style-only; matched-subset data mapping and metric guards implemented anew'))
    (directory/'t16e_pilot_cv_figure_audit.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf8')
    if source.returncode or not all(rebuilt.values()) or any(v['text_returncode'] or v['collision_returncode'] for v in checks.values()):
        raise ValueError('figure gates failed; numerical CSVs protected')
    print('T16E source/alignment/text/collision/CSV-only byte rebuild PASS; actual visual review pending')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-dir',default=str(ROOT/'outputs'))
    p.add_argument('--output-dir',default=str(ROOT/'outputs'));p.add_argument('--audit',action='store_true');a=p.parse_args()
    if a.audit:audit(a.input_dir,a.output_dir)
    else:plot(a.input_dir,a.output_dir)
