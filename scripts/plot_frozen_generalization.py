"""T18A CSV-only interpolation, composed value/geometry and stress figures."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
SKILL=Path('C:/Users/yuhua/.codex/skills/nature-figure/scripts')
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(SKILL))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.pytest_cache/matplotlib-t18a'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_panel_alignment import require_matplotlib_panel_alignment
from src.frozen_affine_validation import INTERPOLATION,EXTRAPOLATION,SNRS
STYLE={'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
    'font.size':10,'legend.fontsize':9,'svg.fonttype':'none','svg.hashsalt':'T18A-fixed',
    'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False}
DATA=('t18a_generalization_nmse.csv','t18a_condition_metrics.csv')
STEMS=('t18a_interpolation_nmse','t18a_composed_gain_geometry','t18a_extrapolation_stress')
METHODS=(('source','Source','0.5','o',':'),('convex','Convex','#E69F00','s','--'),
    ('lr4-affine','LR4 Affine','#0072B2','D','-'),('matched','Matched','#009E73','v','-.'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):
    with Path(path).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
def number(r,k):
    v=float(r[k])
    if not np.isfinite(v):raise ValueError('nonfinite plotted value')
    return v


def series(nmse,conditions):
    perf={};derived={}
    for n in (*INTERPOLATION,*EXTRAPOLATION):
        for s in SNRS:
            for m,*_ in METHODS:
                found=[r for r in nmse if (r['environment'],int(r['snr_db']),r['method'])==(n,s,m)]
                if len(found)!=1 or found[0]['subset']!='frames4..9':raise ValueError('complete primary pooled CSV required')
                perf[n,s,m]=number(found[0],'nmse_db')
            found=[r for r in conditions if (r['environment'],int(r['snr_db']))==(n,s)]
            if len(found)!=1:raise ValueError('unique complete condition CSV required')
            r=found[0];derived[n,s]={k:number(r,k) for k in ('affine_vs_convex_gain_db','lr4_vs_convex_gain_db','outside_simplex_fraction')}
            if not 0<=derived[n,s]['outside_simplex_fraction']<=1:raise ValueError('invalid outside fraction')
            if abs(derived[n,s]['lr4_vs_convex_gain_db']-(perf[n,s,'convex']-perf[n,s,'lr4-affine']))>1e-10:
                raise ValueError('paired LR4 gain formula drift')
    return perf,derived


def export(fig,directory,stem):
    require_matplotlib_panel_alignment(fig,json_out=str(directory/f'{stem}.alignment.json'),strict=True)
    fig.savefig(directory/f'{stem}.png',dpi=300)
    fig.savefig(directory/f'{stem}.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(directory/f'{stem}.svg',metadata={'Date':None});plt.close(fig)


def scaffold(title):
    fig,axes=plt.subplots(1,2,figsize=(7.2,4.8))
    fig.subplots_adjust(left=.12,right=.97,bottom=.29,top=.71,wspace=.37)
    fig.suptitle(title,fontsize=12,y=.98)
    for ax,letter in zip(axes,('a','b')):
        ax.annotate(letter,(0,1),xycoords='axes fraction',xytext=(-18,12),textcoords='offset points',fontweight='bold',fontsize=11)
        ax.grid(axis='y',color='0.9',linewidth=.6);ax.set_axisbelow(True)
    return fig,axes


def plot(input_dir,output_dir):
    input_dir,directory=Path(input_dir),Path(output_dir);directory.mkdir(parents=True,exist_ok=True)
    nmse=rows(input_dir/DATA[0]);conditions=rows(input_dir/DATA[1]);perf,derived=series(nmse,conditions)
    with matplotlib.rc_context(STYLE):
        fig,axes=scaffold('Frozen rank4 affine generalization: interpolation')
        for ax,names,title in zip(axes,(INTERPOLATION[:2],INTERPOLATION[2:]),('Mobility: C60 / C90','Composed: B60 / B90')):
            keys=[(n,s) for n in names for s in (10,20)]
            for m,label,color,marker,style in METHODS:
                ax.plot(range(4),[perf[n,s,m] for n,s in keys],color=color,marker=marker,linestyle=style,label=label,linewidth=1.8 if m=='lr4-affine' else 1.2,markersize=4)
            ax.set_xticks(range(4),[n.replace('CDL-','').replace('-','')+'\n'+str(s) for n,s in keys],fontsize=9)
            ax.set_xlabel('Environment / source SNR label (dB)',fontsize=9);ax.set_title(title,fontsize=10,pad=12)
        axes[0].set_ylabel('Pooled data-RE NMSE (dB)')
        lo=min(perf[n,s,m] for n in INTERPOLATION for s in (10,20) for m,*_ in METHODS)
        hi=max(perf[n,s,m] for n in INTERPOLATION for s in (10,20) for m,*_ in METHODS)
        for ax in axes:ax.set_ylim(lo-1,hi+1)
        fig.legend(*axes[0].get_legend_handles_labels(),ncol=4,loc='upper center',bbox_to_anchor=(.55,.91))
        fig.text(.12,.12,'100 independent trajectories/environment; paired causal frames4..9',fontsize=9)
        fig.text(.12,.055,'Source means; Matched: target means. Pooled estimates without CI.',fontsize=9)
        export(fig,directory,STEMS[0])

        fig,axes=scaffold('Composed interpolation: value and deployed weight geometry')
        keys=[(n,s) for n in INTERPOLATION[2:] for s in (10,20)];x=np.arange(4)
        for offset,field,label,color in ((-.18,'affine_vs_convex_gain_db','Dense Affine','0.5'),(.18,'lr4_vs_convex_gain_db','LR4 Affine','#0072B2')):
            axes[0].bar(x+offset,[derived[n,s][field] for n,s in keys],width=.34,label=label,color=color)
        axes[0].axhline(1,color='0.35',linestyle=':',linewidth=1)
        axes[0].set_ylabel('Gain over Convex (dB)');axes[0].set_title('Paired pooled NMSE gain',fontsize=10,pad=12)
        axes[1].bar(x,[100*derived[n,s]['outside_simplex_fraction'] for n,s in keys],width=.6,color='#0072B2')
        axes[1].axhline(75,color='0.35',linestyle=':',linewidth=1)
        axes[1].set_ylim(0,110);axes[1].set_ylabel('Outside-simplex windows (%)');axes[1].set_title('Pilot-CV optimum geometry',fontsize=10,pad=12)
        for ax in axes:
            ax.set_xticks(x,[n.replace('CDL-','').replace('-','')+'\n'+str(s) for n,s in keys],fontsize=9)
            ax.set_xlabel('Environment / source SNR label (dB)',fontsize=9)
        fig.legend(*axes[0].get_legend_handles_labels(),ncol=2,loc='upper center',bbox_to_anchor=(.55,.91))
        fig.text(.12,.12,'Dashed references: 1 dB value / 75% outside; all 600 windows retained.',fontsize=9)
        fig.text(.12,.055,'Unclipped gains; geometry is descriptive. No CI or causal claim.',fontsize=9)
        export(fig,directory,STEMS[1])

        fig,axes=scaffold('Frozen pipeline: 180 km/h extrapolation stress')
        for ax,n in zip(axes,EXTRAPOLATION):
            for m,label,color,marker,style in METHODS:
                ax.plot(SNRS,[perf[n,s,m] for s in SNRS],color=color,marker=marker,linestyle=style,label=label,linewidth=1.8 if m=='lr4-affine' else 1.2,markersize=4)
            ax.set_xticks(SNRS);ax.set_xlabel('Source deployment SNR label (dB)',fontsize=9)
            ax.set_title(n.replace('CDL-','').replace('-',' / ')+' km/h',fontsize=10,pad=12)
        axes[0].set_ylabel('Pooled data-RE NMSE (dB)')
        lo=min(perf[n,s,m] for n in EXTRAPOLATION for s in SNRS for m,*_ in METHODS)
        hi=max(perf[n,s,m] for n in EXTRAPOLATION for s in SNRS for m,*_ in METHODS)
        for ax in axes:ax.set_ylim(lo-1,hi+1)
        fig.legend(*axes[0].get_legend_handles_labels(),ncol=4,loc='upper center',bbox_to_anchor=(.55,.91))
        fig.text(.12,.12,'Paired frames4..9; pooled NMSE without CI; 5 dB is diagnostic only.',fontsize=9)
        fig.text(.12,.055,'Current symbol-wise CDL abstraction: within-symbol ICI excluded.',fontsize=9)
        export(fig,directory,STEMS[2])
    return dict(input_nmse_rows=len(nmse),input_condition_rows=len(conditions),
        figure1_rows=32,figure2_conditions=4,figure3_rows=24,
        exclusion='Only requested figure subsets plotted; all 18 conditions and six deployed/reference methods retained in CSV.')


def audit(input_dir,output_dir):
    directory=Path(output_dir);selection=plot(input_dir,directory)
    source=subprocess.run([sys.executable,str(SKILL/'validate_figure.py'),str(Path(__file__)),'--json'],capture_output=True,text=True)
    checks={}
    for stem in STEMS:
        text=subprocess.run([sys.executable,str(SKILL/'audit_pdf_text.py'),str(directory/f'{stem}.pdf'),'--min-pt','5','--json'],capture_output=True,text=True)
        collision=subprocess.run([sys.executable,str(SKILL/'audit_figure_collisions.py'),str(directory/f'{stem}.pdf'),'--json-out',str(directory/f'{stem}.collision-audit.json')],capture_output=True,text=True)
        checks[stem]=dict(text_returncode=text.returncode,collision_returncode=collision.returncode,
            text_output=text.stdout+text.stderr,collision_output=collision.stdout+collision.stderr)
    with tempfile.TemporaryDirectory(dir=ROOT/'.pytest_cache') as temp:
        temp=Path(temp)
        for f in DATA:shutil.copy2(Path(input_dir)/f,temp/f)
        plot(temp,temp/'rebuilt')
        rebuilt={f'{stem}.{ext}':sha(directory/f'{stem}.{ext}')==sha(temp/'rebuilt'/f'{stem}.{ext}') for stem in STEMS for ext in ('png','pdf','svg')}
    report=dict(source_returncode=source.returncode,source_output=source.stdout+source.stderr,rendered_checks=checks,
        CSV_only_rebuild=rebuilt,visual_review='pending',all_checks_passed=False,source_data_selection=selection,
        source_data_hashes={f:sha(Path(input_dir)/f) for f in DATA},
        figure_contract=dict(backend='python',archetype='quantitative grid',dimensions_inches=[7.2,4.8],
            panels=['C mobility vs B composed interpolation','Paired gain vs deployed geometry','C180 vs B180 stress'],
            uncertainty='pooled NMSE/dB differences, no CI; window fraction descriptive',
            independent_unit='trajectory; 100 per environment',source_data=list(DATA),
            exports=['PNG300dpi','editable PDF','editable SVG'],adaptation='style-only inheritance; new guarded field mapping'),
        reviewed_warnings=['PNG300dpi/no TIFF: research export, no journal submission requested'])
    (directory/'t18a_figure_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    if source.returncode or not all(rebuilt.values()) or any(v['text_returncode'] or v['collision_returncode'] for v in checks.values()):
        raise ValueError('T18A figure automatic gate failed')
    print('T18A figure source/alignment/text/collision and nine CSV-only byte rebuilds PASS; visual review pending',flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-dir',default=str(ROOT/'outputs'))
    p.add_argument('--output-dir',default=str(ROOT/'outputs'));p.add_argument('--audit',action='store_true');a=p.parse_args()
    if a.audit:audit(a.input_dir,a.output_dir)
    else:plot(a.input_dir,a.output_dir)
