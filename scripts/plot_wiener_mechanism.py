"""Two T17A CSV-only figures: weighted residual and interaction spectrum."""
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
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.pytest_cache/matplotlib-t17a'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment

SNRS=(5,10,20)
DATA=('t17a_mechanism_closure.csv','t17a_interaction_spectrum.csv')
STEMS=('t17a_weighted_residual','t17a_interaction_energy')
CURVES=(('exact_correction','Exact reference','0.5','o',':'),
        ('first_order_additive','First-order additive','#0072B2','s','--'),
        ('interaction_corrected','Interaction-corrected','#D55E00','D','-'),
        ('second_order_additive','Second-order additive','#CC79A7','^','-.'),
        ('second_order_actual_perturbation','Second-order actual','#009E73','v',':'))
STYLE={'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
    'font.size':10,'legend.fontsize':9,'svg.fonttype':'none','pdf.fonttype':42,
    'svg.hashsalt':'T17A-fixed','axes.spines.top':False,'axes.spines.right':False,
    'legend.frameon':False}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))


def number(row,key):
    value=float(row[key])
    if not math.isfinite(value):raise ValueError('nonfinite figure value: '+key)
    return value


def residual_series(data):
    result={}
    for name,*_ in CURVES:
        selected=sorted((v for v in data if v['prediction']==name),key=lambda v:int(v['snr_db']))
        if [int(v['snr_db']) for v in selected]!=list(SNRS):
            raise ValueError('one row for each frozen SNR and model is required')
        values=[number(v,'relative_weighted_residual') for v in selected]
        if any(v<0 for v in values):raise ValueError('residual norm cannot be negative')
        for row,value in zip(selected,values):
            if not math.isclose(1.-value**2,number(row,'closure'),rel_tol=2e-13,abs_tol=1e-12):
                raise ValueError('residual and unclipped closure must agree')
        result[name]=values
    if result['exact_correction']!=[0.,0.,0.]:raise ValueError('exact reference must remain zero')
    return result


def spectrum_series(data):
    result={}
    for snr in SNRS:
        for name in ('R_exact','R_pred_2'):
            selected=sorted((v for v in data if int(v['snr_db'])==snr and v['operator']==name
                and v['weighting']=='target_weighted'),key=lambda v:int(v['mode']))
            modes=[int(v['mode']) for v in selected]
            if modes!=list(range(1,65)):raise ValueError('all64 modes exactly once required')
            values=[number(v,'cumulative_capture') for v in selected]
            if any(b<a-1e-13 for a,b in zip(values,values[1:])) or not math.isclose(values[-1],1.,abs_tol=1e-12):
                raise ValueError('cumulative squared singular energy must be monotone and complete')
            if any(v<0 or v>1.+1e-12 for v in values):raise ValueError('invalid cumulative energy')
            result[snr,name]=(modes,values)
    return result


def export(fig,stem,directory):
    require_matplotlib_panel_alignment(fig,json_out=str(directory/f'{stem}.alignment.json'),strict=True)
    fig.savefig(directory/f'{stem}.png',dpi=300)
    fig.savefig(directory/f'{stem}.pdf',metadata={'CreationDate':None,'ModDate':None})
    target=directory/f'{stem}.svg'
    fig.savefig(target,metadata={'Date':None})
    # Normalize only new generated SVG XML whitespace; historical figures untouched.
    target.write_text('\n'.join(line.rstrip() for line in target.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8')
    plt.close(fig)


def plot(input_dir,output_dir):
    source,directory=Path(input_dir),Path(output_dir)
    closure,spectra=rows(source/DATA[0]),rows(source/DATA[1])
    residual=residual_series(closure);spectrum=spectrum_series(spectra)
    directory.mkdir(parents=True,exist_ok=True)
    with matplotlib.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(7.2,4.8))
        fig.subplots_adjust(left=.15,right=.97,bottom=.27,top=.72)
        for name,label,color,marker,style in CURVES:
            ax.plot(SNRS,residual[name],label=label,color=color,marker=marker,linestyle=style,
                    linewidth=1.5,markersize=4)
        ax.set_yscale('symlog',linthresh=1.)
        ax.set_xticks(SNRS);ax.set_xlim(4,21)
        ax.set_xlabel('Frozen source deployment SNR label (dB)')
        ax.set_ylabel('Relative B120-weighted residual norm')
        ax.grid(axis='y',color='0.9',linewidth=.6);ax.set_axisbelow(True)
        fig.suptitle('B120 finite-shift Wiener approximation residual',fontsize=12,y=.98)
        handles,labels=ax.get_legend_handles_labels()
        fig.legend(handles,labels,ncol=3,loc='upper center',bbox_to_anchor=(.55,.91),columnspacing=1.2)
        fig.text(.15,.105,'Symlog: linear for residual norm <= 1; exact zero retained; lower is better',fontsize=9)
        fig.text(.15,.045,'Three frozen operators; no CI; closure = 1 - residual norm squared, unclipped',fontsize=9)
        export(fig,STEMS[0],directory)

        fig,axes=plt.subplots(1,3,figsize=(7.2,4.8),sharey=True)
        fig.subplots_adjust(left=.12,right=.97,bottom=.27,top=.73,wspace=.30)
        for ax,snr,letter in zip(axes,SNRS,('a','b','c')):
            for name,label,color,style in (('R_exact','Observed interaction','0.3','-'),
                    ('R_pred_2','Predicted interaction','#D55E00','--')):
                modes,capture=spectrum[snr,name]
                if min(modes)<=0:raise ValueError('positive ranks required for log axis')
                ax.plot(modes,capture,color=color,linestyle=style,label=label,linewidth=1.6)
            ax.set_xscale('log',base=2)
            ax.set_xticks([1,4,8,16,32,64],['1','4','8','16','32','64'])
            ax.set_xlim(1,64);ax.set_ylim(-.02,1.04)
            ax.set_xlabel('Mode index (log2)');ax.set_title(f'{snr} dB',fontsize=11,pad=12)
            ax.annotate(letter,xy=(0,1),xycoords='axes fraction',xytext=(-20,12),
                        textcoords='offset points',fontsize=11,fontweight='bold')
            ax.grid(axis='y',color='0.9',linewidth=.6);ax.set_axisbelow(True)
        axes[0].set_ylabel('Cumulative squared singular energy')
        fig.suptitle('Observed vs predicted interaction subspace energy',fontsize=12,y=.98)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,ncol=2,loc='upper center',bbox_to_anchor=(.55,.91))
        fig.text(.12,.105,'B120 target weighting; predicted interaction = L_interaction + Q_cross',fontsize=9)
        fig.text(.12,.045,'Each operator normalized by its own total energy; all64 modes; no CI',fontsize=9)
        export(fig,STEMS[1],directory)
    return dict(input_rows={DATA[0]:len(closure),DATA[1]:len(spectra)},
        plotted_rows={DATA[0]:15,DATA[1]:384},
        selection='Fig1: five fixed predictions at5/10/20; Fig2: R_exact/R_pred_2 target-weighted, all64 modes')


def audit(input_dir,output_dir):
    start=time.perf_counter();directory=Path(output_dir)
    selection=plot(input_dir,directory)
    source=subprocess.run([sys.executable,str(SKILL/'validate_figure.py'),str(Path(__file__)),'--json'],capture_output=True,text=True)
    checks={}
    for stem in STEMS:
        text=subprocess.run([sys.executable,str(SKILL/'audit_pdf_text.py'),str(directory/f'{stem}.pdf'),
                             '--min-pt','5','--json'],capture_output=True,text=True)
        collision=subprocess.run([sys.executable,str(SKILL/'audit_figure_collisions.py'),str(directory/f'{stem}.pdf'),
            '--json-out',str(directory/f'{stem}.collision-audit.json')],capture_output=True,text=True)
        checks[stem]=dict(text_returncode=text.returncode,collision_returncode=collision.returncode,
            text_output=text.stdout+text.stderr,collision_output=collision.stdout+collision.stderr)
    with tempfile.TemporaryDirectory(dir=ROOT/'.pytest_cache/pytest_workspace_temp') as temp:
        temp=Path(temp)
        for name in DATA:shutil.copy2(Path(input_dir)/name,temp/name)
        plot(temp,temp/'rebuilt')
        rebuilt={f'{stem}.{ext}':sha(directory/f'{stem}.{ext}')==sha(temp/'rebuilt'/f'{stem}.{ext}')
                 for stem in STEMS for ext in ('png','pdf','svg')}
    report=dict(source_returncode=source.returncode,source_output=source.stdout+source.stderr,
        rendered_checks=checks,CSV_only_rebuild=rebuilt,source_data_selection=selection,
        source_data_hashes={name:sha(Path(input_dir)/name) for name in DATA},
        figure_contract=dict(backend='python',archetype='mechanistic comparison and failure boundary',
            question='Do fixed local Wiener terms explain the finite B120 correction and its interaction structure?',
            residual_role='Magnitude of finite-shift approximation error including failures and exact zero',
            spectrum_roles=['5dB boundary','10dB primary condition','20dB primary condition'],
            transform='symlog with linthresh1; no numerical clipping; spectrum ranks positive log2',
            uncertainty='deterministic frozen operators, no CI or sampling variance',
            exports=['PNG300dpi','editable PDF','editable SVG'],final_size_inches=[7.2,4.8],
            adaptation='style-only inheritance; mapping and mathematical guards built anew'),
        runtime_plotting_and_QA_seconds=time.perf_counter()-start,visual_review='pending',all_checks_passed=False,
        reviewed_warnings=['No TIFF: research exports, no journal submission requested'])
    (directory/'t17a_figure_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    if source.returncode or not all(rebuilt.values()) or any(v['text_returncode'] or v['collision_returncode'] for v in checks.values()):
        raise ValueError('T17A figure QA failed; numerical CSVs preserved')
    print('T17A source/alignment/text/collision/CSV-only byte rebuild PASS; actual visual review pending')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',default=str(ROOT/'outputs'));parser.add_argument('--output-dir',default=str(ROOT/'outputs'))
    parser.add_argument('--audit',action='store_true');args=parser.parse_args()
    if args.audit:audit(args.input_dir,args.output_dir)
    else:plot(args.input_dir,args.output_dir)
