"""Three T19B CSV-only figures: capacity, budget decomposition and direction reuse."""
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
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.pytest_cache/matplotlib-t19b'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_panel_alignment import require_matplotlib_panel_alignment
from src import operator_space_audit as p
STYLE={'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
       'font.size':9,'legend.fontsize':9,'svg.fonttype':'none','svg.hashsalt':'T19B-fixed',
       'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False}
DATA=('t19b_operator_span_capture.csv','t19b_rank4_span.csv','t19b_exact_mismatch_budget.csv',
      't19b_delay_residual_similarity.csv','t19b_delay_direction_capture.csv')
STEMS=('t19b_pure_delay_capacity','t19b_pure_delay_budget','t19b_delay_direction_reuse')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))


def series(directory):
    result=[]
    for j,name in enumerate(DATA):
        data=rows(Path(directory)/name)
        index={(r['environment'],int(r['snr_db'])):r for r in data}
        expected={(n,s) for n in (p.ENVIRONMENTS if j<3 else p.HELDOUT) for s in p.SNRS}
        if len(data)!=len(expected) or set(index)!=expected:raise ValueError('complete unique CSV population required:'+name)
        result.append(index)
    return result


def export(fig,directory,stem):
    require_matplotlib_panel_alignment(fig,json_out=str(directory/f'{stem}.alignment.json'),strict=True)
    fig.savefig(directory/f'{stem}.png',dpi=300)
    fig.savefig(directory/f'{stem}.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(directory/f'{stem}.svg',metadata={'Date':None});plt.close(fig)


def values(index,conditions,field):
    out=[]
    for condition in conditions:
        cell=index[condition][field]
        if cell=='':raise ValueError('undefined figure metric: preserve explicit diagnostic; do not silently drop')
        value=float(cell)
        if not np.isfinite(value):raise ValueError('nonfinite figure metric')
        out.append(value)
    return out


def plot(input_dir,output_dir):
    directory=Path(output_dir);directory.mkdir(parents=True,exist_ok=True)
    dense,rank,budget,similarity,capture=series(input_dir)
    conditions=[(n,s) for n in p.PURE for s in p.SNRS]
    with matplotlib.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(7.4,4.8));fig.subplots_adjust(left=.13,right=.96,bottom=.24,top=.78)
        for j,(index,field,label,color) in enumerate(((dense,'historical_validation_representation_closure','T19A validation closure','0.5'),
                (dense,'operator_span_capture_dense','Exact dense capture','#0072B2'),(rank,'operator_span_capture_lr4','Exact rank4 capture','#E69F00'))):
            ax.plot(np.arange(4)+(j-1)*.15,values(index,conditions,field),linestyle='none',marker='o',color=color,markersize=6,label=label)
        ax.axhline(.70,color='0.65',linestyle='--',linewidth=.8)
        ax.axhline(.50,color='0.65',linestyle=':',linewidth=.8)
        ax.set_xticks(range(4),[f'C30 / {n.split("-")[-1]} ns\n{s} dB' for n,s in conditions])
        ax.set_ylabel('Closure / exact operator-span capture');ax.set_xlim(-.5,3.5)
        fig.suptitle('Pure-delay validation closure versus exact operator capacity',y=.96,fontsize=11)
        fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.55,.90),ncol=3)
        fig.text(.13,.09,'Exact capture uses target covariance; validation closure uses frozen pooled NMSE.',fontsize=9)
        fig.text(.13,.045,'No clipping; dashed/dotted lines: preregistered 0.70 / 0.50 dense thresholds.',fontsize=9)
        export(fig,directory,STEMS[0])
    with matplotlib.rc_context(STYLE):
        fig,axes=plt.subplots(1,2,figsize=(8.8,4.8),sharey=True);fig.subplots_adjust(left=.11,right=.97,bottom=.28,top=.76,wspace=.12)
        for ax,s in zip(axes,p.SNRS):
            for i,n in enumerate(p.PURE):
                v=budget[n,s];den=float(v['E_total_source'])
                if den<=0:raise ValueError('positive source total required for normalized budget')
                for j,(op,mean) in enumerate((('E_op_source','E_mean_source'),('E_op_oracle2','E_mean_oracle2'),('matched_control_operator_excess','matched_operator_source_mean_floor'))):
                    x=i*4+j
                    eo=float(v[op])/den;em=float(v[mean])/den
                    ax.bar(x,eo,width=.68,color='#0072B2',label='Operator excess' if i==j==0 else None)
                    ax.bar(x,em,bottom=eo,width=.68,color='#CC79A7',label='Mean-bias excess' if i==j==0 else None)
            ax.set_xticks([0,1,2,4,5,6],['Src','O2','Mt','Src','O2','Mt']);ax.set_title(f'{s} dB',pad=8,fontsize=10)
            ax.text(.22,-.18,'C30 / 50 ns',transform=ax.transAxes,ha='center',fontsize=9)
            ax.text(.77,-.18,'C30 / 200 ns',transform=ax.transAxes,ha='center',fontsize=9)
        axes[0].set_ylabel('Excess / source total excess (same condition)')
        fig.suptitle('Operator and source-mean bias contributions to pure-delay mismatch',y=.96,fontsize=11)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.55,.90),ncol=2)
        fig.text(.11,.07,'Src: source operator; O2: operator-only Oracle2; Mt: matched operator with source means.',fontsize=9)
        fig.text(.11,.025,'All bars use one fixed source-total denominator per condition; not absolute expected MSE.',fontsize=9)
        export(fig,directory,STEMS[1])
    with matplotlib.rc_context(STYLE):
        fig,axes=plt.subplots(2,1,figsize=(8.3,6.2),sharex=True);fig.subplots_adjust(left=.14,right=.97,bottom=.24,top=.84,hspace=.30)
        for s,color,marker in ((10,'#0072B2','o'),(20,'#E69F00','s')):
            held=[(n,s) for n in p.HELDOUT]
            axes[0].plot(range(7),values(similarity,held,'cos_delay'),color=color,marker=marker,label=f'{s} dB')
            axes[1].plot(range(7),values(capture,held,'delay_direction_incremental_capture'),color=color,marker=marker)
        axes[0].set_ylabel('Signed source-weighted cosine');axes[0].axhline(0,color='0.65',linewidth=.8)
        axes[1].set_ylabel('Incremental joint 3D capture');axes[1].axhline(0,color='0.65',linewidth=.8)
        axes[1].set_xticks(range(7),[n.replace('CDL-','').replace('-', '\n',1).replace('-', ' / ') for n in p.HELDOUT])
        axes[1].set_xlabel('Held-out profile / km/h / ns',labelpad=8)
        fig.suptitle('Held-out reuse of the C30 / 200 ns residual direction',y=.97,fontsize=11)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.55,.925),ncol=2)
        fig.text(.14,.075,'Top: common source weighting per SNR. Bottom: joint real 3-variable target-weighted LS.',fontsize=9)
        fig.text(.14,.025,'Self-condition excluded; 14 held-out rows shown. Oracle diagnostic; no deployment or CI.',fontsize=9)
        export(fig,directory,STEMS[2])
    return dict(pure_conditions=4,heldout_conditions=14,self_condition_excluded=True,
                budget_normalization='source E_total per condition for all three controls',figure_count=3)


def audit(input_dir,output_dir):
    directory=Path(output_dir);selection=plot(input_dir,directory)
    source=subprocess.run([sys.executable,str(SKILL/'validate_figure.py'),str(Path(__file__)),'--json'],capture_output=True,text=True)
    checks={}
    for stem in STEMS:
        text=subprocess.run([sys.executable,str(SKILL/'audit_pdf_text.py'),str(directory/f'{stem}.pdf'),'--min-pt','5','--json'],capture_output=True,text=True)
        collision=subprocess.run([sys.executable,str(SKILL/'audit_figure_collisions.py'),str(directory/f'{stem}.pdf'),'--json-out',str(directory/f'{stem}.collision-audit.json')],capture_output=True,text=True)
        checks[stem]=dict(text_returncode=text.returncode,collision_returncode=collision.returncode,text_output=text.stdout+text.stderr,collision_output=collision.stdout+collision.stderr)
    with tempfile.TemporaryDirectory(dir=ROOT/'.pytest_cache') as temp:
        temp=Path(temp)
        for f in DATA:shutil.copy2(Path(input_dir)/f,temp/f)
        plot(temp,temp/'rebuilt')
        rebuilt={f'{stem}.{ext}':sha(directory/f'{stem}.{ext}')==sha(temp/'rebuilt'/f'{stem}.{ext}') for stem in STEMS for ext in ('png','pdf','svg')}
    report=dict(source_returncode=source.returncode,source_output=source.stdout+source.stderr,rendered_checks=checks,
                CSV_only_rebuild=rebuilt,source_data_hashes={f:sha(Path(input_dir)/f) for f in DATA},selection=selection,
                figure_contract=dict(backend='python',exports=['PNG300dpi','editable PDF','editable SVG'],glyph_floor_pt=5,
                    uncertainty='no sampling CI for exact frozen-statistics matrix diagnostics',
                    adaptation='style-only inheritance; T19B mappings freshly implemented',
                    panel_roles={'capacity':'empirical versus exact capacity comparison','budget':'operator/mean decomposition controls',
                                 'direction_top':'common-metric alignment','direction_bottom':'held-out joint-oracle utility'}),
                visual_review='pending',all_checks_passed=False)
    (directory/'t19b_figure_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    if source.returncode or not all(rebuilt.values()) or any(v['text_returncode'] or v['collision_returncode'] for v in checks.values()):
        raise ValueError('figure source/render/rebuild calibration failed')
    print('T19B source/alignment/text/collision and nine CSV-only byte rebuilds PASS; visual review pending',flush=True)
    return report


def calibrate():
    from scripts.run_cdl_mismatch import csvsave,save
    directory=ROOT/'outputs/t19b_figure_calibration'
    if directory.exists():raise ValueError('existing calibration protected')
    directory.mkdir();tables=[[] for _ in DATA]
    for j,n in enumerate(p.ENVIRONMENTS):
        for s in p.SNRS:
            key=dict(environment=n,snr_db=s)
            tables[0].append(dict(**key,historical_validation_representation_closure=.3+.03*j,operator_span_capture_dense=.4+.02*j))
            tables[1].append(dict(**key,operator_span_capture_lr4=.35+.02*j))
            tables[2].append(dict(**key,E_op_source=10.,E_mean_source=.2,E_total_source=10.2,E_op_oracle2=6.,E_mean_oracle2=.1,
                                  matched_control_operator_excess=0.,matched_operator_source_mean_floor=.1))
            if n in p.HELDOUT:
                tables[3].append(dict(**key,cos_delay=(-.1+.12*j)+(s-10)*.005))
                tables[4].append(dict(**key,delay_direction_incremental_capture=.02+.03*j+(s-10)*.001))
    for name,table in zip(DATA,tables):csvsave(directory/name,table)
    report=audit(directory,directory)
    save(ROOT/'outputs/t19b_fixture_figure_gate.json',dict(all_checks_passed=True,synthetic_fixture_only=True,
        scientific_data_used=False,production_outputs_written=False,automatic_QA=report))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input-dir',default=str(ROOT/'outputs'))
    parser.add_argument('--output-dir',default=str(ROOT/'outputs'));parser.add_argument('--audit',action='store_true');parser.add_argument('--calibrate',action='store_true');args=parser.parse_args()
    if args.calibrate:calibrate()
    elif args.audit:audit(args.input_dir,args.output_dir)
    else:plot(args.input_dir,args.output_dir)
