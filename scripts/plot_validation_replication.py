"""T18B source-CSV-only paired benefit and interpolation uncertainty figures."""
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
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.pytest_cache/matplotlib-t18b'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_panel_alignment import require_matplotlib_panel_alignment
from src.validation_replication import ENVIRONMENTS,SNRS,REPLICATES
STYLE={'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
    'font.size':9,'legend.fontsize':9,'svg.fonttype':'none','svg.hashsalt':'T18B-fixed',
    'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False}
DATA=('t18b_replicate_metrics.csv','t18b_pooled_metrics.csv','t18b_bootstrap_ci.csv','t18a_condition_metrics.csv')
STEMS=('t18b_composed_replication','t18b_interpolation_uncertainty')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):
    with Path(path).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
def number(r,k):
    v=float(r[k])
    if not np.isfinite(v):raise ValueError('nonfinite figure value')
    return v


def series(directory):
    rr,pp,cc,aa=[rows(Path(directory)/f) for f in DATA]
    replicate={};points={};intervals={};prior={}
    for n in ENVIRONMENTS:
        for s in SNRS:
            for r in REPLICATES:
                found=[v for v in rr if (v['environment'],int(v['snr_db']),int(v['replicate']))==(n,s,r)]
                if len(found)!=1:raise ValueError('all40 independent replicate metrics required')
                replicate[n,s,r]=number(found[0],'lr4_vs_convex_gain_db')
            found=[v for v in pp if (v['environment'],int(v['snr_db']))==(n,s)]
            if len(found)!=1:raise ValueError('all8 pooled conditions required')
            for metric in ('lr4_vs_convex_gain_db','dense_affine_closure','compression_penalty_db'):
                points[n,s,metric]=number(found[0],metric)
                matched=[v for v in cc if (v['environment'],int(v['snr_db']),v['metric'])==(n,s,metric)]
                if len(matched)!=1 or matched[0]['bootstrap_unit']!='trajectory' or int(matched[0]['bootstrap_samples'])!=10000:
                    raise ValueError('paired10000 trajectory CI required')
                v=matched[0];lo=number(v,'lower_2_5');hi=number(v,'upper_97_5')
                if lo>hi or number(v,'point_estimate')!=points[n,s,metric]:raise ValueError('invalid/mismatched plotted CI')
                intervals[n,s,metric]=(lo,hi)
            priorrows=[v for v in aa if (v['environment'],int(v['snr_db']))==(n,s)]
            if len(priorrows)!=1:raise ValueError('frozen T18A external prior required')
            prior[n,s]=number(priorrows[0],'lr4_vs_convex_gain_db')
    return replicate,points,intervals,prior


def export(fig,directory,stem):
    require_matplotlib_panel_alignment(fig,json_out=str(directory/f'{stem}.alignment.json'),strict=True)
    fig.savefig(directory/f'{stem}.png',dpi=300)
    fig.savefig(directory/f'{stem}.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(directory/f'{stem}.svg',metadata={'Date':None});plt.close(fig)


def plot(input_dir,output_dir):
    directory=Path(output_dir);directory.mkdir(parents=True,exist_ok=True)
    rep,point,interval,prior=series(input_dir)
    composed=[(n,s) for n in ENVIRONMENTS if n.startswith('CDL-B') for s in SNRS]
    with matplotlib.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(7.2,4.8));fig.subplots_adjust(left=.14,right=.96,bottom=.23,top=.76)
        ax.axhline(0,color='0.65',linestyle='--',linewidth=.8)
        for x,(n,s) in enumerate(composed):
            ax.scatter(x-.28,prior[n,s],color='#E69F00',marker='s',s=35,label='T18A Prior' if x==0 else None)
            ax.scatter(x+np.linspace(-.18,.18,5),[rep[n,s,r] for r in REPLICATES],color='0.55',s=20,label='New Replicates' if x==0 else None)
            center=point[n,s,'lr4_vs_convex_gain_db'];lo,hi=interval[n,s,'lr4_vs_convex_gain_db']
            ax.plot([x+.30,x+.30],[lo,hi],color='#0072B2',linewidth=1.5)
            ax.errorbar(x+.30,center,yerr=None,color='#0072B2',marker='D',markersize=5,linestyle='none',label='Pooled + 95% CI' if x==0 else None)
        ax.set_xticks(range(4),[f'{n.replace("CDL-","").replace("-","")} / {s} dB' for n,s in composed]);ax.set_xlim(-.6,3.7)
        ax.set_ylabel('LR4 Affine gain over Convex (dB)');ax.set_xlabel('Fixed composed condition',labelpad=10)
        fig.suptitle('Independent replication of positive composed-shift benefit',y=.96,fontsize=11)
        fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.55,.90),ncol=3)
        fig.text(.14,.09,'Five new namespaces ×100 trajectories; pooled sample n=500 per condition.',fontsize=9)
        fig.text(.14,.045,'Paired trajectory bootstrap:10,000 draws,95% percentile CI; T18A excluded.',fontsize=9)
        export(fig,directory,STEMS[0])
    conditions=[(n,s) for n in ENVIRONMENTS for s in SNRS]
    with matplotlib.rc_context(STYLE):
        fig,axes=plt.subplots(2,1,figsize=(7.2,4.8));fig.subplots_adjust(left=.14,right=.96,bottom=.24,top=.84,hspace=.60)
        fig.suptitle('Frozen interpolation performance and compression uncertainty',y=.96,fontsize=11)
        for ax,metric,threshold,label in zip(axes,('dense_affine_closure','compression_penalty_db'),(.70,.50),('Dense closure','LR4 minus Dense (dB)')):
            centers=[point[n,s,metric] for n,s in conditions];bounds=np.array([interval[n,s,metric] for n,s in conditions])
            for x,c,(lo,hi) in zip(range(8),centers,bounds):
                ax.plot([x,x],[lo,hi],color='#0072B2',linewidth=1.2)
                ax.plot(x,c,marker='D',color='#0072B2',markersize=4)
            ax.axhline(threshold,color='0.5',linestyle='--',linewidth=.9)
            ax.set_xticks(range(8),[n.replace('CDL-','').replace('-','')+'\n'+str(s)+' dB' for n,s in conditions]);ax.set_xlim(-.5,7.5)
            ax.set_ylabel(label);ax.text(-.025,1.04,'a' if metric=='dense_affine_closure' else 'b',transform=ax.transAxes,fontweight='bold',fontsize=10)
        fig.text(.14,.09,'New pooled n=500 trajectories/condition; paired95% percentile bootstrap CI.',fontsize=9)
        fig.text(.14,.045,'Dashed: original closure>=0.70 / penalty<=0.50dB thresholds; fixed references.',fontsize=9)
        export(fig,directory,STEMS[1])
    return dict(conditions=8,replicate_conditions=40,composed_conditions=4,composed_replicate_points=20,
        exclusion='Only requested four composed conditions appear in Figure1; all8 conditions/40 replicates retained in CSV.')


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
        CSV_only_rebuild=rebuilt,source_data_hashes={f:sha(Path(input_dir)/f) for f in DATA},source_data_selection=selection,
        visual_review='pending',all_checks_passed=False,
        figure_contract=dict(backend='python',archetype='quantitative grid',dimensions_inches=[7.2,4.8],
            exports=['PNG300dpi','editable PDF','editable SVG'],glyph_floor_pt=5,
            uncertainty='95% percentile paired trajectory bootstrap,10000 draws; conditional fixed-environment inference',
            target_statistics_uncertainty=False,adaptation='style-only inheritance; all statistical mapping implemented anew'),
        reviewed_warnings=['Research export PNG300dpi/no TIFF; no journal submission requested. Marginal, not simultaneous,95% intervals.'])
    (directory/'t18b_figure_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    if source.returncode or not all(rebuilt.values()) or any(v['text_returncode'] or v['collision_returncode'] for v in checks.values()):raise ValueError('T18B automatic figure gate failed')
    print('T18B two figures: source/alignment/text/collision and six CSV-only byte rebuilds PASS; visual review pending',flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input-dir',default=str(ROOT/'outputs'));parser.add_argument('--output-dir',default=str(ROOT/'outputs'));parser.add_argument('--audit',action='store_true');args=parser.parse_args()
    if args.audit:audit(args.input_dir,args.output_dir)
    else:plot(args.input_dir,args.output_dir)
