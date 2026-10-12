"""T19A two CSV-only descriptive factor-challenge figures."""
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
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.pytest_cache/matplotlib-t19a'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_panel_alignment import require_matplotlib_panel_alignment
from src.delay_factor_challenge import ENVIRONMENTS,PURE,SNRS
STYLE={'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
    'font.size':9,'legend.fontsize':9,'svg.fonttype':'none','svg.hashsalt':'T19A-fixed',
    'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False}
DATA=('t19a_factor_nmse.csv','t19a_factor_metrics.csv')
STEMS=('t19a_pure_delay_nmse','t19a_representation_coordinate')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):
    with Path(path).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))


def series(directory):
    nmse,metrics=[rows(Path(directory)/f) for f in DATA]
    expected={(n,s) for n in ENVIRONMENTS for s in SNRS}
    index={(r['environment'],int(r['snr_db'])):r for r in metrics}
    if len(metrics)!=16 or set(index)!=expected:raise ValueError('all sixteen factor conditions required')
    methods=('source','oracle2-affine','dense-affine','lr4-affine','matched')
    pure={};points={};invalid=[]
    for n,s in expected:
        row=index[n,s]
        if row['representation_closure']=='' or row['coordinate_efficiency']=='':
            if row['ratio_anomaly']!='True':raise ValueError('undefined ratios need explicit reason')
            invalid.append(dict(environment=n,snr_db=s,reason=row['ratio_anomaly_reasons']))
        else:
            xy=tuple(float(row[k]) for k in ('representation_closure','coordinate_efficiency'))
            if not np.isfinite(xy).all():raise ValueError('nonfinite figure point')
            points[n,s]=xy
        if n in PURE:
            for m in methods:
                found=[v for v in nmse if (v['environment'],int(v['snr_db']),v['method'])==(n,s,m)]
                if len(found)!=1:raise ValueError('five paired pure-delay method values required')
                value=float(found[0]['nmse_db'])
                if not np.isfinite(value):raise ValueError('nonfinite NMSE')
                pure[n,s,m]=value
    return pure,points,sorted(invalid,key=lambda r:(r['environment'],r['snr_db']))


def export(fig,directory,stem):
    require_matplotlib_panel_alignment(fig,json_out=str(directory/f'{stem}.alignment.json'),strict=True)
    fig.savefig(directory/f'{stem}.png',dpi=300)
    fig.savefig(directory/f'{stem}.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(directory/f'{stem}.svg',metadata={'Date':None});plt.close(fig)


def plot(input_dir,output_dir):
    directory=Path(output_dir);directory.mkdir(parents=True,exist_ok=True)
    pure,points,invalid=series(input_dir)
    methods=('source','oracle2-affine','dense-affine','lr4-affine','matched')
    labels=('Source','Oracle2','Dense','LR4','Matched')
    colors=('0.55','#9467BD','#0072B2','#E69F00','#009E73')
    conditions=[(n,s) for n in PURE for s in SNRS]
    with matplotlib.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(7.2,4.8));fig.subplots_adjust(left=.14,right=.96,bottom=.24,top=.78)
        for j,(m,label,color) in enumerate(zip(methods,labels,colors)):
            ax.plot(np.arange(4)+(j-2)*.13,[pure[n,s,m] for n,s in conditions],
                marker='o',linestyle='none',color=color,markersize=6,label=label)
        ax.set_xticks(range(4),[f'C30 / {n.split("-")[-1]} ns\n{s} dB' for n,s in conditions]);ax.set_xlim(-.6,3.6)
        ax.set_ylabel('Data-RE pooled NMSE (dB)');ax.set_xlabel('Pure delay-factor condition',labelpad=9)
        fig.suptitle('Frozen P/M representation and deployed inference under delay shifts',y=.96,fontsize=11)
        fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.55,.90),ncol=5)
        fig.text(.14,.09,'100 independent validation trajectories per environment; frames4–9.',fontsize=9)
        fig.text(.14,.045,'Pooled error/truth energy; one namespace, no replication or confidence intervals.',fontsize=9)
        export(fig,directory,STEMS[0])
    with matplotlib.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(7.2,4.8));fig.subplots_adjust(left=.14,right=.70,bottom=.24,top=.84)
        palette=('#0072B2','#56B4E9','#009E73','#66AA55','#E69F00','#AA7722','#9467BD','#CC79A7')
        for n,color in zip(ENVIRONMENTS,palette):
            for s in SNRS:
                if (n,s) in points:
                    x,y=points[n,s];ax.scatter(x,y,color=color,marker='o' if s==10 else 's',s=42)
            ax.scatter([],[],color=color,s=42,label=n.replace('CDL-','').replace('-', ' / '))
        ax.axvline(.70,color='0.65',linestyle='--',linewidth=.8);ax.axhline(.80,color='0.65',linestyle='--',linewidth=.8)
        ax.set_xlabel('Representation closure');ax.set_ylabel('Coordinate efficiency')
        ax.legend(loc='upper left',bbox_to_anchor=(1.04,1.02),borderaxespad=0,title='Profile / km/h / ns')
        fig.suptitle('Representation capacity versus pilot-only coordinate inference',y=.96,fontsize=11)
        fig.text(.14,.09,f'Circles:10dB; square markers:20dB. All16 conditions; undefined ratios:{len(invalid)}.',fontsize=9)
        fig.text(.14,.045,'Dashed:0.70 representation /0.80 coordinate thresholds; no clipping or CI.',fontsize=9)
        if invalid:
            fig.text(.14,.155,'Undefined: '+', '.join(r['environment'].replace('CDL-','')+'/'+str(r['snr_db']) for r in invalid),fontsize=7)
        export(fig,directory,STEMS[1])
    return dict(nmse_points=20,ratio_conditions=16,ratio_points=len(points),undefined_conditions=invalid,
        selection='Figure1:requested four pure-delay conditions; Figure2:all16, invalid ratios explicitly retained.')


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
        figure_contract=dict(backend='python',archetype='quantitative grid',dimensions_inches=[7.2,4.8],
            exports=['PNG300dpi','editable PDF','editable SVG'],glyph_floor_pt=5,uncertainty='none:discovery gate,100 trajectories and no replication/bootstrap',
            adaptation='style-only inheritance; different failure-localization statistical mappings implemented anew'),
        visual_review='pending',all_checks_passed=False,reviewed_warnings=['Research PNG300dpi/no TIFF; no journal submission requested. No sampling uncertainty is claimed.'])
    (directory/'t19a_figure_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    if source.returncode or not all(rebuilt.values()) or any(v['text_returncode'] or v['collision_returncode'] for v in checks.values()):raise ValueError('figure QA failed')
    print('T19A source/alignment/text/collision and six CSV-only byte rebuilds PASS; actual review pending',flush=True)
    return report


def calibrate():
    """Isolated synthetic schema/layout fixture; never a scientific result."""
    from scripts.run_cdl_mismatch import csvsave,save
    from src.delay_factor_challenge import METHODS,metric_columns
    directory=ROOT/'outputs/t19a_figure_calibration'
    if directory.exists():raise ValueError('existing calibration protected')
    directory.mkdir();nmse=[];metrics=[]
    for j,n in enumerate(ENVIRONMENTS):
        for s in SNRS:
            v=dict(zip(METHODS,[0.,-.5,-1.,-2.,-1.8,-4.,-3.]))
            v={m:x-j*.1-(s-10)/10 for m,x in v.items()}
            v['oracle2-affine']=v['source']-(1.2+j*.35+(s-10)*.05)
            v['dense-affine']=v['source']-(.5+j*.25+(s-10)*.04)
            nmse.extend(dict(environment=n,snr_db=s,method=m,nmse_db=x) for m,x in v.items())
            metrics.append(dict(environment=n,snr_db=s,**metric_columns(v)))
    csvsave(directory/DATA[0],nmse);csvsave(directory/DATA[1],metrics)
    report=audit(directory,directory)
    save(ROOT/'outputs/t19a_fixture_figure_gate.json',dict(all_checks_passed=True,
        synthetic_fixture_only=True,scientific_data_used=False,production_outputs_written=False,
        fixture_directory='outputs/t19a_figure_calibration',automatic_QA=report))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input-dir',default=str(ROOT/'outputs'));parser.add_argument('--output-dir',default=str(ROOT/'outputs'));parser.add_argument('--audit',action='store_true');parser.add_argument('--calibrate',action='store_true');args=parser.parse_args()
    if args.calibrate:calibrate()
    elif args.audit:audit(args.input_dir,args.output_dir)
    else:plot(args.input_dir,args.output_dir)
