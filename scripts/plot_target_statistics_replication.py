"""T19C two CSV-only figures: independent-statistics captures and paired SNRs."""
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
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.pytest_cache/matplotlib-t19c'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_panel_alignment import require_matplotlib_panel_alignment
from src import target_statistics_replication as p
STYLE={'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
       'font.size':9,'legend.fontsize':9,'svg.fonttype':'none','svg.hashsalt':'T19C-fixed',
       'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False}
DATA=('t19c_operator_capture.csv','t19c_original_reference.csv','t19c_large_stat_reference.csv')
STEMS=('t19c_statistics_capture_replication','t19c_paired_snr_capture')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))


def series(directory):
    actual,original,large=[rows(Path(directory)/n) for n in DATA]
    captures={(r['environment'],int(r['replicate']),int(r['snr_db'])):r for r in actual}
    expected={(n,r,s) for n in p.ENVIRONMENTS for r in p.REPLICATES for s in p.SNRS}
    if len(actual)!=20 or set(captures)!=expected:raise ValueError('twenty unique replicate/SNR points required')
    for n in p.ENVIRONMENTS:
        for r in p.REPLICATES:
            if len({captures[n,r,s]['statistics_sha256'] for s in p.SNRS})!=1:
                raise ValueError('two SNRs must use exactly the same clean statistics')
    reference={(r['environment'],int(r['snr_db'])):r for r in original}
    pooled={(r['environment'],int(r['snr_db'])):r for r in large}
    grid={(n,s) for n in p.ENVIRONMENTS for s in p.SNRS}
    if len(original)!=4 or len(large)!=4 or set(reference)!=grid or set(pooled)!=grid:
        raise ValueError('four original and four secondary large-stat references required')
    for group in (captures,reference,pooled):
        for r in group.values():
            if not np.isfinite(float(r['operator_span_capture_dense'])):raise ValueError('undefined capture; do not drop points')
    return captures,reference,pooled


def export(fig,directory,stem):
    require_matplotlib_panel_alignment(fig,json_out=str(directory/f'{stem}.alignment.json'),strict=True)
    fig.savefig(directory/f'{stem}.png',dpi=300)
    fig.savefig(directory/f'{stem}.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(directory/f'{stem}.svg',metadata={'Date':None});plt.close(fig)


def plot(input_dir,output_dir):
    directory=Path(output_dir);directory.mkdir(parents=True,exist_ok=True)
    actual,original,large=series(input_dir);conditions=[(n,s) for n in p.ENVIRONMENTS for s in p.SNRS]
    with matplotlib.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(7.4,4.8));fig.subplots_adjust(left=.13,right=.96,bottom=.24,top=.78)
        for i,(n,s) in enumerate(conditions):
            y=[float(actual[n,r,s]['operator_span_capture_dense']) for r in p.REPLICATES]
            ax.scatter(i+np.linspace(-.16,.16,5),y,color='#0072B2',s=36,label='New statistics replicates (n=5)' if i==0 else None,zorder=3)
            ax.scatter(i-.26,float(original[n,s]['operator_span_capture_dense']),color='0.35',marker='x',s=55,label='Original T19A500' if i==0 else None,zorder=4)
            ax.scatter(i+.26,float(large[n,s]['operator_span_capture_dense']),color='#009E73',marker='D',s=40,label='New pooled2500 (secondary)' if i==0 else None,zorder=4)
        ax.axhline(.50,color='0.65',linestyle='--',linewidth=.8)
        ax.axhline(.70,color='0.65',linestyle=':',linewidth=.8)
        ax.set_xticks(range(4),[f'C30 / {n.split("-")[-1]} ns\n{s} dB' for n,s in conditions]);ax.set_xlim(-.5,3.5)
        ax.set_ylabel('Exact dense operator-span capture')
        fig.suptitle('Independent target-statistics replication of pure-delay capacity',y=.96,fontsize=11)
        fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.55,.90),ncol=3)
        fig.text(.13,.09,'Independent unit: statistics replicate; each500 trajectories ×10 frames. No bootstrap/CI.',fontsize=9)
        fig.text(.13,.045,'Original and pooled references excluded from the five-replicate aggregate and gate.',fontsize=9)
        export(fig,directory,STEMS[0])
    with matplotlib.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(7.4,4.8));fig.subplots_adjust(left=.13,right=.96,bottom=.24,top=.78)
        for r,color in zip(p.REPLICATES,('#0072B2','#E69F00','#009E73','#CC79A7','#9467BD')):
            ax.plot([10,20],[float(actual['CDL-C-30-200',r,s]['operator_span_capture_dense']) for s in p.SNRS],
                    marker='o',markersize=6,color=color,label=f'r{r}')
        ax.set_xticks([10,20],['10 dB','20 dB']);ax.set_xlim(8,22)
        ax.set_xlabel('Frozen source receive-SNR convention');ax.set_ylabel('Exact dense operator-span capture')
        fig.suptitle('C30 / 200 ns: SNR asymmetry within each statistics replicate',y=.96,fontsize=11)
        fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.55,.90),ncol=5)
        fig.text(.13,.09,'Each line shares one clean covariance estimate; only frozen noise variance differs.',fontsize=9)
        fig.text(.13,.045,'Five independent target-statistics estimates; no validation NMSE or deployment claim.',fontsize=9)
        export(fig,directory,STEMS[1])
    return dict(new_point_count=20,original_point_count=4,large_point_count=4,paired_lines=5,
                independent_unit='target-training-statistics replicate',figure_count=2)


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
        for name in DATA:shutil.copy2(Path(input_dir)/name,temp/name)
        plot(temp,temp/'rebuilt')
        rebuilt={f'{stem}.{ext}':sha(directory/f'{stem}.{ext}')==sha(temp/'rebuilt'/f'{stem}.{ext}') for stem in STEMS for ext in ('png','pdf','svg')}
    report=dict(source_returncode=source.returncode,source_output=source.stdout+source.stderr,rendered_checks=checks,
                CSV_only_rebuild=rebuilt,source_data_hashes={n:sha(Path(input_dir)/n) for n in DATA},selection=selection,
                figure_contract=dict(backend='python',dimensions_inches=[7.4,4.8],exports=['PNG300dpi','editable PDF','editable SVG'],glyph_floor_pt=5,
                    uncertainty='individual five statistics replicates;no trajectory bootstrap/CI',
                    adaptation='scoped style-only reuse; independent-statistics and paired-SNR mappings freshly implemented'),
                visual_review='pending',all_checks_passed=False)
    (directory/'t19c_figure_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    if source.returncode or not all(rebuilt.values()) or any(v['text_returncode'] or v['collision_returncode'] for v in checks.values()):
        raise ValueError('figure source/render/rebuild calibration failed')
    print('T19C source/alignment/text/collision and six CSV-only byte rebuilds PASS; visual review pending',flush=True)
    return report


def calibrate():
    from scripts.run_cdl_mismatch import csvsave,save
    directory=ROOT/'outputs/t19c_figure_calibration'
    if directory.exists():raise ValueError('existing calibration protected')
    directory.mkdir();captures=[];reference=[];large=[]
    for j,n in enumerate(p.ENVIRONMENTS):
        for s in p.SNRS:
            base=.77-j*.3+(s-10)*.015
            for r in p.REPLICATES:captures.append(dict(environment=n,replicate=r,snr_db=s,operator_span_capture_dense=base+(r-3)*.005,statistics_sha256=f'synthetic-only-{n}-{r}'))
            reference.append(dict(environment=n,snr_db=s,operator_span_capture_dense=base+.015))
            large.append(dict(environment=n,snr_db=s,operator_span_capture_dense=base-.007))
    for name,values in zip(DATA,(captures,reference,large)):csvsave(directory/name,values)
    report=audit(directory,directory)
    save(ROOT/'outputs/t19c_fixture_figure_gate.json',dict(all_checks_passed=True,synthetic_fixture_only=True,
        scientific_data_used=False,production_outputs_written=False,automatic_QA=report))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input-dir',default=str(ROOT/'outputs'))
    parser.add_argument('--output-dir',default=str(ROOT/'outputs'));parser.add_argument('--audit',action='store_true');parser.add_argument('--calibrate',action='store_true');args=parser.parse_args()
    if args.calibrate:calibrate()
    elif args.audit:audit(args.input_dir,args.output_dir)
    else:plot(args.input_dir,args.output_dir)
