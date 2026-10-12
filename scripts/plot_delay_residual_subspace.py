"""T19D CSV-only development spectrum and held-out oracle transfer figures."""
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
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.pytest_cache/matplotlib-t19d'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_panel_alignment import require_matplotlib_panel_alignment
from src import delay_residual_subspace as p
STYLE={'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
       'font.size':9,'legend.fontsize':9,'svg.fonttype':'none','svg.hashsalt':'T19D-fixed',
       'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False}
DATA=('t19d_residual_pca.csv','t19d_heldout_transfer.csv')
STEMS=('t19d_pure_delay_pca_energy','t19d_heldout_operator_capture','t19d_heldout_capture_increments')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))


def series(directory):
    spectrum,transfer=[rows(Path(directory)/n) for n in DATA]
    spectra={(int(v['snr_db']),int(v['component'])):v for v in spectrum}
    index={(v['environment'],int(v['snr_db'])):v for v in transfer}
    if len(spectrum)!=20 or set(spectra)!={(s,k) for s in p.SNRS for k in range(1,11)}:
        raise ValueError('all twenty normalized development spectrum values required')
    if len(transfer)!=12 or set(index)!={(n,s) for n in p.HELDOUT for s in p.SNRS}:
        raise ValueError('all twelve held-out conditions required; no development rows')
    for s in p.SNRS:
        ratios=np.array([float(spectra[s,k]['explained_energy_ratio']) for k in range(1,11)])
        cum=np.array([float(spectra[s,k]['cumulative_energy']) for k in range(1,11)])
        if not np.isfinite(ratios).all() or np.any(ratios<0) or not np.allclose(cum,np.cumsum(ratios),rtol=1e-11,atol=1e-11) or not np.isclose(cum[-1],1.,atol=1e-11):
            raise ValueError('cumulative normalized energy schema drift')
    for v in index.values():
        values=[float(v[k]) for k in ('capture_q0','capture_q1','capture_q2','increment_q1','increment_q2','marginal_q2')]
        if not np.isfinite(values).all() or values[3]!=values[1]-values[0] or values[4]!=values[2]-values[0] or values[5]!=values[2]-values[1]:
            raise ValueError('unclipped paired metrics drift')
    return spectra,index


def export(fig,directory,stem):
    require_matplotlib_panel_alignment(fig,json_out=str(directory/f'{stem}.alignment.json'),strict=True)
    fig.savefig(directory/f'{stem}.png',dpi=300)
    fig.savefig(directory/f'{stem}.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(directory/f'{stem}.svg',metadata={'Date':None});plt.close(fig)


def plot(input_dir,output_dir):
    directory=Path(output_dir);directory.mkdir(parents=True,exist_ok=True)
    spectrum,transfer=series(input_dir);conditions=[(n,s) for n in p.HELDOUT for s in p.SNRS]
    labels=[f'{n.split("-")[1]}{n.split("-")[2]} / {n.split("-")[3]} ns / {s} dB' for n,s in conditions]
    with matplotlib.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(7.4,4.8));fig.subplots_adjust(left=.13,right=.96,bottom=.25,top=.78)
        for s,color,marker in ((10,'#0072B2','o'),(20,'#E69F00','s')):
            ax.plot(range(1,11),[float(spectrum[s,k]['cumulative_energy']) for k in range(1,11)],marker=marker,color=color,label=f'{s} dB')
        ax.set_xticks(range(1,11));ax.set_xlim(.7,10.3);ax.set_ylim(0,1.04)
        ax.set_xlabel('Additional module-space dimension q');ax.set_ylabel('Cumulative normalized residual energy')
        fig.suptitle('Pure-delay development residual spectrum',y=.96,fontsize=11)
        fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.55,.89),ncol=2)
        fig.text(.13,.09,'Ten equal-weight unit directions per SNR; uncentered real-augmented SVD.',fontsize=9)
        fig.text(.13,.045,'Development spectrum is diagnostic; held-out transfer decides C12.',fontsize=9)
        export(fig,directory,STEMS[0])
    y=np.arange(12)
    with matplotlib.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(7.4,6.4));fig.subplots_adjust(left=.30,right=.97,bottom=.17,top=.80)
        for i,key in enumerate(conditions):
            v=transfer[key];ax.plot([float(v['capture_q0']),float(v['capture_q2'])],[i,i],color='0.85',linewidth=1,zorder=1)
        for field,color,marker,label in (('capture_q0','0.45','x','P/M (q=0)'),('capture_q1','#0072B2','o','P/M + D1 (q=1)'),('capture_q2','#E69F00','D','P/M + D1,D2 (q=2)')):
            ax.scatter([float(transfer[key][field]) for key in conditions],y,color=color,marker=marker,s=30,label=label,zorder=3)
        ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel('Target-weighted operator-span capture')
        ax.axvline(.70,color='0.65',linestyle=':',linewidth=.8);ax.set_xlim(0,1.04)
        fig.suptitle('Held-out profile / mobility + delay transfer',y=.96,fontsize=11)
        fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.55,.90),ncol=1)
        fig.text(.08,.07,'Joint real LS; q counts additional operator directions, not matrix rank.',fontsize=9)
        fig.text(.08,.035,'All twelve contexts held out from basis construction; oracle diagnostics only.',fontsize=9)
        export(fig,directory,STEMS[1])
    with matplotlib.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(7.4,6.4));fig.subplots_adjust(left=.30,right=.97,bottom=.17,top=.80)
        for field,color,marker,label in (('increment_q1','#0072B2','o','Increment q1 (c1-c0)'),('increment_q2','#E69F00','D','Increment q2 (c2-c0)'),('marginal_q2','#009E73','+','Marginal q2 (c2-c1)')):
            ax.scatter([float(transfer[key][field]) for key in conditions],y,color=color,marker=marker,s=35,label=label,zorder=3)
        ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel('Unclipped incremental operator capture')
        ax.axvline(0,color='0.6',linewidth=.8);ax.axvline(.05,color='0.75',linestyle=':',linewidth=.8);ax.axvline(.10,color='0.75',linestyle='--',linewidth=.8)
        fig.suptitle('Held-out gains from one or two additional directions',y=.96,fontsize=11)
        fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.55,.90),ncol=1)
        fig.text(.08,.07,'Paired within each frozen target covariance; no clipping, bootstrap or CI.',fontsize=9)
        fig.text(.08,.035,'All twelve conditions enter the pre-registered factor-transfer gate.',fontsize=9)
        export(fig,directory,STEMS[2])
    return dict(spectrum_points=20,heldout_conditions=12,capture_points=36,increment_points=36,figure_count=3,
                independent_unit='frozen target statistics;deterministic oracle diagnostics;no CI')


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
    report=dict(source_returncode=source.returncode,source_output=source.stdout+source.stderr,rendered_checks=checks,CSV_only_rebuild=rebuilt,
        source_data_hashes={n:sha(Path(input_dir)/n) for n in DATA},selection=selection,
        figure_contract=dict(backend='python',exports=['PNG300dpi','editable PDF','editable SVG'],glyph_floor_pt=5,
            uncertainty='all frozen deterministic conditions; no aggregate CI',adaptation='style-only inheritance;fresh development/transfer/increment mapping'),
        visual_review='pending',all_checks_passed=False)
    (directory/'t19d_figure_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    if source.returncode or not all(rebuilt.values()) or any(v['text_returncode'] or v['collision_returncode'] for v in checks.values()):
        raise ValueError('figure source/render/rebuild gate failed')
    print('T19D source/alignment/text/collision and nine CSV-only rebuilds PASS; visual review pending',flush=True)
    return report


def calibrate():
    from scripts.run_cdl_mismatch import csvsave,save
    directory=ROOT/'outputs/t19d_figure_calibration'
    if directory.exists():raise ValueError('existing fixture protected')
    directory.mkdir();spectrum=[];transfer=[]
    for s in p.SNRS:
        ratios=[.75,.20]+[.00625]*8 if s==10 else [.85,.12]+[.00375]*8
        for k,(ratio,cum) in enumerate(zip(ratios,np.cumsum(ratios)),1):
            spectrum.append(dict(snr_db=s,component=k,explained_energy_ratio=ratio,cumulative_energy=float(cum)))
    for i,n in enumerate(p.HELDOUT):
        for s in p.SNRS:
            a=.45+i*.025;b=.70+i*.008;c=.82+i*.006
            transfer.append(dict(environment=n,snr_db=s,capture_q0=a,capture_q1=b,capture_q2=c,increment_q1=b-a,increment_q2=c-a,marginal_q2=c-b))
    for name,value in zip(DATA,(spectrum,transfer)):csvsave(directory/name,value)
    report=audit(directory,directory)
    save(ROOT/'outputs/t19d_fixture_figure_gate.json',dict(all_checks_passed=True,synthetic_fixture_only=True,scientific_data_used=False,production_outputs_written=False,automatic_QA=report))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input-dir',default=str(ROOT/'outputs'));parser.add_argument('--output-dir',default=str(ROOT/'outputs'))
    parser.add_argument('--audit',action='store_true');parser.add_argument('--calibrate',action='store_true');args=parser.parse_args()
    if args.calibrate:calibrate()
    elif args.audit:audit(args.input_dir,args.output_dir)
    else:plot(args.input_dir,args.output_dir)
