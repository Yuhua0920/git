"""Draw only persisted EM16 reconciliation data; no fitting, draws or inference.

Python structural adaptation of an existing certified five-condition plot.
Both data sets have been disclosed and are development diagnostics.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CONDITIONS = ('ID','Profile','Mobility','Composed','Boundary')
SNRS = (10,20,30)
DATASETS = ('V_new','D_eval')
METHODS = ('ORIGINAL_EM16','PPCA_DIRECT_R16','PPCA_DIRECT_R32',
           'PPCA_DIRECT_R64','POOLED_NATURAL','LR4_TW1','LR4_TW5')
STYLES = {
    'ORIGINAL_EM16': ('Original EM16','#000000','o','-'),
    'PPCA_DIRECT_R16': ('Direct PPCA16','#0072B2','s','--'),
    'PPCA_DIRECT_R32': ('Direct PPCA32','#009E73','D','-'),
    'PPCA_DIRECT_R64': ('Direct PPCA64','#E69F00','P',':'),
    'POOLED_NATURAL': ('Pooled','#666666','x','--'),
    'LR4_TW1': ('LR4, Tw=1','#CC79A7','v','-.'),
    'LR4_TW5': ('LR4, Tw=5','#D55E00','^','-.')}
INTERVAL = 'Unadjusted exploratory 95% trajectory paired percentile; both sets already disclosed'
SIGN = 'first minus second; positive favors second'
ROLE = dict(a='source-condition control',b='profile mismatch',c='mobility mismatch',
            d='joint-mismatch boundary',e='delay-spread boundary')

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read(path):
    with Path(path).open(encoding='utf-8-sig') as f:return json.load(f)

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')

def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:r=list(csv.DictReader(f))
    if not r:raise ValueError('EMPTY_INPUT_TABLE')
    return r

def csvout(path,values):
    with Path(path).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)

def number(row,field):
    v=float(row[field])
    if not math.isfinite(v):raise ValueError('NONFINITE_TABLE:'+field)
    return v

def table_indices(nmse,contrasts):
    ix={};cx={}
    for row in nmse:
        key=(row['dataset'],row['condition'],int(row['snr_db']),row['method'])
        if key in ix or key[0] not in DATASETS or key[1] not in CONDITIONS or key[2] not in SNRS:
            raise ValueError('NMSE_IDENTITY')
        if int(row['trajectory_count'])!=100 or int(row['scored_frames'])!=600:
            raise ValueError('NMSE_CARDINALITY')
        for f in ('nmse_linear','nmse_db','CI_lower','CI_upper'):number(row,f)
        # Guard: the linear NMSE must be strictly positive before log10.
        if number(row,'nmse_linear')<=0 or number(row,'CI_lower')>number(row,'CI_upper'):
            raise ValueError('NMSE_RANGE')
        if abs(10*math.log10(number(row,'nmse_linear'))-number(row,'nmse_db'))>1e-10:
            raise ValueError('NMSE_TRANSFORM')
        ix[key]=row
    for row in contrasts:
        key=(row['dataset'],row['condition'],int(row['snr_db']),row['first'],row['second'])
        if key in cx or key[0] not in DATASETS or key[1] not in CONDITIONS or key[2] not in SNRS:
            raise ValueError('CONTRAST_IDENTITY')
        if row['interval']!=INTERVAL or row['sign']!=SIGN or int(row['bootstrap'])!=2000 or int(row['trajectory_count'])!=100:
            raise ValueError('CONTRAST_STATISTICAL_SCOPE')
        for f in ('estimate_db','CI_lower','CI_upper'):number(row,f)
        if number(row,'CI_lower')>number(row,'CI_upper'):raise ValueError('CONTRAST_INTERVAL_ORDER')
        if abs(number(row,'estimate_db')-(number(ix[key[:3]+(key[3],)],'nmse_db')-number(ix[key[:3]+(key[4],)],'nmse_db')))>1e-10:
            raise ValueError('CONTRAST_DIRECTION_OR_VALUE')
        cx[key]=row
    for ds in DATASETS:
        for c in CONDITIONS:
            for s in SNRS:
                for m in METHODS:
                    if (ds,c,s,m) not in ix:raise ValueError('MISSING_REQUIRED_METHOD_CELL')
                if (ds,c,s,'ORIGINAL_EM16','PPCA_DIRECT_R16') not in cx:raise ValueError('MISSING_EM_DIRECT16_CONTRAST')
    return ix,cx

def helper(tools):
    spec=importlib.util.spec_from_file_location('mr03b_alignment',tools/'audit_panel_alignment.py')
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);return mod

def label(ax,text):
    ax.annotate(text,(0,1),xycoords='axes fraction',xytext=(-22,14),textcoords='offset points',
                fontsize=8,fontweight='bold',ha='left',va='bottom')

def limits(values,include_zero=False):
    values=list(values)+([0.] if include_zero else [])
    lo,hi=min(values),max(values);pad=.075*max(hi-lo,1.)
    return lo-pad,hi+pad

def mark_intervals(ax,x,lower,upper,color):
    # Percentile bounds are displayed independently; no forced nonnegative yerr.
    ax.vlines(x,lower,upper,color=color,linewidth=.7,alpha=.55,zorder=2)
    ax.hlines(lower,np.asarray(x)-.24,np.asarray(x)+.24,color=color,linewidth=.7,alpha=.55)
    ax.hlines(upper,np.asarray(x)-.24,np.asarray(x)+.24,color=color,linewidth=.7,alpha=.55)

def grid():
    fig,axes=plt.subplots(2,3,figsize=(7.0866141732,5.4330708661))
    fig.subplots_adjust(left=.115,right=.98,bottom=.105,top=.90,wspace=.52,hspace=.65)
    return fig,axes

def finish(fig,axes,stem,out,tools,alignment):
    target=out/'figures';fig.canvas.draw()
    aligned=alignment.require_matplotlib_panel_alignment(fig,json_out=target/(stem+'.alignment.json'),
        overlay_svg=target/(stem+'.alignment.svg'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,
        require_panel_labels=True,strict=True,exclude_axes=[axes.flat[-1]])
    paths=[target/(stem+'.pdf'),target/(stem+'.svg'),target/(stem+'.png')]
    for p in paths:
        fig.savefig(p,dpi=600,facecolor='white')
    plt.close(fig)
    checks=[]
    commands=[['audit_pdf_text.py',str(paths[0]),'--min-pt','5','--json'],
              ['audit_figure_collisions.py',str(paths[0]),'--json-out',str(target/(stem+'.collision.json')),
               '--overlay-pdf',str(target/(stem+'.collision.pdf'))]]
    for i,cmd in enumerate(commands):
        result=subprocess.run([sys.executable,str(tools/cmd[0]),*cmd[1:]],capture_output=True,text=True,encoding='utf8',errors='replace')
        p=target/(stem+('.text-audit.log' if i==0 else '.collision.log'))
        p.write_text(result.stdout+'\n'+result.stderr,encoding='utf8')
        checks.append(dict(command=cmd,returncode=result.returncode,log=p.name))
    answer=dict(figure=stem,alignment=aligned,rendered_checks=checks,
        exports=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in paths],
        PNG_dpi=600,editable_text=True,manual_visual_QA='PENDING_ACTUAL_INSPECTION')
    write(target/(stem+'.qa.json'),answer)
    if any(c['returncode'] for c in checks):raise RuntimeError('RENDERED_QA_FAILED:'+stem)
    return answer

def axis(ax,condition,k,ylim,ylabel):
    label(ax,chr(97+k));ax.set_title(condition,fontsize=8,pad=6)
    ax.set_xlim(8,32);ax.set_xticks(SNRS);ax.set_ylim(*ylim)
    ax.set_xlabel('SNR (dB)',fontsize=8);ax.set_ylabel(ylabel,fontsize=8)
    ax.tick_params(labelsize=7.2);ax.grid(axis='y',color='#e6e6e6',linewidth=.4)

def absolute(ix,ds,out,tools,alignment,ylim):
    fig,axes=grid();source=[]
    for k,c in enumerate(CONDITIONS):
        ax=axes.flat[k];axis(ax,c,k,ylim,'Pooled NMSE (dB)')
        for m in METHODS:
            display,color,marker,style=STYLES[m]
            values=[ix[ds,c,s,m] for s in SNRS]
            ax.plot(SNRS,[number(v,'nmse_db') for v in values],color=color,marker=marker,
                    linestyle=style,linewidth=1.2 if m=='ORIGINAL_EM16' else .9,
                    markersize=3.7 if m=='ORIGINAL_EM16' else 3.4,markeredgewidth=.65)
            mark_intervals(ax,SNRS,[number(v,'CI_lower') for v in values],[number(v,'CI_upper') for v in values],color)
            source.extend([dict(panel=chr(97+k),**v) for v in values])
    leg=axes.flat[-1];leg.axis('off')
    handles=[Line2D([],[],color=STYLES[m][1],marker=STYLES[m][2],linestyle=STYLES[m][3],linewidth=1.,markersize=4,label=STYLES[m][0]) for m in METHODS]
    leg.legend(handles=handles,loc='upper left',bbox_to_anchor=(-.02,1.03),fontsize=7.5,labelspacing=.7,handlelength=2.1)
    leg.text(0,.055,'100 trajectories / condition\n6 scored frames / trajectory\n95% exploratory intervals',
             transform=leg.transAxes,fontsize=7.2,va='top',ha='left',linespacing=1.4)
    stem='em16_capacity_'+ds;csvout(out/'figures/source_data'/(stem+'.csv'),source)
    caption=('Absolute channel-estimation NMSE for the original final EM16 model and fixed capacity/information controls. '
             'Panels a–e show ID, Profile, Mobility, Composed and Boundary, respectively. '
             f'The {ds} data comprise 100 physical trajectories per condition, with frames 4–9 scored at 10, 20 and 30 dB. '
             'Linear pooled error/truth NMSE is converted to dB; lower is better. Vertical marks are unadjusted exploratory '
             '95% trajectory-paired percentile intervals from the same 2000 frozen resamples. Both data sets have already been disclosed '
             'and serve development and consistency diagnosis, not independent final confirmation. EM16, direct PPCA and Pooled use '
             'current-frame pilot observations; LR4 Tw=1 and Tw=5 preserve their respective causal fitting windows. '
             'Near-coincident curves are not displaced; full-precision source data resolve these comparisons. Marginal interval overlap '
             'does not establish a paired-difference result. Equal evaluation inputs do not equalize all offline development budgets. '
             'The plotted ordering does not identify a causal mechanism or establish superiority over the entire MFA family.')
    (out/'figures'/(stem+'.caption.md')).write_text(caption+'\n',encoding='utf8')
    write(out/'figures'/(stem+'.brief.json'),dict(question='Does the original final EM16 model agree in channel risk with direct PPCA and fixed same-input controls?',
        conclusion='Same-input comparisons establish the observed performance ordering and condition-specific boundaries; no ordering is presumed.',
        archetype='quantitative grid',panel_roles=ROLE,dataset=ds,methods=list(METHODS),
        scientific_scope='already-disclosed development diagnostics',uncertainty=INTERVAL,source_rows=len(source),
        omitted_methods=['SOURCE','DENSE_TW5','DENSE_TW1','WHOLE_POOL_M2_R23','WHOLE_POOL_M2_R24'],
        omission_reason='fixed seven-method model-reconciliation figure; all methods remain in the authoritative tables',
        no_condition_SNR_or_trajectory_exclusions=True))
    return finish(fig,axes,stem,out,tools,alignment)

def paired(cx,out,tools,alignment):
    fig,axes=grid();source=[];styles={'V_new':('V_new','#0072B2','o','-'),'D_eval':('D_eval','#D55E00','s','--')}
    selected=[cx[ds,c,s,'ORIGINAL_EM16','PPCA_DIRECT_R16'] for ds in DATASETS for c in CONDITIONS for s in SNRS]
    ylim=limits([number(v,f) for v in selected for f in ('estimate_db','CI_lower','CI_upper')],True)
    for k,c in enumerate(CONDITIONS):
        ax=axes.flat[k];axis(ax,c,k,ylim,'EM16 − Direct16 (dB)');ax.axhline(0,color='#777777',linewidth=.65,linestyle=':')
        for ds in DATASETS:
            display,color,marker,style=styles[ds];values=[cx[ds,c,s,'ORIGINAL_EM16','PPCA_DIRECT_R16'] for s in SNRS]
            ax.plot(SNRS,[number(v,'estimate_db') for v in values],color=color,marker=marker,linestyle=style,linewidth=1.,markersize=4)
            mark_intervals(ax,SNRS,[number(v,'CI_lower') for v in values],[number(v,'CI_upper') for v in values],color)
            source.extend([dict(panel=chr(97+k),**v) for v in values])
    leg=axes.flat[-1];leg.axis('off')
    leg.legend(handles=[Line2D([],[],color=v[1],marker=v[2],linestyle=v[3],linewidth=1.,markersize=4,label=v[0]) for v in styles.values()],
               loc='upper left',bbox_to_anchor=(-.02,1.03),fontsize=8,labelspacing=.8,handlelength=2.2)
    leg.text(0,.48,'Positive: Direct16 lower NMSE\nZero: equal pooled NMSE\n100 trajectories / condition\n2000 paired resamples\n95% exploratory intervals',
             transform=leg.transAxes,fontsize=7.2,ha='left',va='top',linespacing=1.45)
    stem='em16_direct16_paired';csvout(out/'figures/source_data'/(stem+'.csv'),source)
    caption=('Paired pooled-NMSE difference between the original final EM16 model and direct PPCA16 on the same evaluation inputs. '
             'Positive EM16 minus Direct16 values indicate lower NMSE for Direct16. Panels a–e show ID, Profile, Mobility, '
             'Composed and Boundary. Each cell contains 100 physical trajectories with six scored frames; the two methods share '
             'channel realizations, pilot observations, noise and truth coordinates. Vertical marks are unadjusted exploratory '
             '95% trajectory-paired percentile intervals from the original 2000 frozen resamples. The 30 displayed intervals '
             'have no multiplicity adjustment. Across-SNR evaluations repeat the same physical trajectories. V_new and D_eval '
             'have both already been disclosed and are development diagnostics. These paired output differences do not alone '
             'separate training quality, covariance parameterization and numerical inference effects; the independent model and '
             'conditional-mean checks are required for that interpretation.')
    (out/'figures'/(stem+'.caption.md')).write_text(caption+'\n',encoding='utf8')
    write(out/'figures'/(stem+'.brief.json'),dict(question='How large is the same-input EM16-to-direct16 risk difference and where does it change across SNR/conditions?',
        conclusion='Displayed signs and frozen exploratory intervals characterize the paired difference without causal attribution.',
        archetype='quantitative grid',panel_roles=ROLE,datasets=list(DATASETS),source_rows=len(source),
        uncertainty=INTERVAL,sign='positive favors Direct16',no_exclusions=True))
    return finish(fig,axes,stem,out,tools,alignment)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--qa-tools',type=Path,default=Path(__file__).parent/'figure_qa');a=p.parse_args();out=a.out.resolve();tools=a.qa_tools.resolve()
    if not read(out/'reports/analysis.json')['PASS']:raise ValueError('COMPLETED_ANALYSIS_REQUIRED')
    paths=[out/'tables/capacity_comparison.csv',out/'tables/paired_contrasts.csv',out/'reports/analysis.json']
    source={x.relative_to(out).as_posix():sha(x) for x in paths}
    ix,cx=table_indices(rows(paths[0]),rows(paths[1]));alignment=helper(tools)
    (out/'figures/source_data').mkdir(parents=True,exist_ok=True)
    rc={'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,
        'axes.labelsize':8,'xtick.labelsize':7.2,'ytick.labelsize':7.2,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.65,'legend.frameon':False,'svg.fonttype':'none',
        'pdf.fonttype':42,'savefig.dpi':600,'text.usetex':False}
    # One common vertical range preserves comparisons between the two data sets.
    ylim=limits([number(ix[ds,c,s,m],f) for ds in DATASETS for c in CONDITIONS for s in SNRS for m in METHODS for f in ('nmse_db','CI_lower','CI_upper')])
    with mpl.rc_context(rc):
        plots=[absolute(ix,ds,out,tools,alignment,ylim) for ds in DATASETS]
        plots.append(paired(cx,out,tools,alignment))
    for path,h in source.items():
        if sha(out/path)!=h:raise ValueError('SOURCE_TABLE_MODIFIED')
    write(out/'figures/source_data_manifest.json',dict(source_hashes=source,plots=plots,
        new_inference_executed=False,new_bootstrap_executed=False,scientific_results_modified=False,
        transformations='plot persisted dB and intervals only; no inferred or interpolated values',
        adaptation='structural adaptation of existing project Python five-condition layout',
        backend='Python',manual_visual_QA='PENDING_ACTUAL_IMAGE_INSPECTION'))
    print(json.dumps(dict(figures=len(plots),source_hashes=source,rendered_QA='PASS',manual_visual_QA='PENDING_ACTUAL_INSPECTION'),indent=2))

if __name__=='__main__':main()
