"""Plot persisted MR03 development/validation tables; performs no inference/statistics.

Python-only matplotlib exports. Input-table values and frozen percentile limits
are copied into figure source-data files; no bootstrap draws are regenerated.
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

CONDITIONS = ('ID', 'Profile', 'Mobility', 'Composed', 'Boundary')
SNRS = (10, 20, 30)
RANKS = (16, 32, 64)
INTERVAL = 'Unadjusted exploratory 95% trajectory-paired percentile interval; not confirmatory'
STYLES = {
    'POOLED_NATURAL': ('Pooled', '#666666', 'x', '--'),
    'PPCA_DIRECT_R16': ('Direct r16', '#0072B2', 'o', '-'),
    'PPCA_DIRECT_R32': ('Direct r32', '#009E73', 's', '-'),
    'PPCA_DIRECT_R64': ('Direct r64', '#E69F00', 'D', '-'),
    'LR4_TW1': ('LR4, Tw=1', '#CC79A7', 'v', '-.'),
    'LR4_TW5': ('LR4, Tw=5', '#D55E00', '^', '-.'),
    'DENSE_TW5': ('Dense, Tw=5', '#000000', '+', '-'),
    'WHOLE_POOL_M2_R23': ('Whole r23', '#56B4E9', 'P', ':'),
    'WHOLE_POOL_M2_R24': ('Whole r24', '#882255', '*', ':'),
}
GROUPS = {
    'capacity': ('POOLED_NATURAL', 'PPCA_DIRECT_R16', 'PPCA_DIRECT_R32',
                 'PPCA_DIRECT_R64', 'WHOLE_POOL_M2_R24'),
    'current_frame': ('POOLED_NATURAL', 'PPCA_DIRECT_R16', 'PPCA_DIRECT_R32',
                      'PPCA_DIRECT_R64', 'LR4_TW1', 'LR4_TW5'),
    'deployment_boundaries': ('DENSE_TW5', 'LR4_TW5', 'WHOLE_POOL_M2_R23',
                              'WHOLE_POOL_M2_R24', 'POOLED_NATURAL', 'PPCA_DIRECT_R64'),
}
QUESTIONS = {
    'capacity': 'How do fixed single-component PPCA capacities compare with Pooled and a fixed whole-operator compression across conditions?',
    'current_frame': 'How do current-frame PPCA/Pooled and the same LR4 method with current-frame or five-frame fitting compare?',
    'deployment_boundaries': 'Which conditions expose differences between dense adaptation, two compression placements and current-frame statistical controls?',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('EMPTY_TABLE:' + str(path))
    return rows


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    with Path(path).open('w', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def finite(row, field):
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError('NONFINITE_TABLE:' + field)
    return value


def validated_index(rows, dataset):
    index = {}
    for row in rows:
        key = (row['condition'], int(row['snr_db']), row['method'])
        if row['dataset'] != dataset or key in index or key[:2] not in [(c,s) for c in CONDITIONS for s in SNRS]:
            raise ValueError('NMSE_DUPLICATE_OR_DATASET_IDENTITY:' + str(key))
        if int(row['trajectory_count']) != 100 or int(row['scored_frames']) != 600 or int(row['bootstrap']) != 2000 or row['interval'] != INTERVAL:
            raise ValueError('NMSE_SCOPE_OR_INTERVAL')
        for field in ('error_energy','truth_energy','nmse_linear','nmse_db','CI_lower','CI_upper'):
            finite(row,field)
        if finite(row,'nmse_linear') <= 0 or finite(row,'truth_energy') <= 0 or finite(row,'error_energy') < 0 or finite(row,'CI_lower') > finite(row,'CI_upper'):
            raise ValueError('NMSE_ENERGY_OR_INTERVAL')
        if abs(10*math.log10(finite(row,'nmse_linear'))-finite(row,'nmse_db')) > 1e-10:
            raise ValueError('TABLE_DB_TRANSFORM_DRIFT')
        index[key] = row
    for method in set(m for methods in GROUPS.values() for m in methods):
        for condition in CONDITIONS:
            for snr in SNRS:
                if (condition,snr,method) not in index:
                    raise ValueError('MISSING_REQUIRED_CURVE:' + str((condition,snr,method)))
    return index


def validated_contrasts(rows, dataset):
    index={}
    for row in rows:
        key=(row['condition'],int(row['snr_db']),row['first'],row['second'])
        if row['dataset'] != dataset or key in index:
            raise ValueError('CONTRAST_IDENTITY')
        for field in ('estimate_db','CI_lower','CI_upper'):
            finite(row,field)
        if row['interval'] != INTERVAL or int(row['bootstrap']) != 2000 or int(row['trajectory_count']) !=100:
            raise ValueError('CONTRAST_SCOPE')
        if row['sign'] != 'first minus second NMSE(dB); positive favors second' or finite(row,'CI_lower') > finite(row,'CI_upper'):
            raise ValueError('CONTRAST_SIGN_OR_INTERVAL')
        index[key]=row
    return index


def label(ax, text):
    ax.annotate(text, (0,1), xycoords='axes fraction', xytext=(-22,14),
                textcoords='offset points', ha='left', va='bottom',
                fontweight='bold', fontsize=8)


def line(ax, method, x, y):
    display,color,marker,style=STYLES[method]
    return ax.plot(x,y,label=display,color=color,marker=marker,linestyle=style,
                   linewidth=1.0,markersize=3.5,markeredgewidth=.65)[0]


def interval_marks(ax,x,lo,hi,color):
    # Percentile intervals need not contain the observed estimate; draw the
    # interval independently instead of forcing a nonnegative yerr array.
    ax.vlines(x,lo,hi,color=color,alpha=.5,linewidth=.65)
    ax.hlines(lo,np.array(x)-.25,np.array(x)+.25,color=color,alpha=.5,linewidth=.65)
    ax.hlines(hi,np.array(x)-.25,np.array(x)+.25,color=color,alpha=.5,linewidth=.65)


def qa_module(tools):
    helper=tools/'audit_panel_alignment.py'
    if not helper.is_file():
        raise FileNotFoundError('ALIGNMENT_HELPER_REQUIRED:' + str(helper))
    spec=importlib.util.spec_from_file_location('mr03_panel_alignment',helper)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def run_qa(command, out):
    result=subprocess.run(command,capture_output=True,text=True,encoding='utf8',errors='replace')
    out.write_text(result.stdout+'\n'+result.stderr,encoding='utf8')
    return dict(command=command,returncode=result.returncode,log=str(out))


def save(fig, stem, figure_dir, tools, alignment, exclude=()):
    fig.canvas.draw()
    audit=alignment.require_matplotlib_panel_alignment(
        fig,json_out=figure_dir/(stem+'.alignment.json'),
        overlay_svg=figure_dir/(stem+'.alignment.svg'),
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=True,
        strict=True,exclude_axes=list(exclude))
    paths=[figure_dir/(stem+'.pdf'),figure_dir/(stem+'.svg'),figure_dir/(stem+'.png')]
    # Preserve the declared 180-mm physical canvas; do not crop differently
    # across formats after the render-time rectangle measurements.
    for path in paths:
        fig.savefig(path,dpi=600,facecolor='white')
    plt.close(fig)
    text_report=run_qa([sys.executable,str(tools/'audit_pdf_text.py'),str(paths[0]),'--min-pt','5','--json'],figure_dir/(stem+'.text-audit.log'))
    collision_report=run_qa([sys.executable,str(tools/'audit_figure_collisions.py'),str(paths[0]),'--json-out',str(figure_dir/(stem+'.collision.json')),'--overlay-pdf',str(figure_dir/(stem+'.collision.pdf'))],figure_dir/(stem+'.collision.log'))
    answer=dict(stem=stem,alignment=audit,PDF_text=text_report,collision=collision_report,
                exports=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in paths],
                manual_visual_QA='PENDING_ACTUAL_IMAGE_INSPECTION',
                editable_text=dict(svg_fonttype='none',pdf_fonttype=42),PNG_dpi=600)
    write_json(figure_dir/(stem+'.qa.json'),answer)
    if text_report['returncode'] or collision_report['returncode']:
        raise RuntimeError('FIGURE_QA_NOT_PASSED:' + stem)
    return answer


def draw_grid(index, contrasts, dataset, group, out, figure_dir, source_dir, tools, alignment):
    methods=GROUPS[group]
    fig,axes=plt.subplots(2,3,figsize=(7.0866141732,5.1968503937),sharex=True,sharey=True)
    fig.subplots_adjust(left=.115,right=.98,bottom=.11,top=.90,wspace=.52,hspace=.64)
    source=[];contrast_source=[]
    extents=[]
    for condition in CONDITIONS:
        for method in methods:
            for snr in SNRS:
                row=index[condition,snr,method]
                extents += [finite(row,'CI_lower'),finite(row,'CI_upper'),finite(row,'nmse_db')]
    minimum,maximum=min(extents),max(extents);span=max(maximum-minimum,1.)
    limits=(minimum-.08*span,maximum+.08*span)
    for k,condition in enumerate(CONDITIONS):
        ax=axes.flat[k];label(ax,chr(ord('a')+k));ax.set_title(condition,pad=6,fontsize=8)
        for method in methods:
            rows=[index[condition,snr,method] for snr in SNRS]
            line(ax,method,SNRS,[finite(row,'nmse_db') for row in rows])
            interval_marks(ax,SNRS,[finite(row,'CI_lower') for row in rows],
                           [finite(row,'CI_upper') for row in rows],STYLES[method][1])
            for row in rows:
                source.append(dict(figure=group,panel=chr(ord('a')+k),**row))
        ax.set_xlim(8,32);ax.set_xticks(SNRS);ax.set_ylim(*limits)
        ax.tick_params(axis='both',labelsize=7.2,labelbottom=True,labelleft=True)
        ax.set_xlabel('SNR (dB)',fontsize=8)
        ax.set_ylabel('Pooled NMSE (dB)',fontsize=8)
        ax.grid(axis='y',color='#e6e6e6',linewidth=.4,zorder=0)
        # Associate existing paired contrasts without manufacturing intervals
        # for method pairs absent from the frozen development analysis.
        for key,row in contrasts.items():
            if key[0]==condition and row['first'] in methods and row['second'] in methods:
                contrast_source.append(dict(figure=group,panel=chr(ord('a')+k),**row))
    legend_ax=axes.flat[-1];legend_ax.set_axis_off()
    handles=[Line2D([],[],color=STYLES[m][1],marker=STYLES[m][2],
                    linestyle=STYLES[m][3],linewidth=1.,markersize=4,label=STYLES[m][0]) for m in methods]
    legend_ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(-.02,1.03),
                     fontsize=8,labelspacing=.65,handlelength=2.2,handletextpad=.7)
    legend_ax.text(0,.17,'n = 100 trajectories / condition\n6 scored frames / trajectory\n95% exploratory intervals',
                   transform=legend_ax.transAxes,fontsize=7.2,ha='left',va='top',linespacing=1.4)
    stem=group+'_'+dataset
    write_csv(source_dir/(stem+'.csv'),source)
    if contrast_source:
        write_csv(source_dir/(stem+'_paired_contrasts.csv'),contrast_source)
    dataset_scope=('Independent validation trajectories; not final confirmation.' if dataset=='V_new' else 'Previously revealed development evaluation trajectories; not an independent final confirmation.')
    temporal=('PPCA and Pooled use only the current frame. LR4 Tw=1 and Tw=5 differ only in their cross-pilot fitting windows; all methods use identical scored frames. Equal current-frame observations do not equalize all offline development budgets.' if group=='current_frame' else
              'PPCA and Pooled use current-frame observations. Dense, LR4 and Whole use the fixed five-frame causal fit where displayed. Whole r23/r24 constrain conditional-estimator rank, while direct PPCA r16/r32/r64 constrain generative loading rank; these ranks are different objects.')
    caption=(f'{QUESTIONS[group]} Panels a-e show ID, Profile, Mobility, Composed and Boundary, respectively. '
             'Curves are data-domain linear pooled error/truth NMSE converted to dB; lower is better. '
             'Each cell contains 100 independent physical trajectories with frames 4-9 scored. '
             'Vertical marks are unadjusted exploratory 95% trajectory-paired bootstrap percentile intervals from 2000 frozen resamples. '
             'Across-SNR observations reuse physical trajectories and are not extra independent replicates. '
             f'{dataset_scope} {temporal} Marginal interval overlap is not a paired-difference test; existing paired contrasts are supplied separately as source data. '
             'No original EM-model results are inserted when its parameter arrays are unavailable. '
             'The plotted values establish the comparisons only; they do not by themselves identify a causal mechanism or family-wide MFA superiority.')
    (figure_dir/(stem+'.caption.md')).write_text(caption+'\n',encoding='utf8')
    brief=dict(figure=stem,question=QUESTIONS[group],archetype='quantitative grid',
               conclusion='Empirical comparison to be interpreted after reviewing the actual data; no performance ordering assumed.',
               dataset=dataset,conditions=list(CONDITIONS),methods=list(methods),snrs=list(SNRS),
               panel_roles=dict(a='source-condition anchor',b='profile mismatch',c='mobility mismatch',
                                d='joint-mismatch boundary',e='delay-spread boundary'),
               rows_plotted=len(source),source_data=str(source_dir/(stem+'.csv')),
               uncertainty=INTERVAL,independent_unit='physical trajectory',
               source_rows_total=len(index),plot_rows=len(source),
               method_selection_rule='explicit fixed scientific comparison group',
               omitted_methods=sorted(set(k[2] for k in index)-set(methods)),
               no_condition_SNR_or_trajectory_exclusions=True,
               unavailable_original_EM='not substituted or synthesized',
               placement='development/capacity evidence; manuscript placement requires joint scientific review')
    write_json(figure_dir/(stem+'.brief.json'),brief)
    return save(fig,stem,figure_dir,tools,alignment,[legend_ax])


def draw_overview(spectral, macros, figure_dir, source_dir, tools, alignment):
    by={int(row['rank']):row for row in spectral}
    if len(spectral)!=3 or set(by)!=set(RANKS):raise ValueError('SPECTRAL_CAPACITY_CARDINALITY')
    mm={(row['scope'],int(row['rank'])):row for row in macros}
    if len(macros)!=6 or set(mm)!={(scope,r) for scope in ('origin-only','expanded') for r in RANKS}:
        raise ValueError('VALIDATION_MACRO_CARDINALITY')
    fig,axes=plt.subplots(2,2,figsize=(7.0866141732,5.1181102362))
    fig.subplots_adjust(left=.115,right=.97,bottom=.16,top=.91,wspace=.52,hspace=.65)
    source=[]
    fields=[('training_average_complex_dimension_logscore','Marginal log score / dim.',1),
            ('nu','Residual variance ν',1),('model_array_bytes','Model arrays (MiB)',1/2**20)]
    for k,(field,ylabel,scale) in enumerate(fields):
        ax=axes.flat[k];label(ax,chr(ord('a')+k))
        y=[finite(by[r],field)*scale for r in RANKS]
        ax.plot(RANKS,y,color='#0072B2',marker='o',linewidth=1.,markersize=4)
        ax.set_ylabel(ylabel,fontsize=8);ax.set_xlabel('Loading rank r',fontsize=8)
        ax.set_xticks(RANKS);ax.tick_params(labelsize=7.2)
        ax.grid(axis='y',color='#e6e6e6',linewidth=.4)
        if field=='nu':
            if min(y)<=0:raise ValueError('NONPOSITIVE_VARIANCE')
            ax.set_yscale('log')
        if field=='model_array_bytes':ax.set_ylim(bottom=0)
        for r in RANKS:
            source.append(dict(panel=chr(ord('a')+k),scope='same prepared P',rank=r,
                               measure=field,value=finite(by[r],field)*scale,
                               unit=('MiB' if scale!=1 else 'source units'),
                               lower='',upper='',source='tables/spectral_models.csv'))
    ax=axes.flat[3];label(ax,'d');handles=[]
    for scope,color,marker in [('origin-only','#0072B2','o'),('expanded','#D55E00','s')]:
        rows=[mm[scope,r] for r in RANKS]
        y=[];lo=[];hi=[]
        for r,row in zip(RANKS,rows):
            values=[finite(row,x) for x in ('macro_linear_pooled_nmse','CI_lower_linear','CI_upper_linear')]
            if min(values)<=0 or values[1]>values[2]:raise ValueError('MACRO_INTERVAL')
            y.append(10*math.log10(values[0]));lo.append(10*math.log10(values[1]));hi.append(10*math.log10(values[2]))
            source.append(dict(panel='d',scope=scope,rank=r,measure='db_of_macro_linear',value=y[-1],unit='dB',lower=lo[-1],upper=hi[-1],source='tables/validation_capacity_macros.csv'))
        handle=ax.plot(RANKS,y,color=color,marker=marker,linewidth=1.,markersize=4,label=('Origin-only' if scope=='origin-only' else 'Expanded'))[0]
        interval_marks(ax,RANKS,lo,hi,color);handles.append(handle)
    ax.set_xlabel('Loading rank r',fontsize=8);ax.set_ylabel('Validation macro NMSE (dB)',fontsize=8)
    ax.set_xticks(RANKS);ax.tick_params(labelsize=7.2);ax.grid(axis='y',color='#e6e6e6',linewidth=.4)
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.55,.025),ncol=2,fontsize=8,handlelength=2.2)
    stem='capacity_overview';write_csv(source_dir/(stem+'.csv'),source)
    caption=('Single-component proper-complex PPCA capacity controls on the same authenticated preparation array. '
             '(a) Empirical mean frame marginal log score per complex dimension, (b) clean-channel residual variance ν, '
             '(c) stored mean/loading/variance array bytes, and (d) independent-validation macro linear pooled NMSE converted to dB. '
             'Panels a-c are deterministic quantities of the fitted empirical construction and have no sampling intervals. '
             'Panel d uses equal condition-SNR cell weights: origin-only covers the three preparation-origin conditions at 10/20/30 dB, '
             'while expanded covers all five conditions at those SNRs. Limits in d are unadjusted exploratory 95% intervals from the frozen '
             '2000 trajectory resamples; each condition has 100 physical trajectories. The two scopes do not create independent validation sets. '
             'The direct construction optimizes the feasible PPCA covariance family with ν≥1e-6; it does not replay the author EM residual-floor update. '
             'Stored model arrays exclude cached conditional operators, processing buffers and process RSS. '
             'Improved training marginal score does not guarantee better conditional channel risk. Missing original EM arrays are not replaced by synthetic parameters.')
    (figure_dir/(stem+'.caption.md')).write_text(caption+'\n',encoding='utf8')
    write_json(figure_dir/(stem+'.brief.json'),dict(figure=stem,question='Does increased generative loading capacity change training fit, residual variance, storage and independent-validation risk together?',
               archetype='quantitative grid',panel_roles=dict(a='training objective',b='isotropic residual',c='parameter cost',d='independent capacity validation'),
               conclusion='Capacity must be assessed jointly through training fit, validation risk and cost; empirical ordering is not presumed.',
               rows_plotted=len(source),source_data=str(source_dir/(stem+'.csv')),
               fixed_rank_grid=list(RANKS),sampling_intervals_only_panel='d',
               no_exclusions=True,manual_visual_QA='PENDING_ACTUAL_IMAGE_INSPECTION'))
    return save(fig,stem,figure_dir,tools,alignment)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True,help='Completed MR03 output root; input only except figures/')
    parser.add_argument('--datasets',nargs='+',choices=['V_new','D_eval'],default=['V_new','D_eval'])
    parser.add_argument('--qa-tools',type=Path,default=Path(__file__).parent/'figure_qa')
    args=parser.parse_args();out=args.out.resolve();tools=args.qa_tools.resolve()
    figure_dir=out/'figures';source_dir=figure_dir/'source_data'
    figure_dir.mkdir(exist_ok=True);source_dir.mkdir(exist_ok=True)
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':8,'axes.labelsize':8,'xtick.labelsize':7.2,'ytick.labelsize':7.2,
                         'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':.65,
                         'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42,
                         'savefig.dpi':600,'text.usetex':False})
    alignment=qa_module(tools);sources={};figures=[]
    # Validation macro figures require the completed validation analysis too,
    # even if the caller requests development curves only.
    for dataset in sorted(set(args.datasets)|{'V_new'}):
        receipt=out/f'reports/analysis_{dataset}.json'
        with receipt.open(encoding='utf-8-sig') as stream:
            meta=json.load(stream)
        if not meta['PASS'] or meta['physical_trajectories']!=500 or meta['scored_positions_per_method']!=9000 or meta['bootstrap']!=2000:
            raise ValueError('DATASET_ANALYSIS_NOT_PASS:'+dataset)
        bindings={row['path']:row['sha256'] for row in meta['files']}
        for name in ('nmse','contrasts'):
            relative=f'tables/{name}_{dataset}.csv'
            if relative not in bindings or sha(out/relative)!=bindings[relative]:
                raise ValueError('ANALYSIS_TABLE_HASH_DRIFT:'+relative)
        sources[str(receipt.relative_to(out))]=sha(receipt)
    # All scientific quantities must already be persisted and certified.
    construction=out/'reports/spectral_construction.json'
    with construction.open(encoding='utf-8-sig') as stream:
        if not json.load(stream)['PASS']:raise ValueError('SPECTRAL_CONSTRUCTION_NOT_PASS')
    sources[str(construction.relative_to(out))]=sha(construction)
    spectral_path=out/'tables/spectral_models.csv';macro_path=out/'tables/validation_capacity_macros.csv'
    sources[str(spectral_path.relative_to(out))]=sha(spectral_path);sources[str(macro_path.relative_to(out))]=sha(macro_path)
    spectral=read_csv(spectral_path);macros=read_csv(macro_path)
    figures.append(draw_overview(spectral,macros,figure_dir,source_dir,tools,alignment))
    for dataset in args.datasets:
        receipt=out/f'reports/analysis_{dataset}.json'
        with receipt.open(encoding='utf-8-sig') as stream:
            meta=json.load(stream)
        if not meta['PASS'] or meta['physical_trajectories']!=500 or meta['scored_positions_per_method']!=9000 or meta['bootstrap']!=2000:
            raise ValueError('DATASET_ANALYSIS_NOT_PASS:'+dataset)
        sources[str(receipt.relative_to(out))]=sha(receipt)
        path=out/f'tables/nmse_{dataset}.csv';cpath=out/f'tables/contrasts_{dataset}.csv'
        sources[str(path.relative_to(out))]=sha(path);sources[str(cpath.relative_to(out))]=sha(cpath)
        index=validated_index(read_csv(path),dataset)
        contrasts=validated_contrasts(read_csv(cpath),dataset)
        for group in GROUPS:
            figures.append(draw_grid(index,contrasts,dataset,group,out,figure_dir,source_dir,tools,alignment))
    for relative,expected in sources.items():
        if sha(out/relative)!=expected:raise ValueError('SOURCE_TABLE_MODIFIED_DURING_PLOTTING:'+relative)
    write_json(figure_dir/'source_data_manifest.json',dict(source_hashes=sources,
               source_tables_read_only=True,new_inference_executed=False,new_bootstrap_executed=False,
               transformations='authoritative linear-to-dB display and byte-to-MiB display only; fixed table intervals',
               source_rows_not_deleted=True,plot_method_subsets_explicit_in_briefs=True,
               all_planned_conditions_SNRs_and_ranks_displayed=True,
               plots=figures,manual_visual_QA='PENDING_ROOT_ACTUAL_INSPECTION'))
    print(json.dumps(dict(figures=len(figures),source_hashes=sources,manual_visual_QA='PENDING_ROOT_ACTUAL_INSPECTION'),indent=2))


if __name__=='__main__':
    main()
