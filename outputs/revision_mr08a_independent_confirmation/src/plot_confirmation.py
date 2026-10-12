"""Final MR08A figures from complete certified tables only: no inference/draws.

All35 condition-SNR cells and all10 method cells remain visible. Only the three
predeclared families have adjusted intervals; compression penalties are purely
descriptive point differences. Matplotlib is the sole graphics backend.
"""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,math,subprocess,sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CONDITIONS=('ID','Profile','Mobility','Composed','Boundary');SNRS=(0,5,10,15,20,25,30)
METHODS=('SOURCE','POOLED_NATURAL','PPCA_DIRECT_R32','GMM_PREPARED_3','HARD_TW5','CONVEX_TW5','LR4_TW1','LR4_TW5','DENSE_TW5','WHOLE_POOL_M2_R24')
FAMILIES={'A':('POOLED_NATURAL','LR4_TW5'),'B':('GMM_PREPARED_3','LR4_TW1'),'C':('LR4_TW5','WHOLE_POOL_M2_R24')};TAIL=.05/(2*35)
STYLE={
'SOURCE':('Source','#999999','x',':'),
'POOLED_NATURAL':('Pooled','#555555','+','--'),
'PPCA_DIRECT_R32':('Direct PPCA r32','#009E73','s','--'),
'GMM_PREPARED_3':('Prepared GMM-3','#0072B2','o','-'),
'HARD_TW5':('Hard, Tw=5','#56B4E9','P',':'),
'CONVEX_TW5':('Convex, Tw=5','#AA6600','h',':'),
'LR4_TW1':('LR4, Tw=1','#CC79A7','v','-.'),
'LR4_TW5':('LR4, Tw=5','#D55E00','^','-.'),
'DENSE_TW5':('Dense, Tw=5','#111111','D','-'),
'WHOLE_POOL_M2_R24':('Whole r24, Tw=5','#882255','*',':')}
COST={'DENSE_TW5':807936,'LR4_TW5':332288,'WHOLE_POOL_M2_R24':336384}

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def js(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def dump(path,v):Path(path).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def csvrows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csvwrite(path,rows):
    with Path(path).open('w',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def finite(row,key):
    x=float(row[key])
    if not math.isfinite(x):raise ValueError('NONFINITE_FIGURE_TABLE:'+key)
    return x

def read_tables(out):
    analysis=js(out/'reports/formal_analysis.json');raw=js(out/'reports/formal_receipt.json')
    if analysis['PASS'] is not True or raw['PASS'] is not True or raw['identity_PASS'] is not True or analysis['physical_trajectories']!=5000 or analysis['bootstrap_replicates']!=100000 or analysis['NMSE_rows']!=350 or analysis['primary_rows']!=105:raise ValueError('FORMAL_RESULTS_NOT_COMPLETE')
    nm=csvrows(out/'tables/formal_nmse.csv');ci=csvrows(out/'tables/formal_primary_contrasts.csv');ix={(z['condition'],int(z['snr_db']),z['method']):z for z in nm};cx={(z['family'],z['condition'],int(z['snr_db'])):z for z in ci}
    if len(nm)!=350 or set(ix)!={(c,s,m) for c in CONDITIONS for s in SNRS for m in METHODS} or len(ci)!=105 or set(cx)!={(f,c,s) for f in FAMILIES for c in CONDITIONS for s in SNRS}:raise ValueError('ALL_METHOD_AND_PRIMARY_GRIDS')
    for z in nm:
        if int(z['trajectory_count'])!=1000 or int(z['scored_frames'])!=6000 or finite(z,'nmse_linear')<=0:raise ValueError('NMSE_UNIT')
    for z in ci:
        if (z['first'],z['second'])!=FAMILIES[z['family']] or int(z['family_size'])!=35 or int(z['bootstrap'])!=100000 or finite(z,'tail_probability')!=TAIL or finite(z,'CI_lower')>finite(z,'CI_upper'):raise ValueError('PRIMARY_SCOPE')
    return ix,cx

def labels(ax,letter):
    ax.annotate(letter,(0,1),xycoords='axes fraction',xytext=(-24,14),textcoords='offset points',ha='left',va='bottom',fontweight='bold',fontsize=8)

def setup_grid():
    fig,axes=plt.subplots(2,3,figsize=(7.0866141732,5.905511811),sharex=True,sharey=True)
    fig.subplots_adjust(left=.13,right=.98,top=.90,bottom=.12,wspace=.53,hspace=.64)
    for k,c in enumerate(CONDITIONS):
        ax=axes.flat[k];labels(ax,chr(97+k));ax.set_title(c,fontsize=8,pad=6);ax.set_xlim(-1.5,31.5);ax.set_xticks(SNRS);ax.set_xlabel('SNR (dB)',fontsize=8);ax.tick_params(labelsize=7.0,labelleft=True,labelbottom=True);ax.grid(axis='y',color='#e6e6e6',linewidth=.4,zorder=0)
    axes.flat[5].set_axis_off();return fig,axes

def interval(ax,x,lo,hi,color):
    ax.vlines(x,lo,hi,color=color,linewidth=.85,alpha=.85,zorder=4)
    ax.hlines(lo,np.asarray(x)-.65,np.asarray(x)+.65,color=color,linewidth=.85,alpha=.85,zorder=4)
    ax.hlines(hi,np.asarray(x)-.65,np.asarray(x)+.65,color=color,linewidth=.85,alpha=.85,zorder=4)

def helper(tools):
    spec=importlib.util.spec_from_file_location('mr08_alignment',tools/'audit_panel_alignment.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def run(cmd,log):
    p=subprocess.run(cmd,capture_output=True,text=True,encoding='utf8',errors='replace');log.write_text(p.stdout+'\n'+p.stderr,encoding='utf8');return dict(command=cmd,returncode=p.returncode,log=str(log))

def save(fig,stem,out,tools,alignment,legend_ax):
    folder=out/'figures';fig.canvas.draw();a=alignment.require_matplotlib_panel_alignment(fig,json_out=folder/(stem+'.alignment.json'),overlay_svg=folder/(stem+'.alignment.svg'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=True,strict=True,exclude_axes=[legend_ax])
    paths=[]
    for suffix in ('.pdf','.svg','.png'):
        p=folder/(stem+suffix);fig.savefig(p,dpi=600,facecolor='white');paths.append(p)
    plt.close(fig)
    text=run([sys.executable,str(tools/'audit_pdf_text.py'),str(paths[0]),'--min-pt','5','--json'],folder/(stem+'.text.log'));collision=run([sys.executable,str(tools/'audit_figure_collisions.py'),str(paths[0]),'--json-out',str(folder/(stem+'.collision.json')),'--overlay-pdf',str(folder/(stem+'.collision.pdf'))],folder/(stem+'.collision.log'))
    answer=dict(stem=stem,alignment=a,text=text,collision=collision,exports=[dict(path=str(p),SHA256=sha(p),bytes=p.stat().st_size) for p in paths],physical_width_mm=180,editable_PDF_SVG_text=True,PNG_dpi=600,manual_QA='PENDING_ACTUAL_VIEW')
    dump(folder/(stem+'.qa.json'),answer)
    if text['returncode'] or collision['returncode']:raise ValueError('FIGURE_RENDERED_QA_FAILED:'+stem)
    return answer

def caption(stem,text,out):
    (out/'figures'/(stem+'.caption.md')).write_text(text+'\n',encoding='utf8')

def absolute(out,nm,tools,alignment):
    fig,axes=setup_grid();values=[finite(nm[c,s,m],'nmse_db') for c in CONDITIONS for s in SNRS for m in METHODS];span=max(max(values)-min(values),1);limits=(min(values)-.07*span,max(values)+.07*span);source=[]
    for k,c in enumerate(CONDITIONS):
        ax=axes.flat[k];ax.set_ylim(*limits);ax.set_ylabel('Pooled NMSE (dB)',fontsize=8)
        for m in METHODS:
            display,color,marker,style=STYLE[m];rows=[nm[c,s,m] for s in SNRS]
            ax.plot(SNRS,[finite(r,'nmse_db') for r in rows],color=color,marker=marker,linestyle=style,linewidth=.9,markersize=3,markeredgewidth=.6,label=display)
            source.extend(dict(panel=chr(97+k),**r) for r in rows)
    leg=axes.flat[5];handles=[Line2D([],[],color=STYLE[m][1],marker=STYLE[m][2],linestyle=STYLE[m][3],linewidth=.9,markersize=3.5,label=STYLE[m][0]) for m in METHODS]
    leg.legend(handles=handles,loc='upper left',bbox_to_anchor=(-.04,1.03),fontsize=7.2,labelspacing=.35,handlelength=2.2,handletextpad=.7)
    leg.text(0,.15,'1000 trajectories / condition\n6 scored frames / trajectory\nAll ten frozen methods\nPrimary intervals shown separately',transform=leg.transAxes,ha='left',va='top',fontsize=7,linespacing=1.35)
    stem='confirmation_all_methods_nmse';csvwrite(out/'figures/source_data'/(stem+'.csv'),source)
    caption(stem,'Independent confirmation of all ten frozen estimators. Panels a–e show ID, Profile, Mobility, Composed and Boundary on all seven SNR values. Points are pooled linear error/truth NMSE converted to dB; lower is better. Each cell has 1000 independent physical trajectories and six scored frames per trajectory. The same trajectories are reused across SNR and methods, which are repeated measurements rather than additional independent replicates. All350 method cells are displayed without exclusions or visual offsets. This figure shows point estimates; uncertainty for the three prespecified primary families is shown separately using 100000 trajectory-paired bootstrap draws and Bonferroni adjustment within each 35-comparison family. Source, Pooled, Direct PPCA32, GMM-3 and LR4 Tw=1 use the current frame only; the remaining adaptive systems use four additional history frames. Offline model structures and assets differ. The confirmation uses new trajectories from five known channel distributions and does not establish unseen-distribution generalization.',out)
    dump(out/'figures'/(stem+'.brief.json'),dict(question='Do the preparation, online refinement and deployment comparisons reproduce on independent trajectories?',archetype='quantitative grid',methods=list(METHODS),cells=350,all_conditions_and_SNRs=True,intervals='None: point-estimate panel; primary paired intervals displayed in separate figures',no_exclusions=True))
    return save(fig,stem,out,tools,alignment,leg)

def primary(out,cs,family,tools,alignment):
    first,second=FAMILIES[family];fig,axes=setup_grid();source=[];bounds=[finite(cs[family,c,s],key) for c in CONDITIONS for s in SNRS for key in ('estimate_db','CI_lower','CI_upper')];low=min(min(bounds),0);high=max(max(bounds),0);span=max(high-low,1);lim=(low-.08*span,high+.08*span);color=STYLE[second][1]
    for k,c in enumerate(CONDITIONS):
        ax=axes.flat[k];rows=[cs[family,c,s] for s in SNRS];ax.set_ylim(*lim);ax.set_ylabel('First − second NMSE (dB)',fontsize=8);ax.axhline(0,color='#777777',linewidth=.7,zorder=0)
        ax.plot(SNRS,[finite(r,'estimate_db') for r in rows],color=color,marker='o',linewidth=1.,markersize=2.6);interval(ax,SNRS,[finite(r,'CI_lower') for r in rows],[finite(r,'CI_upper') for r in rows],color);source.extend(dict(panel=chr(97+k),**r) for r in rows)
    leg=axes.flat[5];leg.text(0,.98,'Family '+family+'\n\nFirst: '+STYLE[first][0]+'\nSecond: '+STYLE[second][0]+'\n\nPositive favors second\nNegative favors first\n\nBonferroni-adjusted intervals\n35 comparisons in this family\n100000 paired trajectory draws\nNo combined 105-item guarantee',transform=leg.transAxes,fontsize=7.3,ha='left',va='top',linespacing=1.45)
    stem='confirmation_primary_family_'+family;csvwrite(out/'figures/source_data'/(stem+'.csv'),source)
    caption(stem,'Prespecified primary family '+family+': '+STYLE[first][0]+' minus '+STYLE[second][0]+'. Panels a–e show all 35 condition-SNR contrasts across ID, Profile, Mobility, Composed and Boundary. Positive differences favor the second estimator; the zero line denotes equal pooled NMSE. Points are differences of pooled-energy NMSE in dB; vertical segments are two-sided Bonferroni-adjusted percentile intervals using 100000 paired physical-trajectory bootstrap draws per condition, shared across all SNR and methods. Each cell has 1000 trajectories with six frames retained together. Tail probabilities are 0.05/(2×35) and 1−0.05/(2×35). The nominal 95% familywise scope applies separately to this35-comparison family; no joint 95% guarantee is claimed for all 105 comparisons. An interval crossing zero does not establish equivalence. This is independent-trajectory confirmation within five previously studied distributions, not an unseen-profile experiment.',out)
    dump(out/'figures'/(stem+'.brief.json'),dict(question='Which independently confirmed contrasts favor each comparator in family '+family+'?',archetype='quantitative grid',family=family,first=first,second=second,cells=35,tail=TAIL,interval='Bonferroni-adjusted two-sided percentile; separate35-comparison family',no_joint105_guarantee=True,no_exclusions=True))
    return save(fig,stem,out,tools,alignment,leg)

def compression(out,nm,tools,alignment):
    fig,axes=setup_grid();values=[];source=[]
    for c in CONDITIONS:
        for s in SNRS:
            for m in ('LR4_TW5','WHOLE_POOL_M2_R24'):values.append(finite(nm[c,s,m],'nmse_db')-finite(nm[c,s,'DENSE_TW5'],'nmse_db'))
    low=min(min(values),0);high=max(max(values),0);span=max(high-low,1);limits=(low-.08*span,high+.08*span)
    for k,c in enumerate(CONDITIONS):
        ax=axes.flat[k];ax.set_ylim(*limits);ax.set_ylabel('Compressed − Dense NMSE (dB)',fontsize=8);ax.axhline(0,color='#777777',linewidth=.6,zorder=0)
        for m in ('LR4_TW5','WHOLE_POOL_M2_R24'):
            y=[finite(nm[c,s,m],'nmse_db')-finite(nm[c,s,'DENSE_TW5'],'nmse_db') for s in SNRS];display,color,marker,style=STYLE[m];ax.plot(SNRS,y,color=color,marker=marker,linestyle=style,linewidth=1.,markersize=3.5)
            source.extend(dict(panel=chr(97+k),condition=c,snr_db=s,method=m,reference='DENSE_TW5',descriptive_penalty_db=v,method_nmse_db=finite(nm[c,s,m],'nmse_db'),Dense_nmse_db=finite(nm[c,s,'DENSE_TW5'],'nmse_db'),interval='None: descriptive secondary point contrast; not a new primary family') for s,v in zip(SNRS,y))
    leg=axes.flat[5];handles=[Line2D([],[],color=STYLE[m][1],marker=STYLE[m][2],linestyle=STYLE[m][3],linewidth=1.,markersize=4,label=STYLE[m][0]) for m in ('LR4_TW5','WHOLE_POOL_M2_R24')];leg.legend(handles=handles,loc='upper left',bbox_to_anchor=(-.04,1.03),fontsize=7.5,handlelength=2.3)
    leg.text(0,.70,'Dominant complex-MAC proxy\nper scored estimator output\n\nDense: 807936\nLR4: 332288\nWhole24: 336384\n\nWhole24: +1.23% versus LR4\n\nDescriptive risk differences\nNo new confidence interval\nMAC counts are not runtime',transform=leg.transAxes,fontsize=7.2,ha='left',va='top',linespacing=1.4)
    stem='confirmation_compression_tradeoff';csvwrite(out/'figures/source_data'/(stem+'.csv'),source);csvwrite(out/'figures/source_data'/(stem+'_MAC.csv'),[dict(method=m,total_dominant_complex_MAC=mac,source='Fixed implementation arithmetic: fit46080; deployment Source+2rank4,3dense maps or3rank24 whole maps',runtime_claim=False) for m,mac in COST.items()])
    caption(stem,'Correction-only versus whole-bank compression relative to the shared dense adaptive estimator. Panels a–e show all 35 condition-SNR cells; points are the compressed estimator pooled NMSE(dB) minus Dense pooled NMSE(dB). Positive values indicate higher risk than Dense; negative values indicate lower observed risk. These are descriptive secondary point differences and no additional primary comparison family or interval is introduced. Both compressed methods retain the frozen five-frame cross-pilot fit. Whole24 preserves the original common Pooled second-moment weighting and Source centering. The adjacent costs are dominant complex-MAC proxies, including the common 46,080 fitting term: Dense 807,936, LR4 332,288, and Whole24 336,384. Whole24 is 1.2327% above LR4 in this proxy. Gram construction, the small real solve and scalar additions/products are excluded consistently. These arithmetic counts are not measured latency, memory or energy; previously measured runtime evidence is interpreted separately with its own endpoint.',out)
    dump(out/'figures'/(stem+'.brief.json'),dict(question='What risk-cost differences follow from the two frozen compression placements?',archetype='quantitative grid',cells=35,descriptive_secondary_contrasts=2,intervals='None: no new primary family',cost_source='Existing certified MR05 arithmetic, formulas unchanged',costs=COST,no_exclusions=True))
    return save(fig,stem,out,tools,alignment,leg)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--qa-tools',type=Path,default=Path(__file__).parent/'figure_qa');a=p.parse_args();out=a.out.resolve();tools=a.qa_tools.resolve();(out/'figures/source_data').mkdir(parents=True,exist_ok=True)
    # No scientific work is performed by this entry. It requires completed raw
    # and statistical receipts before any table is rendered.
    ix,cx=read_tables(out);sources={q.relative_to(out).as_posix():sha(q) for q in [out/'tables/formal_nmse.csv',out/'tables/formal_primary_contrasts.csv',out/'reports/formal_analysis.json',out/'reports/formal_receipt.json']}
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,'axes.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42,'savefig.dpi':600,'text.usetex':False})
    alignment=helper(tools);plots=[absolute(out,ix,tools,alignment)]
    for f in FAMILIES:plots.append(primary(out,cx,f,tools,alignment))
    plots.append(compression(out,ix,tools,alignment))
    for name,h in sources.items():
        if sha(out/name)!=h:raise ValueError('READ_ONLY_FIGURE_SOURCE_CHANGED')
    dump(out/'figures/source_data_manifest.json',dict(source_hashes=sources,all10_methods_and35cells_retained=True,primary_families_separate=True,no_new_CI_family_or_statistics=True,new_predictions=0,new_random_draws=0,figures=plots,manual_QA='PENDING_ACTUAL_VIEW'))
    print(json.dumps(dict(PASS=True,figures=len(plots),manual_QA='PENDING_ACTUAL_VIEW'),indent=2))
if __name__=='__main__':main()


