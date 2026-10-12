"""Publication figures from certified persisted MR03C tables only.

Python/matplotlib; no inference, fitting, random sampling or performance edits.
Quantitative-grid figures preserve all five conditions and all three SNRs.
"""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,math,subprocess,sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CONDITIONS=('ID','Profile','Mobility','Composed','Boundary')
SNRS=(10,20,30)
METHODS=('SOURCE','POOLED_NATURAL','GMM_PREPARED_3','PPCA_DIRECT_R16','PPCA_DIRECT_R32','LR4_TW1','LR4_TW5','DENSE_TW5','WHOLE_POOL_M2_R24')
ABSOLUTE=('SOURCE','POOLED_NATURAL','GMM_PREPARED_3','PPCA_DIRECT_R32','LR4_TW1','LR4_TW5','DENSE_TW5','WHOLE_POOL_M2_R24')
SECOND=('POOLED_NATURAL','LR4_TW1','LR4_TW5')
STYLE={
'SOURCE':('Source','#999999','x',':'),
'POOLED_NATURAL':('Pooled','#555555','+','--'),
'GMM_PREPARED_3':('Prepared GMM-3','#0072B2','o','-'),
'PPCA_DIRECT_R32':('Direct PPCA r32','#009E73','s','--'),
'LR4_TW1':('LR4, Tw=1','#CC79A7','v','-.'),
'LR4_TW5':('LR4, Tw=5','#D55E00','^','-.'),
'DENSE_TW5':('Dense, Tw=5','#111111','D','-'),
'WHOLE_POOL_M2_R24':('Whole r24, Tw=5','#882255','*',':'),
}
PAIR_STYLE={'POOLED_NATURAL':('#555555','s','--'),'LR4_TW1':('#CC79A7','v','-.'),'LR4_TW5':('#D55E00','^','-')}

def digest(path):
    with Path(path).open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()

def js(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as h:r=list(csv.DictReader(h))
    if not r:raise ValueError('EMPTY_TABLE')
    return r

def csvwrite(path,records):
    with Path(path).open('w',encoding='utf8',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

def val(row,key):
    x=float(row[key])
    if not math.isfinite(x):raise ValueError('NONFINITE_TABLE:'+key)
    return x

def nmse_table(path,dataset):
    r=[z for z in rows(path) if z['dataset']==dataset];ix={(z['condition'],int(z['snr_db']),z['method']):z for z in r}
    if len(r)!=135 or len(ix)!=135 or set(ix)!={(c,s,m) for c in CONDITIONS for s in SNRS for m in METHODS}:raise ValueError('NMSE_GRID')
    for z in r:
        if z['dataset']!=dataset or int(z['trajectory_count'])!=100 or int(z['scored_frames'])!=600:raise ValueError('NMSE_SCOPE')
        if val(z,'nmse_linear')<=0 or val(z,'CI_lower')>val(z,'CI_upper'):raise ValueError('NMSE_NUMERIC')

    return ix

def contrast_table(path,dataset):
    r=[z for z in rows(path) if z['dataset']==dataset];ix={(z['condition'],int(z['snr_db']),z['first'],z['second']):z for z in r}
    if len(ix)!=len(r):raise ValueError('CONTRAST_DUPLICATE')
    for c in CONDITIONS:
        for s in SNRS:
            for second in SECOND:
                z=ix[c,s,'GMM_PREPARED_3',second]
                if z['dataset']!=dataset or int(z['trajectory_count'])!=100 or val(z,'CI_lower')>val(z,'CI_upper'):raise ValueError('CONTRAST_SCOPE')
                if 'positive favors comparator' not in z['sign']:raise ValueError('CONTRAST_DIRECTION')
    return ix

def letters(ax,t):
    ax.annotate(t,(0,1),xycoords='axes fraction',xytext=(-24,14),textcoords='offset points',ha='left',va='bottom',fontsize=8,fontweight='bold')

def intervals(ax,x,lo,hi,color):
    ax.vlines(x,lo,hi,color=color,linewidth=.65,alpha=.55)
    ax.hlines(lo,np.array(x)-.2,np.array(x)+.2,color=color,linewidth=.65,alpha=.55)
    ax.hlines(hi,np.array(x)-.2,np.array(x)+.2,color=color,linewidth=.65,alpha=.55)

def load_alignment(tools):
    spec=importlib.util.spec_from_file_location('gmm_alignment',tools/'audit_panel_alignment.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module

def run(command,path):
    p=subprocess.run(command,capture_output=True,text=True,encoding='utf8',errors='replace');Path(path).write_text(p.stdout+'\n'+p.stderr,encoding='utf8')
    return dict(command=command,returncode=p.returncode,log=str(path))

def save(fig,stem,out,tools,alignment,exclude=()):
    target=out/'figures';fig.canvas.draw()
    align=alignment.require_matplotlib_panel_alignment(fig,json_out=target/(stem+'.alignment.json'),overlay_svg=target/(stem+'.alignment.svg'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=True,strict=True,exclude_axes=list(exclude))
    paths=[]
    for suffix in ('.pdf','.svg','.png'):
        p=target/(stem+suffix);fig.savefig(p,dpi=600,facecolor='white');paths.append(p)
    plt.close(fig)
    text=run([sys.executable,str(tools/'audit_pdf_text.py'),str(paths[0]),'--min-pt','5','--json'],target/(stem+'.text.log'))
    collision=run([sys.executable,str(tools/'audit_figure_collisions.py'),str(paths[0]),'--json-out',str(target/(stem+'.collision.json')),'--overlay-pdf',str(target/(stem+'.collision.pdf'))],target/(stem+'.collision.log'))
    result=dict(stem=stem,alignment=align,text=text,collision=collision,exports=[dict(path=str(p),sha256=digest(p),bytes=p.stat().st_size) for p in paths],manual_visual_QA='PENDING_ACTUAL_INSPECTION',PNG_dpi=600,editable_PDF_SVG_text=True)
    dump(target/(stem+'.qa.json'),result)
    if text['returncode'] or collision['returncode']:raise ValueError('RENDERED_QA_FAILED:'+stem)
    return result

def grid(out,dataset,ix,ci,kind,tools,alignment):
    fig,axs=plt.subplots(2,3,figsize=(7.0866141732,5.5118110236),sharex=True,sharey=True)
    fig.subplots_adjust(left=.12,right=.98,bottom=.115,top=.905,wspace=.52,hspace=.65)
    selected=ABSOLUTE if kind=='absolute' else SECOND
    bounds=[];records=[]
    for c in CONDITIONS:
        for s in SNRS:
            for m in selected:
                z=ix[c,s,m] if kind=='absolute' else ci[c,s,'GMM_PREPARED_3',m]
                bounds.extend([val(z,'CI_lower'),val(z,'CI_upper'),val(z,'nmse_db' if kind=='absolute' else 'estimate_db')])
    low,high=min(bounds),max(bounds);span=max(high-low,1.);lim=(low-.07*span,high+.07*span)
    for k,c in enumerate(CONDITIONS):
        ax=axs.flat[k];letters(ax,chr(97+k));ax.set_title(c,fontsize=8,pad=6)
        for m in selected:
            z=[ix[c,s,m] if kind=='absolute' else ci[c,s,'GMM_PREPARED_3',m] for s in SNRS]
            if kind=='absolute':display,color,marker,linestyle=STYLE[m]
            else:
                color,marker,linestyle=PAIR_STYLE[m];display='GMM-3 − '+STYLE[m][0]
            y=[val(t,'nmse_db' if kind=='absolute' else 'estimate_db') for t in z]
            ax.plot(SNRS,y,color=color,marker=marker,linestyle=linestyle,linewidth=1.2 if m=='GMM_PREPARED_3' else .9,markersize=3.5,label=display)
            intervals(ax,SNRS,[val(t,'CI_lower') for t in z],[val(t,'CI_upper') for t in z],color)
            records.extend(dict(panel=chr(97+k),**t) for t in z)
        if kind!='absolute':ax.axhline(0,color='#777777',linewidth=.6,zorder=0)
        ax.set_xlim(8,32);ax.set_ylim(*lim);ax.set_xticks(SNRS);ax.set_xlabel('SNR (dB)',fontsize=8)
        ax.set_ylabel('Pooled NMSE (dB)' if kind=='absolute' else 'GMM-3 − comparator (dB)',fontsize=8)
        ax.tick_params(labelbottom=True,labelleft=True,labelsize=7.2);ax.grid(axis='y',color='#e5e5e5',linewidth=.4,zorder=0)
    leg=axs.flat[5];leg.set_axis_off();handles=[]
    for m in selected:
        if kind=='absolute':display,color,marker,linestyle=STYLE[m]
        else:color,marker,linestyle=PAIR_STYLE[m];display='GMM-3 − '+STYLE[m][0]
        handles.append(Line2D([],[],color=color,marker=marker,linestyle=linestyle,linewidth=.9,markersize=3.5,label=display))
    leg.legend(handles=handles,loc='upper left',bbox_to_anchor=(-.04,1.05),fontsize=7.4,labelspacing=.55,handlelength=2.3,handletextpad=.7)
    ypos=.10 if kind=='absolute' else .47
    leg.text(0,ypos,'100 trajectories / condition\n6 scored frames / trajectory\n95% exploratory intervals\nPreviously revealed development data',transform=leg.transAxes,fontsize=7.0,ha='left',va='top',linespacing=1.4)
    stem='gmm_'+kind+'_'+dataset;csvwrite(out/'figures/source_data'/(stem+'.csv'),records)
    dump(out/'figures'/(stem+'.brief.json'),dict(core_question=('Does observation-dependent prepared-origin mixing add value relative to fixed statistics and online coordinates?' if kind=='absolute' else 'Which paired comparisons favor prepared-origin mixing or coordinate refinement?'),dataset=dataset,archetype='quantitative grid',panels={chr(97+k):c for k,c in enumerate(CONDITIONS)},methods_displayed=list(selected),omitted_methods=(['PPCA_DIRECT_R16'] if kind=='absolute' else ['DENSE_TW5','WHOLE_POOL_M2_R24']),reason_for_subset=('All nine methods remain in full performance table; selected curves show current-frame and full-system controls.' if kind=='absolute' else 'Three core contrasts separate fixed statistics, same-frame online refinement, and full-system comparison; Dense/Whole contrasts remain in the complete table.'),physical_width_mm=180,all_conditions_snrs_and_trajectories_retained=True))
    caption=('Prepared-origin posterior mixture '+('absolute risk' if kind=='absolute' else 'paired risk differences')+f' on {dataset}. Panels a–e correspond to ID, Profile, Mobility, Composed and Boundary. Each point uses 100 physical trajectories with frames 4–9 scored, identical received pilots and truth for every method. '+('The metric is pooled linear error/truth NMSE converted to dB, lower is better.' if kind=='absolute' else 'Differences are GMM-3 NMSE(dB) minus comparator NMSE(dB); positive values favor the comparator. The zero line marks equal pooled risk.')+' Vertical marks are unadjusted exploratory 95% trajectory-paired percentile intervals from the same 2000 existing draws. Both datasets have been revealed and are development diagnostics, not final independent confirmation. Source, Pooled, PPCA, GMM-3 and LR4 Tw=1 use only current-frame observations; LR4 Tw=5, Dense and Whole r24 use four additional historical frames. Offline models and preparation interfaces also differ. Curves do not establish a causal mechanism or method-family superiority. Full nine-method tables and all five planned GMM contrast definitions are supplied separately.')
    (out/'figures'/(stem+'.caption.md')).write_text(caption+'\n',encoding='utf8')
    return save(fig,stem,out,tools,alignment,[leg])

def posterior(out,by_dataset,tools,alignment):
    fig,axs=plt.subplots(2,2,figsize=(7.0866141732,6.2992125984))
    fig.subplots_adjust(left=.16,right=.95,top=.93,bottom=.11,wspace=.55,hspace=.40)
    all_records=[]
    for di,dataset in enumerate(('V_new','D_eval')):
        ix={(z['condition'],int(z['snr_db'])):z for z in by_dataset[dataset]}
        if len(ix)!=15 or set(ix)!={(c,s) for c in CONDITIONS for s in SNRS}:raise ValueError('POSTERIOR_GRID')
        records=[ix[c,s] for c in CONDITIONS for s in SNRS]
        weights=np.array([[val(z,'mean_weight_Source'),val(z,'mean_weight_Profile'),val(z,'mean_weight_Mobility')] for z in records])
        certainty=np.array([[val(z,'mean_entropy_nats')/math.log(3),val(z,'mean_max_weight'),val(z,'hardlike_fraction_gamma_ge_095')] for z in records])
        if not np.allclose(weights.sum(axis=1),1,atol=1e-12,rtol=0) or min(weights.min(),certainty.min()) < -1e-12 or max(weights.max(),certainty.max()) > 1+1e-12:raise ValueError('POSTERIOR_BOUNDS')
        for j,(matrix,columns,title) in enumerate(((weights,('Source','Profile','Mobility'),dataset+' mean weights'),(certainty,('H / ln 3','Mean max','Max ≥ 0.95'),dataset+' concentration'))):
            ax=axs[di,j];letters(ax,chr(97+2*di+j));ax.set_title(title,fontsize=8,pad=8)
            image=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=1,aspect='auto',interpolation='none')
            ax.set_xticks(range(3),columns,fontsize=7.2);ax.tick_params(axis='x',length=0,pad=5)
            ax.set_yticks(range(15),[f'{z["condition"]} {z["snr_db"]}' for z in records],fontsize=7.0);ax.tick_params(axis='y',length=0,pad=5)
            for row in range(15):
                for col in range(3):ax.text(col,row,f'{matrix[row,col]:.2f}',ha='center',va='center',fontsize=7.0,color='white' if matrix[row,col]>.58 else '#111111')
            for edge in (2.5,5.5,8.5,11.5):ax.axhline(edge,color='white',linewidth=.7)
            all_records.extend(dict(panel=chr(97+2*di+j),kind='weights' if j==0 else 'concentration',**z) for z in records)
    cax=fig.add_axes([.34,.07,.45,.015]);bar=fig.colorbar(image,cax=cax,orientation='horizontal');bar.set_ticks([0,.25,.5,.75,1]);bar.ax.tick_params(labelsize=7.2,length=2);bar.set_label('Mean probability, normalized entropy or fraction',fontsize=8,labelpad=5)
    stem='gmm_posterior_diagnostics';csvwrite(out/'figures/source_data'/(stem+'.csv'),all_records)
    caption=('Prepared-origin posterior summaries across all development cells. Panels a,c show means of the three posterior responsibilities on V_new and D_eval. Panels b,d show natural-log entropy divided by ln(3) for a common 0–1 display, mean maximum posterior weight, and the fraction of scored frames with maximum weight ≥0.95. Original entropy in nats is retained in the source table. Each cell summarizes all 600 scored frames from 100 physical trajectories; these are descriptive summaries without confidence intervals or independent-frame claims. ID/Profile/Mobility have known offline correspondence to preparation origins, but condition labels never enter online inference. Composed and Boundary have no unique prepared-origin ground-truth class. Concentrated or low-entropy weights alone do not prove correct selection, error causation, or calibrated target probabilities. Uniform priors are fixed at 1/3; no value was tuned on either revealed dataset.')
    (out/'figures'/(stem+'.caption.md')).write_text(caption+'\n',encoding='utf8')
    dump(out/'figures'/(stem+'.brief.json'),dict(core_question='How does prepared-origin posterior concentration vary with mismatch condition and SNR?',archetype='quantitative grid',panels=dict(a='V_new origin allocation',b='V_new concentration',c='D_eval origin allocation',d='D_eval concentration'),all30_cells_retained=True,transform='Entropy is divided by ln(3) only for plotting; originalnats unchanged',intervals='none: descriptive frozen-input summaries',physical_width_mm=180))
    return save(fig,stem,out,tools,alignment,[cax])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--qa-tools',type=Path,default=Path(__file__).parent/'figure_qa');args=parser.parse_args();out=args.out.resolve();tools=args.qa_tools.resolve()
    (out/'figures/source_data').mkdir(parents=True,exist_ok=True)
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,'axes.labelsize':8,'xtick.labelsize':7.2,'ytick.labelsize':7.2,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42,'savefig.dpi':600,'text.usetex':False})
    sources={};plots=[];post={};alignment=load_alignment(tools)
    for dataset in ('V_new','D_eval'):
        receipt=out/'reports/analysis.json';meta=js(receipt)
        if meta['PASS'] is not True:raise ValueError('ANALYSIS_NOT_PASS')
        sources[str(receipt.relative_to(out))]=digest(receipt)
        p=out/'tables/nmse.csv';c=out/'tables/paired_contrasts.csv';q=out/'tables/posterior_diagnostics.csv'
        for path in (p,c,q):sources[str(path.relative_to(out))]=digest(path)
        ix=nmse_table(p,dataset);ci=contrast_table(c,dataset);post[dataset]=[r for r in rows(q) if r['dataset']==dataset]
        for kind in ('absolute','paired'):plots.append(grid(out,dataset,ix,ci,kind,tools,alignment))
    plots.append(posterior(out,post,tools,alignment))
    for relative,expected in sources.items():
        if digest(out/relative)!=expected:raise ValueError('READ_ONLY_SOURCE_DRIFT:'+relative)
    dump(out/'figures/source_data_manifest.json',dict(source_hashes=sources,source_tables_read_only=True,inference_executed=False,new_bootstrap_draws=0,all30_condition_SNR_cells_retained=True,figures=plots,manual_visual_QA='PENDING_ACTUAL_IMAGE_INSPECTION'))
    print(json.dumps(dict(PASS=True,figures=len(plots),manual_visual_QA='PENDING_ACTUAL_IMAGE_INSPECTION'),indent=2))

if __name__=='__main__':main()



