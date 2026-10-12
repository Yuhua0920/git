"""CSV-only quantitative grids: overhead response, paired tradeoff, receiver and stability.

Six development figures; 183mm; median/trajectory IQR, all paired values in D.
No fitted sample-efficiency curve, formal intervals, exclusions or throughput metric.
Style-only inheritance; statistical logic and overhead mapping built for SP-U5A.
"""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u5a_common import *
os.environ['MPLCONFIGDIR']=str(OUT/'sp_u5a_mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SKILL=Path('C:/Users/yuhua/.codex/skills/nature-figure');sys.path.insert(0,str(SKILL/'scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment
STYLE={'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
    'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'SP-U5A','savefig.dpi':300}
FOUT=OUT/'sp_u5a_figures';DFS=[32,16,8];X=np.array([1/64,1/32,1/16]);METHODS=['Source','LR4'];COLORS={'Source':'#747474','LR4':'#3068A6'}
def canvas(rows=1,cols=2,note=''):
    fig,axs=plt.subplots(rows,cols,figsize=(7.20472440945,3.6 if rows==1 else 5.5),squeeze=False)
    fig.subplots_adjust(left=.11,right=.975,bottom=.25 if rows==1 else .20,top=.82 if rows==1 else .88,wspace=.36,hspace=.58)
    for i,ax in enumerate(axs.ravel()):
        ax.text(-.14,1.07,chr(97+i),transform=ax.transAxes,fontweight='bold',fontsize=9)
        ax.set_xticks(X,['1/64\ndF32','1/32\ndF16','1/16\ndF8']);ax.set_xlabel('Actual union pilot fraction')
    fig.text(.11,.035,note,fontsize=7,va='bottom');return fig,axs
def export(fig,name):
    stem=FOUT/name;fig.canvas.draw();require_matplotlib_panel_alignment(fig,json_out=str(stem)+'.alignment.json',overlay_svg=str(stem)+'.alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(str(stem)+'.png',dpi=300,metadata={'Software':'SP-U5A CSV-only'});fig.savefig(str(stem)+'.pdf',metadata={'CreationDate':None,'ModDate':None});fig.savefig(str(stem)+'.svg',metadata={'Date':None});plt.close(fig)
def legend(fig,ax):
    h,l=ax.get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.55,.995),ncol=4,frameon=False)
def pick(rows,**kw):return next(r for r in rows if all(r[k]==str(v) for k,v in kw.items()))
def errorline(ax,x,rr,field='median',lo='q25',hi='q75',**kw):
    m=np.array([float(r[field]) for r in rr]);low=np.array([float(r[lo]) for r in rr]);high=np.array([float(r[hi]) for r in rr])
    ax.errorbar(x,m,yerr=np.vstack((m-low,high-m)),marker='o',ms=3,lw=1,capsize=2,**kw)
def main():
    firewall();FOUT.mkdir(exist_ok=True);nm=csvread(OUT/'sp_u5a_nmse.csv');ga=csvread(OUT/'sp_u5a_table4_gains.csv');bi=csvread(OUT/'sp_u5a_table6_BER.csv');st=csvread(OUT/'sp_u5a_stability.csv');pa=csvread(OUT/'sp_u5a_paired_individual.csv');manifest=[]
    with plt.rc_context(STYLE):
        for cond,name in [('Composed','A_Composed_NMSE'),('ID','B_ID_NMSE')]:
            fig,axs=canvas(note='100 new paired trajectories per condition; trajectory NMSE median and IQR.\nEach geometry uses its actual data target. Lines guide the eye; no fitted overhead curve.')
            for ax,s in zip(axs.ravel(),(10,20)):
                for m in METHODS:errorline(ax,X,[pick(nm,condition=cond,snr_db=s,dF=df,method=m) for df in DFS],color=COLORS[m],label=m)
                ax.set_title(f'{cond}: {s} dB',fontsize=9);ax.set_ylabel('Data-domain NMSE (dB)')
            legend(fig,axs[0,0]);export(fig,name);manifest.append(dict(figure=name,panels=2,data='sp_u5a_nmse.csv',question=f'{cond} performance response',spread='trajectory IQR'))
        fig,axs=canvas(2,2,note='Same-density Source minus LR4 gain; median and trajectory IQR.\nPositive means adaptation improves NMSE;100 paired trajectories in each panel.')
        for row,c in enumerate(('ID','Composed')):
            for col,s in enumerate((10,20)):
                ax=axs[row,col];errorline(ax,X,[pick(ga,condition=c,snr_db=s,dF=df) for df in DFS],color=COLORS['LR4']);ax.axhline(0,color='#BBBBBB',lw=.6)
                ax.set_title(f'{c}: {s} dB',fontsize=9);ax.set_ylabel('Adaptation gain (dB)')
        export(fig,'C_Adaptation_gain');manifest.append(dict(figure='C_Adaptation_gain',panels=4,data='sp_u5a_table4_gains.csv',question='benefit survival and ID boundary',spread='trajectory IQR'))
        fig,axs=canvas(2,2,note='All100 paired trajectories per method; dF32 minus dF16 NMSE dB.\nBars: median and trajectory IQR. Positive means pilot reduction worsens NMSE.')
        for row,c in enumerate(('ID','Composed')):
            for col,s in enumerate((10,20)):
                ax=axs[row,col]
                for mi,m in enumerate(METHODS):
                    v=np.array([float(r['difference_db']) for r in pa if (r['condition'],r['snr_db'],r['contrast'],r['method'])==(c,str(s),'reduction32_minus16',m)])
                    q=np.quantile(v,[.25,.5,.75]);ax.scatter(mi+np.linspace(-.14,.14,len(v)),v,s=5,color=COLORS[m],alpha=.30)
                    ax.errorbar([mi],[q[1]],yerr=[[q[1]-q[0]],[q[2]-q[1]]],color=COLORS[m],marker='o',ms=4,capsize=4,lw=1.5)
                ax.axhline(0,color='#BBBBBB',lw=.6);ax.set_xticks([0,1],METHODS);ax.set_xlabel('Same paired density contrast');ax.set_ylabel('Pilot-reduction penalty (dB)');ax.set_title(f'{c}: {s} dB',fontsize=9)
        export(fig,'D_Paired_reduction');manifest.append(dict(figure='D_Paired_reduction',panels=4,data='sp_u5a_paired_individual.csv',question='paired reduction cost',spread='all100 points and IQR; no point sampling'))
        fig,axs=canvas(2,2,note='Frozen ZF/QPSK receiver on each actual data mask;100 trajectories per condition.\nLines: total errors / decoded bits. Vertical segments: trajectory BER IQR; no throughput metric.')
        for row,c in enumerate(('ID','Composed')):
            for col,s in enumerate((10,20)):
                ax=axs[row,col]
                for m in METHODS:
                    rr=[pick(bi,condition=c,snr_db=s,dF=df,method=m) for df in DFS]
                    ax.plot(X,[float(r['BER']) for r in rr],'o-',color=COLORS[m],ms=3,lw=1,label=m)
                    ax.vlines(X,[float(r['trajectory_BER_q25']) for r in rr],[float(r['trajectory_BER_q75']) for r in rr],color=COLORS[m],alpha=.35,lw=2)
                ax.set_title(f'{c}: {s} dB',fontsize=9);ax.set_ylabel('BER')
        legend(fig,axs[0,0]);export(fig,'E_BER');manifest.append(dict(figure='E_BER',panels=4,data='sp_u5a_table6_BER.csv',question='receiver consequences on actual bits',spread='trajectory BER IQR; aggregate center'))
        fig,axs=canvas(2,3,note='Descriptive600 causal windows per condition/density/SNR; correlated within100 trajectories.\nMedian and IQR; ID dashed, Composed solid; coordinates are unrestricted. No causal conditioning claim.')
        for row,s in enumerate((10,20)):
            for col,field in enumerate(('condition_number','coordinate_norm','alpha')):
                ax=axs[row,col]
                for ci,c in enumerate(('ID','Composed')):
                    rr=[pick(st,condition=c,snr_db=s,dF=df) for df in DFS];style='--' if ci==0 else '-'
                    if col<2:errorline(ax,X,rr,field+'_median',field+'_q25',field+'_q75',color=['#747474','#3068A6'][ci],ls=style,label=c)
                    else:
                        for ff,color in [('alpha','#9A6C9E'),('beta','#4B8C79')]:errorline(ax,X,rr,ff+'_median',ff+'_q25',ff+'_q75',color=color,ls=style,label=ff.capitalize()+' '+c)
                ax.set_title(f'{["Gram condition","Coordinate norm","Alpha/Beta"][col]}: {s} dB',fontsize=8)
                ax.set_xlabel('Pilot fraction');ax.set_ylabel(['Condition number','Coordinate norm','Coordinate'][col])
        # One explicit legend for condition line styles and coordinate colors.
        from matplotlib.lines import Line2D
        handles=[Line2D([],[],color='#747474',ls='--',label='ID dashed'),Line2D([],[],color='#3068A6',label='Composed solid'),Line2D([],[],color='#9A6C9E',label='Alpha'),Line2D([],[],color='#4B8C79',label='Beta')]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,.995),ncol=4,frameon=False)
        export(fig,'F_Stability');manifest.append(dict(figure='F_Stability',panels=6,data='sp_u5a_stability.csv',question='numerical robustness and coordinates',spread='correlated-window descriptive IQR; rank failures separately counted'))
    save(OUT/'sp_u5a_figure_manifest.json',dict(figures=manifest,backend='python',formats=['PNG300dpi','PDF','SVG'],no_exclusions=True,no_formal_intervals=True,style_only_inheritance=True))
if __name__=='__main__':main()
