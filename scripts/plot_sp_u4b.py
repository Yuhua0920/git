"""Five CSV-only replication figures, showing all five subset realizations.

Quantitative grids183mm wide; no replicate averaging hides raw curves.
A paired penalties; B Composed gains; C budget signal versus subset spread;
D operator decomposition; E ID method penalty and budget-tail diagnostics.
Development only; five offline subset realizations on100 reused trajectories.
"""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u4b_common import *
os.environ['MPLCONFIGDIR']=str(OUT/'sp_u4b_mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SKILL=Path('C:/Users/yuhua/.codex/skills/nature-figure');sys.path.insert(0,str(SKILL/'scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment
STYLE={'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,'axes.labelsize':8,
    'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'SP-U4B','savefig.dpi':300}
COLORS=['#3068A6','#BA7B3E','#4B8C79','#9A6C9E','#747474'];REPS=['R0','R1','R2','R3','R4'];X=np.array([50,100,250,500])
FOUT=OUT/'sp_u4b_figures'


def canvas(rows=1,cols=2,note=''):
    fig,axes=plt.subplots(rows,cols,figsize=(183/25.4,(85 if rows==1 else 135)/25.4),squeeze=False)
    fig.subplots_adjust(left=.10,right=.975,bottom=.24 if rows==1 else .18,top=.79 if rows==1 else .86,wspace=.35,hspace=.55)
    for i,ax in enumerate(axes.ravel()):
        ax.text(-.14,1.07,chr(97+i),transform=ax.transAxes,fontsize=9,fontweight='bold');ax.set_xticks(X)
        ax.set_xlabel('Clean trajectories per environment' if cols==2 else 'Nstat')
    fig.text(.10,.035,note,fontsize=7,va='bottom')
    return fig,axes


def legend(fig,axes):
    h,l=axes.ravel()[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.54,.99),ncol=5,frameon=False)


def export(fig,name):
    stem=FOUT/name;fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(stem)+'.alignment.json',overlay_svg=str(stem)+'.alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(str(stem)+'.png',dpi=300,metadata={'Software':'SP-U4B CSV-only'})
    fig.savefig(str(stem)+'.pdf',metadata={'CreationDate':None,'ModDate':None});fig.savefig(str(stem)+'.svg',metadata={'Date':None});plt.close(fig)


def main():
    firewall();FOUT.mkdir(exist_ok=True)
    summary=csvread(OUT/'sp_u4b_summary.csv');gains=csvread(OUT/'sp_u4b_table3_gains.csv');across=csvread(OUT/'sp_u4b_across_replicates.csv');ops=csvread(OUT/'sp_u4b_table5_operators.csv');manifest=[]
    def sel(rows,**kwargs):return sorted([r for r in rows if all(str(r[k])==str(v) for k,v in kwargs.items())],key=lambda r:int(r['budget']))
    with plt.rc_context(STYLE):
        fig,axes=canvas(note='Five actual subset realizations; n = 100 paired trajectories per curve point.\nMarkers: median paired LR4 penalty; whiskers: trajectory IQR; positive = worse than common N500.')
        for ax,s in zip(axes.ravel(),(10,20)):
            for ri,rep in enumerate(REPS):
                r=sel(summary,replicate=rep,condition='Composed',snr_db=s,method='LR4');m=np.array([float(v['penalty_median']) for v in r])
                ax.plot(X,m,'o-',color=COLORS[ri],ms=3,label=rep,lw=1)
                ax.errorbar(X+(ri-2)*2,m,yerr=np.array([[m[j]-float(v['penalty_q25']) for j,v in enumerate(r)],[float(v['penalty_q75'])-m[j] for j,v in enumerate(r)]]),fmt='none',ecolor=COLORS[ri],alpha=.3,lw=.7)
            ax.axhline(0,color='#BBBBBB',lw=.6);ax.set_title(f'Composed: {s} dB',fontsize=9);ax.set_ylabel('Paired budget penalty (dB)')
        legend(fig,axes);export(fig,'A_Composed_budget_penalty');manifest.append(dict(figure='A_Composed_budget_penalty',data='sp_u4b_summary.csv',spread='trajectory IQR in each preparation',alternative='ordering is based on paired differences, not marginal medians'))
        fig,axes=canvas(note='Five actual subset realizations,100 reused Composed trajectories each.\nMedian same-budget Source minus LR4 gain; whiskers: trajectory IQR; connecting lines guide the eye.')
        for ax,s in zip(axes.ravel(),(10,20)):
            for ri,rep in enumerate(REPS):
                r=sel(gains,replicate=rep,condition='Composed',snr_db=s);m=np.array([float(v['median']) for v in r]);ax.plot(X,m,'o-',label=rep,color=COLORS[ri],ms=3,lw=1)
                ax.errorbar(X+(ri-2)*2,m,yerr=np.array([[m[j]-float(v['q25']) for j,v in enumerate(r)],[float(v['q75'])-m[j] for j,v in enumerate(r)]]),fmt='none',ecolor=COLORS[ri],alpha=.3,lw=.7)
            ax.set_title(f'Composed: {s} dB',fontsize=9);ax.set_ylabel('Same-budget adaptation gain (dB)')
        legend(fig,axes);export(fig,'B_Composed_adaptation_gain');manifest.append(dict(figure='B_Composed_adaptation_gain',data='sp_u4b_table3_gains.csv',spread='trajectory IQR; no formal interval',alternative='all preparations share fixed evaluation conditions'))
        fig,axes=canvas(note='LR4 Composed central metric = trajectory NMSE median; all five displacements shown.\nB = median displacement; V = IQR across five preparation medians; no variance-component theorem.')
        for ax,s in zip(axes.ravel(),(10,20)):
            r=sel(across,condition='Composed',snr_db=s,method='LR4')
            for ri in range(5):
                raw=np.array([json.loads(v['raw5_central_medians'])[ri] for v in r]);raw-=raw[-1]
                ax.scatter(X+(ri-2)*3,raw,color=COLORS[ri],s=10,alpha=.55)
            ax.plot(X,[float(v['budget_displacement_B']) for v in r],'o-',color='#202020',label='B: Median displacement',ms=3)
            ax.plot(X,[float(v['subset_spread_IQR_V']) for v in r],'s--',color='#BA7B3E',label='V: Subset IQR',ms=3)
            ax.axhline(0,color='#BBBBBB',lw=.6);ax.set_title(f'{s} dB',fontsize=9);ax.set_ylabel('Displacement / subset spread (dB)')
        h,l=axes.ravel()[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.54,.99),ncol=2,frameon=False)
        export(fig,'C_Budget_vs_subset');manifest.append(dict(figure='C_Budget_vs_subset',data='sp_u4b_across_replicates.csv',spread='all5 raw points and subset IQR',alternative='median displacement differs from median paired penalty'))
        fig,axes=canvas(2,4,note='Five curves per module: R0-R4 colors. Corrections/CV: P solid, M dashed.\nRelative Frobenius errors versus common N500; effective rank-4 products; one offline object per preparation.')
        families=[('Source W',['Ws']),('Dense P/M',['BP','BM']),('Rank-4 P/M',['BP4','BM4']),('CV P/M',['DeltaKP','DeltaKM'])]
        for row,s in enumerate((10,20)):
            for col,(title,fields) in enumerate(families):
                ax=axes[row,col]
                for ri,rep in enumerate(REPS):
                    r=sel(ops,replicate=rep,snr_db=s)
                    for fi,field in enumerate(fields):ax.plot(np.arange(4),[float(v[field]) for v in r]+[0],marker='o',ms=2,color=COLORS[ri],ls='-' if fi==0 else '--',lw=.8,label=rep if fi==0 else None)
                ax.set_xticks(np.arange(4),X)
                ax.set_title(f'{title}: {s} dB',fontsize=8)
                if col==0:ax.set_ylabel('Relative error')
        legend(fig,axes);export(fig,'D_Operator_replication');manifest.append(dict(figure='D_Operator_replication',data='sp_u4b_table5_operators.csv',spread='all5 actual preparations; no trajectory uncertainty for fixed operator',alternative='different reference norms; weighted compression co-varies'))
        fig,axes=canvas(2,2,note='Top: ID median LR4-minus-Source degradation, with trajectory IQR.\nBottom: positive LR4 budget penalty versus common N500; P95 dashed, maximum solid; all five preparations.')
        for col,s in enumerate((10,20)):
            for ri,rep in enumerate(REPS):
                r=sel(gains,replicate=rep,condition='ID',snr_db=s);m=np.array([float(v['degradation_median']) for v in r]);axes[0,col].plot(X,m,'o-',color=COLORS[ri],ms=3,label=rep,lw=1)
                lo=np.array([-float(v['q75']) for v in r]);hi=np.array([-float(v['q25']) for v in r])
                axes[0,col].errorbar(X+(ri-2)*2,m,yerr=np.vstack((m-lo,hi-m)),fmt='none',ecolor=COLORS[ri],alpha=.3,lw=.7)
                r=sel(summary,replicate=rep,condition='ID',snr_db=s,method='LR4')
                axes[1,col].plot(X,[float(v['maximum_positive_penalty']) for v in r],'o-',color=COLORS[ri],ms=3,lw=1)
                axes[1,col].plot(X,[max(0,float(v['penalty_p95'])) for v in r],'--',color=COLORS[ri],lw=.8)
            axes[0,col].set_title(f'ID method penalty: {s} dB',fontsize=9);axes[0,col].set_ylabel('LR4 minus Source (dB)')
            axes[1,col].set_title(f'ID budget tails: {s} dB',fontsize=9);axes[1,col].set_ylabel('Positive budget penalty (dB)')
        legend(fig,axes);export(fig,'E_ID_and_tails');manifest.append(dict(figure='E_ID_and_tails',data='sp_u4b_table3_gains.csv + sp_u4b_summary.csv',spread='trajectory IQR and empirical tail endpoints',alternative='method penalty versus subset-budget interaction'))
    save(OUT/'sp_u4b_figure_manifest.json',dict(figures=manifest,backend='python',formats=['300dpi PNG','editable PDF','editable SVG'],all5_replicates_displayed=True,no_exclusions=True))


if __name__=='__main__':main()
