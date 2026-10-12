"""Six CSV-only development figures. Median/IQR and P05/P95 are descriptive.

Contract: A/B quantify primary/safeguard sensitivity; C paired adaptation gains;
D operator changes; E paired coordinate deviations; F measured offline costs.
Quantitative grids, 183 mm wide, editable vector exports, no excluded trajectories.
Single nested subset realization; no inferential confidence interval or tuning.
"""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u4a_common import *
os.environ['MPLCONFIGDIR']=str(OUT/'sp_u4a_mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SKILL=Path('C:/Users/yuhua/.codex/skills/nature-figure')
sys.path.insert(0,str(SKILL/'scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

STYLE={'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
    'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'SP_U4A_STAT_DEV_v1',
    'savefig.dpi':300}
COLORS={'Source':'#858585','LR4':'#3068A6','ID':'#858585','Composed':'#3068A6'}
X=np.array([50,100,250,500]);FOUT=OUT/'sp_u4a_figures'


def canvas(note):
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,85/25.4))
    fig.subplots_adjust(left=.10,right=.97,bottom=.27,top=.79,wspace=.34)
    for i,ax in enumerate(axes):
        ax.text(-.14,1.06,chr(97+i),transform=ax.transAxes,fontweight='bold',fontsize=9)
        ax.set_xticks(X);ax.set_xlabel('Clean trajectories per environment')
    fig.text(.1,.07,note,fontsize=7,va='bottom')
    return fig,axes


def spread(ax,rows,label,color,offset=0.):
    rows=sorted(rows,key=lambda r:int(r['budget']));x=np.array([int(r['budget']) for r in rows],float)+offset
    med=np.array([float(r['median']) for r in rows]);lo=np.array([float(r['q25']) for r in rows]);hi=np.array([float(r['q75']) for r in rows])
    p05=np.array([float(r['p05']) for r in rows]);p95=np.array([float(r['p95']) for r in rows])
    ax.plot(x,med,'o-',color=color,label=label,lw=1.2,ms=3)
    ax.errorbar(x,med,yerr=np.vstack((med-p05,p95-med)),fmt='none',ecolor=color,alpha=.35,lw=.65,capsize=2)
    ax.errorbar(x,med,yerr=np.vstack((med-lo,hi-med)),fmt='none',ecolor=color,lw=2,capsize=0)


def export(fig,name):
    stem=FOUT/name
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(stem)+'.alignment.json',overlay_svg=str(stem)+'.alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(str(stem)+'.png',dpi=300,metadata={'Software':'SP-U4A CSV-only renderer'})
    fig.savefig(str(stem)+'.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(str(stem)+'.svg',metadata={'Date':None})
    plt.close(fig)


def main():
    firewall();FOUT.mkdir(exist_ok=True)
    nmse=csvread(OUT/'sp_u4a_nmse_summary.csv');gains=csvread(OUT/'sp_u4a_table4_gains.csv')
    ops=csvread(OUT/'sp_u4a_table5_operators.csv');coords=csvread(OUT/'sp_u4a_coordinate_summary.csv');cost=csvread(OUT/'sp_u4a_table1_costs.csv')
    manifest=[]
    with plt.rc_context(STYLE):
        for cond,name in (('Composed','A_Composed_NMSE'),('ID','B_ID_NMSE')):
            fig,axes=canvas('n = 100 trajectories per condition/SNR; median, thick IQR, thin P05/P95.\nSingle nested subset realization; all trajectories retained; connecting lines guide the eye.')
            for ax,s in zip(axes,(10,20)):
                for method in ('Source','LR4'):
                    rows=[r for r in nmse if (r['condition'],int(r['snr_db']),r['method'])==(cond,s,method) and r['budget']!='Original500']
                    spread(ax,rows,'Source-LMMSE' if method=='Source' else 'LR4-Affine',COLORS[method],-3 if method=='Source' else 3)
                ax.set_title(f'{cond}: {s} dB',fontsize=9);ax.set_ylabel('Trajectory NMSE (dB)')
            handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,.99),ncol=2,frameon=False)
            export(fig,name);manifest.append(dict(figure=name,data='sp_u4a_nmse_summary.csv',claim=f'{cond} budget sensitivity',spread='trajectory median/IQR/P05/P95',alternative='single subset and condition channel-strength variation'))
        fig,axes=canvas('n = 100 paired trajectories; same-budget Source minus LR4, positive = improvement.\nMedian, thick IQR, thin P05/P95; no formal-test interpretation.')
        for ax,s in zip(axes,(10,20)):
            for cond in ('ID','Composed'):spread(ax,[r for r in gains if (r['condition'],int(r['snr_db']))==(cond,s)],cond,COLORS[cond],-3 if cond=='ID' else 3)
            ax.axhline(0,color='#BBBBBB',lw=.65,zorder=0);ax.set_title(f'{s} dB',fontsize=9);ax.set_ylabel('Same-budget adaptation gain (dB)')
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,.99),ncol=2,frameon=False)
        export(fig,'C_Adaptation_gain');manifest.append(dict(figure='C_Adaptation_gain',data='sp_u4a_table4_gains.csv',claim='gain survival under reduced budget',spread='paired trajectory median/IQR/P05/P95',alternative='compression and fitting kernels change jointly'))
        fig,axes=canvas('One frozen nested subset sequence; relative Frobenius errors versus rebuilt N = 500.\nEffective rank-4 products compared; factors are not compared directly; no uncertainty interval.')
        series=[('WSource_relative_error','Source W','#858585','-'),('BP_relative_error','Dense P','#D1976A','-'),('BM_relative_error','Dense M','#8D729D','-'),('BP4_relative_error','Rank-4 P','#B76D30','--'),('BM4_relative_error','Rank-4 M','#67457E','--')]
        for ax,s in zip(axes,(10,20)):
            rows=sorted([r for r in ops if int(r['snr_db'])==s],key=lambda r:int(r['budget']))
            for field,label,color,ls in series:ax.plot(X,[float(r[field]) for r in rows],marker='o',ms=3,color=color,ls=ls,label=label,lw=1.1)
            ax.set_title(f'{s} dB',fontsize=9);ax.set_ylabel('Relative operator error')
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,1),ncol=3,frameon=False)
        export(fig,'D_Operator_errors');manifest.append(dict(figure='D_Operator_errors',data='sp_u4a_table5_operators.csv',claim='which operators are sensitive',spread='none: one offline subset sequence',alternative='different denominators and origin-weighted SVD sensitivity'))
        fig,axes=canvas('n = 100 trajectories; each value averages six paired causal-window deviations.\nMedian, thick IQR, thin P05/P95; exploratory diagnostic, no causal attribution.')
        for ax,s in zip(axes,(10,20)):
            for cond in ('ID','Composed'):spread(ax,[r for r in coords if (r['condition'],int(r['snr_db']))==(cond,s)],cond,COLORS[cond],-3 if cond=='ID' else 3)
            ax.set_title(f'{s} dB',fontsize=9);ax.set_ylabel('Coordinate deviation from N = 500')
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,.99),ncol=2,frameon=False)
        export(fig,'E_Coordinate_deviation');manifest.append(dict(figure='E_Coordinate_deviation',data='sp_u4a_coordinate_summary.csv',claim='online coordinate sensitivity',spread='trajectory mean-window deviation distribution',alternative='current pilot observations and co-varying CV kernels'))
        fig,axes=canvas('One fresh construction process per budget on this machine; no runtime replicates.\nProcess peak includes corpus mapping, construction and serialization; clean generation is separate.')
        bottom=np.zeros(4)
        for field,label,color in (('stats_seconds','Statistics','#3068A6'),('operator_seconds','Operators','#9AB6D5'),('compression_seconds','Compression','#D1976A')):
            y=np.array([float(r[field]) for r in cost]);axes[0].bar(np.arange(4),y,bottom=bottom,color=color,label=label,width=.62);bottom+=y
        axes[0].set_xticks(np.arange(4),X);axes[0].set_ylabel('Construction wall time (s)')
        axes[1].plot(X,[float(r['peak_working_set_bytes'])/2**30 for r in cost],'o-',label='Peak working set',color='#3068A6',ms=3)
        axes[1].plot(X,[float(r['peak_commit_bytes'])/2**30 for r in cost],'s--',label='Peak commit',color='#858585',ms=3)
        axes[1].set_ylabel('Process peak memory (GiB)')
        handles0,labels0=axes[0].get_legend_handles_labels();handles1,labels1=axes[1].get_legend_handles_labels()
        fig.legend(handles0+handles1,labels0+labels1,loc='upper center',bbox_to_anchor=(.53,1),ncol=3,frameon=False)
        export(fig,'F_Offline_cost');manifest.append(dict(figure='F_Offline_cost',data='sp_u4a_table1_costs.csv',claim='measured offline cost',spread='none: single process execution per budget',alternative='filesystem cache and verification overhead; no hardware-general claim'))
    save(OUT/'sp_u4a_figure_manifest.json',dict(backend='python',figures=manifest,export_formats=['300dpi PNG','editable PDF','editable SVG'],
        no_data_exclusions=True,visual_review_pending=True,script_sha256=sha(Path(__file__))))


if __name__=='__main__':main()
