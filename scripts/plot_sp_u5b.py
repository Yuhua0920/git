"""Three CSV-only measurement-control quantitative grids; style-only inheritance.

A: paired native/common penalty, all100 trajectories per method.
B: common adaptation gain; C: common BER with identical decoded-bit counts.
183mm, editable vector text, median/IQR or declared aggregate BER center.
"""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u5b_common import *
os.environ['MPLCONFIGDIR']=str(OUT/'sp_u5b_mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SKILL=Path('C:/Users/yuhua/.codex/skills/nature-figure');sys.path.insert(0,str(SKILL/'scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment
STYLE={'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
    'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'SP-U5B','savefig.dpi':300}
FOUT=OUT/'sp_u5b_figures';DFS=[32,16,8];X=np.array([1/64,1/32,1/16]);METHODS=['Source','LR4'];COLORS={'Source':'#747474','LR4':'#3068A6'}

def canvas(note):
    fig,axs=plt.subplots(2,2,figsize=(7.20472440945,5.5))
    fig.subplots_adjust(left=.11,right=.975,bottom=.20,top=.88,wspace=.36,hspace=.58)
    for i,ax in enumerate(axs.ravel()):ax.text(-.14,1.07,chr(97+i),transform=ax.transAxes,fontweight='bold',fontsize=9)
    fig.text(.11,.035,note,fontsize=7,va='bottom');return fig,axs

def export(fig,name):
    stem=FOUT/name;fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(stem)+'.alignment.json',overlay_svg=str(stem)+'.alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(str(stem)+'.png',dpi=300,metadata={'Software':'SP-U5B CSV-only'});fig.savefig(str(stem)+'.pdf',metadata={'CreationDate':None,'ModDate':None});fig.savefig(str(stem)+'.svg',metadata={'Date':None});plt.close(fig)

def pick(rows,**kw):return next(r for r in rows if all(r[k]==str(v) for k,v in kw.items()))
def legend(fig,ax):
    h,l=ax.get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.55,.995),ncol=4,frameon=False)
def errorline(ax,rr,**kw):
    v=np.array([float(r['median']) for r in rr]);lo=np.array([float(r['q25']) for r in rr]);hi=np.array([float(r['q75']) for r in rr])
    ax.errorbar(X,v,yerr=[v-lo,hi-v],marker='o',ms=3,lw=1,capsize=2,**kw)
def fractions(ax):
    ax.set_xticks(X,['1/64\ndF32','1/32\ndF16','1/16\ndF8']);ax.set_xlabel('Actual union pilot fraction')

def main():
    firewall();FOUT.mkdir(exist_ok=True)
    pa=csvread(OUT/'sp_u5b_paired_individual.csv');ga=csvread(OUT/'sp_u5b_gains.csv');bi=csvread(OUT/'sp_u5b_BER.csv');manifest=[]
    with plt.rc_context(STYLE):
        fig,axs=canvas('All100 paired trajectories; fixed U5A coordinates/operators. Native vs common dF32 minus dF16.\nThin lines connect the same trajectory; large markers/bars: median/trajectory IQR; no formal intervals.')
        for row,c in enumerate(('ID','Composed')):
            for col,s in enumerate((10,20)):
                ax=axs[row,col]
                for mi,m in enumerate(METHODS):
                    rr=sorted([r for r in pa if (r['condition'],r['snr_db'],r['contrast'],r['method'])==(c,str(s),'reduction32_minus16',m)],key=lambda r:int(r['trajectory_id']))
                    v=np.array([[float(r['native_db']),float(r['common_db'])] for r in rr]);offsets=np.linspace(-.09,.09,100);xs=np.array([mi*1.5,mi*1.5+.55])
                    for j in range(100):ax.plot(xs+offsets[j],v[j],color='#B8B8B8',lw=.35,alpha=.4)
                    for di,(domain,color) in enumerate((('Native','#747474'),('Common','#3068A6'))):
                        ax.scatter(xs[di]+offsets,v[:,di],s=4,color=color,alpha=.4)
                        q=np.quantile(v[:,di],[.25,.5,.75]);ax.errorbar([xs[di]],[q[1]],yerr=[[q[1]-q[0]],[q[2]-q[1]]],color=color,marker='o',ms=4,lw=1.5,capsize=3,label=domain if mi==0 else None)
                ax.axhline(0,color='#BBBBBB',lw=.6);ax.set_xticks([.275,1.775],METHODS);ax.set_xlabel('Density-specific estimator');ax.set_ylabel('Reduction penalty (dB)');ax.set_title(f'{c}: {s} dB',fontsize=9)
        legend(fig,axs[0,0]);export(fig,'A_Native_Common');manifest.append(dict(figure='A_Native_Common',panels=4,data='sp_u5b_paired_individual.csv',center='median',spread='all100 paired values and trajectory IQR'))
        fig,axs=canvas('Common data-domain Source minus LR4 gain;100 reused trajectories per condition.\nMedian and trajectory IQR;3840 identical channel targets/frame; lines guide the eye, no fitted curve.')
        for row,c in enumerate(('ID','Composed')):
            for col,s in enumerate((10,20)):
                ax=axs[row,col];errorline(ax,[pick(ga,condition=c,snr_db=s,dF=df) for df in DFS],color=COLORS['LR4'])
                ax.axhline(0,color='#BBBBBB',lw=.6);fractions(ax);ax.set_ylabel('Common adaptation gain (dB)');ax.set_title(f'{c}: {s} dB',fontsize=9)
        export(fig,'B_Common_Gain');manifest.append(dict(figure='B_Common_Gain',panels=4,data='sp_u5b_gains.csv',center='median',spread='trajectory IQR'))
        fig,axs=canvas('Frozen ZF/QPSK receiver; identical common REs and bits,3840 decoded bits/scored frame.\nLines: total errors/decoded bits. Vertical segments: trajectory BER IQR;100 reused trajectories/condition.')
        for row,c in enumerate(('ID','Composed')):
            for col,s in enumerate((10,20)):
                ax=axs[row,col]
                for m in METHODS:
                    rr=[pick(bi,condition=c,snr_db=s,dF=df,method=m) for df in DFS]
                    ax.plot(X,[float(r['BER']) for r in rr],'o-',color=COLORS[m],ms=3,lw=1,label=m)
                    ax.vlines(X,[float(r['trajectory_BER_q25']) for r in rr],[float(r['trajectory_BER_q75']) for r in rr],color=COLORS[m],alpha=.35,lw=2)
                fractions(ax);ax.set_ylabel('Common-domain BER');ax.set_title(f'{c}: {s} dB',fontsize=9)
        legend(fig,axs[0,0]);export(fig,'C_Common_BER');manifest.append(dict(figure='C_Common_BER',panels=4,data='sp_u5b_BER.csv',center='aggregate errors/bits',spread='trajectory BER IQR segments'))
    save(OUT/'sp_u5b_figure_manifest.json',dict(figures=manifest,panels=12,backend='python',width_mm=183,formats=['PNG300dpi','PDF','SVG'],no_exclusions=True,no_formal_intervals=True,style_only_inheritance=True))
if __name__=='__main__':main()
