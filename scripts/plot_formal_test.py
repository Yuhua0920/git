"""CSV-only T20B figures, all five conditions; no generator or inference imports."""
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SKILL=Path('C:/Users/yuhua/.codex/skills/nature-figure/scripts')
sys.path.insert(0,str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment

CONDITIONS=('CDL-C-30-100','CDL-B-30-100','CDL-C-120-100','CDL-B-120-100','CDL-C-30-200')
LABELS=('ID C30/100','Profile B30/100','Mobility C120/100','Composed B120/100','Boundary C30/200')
METHODS=('Perfect CSI','R3 LS+linear','Source LMMSE','Hard','Convex','Dense Affine','Final LR4 Affine','Matched LMMSE reference')
DISPLAY=('Perfect CSI','LS+Linear','Source','Hard','Convex','Dense affine','Final LR4 affine','Matched reference')
COLORS=('#777777','#aaaaaa','#444444','#d6a945','#0072b2','#56b4e9','#d55e00','#009e73')
STYLE={'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,
       'axes.labelsize':7,'axes.titlesize':7,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,
       'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False}


def read(path):
    with Path(path).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))


def export(fig,path):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(path)+'.alignment.json',
        overlay_svg=str(path)+'.alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(str(path)+'.pdf')
    fig.savefig(str(path)+'.svg')
    fig.savefig(str(path)+'.png',dpi=600)
    plt.close(fig)


def plot(out):
    out=Path(out)
    boot=read(out/'t20b_formal_bootstrap.csv')
    ci={(r['environment'],int(r['snr_db']),r['comparison'],r['metric']):r for r in boot}
    with plt.rc_context(STYLE):
        for metric,title in (('nmse_db','Pooled data NMSE (dB)'),('ber','Data BER')):
            fig,axes=plt.subplots(1,5,figsize=(7.0866,3.2283),sharey=True)
            fig.subplots_adjust(left=.085,right=.99,bottom=.31,top=.86,wspace=.35)
            for column,(name,label,ax) in enumerate(zip(CONDITIONS,LABELS,axes)):
                for mi,(method,display,color) in enumerate(zip(METHODS,DISPLAY,COLORS)):
                    if metric=='nmse_db' and method=='Perfect CSI':continue
                    entries=[ci[name,s,method,metric] for s in (10,20)]
                    y=np.array([float(r['estimate']) for r in entries]);lo=np.array([float(r['ci95_lower']) for r in entries]);hi=np.array([float(r['ci95_upper']) for r in entries])
                    # Endpoints may not center the percentile interval exactly;
                    # draw absolute interval geometry without clipping to point.
                    ax.plot([10,20],y,marker='o',markersize=2.5,lw=1.2 if method=='Final LR4 Affine' else .8,color=color,label=display)
                    ax.vlines([10,20],lo,hi,color=color,lw=.7)
                ax.set_title(label,pad=14)
                ax.text(0,1.03,chr(97+column),transform=ax.transAxes,fontweight='bold',ha='left',va='bottom')
                ax.set_xticks([10,20]);ax.set_xlim(8,22);ax.set_xlabel('SNR (dB)')
                if metric=='ber' and all(float(r['estimate'])>0 and float(r['ci95_lower'])>0 for r in boot if r['metric']=='ber'):
                    ax.set_yscale('log')
            axes[0].set_ylabel(title)
            handles,labels=axes[-1].get_legend_handles_labels()
            fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.055),ncol=4,frameon=False,columnspacing=1.)
            fig.text(.5,.012,'1000 independent trajectories/condition; paired trajectory percentile 95% CI',ha='center',fontsize=7)
            if metric=='nmse_db':fig.text(.5,.965,'Perfect CSI: zero NMSE (dB N/A)',ha='center',fontsize=7)
            export(fig,out/('t20b_formal_nmse' if metric=='nmse_db' else 't20b_formal_ber'))
        fig,axes=plt.subplots(3,1,figsize=(7.0866,5.9055),sharex=True)
        fig.subplots_adjust(left=.12,right=.98,bottom=.18,top=.96,hspace=.65)
        comparisons=(('source_gain','Source − LR4 gain (dB)',0.),('convex_gain','Convex − LR4 gain (dB)',0.),('compression_penalty','LR4 − Dense penalty (dB)',.5))
        for ai,(ax,(comparison,label,threshold)) in enumerate(zip(axes,comparisons)):
            ax.axhline(threshold,color='#888888',ls='--',lw=.6)
            for index,name in enumerate(CONDITIONS):
                for si,snr in enumerate((10,20)):
                    r=ci[name,snr,comparison,'nmse_difference_db'];x=2*index+si
                    ax.vlines(x,float(r['ci95_lower']),float(r['ci95_upper']),color='#999999',lw=1.5)
                    ax.plot(x,float(r['estimate']),'o',color='#d55e00' if r['primary_endpoint'] else '#444444',ms=3)
                    if r['primary_endpoint']:
                        bound=float(r['primary_upper'] if comparison=='compression_penalty' else r['primary_lower'])
                        ax.plot(x,bound,'_',color='#0072b2',ms=7)
            ax.set_ylabel(label);ax.set_xlim(-.6,9.6)
            ax.text(-.10,1.05,chr(97+ai),transform=ax.transAxes,fontweight='bold',ha='left')
        ticks=[f'{label.split()[0]}\n{snr} dB' for label in LABELS for snr in (10,20)]
        axes[-1].set_xticks(range(10),ticks)
        fig.text(.5,.07,'Grey: paired trajectory 95% CI; blue: primary one-sided Bonferroni bound',ha='center',fontsize=7)
        fig.text(.5,.035,'All five conditions retained. Dashed: zero gain / preregistered 0.50-dB margin.',ha='center',fontsize=7)
        export(fig,out/'t20b_formal_endpoints')


if __name__=='__main__':
    plot(ROOT/'outputs' if len(sys.argv)==1 else Path(sys.argv[1]))
