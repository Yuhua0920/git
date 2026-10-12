"""Publication diagnostic figures from frozen MR10 tables/raw arrays, no new statistics."""
import json,csv,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from run_mr10 import OUT,COND,read,csvrows
from audit_panel_alignment import require_matplotlib_panel_alignment
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00']
MARKERS=['o','s','^','D','v']
STYLE={'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
       'legend.fontsize':8,'axes.titlesize':9,'axes.linewidth':.7,'lines.linewidth':1.4,
       'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':600}
def ecdf(ax,v,label,color,style='-',marker=None):
    # Display empirical quantiles at 1001 fixed probability levels; no data truncation.
    xx=np.asarray(v).ravel();p=np.linspace(0,1,1001)
    ax.plot(np.quantile(xx,p),p,label=label,color=color,linestyle=style,marker=marker,markevery=100,markersize=3)
def decorate(ax,panel):
    ax.text(0,1.04,panel,transform=ax.transAxes,fontweight='bold',fontsize=10)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(direction='out',length=3)
def export(fig,name,axes,rows,cols):
    fig.canvas.draw()
    panels=[]
    for k,ax in enumerate(axes):
        box=ax.get_position();w,h=fig.get_size_inches()*72
        panels.append({'id':chr(97+k),'bbox_pt':[box.x0*w,box.y0*h,box.x1*w,box.y1*h],
          'grid_id':name,'row_start':k//cols,'row_stop':k//cols+1,'col_start':k%cols,'col_stop':k%cols+1})
    layout={'schema_version':1,'backend':'matplotlib','figure':{'width_pt':float(w),'height_pt':float(h)},'panels':panels}
    (OUT/'reports'/f'{name}_layout.json').write_text(json.dumps(layout,indent=2),encoding='utf8')
    require_matplotlib_panel_alignment(fig,json_out=OUT/'reports'/f'{name}_render_alignment.json',tolerance_pt=1.5)
    fig.savefig(OUT/'figures'/f'{name}.pdf',facecolor='white')
    fig.savefig(OUT/'figures'/f'{name}.svg',facecolor='white')
    fig.savefig(OUT/'figures'/f'{name}.png',dpi=600,facecolor='white')
    plt.close(fig)
def figure_receiver_risk(raw):
    power=csvrows(OUT/'tables/receive_power_actual_vs_reference.csv')
    with plt.rc_context(STYLE):
        fig,axs=plt.subplots(2,2,figsize=(7.2,5.6));fig.subplots_adjust(left=.095,right=.97,bottom=.10,top=.95,wspace=.32,hspace=.43)
        ax=axs[0,0]
        for s,color,style in zip((10,20),COLORS,('-','--')):ecdf(ax,[float(r['gamma_actual_realized_db']) for r in power if int(r['snr_db'])==s],f'Source SNR {s} dB',color,style)
        ax.set(xlabel='Useful received SNR (dB)',ylabel='Trajectory empirical CDF',ylim=(0,1));ax.legend(loc='lower right',frameon=False)
        ax=axs[0,1]
        for si,s in enumerate((10,20)):ecdf(ax,np.stack([r['zf_effective_esn0_db'][si] for r in raw]),f'Source SNR {s} dB',COLORS[si],('-','--')[si])
        ax.axvline(0,color='0.5',linestyle=':',linewidth=.8);ax.set(xlabel='Post-ZF stream Es/N0 (dB)',ylabel='Stream/RE empirical CDF',ylim=(0,1));ax.legend(loc='upper left',frameon=False)
        ax=axs[1,0];ecdf(ax,np.stack([r['zf_noise_gain'].sum(axis=-1) for r in raw]),'True-channel ZF gain',COLORS[2])
        ax.set(xscale='log',xlabel='ZF noise gain trace',ylabel='RE empirical CDF',ylim=(0,1))
        ax=axs[1,1]
        rr=csvrows(OUT/'tables/receive_power_SNR_trajectory.csv')
        for s,color,marker in zip((10,20),COLORS,MARKERS):
            v=[r for r in rr if int(r['snr_db'])==s]
            ax.scatter([float(r['conditional_ZF_expected_BER']) for r in v],[float(r['observed_Perfect_ZF_BER']) for r in v],s=13,marker=marker,color=color,alpha=.7,label=f'{s} dB')
        ax.plot([0,.5],[0,.5],color='0.35',linestyle='--',linewidth=.8)
        ax.set(xlabel='Conditional predicted Perfect-ZF BER',ylabel='Observed trajectory BER',xlim=(0,.5),ylim=(0,.5));ax.legend(loc='upper left',frameon=False)
        for k,ax in enumerate(axs.ravel()):decorate(ax,chr(97+k))
        export(fig,'receiver_SNR_ZF_risk',list(axs.ravel()),2,2)
def figure_detectors():
    rr=csvrows(OUT/'tables/detector_BER.csv');names=['Perfect CSI / ZF','Perfect CSI / MMSE','Perfect CSI / bit-MAP','LS-LIN / ZF','LR4 / ZF','LS-LIN / MMSE','LR4 / MMSE']
    colors=[COLORS[0],COLORS[0],COLORS[0],COLORS[1],COLORS[2],COLORS[1],COLORS[2]];detector_markers=['o','s','D','o','o','s','s']
    with plt.rc_context(STYLE):
        fig,axs=plt.subplots(1,2,figsize=(7.2,3.6));fig.subplots_adjust(left=.225,right=.98,bottom=.15,top=.86,wspace=.90)
        for si,(s,ax) in enumerate(zip((10,20),axs)):
            v=[float(r['BER']) for r in rr if int(r['snr_db'])==s]
            bars=ax.barh(np.arange(7),v,color=colors,edgecolor='0.15',linewidth=.4,height=.62)
            for k,value in enumerate(v):ax.plot(value,k,marker=detector_markers[k],color='black',markersize=3,linestyle='none')
            for k,value in enumerate(v):ax.text(value+.008,k,f'{value:.3f}',va='center',fontsize=8)
            ax.set_yticks(np.arange(7),names);ax.invert_yaxis();ax.set(xlabel='BER',xlim=(0,.5),title=f'{s} dB')
            decorate(ax,chr(97+si))
        export(fig,'same_input_detector_BER',list(axs),1,2)
def figure_coordinates():
    with np.load(OUT/'data/coordinate_windows.npz',allow_pickle=False) as z:dat={k:z[k] for k in z.files}
    with plt.rc_context(STYLE):
        fig,axs=plt.subplots(2,2,figsize=(7.2,5.7));fig.subplots_adjust(left=.10,right=.97,bottom=.16,top=.95,wspace=.31,hspace=.44)
        for ci,co in enumerate(COND):
            coef=dat['coordinates'][ci];kap=dat['kappa'][ci]
            ecdf(axs[0,0],np.linalg.norm(coef,axis=-1),co,COLORS[ci],['-','--','-.',':','-'][ci],MARKERS[ci])
            ecdf(axs[0,1],kap,co,COLORS[ci],['-','--','-.',':','-'][ci],MARKERS[ci])
            axs[1,0].scatter(coef[...,0].ravel(),coef[...,1].ravel(),s=4,color=COLORS[ci],marker=MARKERS[ci],alpha=.25,rasterized=True)
            tx=np.median(kap,axis=-1).ravel();yy=(dat['LR4_error'][ci].sum(axis=-1)/dat['truth'][ci].sum(axis=-1)).ravel()
            axs[1,1].scatter(tx,yy,s=9,color=COLORS[ci],marker=MARKERS[ci],alpha=.4,rasterized=True)
        axs[0,0].set(xlabel='Coordinate norm',ylabel='Window empirical CDF',ylim=(0,1))
        axs[0,1].set(xlabel='Gram condition number',ylabel='Window empirical CDF',xscale='log',ylim=(0,1))
        axs[1,0].set(xlabel='alpha',ylabel='beta')
        axs[1,1].set(xlabel='Trajectory-median Gram condition',ylabel='Trajectory pooled LR4 NMSE',xscale='log',yscale='log')
        legend=[Line2D([],[],color=COLORS[k],marker=MARKERS[k],linestyle='-',label=co,markersize=4) for k,co in enumerate(COND)]
        fig.legend(handles=legend,ncol=5,loc='lower center',bbox_to_anchor=(.5,.025),frameon=False,columnspacing=1.3)
        for k,ax in enumerate(axs.ravel()):decorate(ax,chr(97+k))
        export(fig,'coordinate_Gram_risk',list(axs.ravel()),2,2)
def main():
    assert read(OUT/'reports/independent_numeric_review.json')['PASS']
    raw=[]
    for i in range(100):
        with np.load(OUT/f'data/receiver/{i:04d}.npz',allow_pickle=False) as z:raw.append({k:z[k] for k in z.files})
    figure_receiver_risk(raw);figure_detectors();figure_coordinates()
    print('Three figures exported as PDF, SVG and 600dpi PNG; final layout manifests saved')
if __name__=='__main__':main()
