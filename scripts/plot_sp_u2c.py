"""Six Python figures, all conditional windows/noise repeats retained.

Contract: A/B distinguish absolute excitation and condition; C decomposes
conditional signed risk using all repeated Q,C,delta observations; D shows
conditional persistence; E compares M-metric cosine by signed-risk group; F
compares surrogate gap and magnitude of conditional noise-coordinate effect.
Quantitative diagnostic grid, development only; 183 mm width, 300 dpi PNG,
editable PDF/SVG, 8 pt minimum text. No inferential intervals: raw conditional
points, full empirical distributions and repeated-noise heatmaps are shown.
"""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import sp_u2c_diagnostic as d
d.install_guard()
os.environ.setdefault('MPLCONFIGDIR',str(d.OUT/'sp_u2c_mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,TwoSlopeNorm
import numpy as np
SKILL=Path('C:/Users/yuhua/.codex/skills/nature-figure/scripts')
sys.path.insert(0,str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment

STYLE={'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,
       'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
       'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':.8,
       'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'SP-U2C-conditional-v1','legend.frameon':False}
COLORS={10:'#527CA5',20:'#BD794D'}
MARKERS={10:'o',20:'^'}


def simple(letter,title):
    fig,ax=plt.subplots(figsize=(183/25.4,125/25.4))
    fig.subplots_adjust(left=.17,right=.96,bottom=.22,top=.79)
    fig.text(.17,.94,f'Fig {letter} | {title}',fontsize=10)
    fig.text(.17,.07,'20 trajectory clusters; 6 overlapping windows; 20 noise repeats; paired 10/20 dB',fontsize=8)
    return fig,ax


def export(letter,fig,source,count,excluded,exclude_axes=()):
    path=d.OUT/f'sp_u2c_figures/fig_{letter}'
    path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.with_suffix('.pdf').exists()
    report=require_matplotlib_panel_alignment(fig,json_out=str(path)+'.alignment.json',
             overlay_svg=str(path)+'.alignment.svg',strict=True,exclude_axes=list(exclude_axes))
    fig.savefig(str(path)+'.png',dpi=300)
    fig.savefig(str(path)+'.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(str(path)+'.svg',metadata={'Date':None})
    plt.close(fig)
    return dict(figure=letter,png=str(path)+'.png',pdf=str(path)+'.pdf',svg=str(path)+'.svg',
                raw_source=str(source),raw_source_sha256=d.sha(source),count=count,excluded=excluded,
                cluster='20 trajectories; overlapping windows; independent noise repeats conditional on fixed window; SNR paired',
                uncertainty='descriptive full empirical observations; no inferential error bars',alignment_verdict=report.get('verdict'),
                script_sha256=d.sha(Path(__file__)),visual_QA='pending')


def main():
    conditional=d.read_rows(d.OUT/'sp_u2c_conditional_windows.csv')
    raw=d.read_rows(d.OUT/'sp_u2c_repeat_rows.csv')
    assert len(conditional)==240 and len(raw)==4800
    manifest=[]
    with plt.rc_context(STYLE):
        for letter,pred,title in [('A','clean_sigma_min','Clean excitation and repeat sensitivity'),('B','clean_kappa_G','Condition and repeat sensitivity')]:
            fig,ax=simple(letter,title)
            for s in d.SNRS:
                group=[a for a in conditional if a['snr_db']==s]
                x=np.array([a[pred] for a in group]);y=np.array([a['RMS_c'] for a in group])
                assert len(x)==120 and np.all(x>0) and np.all(y>0)
                ax.scatter(x,y,s=18,color=COLORS[s],marker=MARKERS[s],alpha=.6,edgecolors='none',label=f'{s} dB (120 windows)')
            ax.set_xscale('log');ax.set_yscale('log')
            ax.set_xlabel('Clean design sigma min' if letter=='A' else 'Clean design condition (G)')
            ax.set_ylabel('Conditional RMS coordinate displacement')
            ax.legend(loc='upper center',bbox_to_anchor=(.5,1.25),ncol=2)
            manifest.append(export(letter,fig,d.OUT/'sp_u2c_conditional_windows.csv',240,0))
        fig,axes=plt.subplots(2,2,figsize=(183/25.4,210/25.4))
        fig.subplots_adjust(left=.13,right=.88,bottom=.13,top=.85,wspace=.48,hspace=.63)
        fig.text(.13,.955,'Fig C | Movement cost and oracle cross term',fontsize=10)
        fig.text(.13,.04,'All 240 fixed windows x 20 repeats; rows ordered by trajectory, SNR, score frame',fontsize=8)
        for label,ax in zip('abcd',axes.flat):
            ax.annotate(label,xy=(0,1),xytext=(-17,12),textcoords='offset points',xycoords='axes fraction',fontweight='bold')
        ax=axes[0,0]
        for s in d.SNRS:
            group=[a for a in conditional if a['snr_db']==s]
            ax.scatter([a['mean_Q'] for a in group],[a['mean_C'] for a in group],s=10,alpha=.65,
                       color=COLORS[s],marker=MARKERS[s],edgecolors='none',label=f'{s} dB')
        allQ=[a['mean_Q'] for a in conditional];lo,hi=min(allQ),max(allQ)
        ax.plot([lo,hi],[-lo,-hi],color='.4',linewidth=.8,linestyle='--')
        ax.set_xscale('log');ax.set_yscale('symlog',linthresh=1e-5)
        ax.set_xlabel('Conditional mean Q');ax.set_ylabel('Conditional mean C')
        ax.set_title('Dashed: C = -Q',fontsize=8,pad=16)
        fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.915),ncol=2)
        keyorder=[(int(a['trajectory_id']),int(a['snr_db']),int(a['frame_id'])) for a in conditional]
        lookup={(int(a['trajectory_id']),int(a['snr_db']),int(a['frame_id']),int(a['repeat_id'])):a for a in raw}
        caxes=[]
        for ax,key,label in zip([axes[0,1],axes[1,0],axes[1,1]],['Q','C','delta_R_noise'],['Q: movement cost','C: oracle cross term','Q + C: signed risk change']):
            data=np.array([[lookup[(*identity,r)][key] for r in range(20)] for identity in keyorder])
            assert data.shape==(240,20) and np.isfinite(data).all()
            if key=='Q':
                norm=Normalize(vmin=0,vmax=float(data.max()));cmap='Blues'
            else:
                extent=float(np.max(abs(data)));norm=TwoSlopeNorm(vmin=-extent,vcenter=0,vmax=extent);cmap='RdBu_r'
            im=ax.imshow(data,aspect='auto',interpolation='nearest',norm=norm,cmap=cmap,rasterized=True)
            ax.set_xticks([0,5,10,15,19]);ax.set_xlabel('Noise repeat ID')
            ax.set_yticks([0,60,120,180,239]);ax.set_ylabel('Fixed-window index')
            ax.set_title(label,fontsize=8,pad=16)
            bb=ax.get_position();ca=fig.add_axes([bb.x1+.013,bb.y0,.012,bb.height]);caxes.append(ca)
            cb=fig.colorbar(im,cax=ca);cb.ax.tick_params(labelsize=8)
        manifest.append(export('C',fig,d.OUT/'sp_u2c_repeat_rows.csv',dict(mean_points=240,heatmap_values_per_quantity=4800),0,caxes))
        fig,ax=simple('D','Distribution of conditional signed risk change')
        values=np.array([a['mean_delta_R_noise'] for a in conditional]);lo,hi=float(values.min()),float(values.max())
        bins=np.linspace(lo,hi,25)
        for s in d.SNRS:
            group=[a['mean_delta_R_noise'] for a in conditional if a['snr_db']==s]
            ax.hist(group,bins=bins,histtype='step',linewidth=1.6,color=COLORS[s],label=f'{s} dB (120 windows)')
        ax.axvline(0,color='.5',linewidth=.8)
        ax.set_xlabel('Conditional mean delta R noise (20 repeats)');ax.set_ylabel('Fixed-window count')
        ax.legend(loc='upper center',bbox_to_anchor=(.5,1.25),ncol=2)
        manifest.append(export('D',fig,d.OUT/'sp_u2c_conditional_windows.csv',240,0))
        fig,ax=simple('E','Risk-metric alignment by signed-risk group')
        undefined=sum(a['alignment_undefined'] for a in raw)
        for s in d.SNRS:
            for sign,style in [('Negative','-'),('Positive','--')]:
                group=[a for a in raw if a['snr_db']==s and not a['alignment_undefined'] and
                       (a['delta_R_noise']<0 if sign=='Negative' else a['delta_R_noise']>0)]
                x=np.sort([a['alignment_cosine'] for a in group]);y=np.arange(1,len(x)+1)/len(x)
                ax.step(x,y,where='post',color=COLORS[s],linestyle=style,linewidth=1.4,label=f'{s} dB {sign} (n={len(x)})')
        ax.set_xlim(-1.03,1.03);ax.set_ylim(0,1.03)
        ax.set_xlabel('M-metric alignment cosine');ax.set_ylabel('Empirical cumulative fraction')
        ax.legend(loc='upper center',bbox_to_anchor=(.5,1.27),ncol=2)
        fig.text(.17,.02,f'Undefined cosines retained as flags in raw CSV: {undefined}; ECDF is descriptive',fontsize=8)
        manifest.append(export('E',fig,d.OUT/'sp_u2c_repeat_rows.csv',4800-undefined,undefined))
        fig,ax=simple('F','Surrogate gap and conditional noise effect')
        for s in d.SNRS:
            group=[a for a in conditional if a['snr_db']==s]
            x=np.array([abs(a['mean_delta_R_noise']) for a in group]);y=np.array([a['mean_surrogate_gap'] for a in group])
            assert np.all(x>0) and np.all(y>0)
            ax.scatter(x,y,s=18,color=COLORS[s],marker=MARKERS[s],alpha=.65,edgecolors='none',label=f'{s} dB (120 windows)')
        values=[abs(a['mean_delta_R_noise']) for a in conditional]+[a['mean_surrogate_gap'] for a in conditional]
        lo,hi=min(values),max(values);ax.plot([lo,hi],[lo,hi],color='.5',linestyle='--',linewidth=.8)
        ax.set_xscale('log');ax.set_yscale('log')
        ax.set_xlabel('Absolute conditional mean noise-coordinate effect');ax.set_ylabel('Conditional mean surrogate risk gap')
        ax.legend(loc='upper center',bbox_to_anchor=(.5,1.25),ncol=2)
        manifest.append(export('F',fig,d.OUT/'sp_u2c_conditional_windows.csv',240,0))
    d.save(d.OUT/'sp_u2c_figure_manifest.json',manifest)
    print('Six figures exported; CSV counts preserved; visual QA pending')


if __name__=='__main__':main()
