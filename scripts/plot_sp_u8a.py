"""Three diagnostic figures; all quantities trace to notice-first U8A CSVs.

Figure contract: show frozen BER ranking, distinguish relative conditioning from
absolute inverse noise sensitivity, then inspect only the seven user-fixed IDs.
Quantitative grids, 183 mm width, editable PDF/SVG, 600 dpi TIFF and 300 dpi PNG.
No confidence interval is implied: boxes show empirical trajectory distributions.
"""
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import sp_u8a_common as c
os.environ['MPLCONFIGDIR']=str(c.OUT/'mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
SKILL=Path('C:/Users/yuhua/.codex/skills/nature-figure/scripts')
sys.path.insert(0,str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment
from audit_pdf_text import audit_pdf
from audit_figure_collisions import audit_pdf as collision_audit

COLORS={'Perfect':'#333333','LS':'#738A9C','LR4':'#337CA0','Matched':'#BD7954'}
DISPLAY=('Perfect','Matched','LS','LR4')
STYLE={'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
    'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'legend.fontsize':7,
    'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.right':False,'axes.spines.top':False,
    'legend.frameon':False,'axes.linewidth':.7}

def label(ax,value):
    ax.annotate(value,xy=(0,1),xycoords='axes fraction',xytext=(-22,15),textcoords='offset points',
        fontsize=10,fontweight='bold',annotation_clip=False)

def export(fig,name,caption):
    fig.text(.02,.99,c.NOTICE,fontsize=5.8,va='top')
    fig.text(.02,.015,caption,fontsize=7,va='bottom')
    fig.canvas.draw()
    p=c.OUT/'figures'/name
    require_matplotlib_panel_alignment(fig,json_out=str(p)+'.alignment.json',tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    # Keep the required notice first in the generated alignment audit as well.
    c.write(str(p)+'.alignment.json',c.read(str(p)+'.alignment.json'))
    fig.savefig(str(p)+'.pdf')
    fig.savefig(str(p)+'.svg')
    fig.savefig(str(p)+'.png',dpi=300)
    fig.savefig(str(p)+'.tiff',dpi=600)
    text=audit_pdf(Path(str(p)+'.pdf').read_bytes(),minimum_pt=5.)
    collision=collision_audit(Path(str(p)+'.pdf'))
    c.write(str(p)+'.text.json',text);c.write(str(p)+'.collision.json',collision)
    plt.close(fig)

def main():
    c.install_guard();assert c.read(c.OUT/'replay_gate.json')['PASS']
    (c.OUT/'figures').mkdir(exist_ok=True)
    analysis=c.read(c.OUT/'analysis.json');tr=c.csv_read(c.OUT/'trajectories.csv')
    channels={int(r['trajectory_id']):r for r in c.csv_read(c.OUT/'channel_trajectories.csv')}
    bank={(int(r['snr_db']),r['method'],int(r['trajectory_id'])):float(r['BER']) for r in tr}
    contract=dict(archetype='quantitative grid',backend='Python',width_mm=183,
        source_data=['trajectories.csv','channel_trajectories.csv','fixed_ID_diagnostics.csv'],
        figures=[dict(name='Figure1',question='How does Perfect ZF relate to frozen comparator BER?',
            panels='10/20 dB empirical distributions with identical summary rules'),
            dict(name='Figure2',question='Do relative conditioning and absolute inverse sensitivity track Perfect BER?',
            panels='upper: log10 condition median; lower: noise enhancement median; left/right: SNR10/20'),
            dict(name='Figure3',question='How do estimates change inverse norms on the fixed diagnostic IDs?',
            panels='10/20 dB equalizer squared Frobenius median')],
        sample_unit='1000 USED independent trajectories; 6 scored frames each; no exclusions',
        intervals='empirical median/IQR with P05/P95 whiskers in Figure1; no bootstrap/confidence intervals',
        figure_count_limit=3,Perfect_label='POST-HOC DIAGNOSTIC ONLY',statistical_tests='none')
    c.write(c.OUT/'figure_contract.json',contract)
    with plt.rc_context(STYLE):
        fig,axs=plt.subplots(1,2,figsize=(183/25.4,3.5))
        fig.subplots_adjust(left=.09,right=.98,bottom=.22,top=.80,wspace=.30)
        for si,s in enumerate(c.SNRS):
            ax=axs[si];label(ax,chr(97+si));ax.set_title(f'{s} dB')
            values=[[bank[s,m,i] for i in range(1000)] for m in DISPLAY]
            boxes=ax.boxplot(values,whis=(5,95),showfliers=False,patch_artist=True,widths=.5)
            for patch,m in zip(boxes['boxes'],DISPLAY):patch.set_facecolor(COLORS[m]);patch.set_alpha(.25)
            for patch in boxes['medians']:patch.set_color('#222222')
            for k,m in enumerate(DISPLAY,1):ax.plot(k,np.mean(values[k-1]),marker='D',color=COLORS[m],markersize=4)
            ax.set_xticks(range(1,5),DISPLAY);ax.set_ylabel('BER');ax.set_ylim(0,.52)
        export(fig,'Figure1_BER','n = 1000 trajectories; box: median/IQR; whiskers: P05/P95; diamond: pooled BER.\nPerfect-CSI: POST-HOC DIAGNOSTIC ONLY. Nonfinite/excluded trajectories: 0.')
        fig,axs=plt.subplots(2,2,figsize=(183/25.4,5.4))
        fig.subplots_adjust(left=.12,right=.97,bottom=.19,top=.87,wspace=.32,hspace=.65)
        for row,key in enumerate(('log10_condition_median','noise_enhancement_median')):
            for si,s in enumerate(c.SNRS):
                ax=axs[row,si];label(ax,chr(97+row*2+si));ax.set_title(f'{s} dB')
                xx=np.array([float(channels[i][key]) for i in range(1000)]);yy=np.array([bank[s,'Perfect',i] for i in range(1000)])
                assert np.all(np.isfinite(xx)) and np.all(np.isfinite(yy))
                ax.scatter(xx,yy,s=7,color='#445F70',alpha=.45,linewidths=0,rasterized=True)
                if row==1:
                    if np.any(xx<=0):raise ValueError('strictly positive noise enhancement required')
                    ax.set_xscale('log');ax.set_xlabel('Median noise enhancement')
                else:ax.set_xlabel('Median log10 condition number')
                ax.set_ylabel('Perfect-CSI BER');ax.set_ylim(0,.52)
        export(fig,'Figure2_Conditioning','All 1000 trajectories shown; x: median over 5952 true-channel data REs per trajectory.\nDescriptive association only; Perfect-CSI: POST-HOC DIAGNOSTIC ONLY.')
        fixed=analysis['fixed_ID_diagnostics'];idx={(r['trajectory_id'],r['snr_db'],r['method']):r for r in fixed}
        fig,axs=plt.subplots(1,2,figsize=(183/25.4,3.6),sharey=True)
        fig.subplots_adjust(left=.12,right=.98,bottom=.23,top=.79,wspace=.34)
        for si,s in enumerate(c.SNRS):
            ax=axs[si];label(ax,chr(97+si));ax.set_title(f'{s} dB')
            for mi,m in enumerate(DISPLAY):
                xx=np.arange(len(c.IDS))+(mi-1.5)*.12
                yy=[idx[i,s,m]['equalizer_squared_Frobenius_median'] for i in c.IDS]
                assert np.all(np.asarray(yy)>0)
                ax.plot(xx,yy,linestyle='none',marker=('o','s','^','D')[mi],color=COLORS[m],markersize=4,label=m)
            ax.set_yscale('log');ax.set_xticks(range(len(c.IDS)),[str(i) for i in c.IDS]);ax.set_xlabel('Fixed trajectory ID')
            ax.set_ylabel('Median squared equalizer norm')
        handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.94),ncol=4)
        export(fig,'Figure3_Fixed_IDs','Seven user-fixed IDs only; each point summarizes 5952 REs; no ID selection from this audit.\nEqualizer norm = noise enhancement; Perfect-CSI: POST-HOC DIAGNOSTIC ONLY.')
    print(c.NOTICE,flush=True);print('3 diagnostic figures exported; rendered QA records saved',flush=True)

if __name__=='__main__':main()
