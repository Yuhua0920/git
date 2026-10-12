"""Five figures from frozen CSVs; trajectory-cluster distributions only.

Contract: A influence, B alignment and C risk are separate diagnostic layers;
D preserves paired current-vs-past trajectory contrasts; E displays complete
normalized age profiles without imposing monotonicity. Quantitative diagnostic
grid; Python only;183mm width,125mm height;8pt minimum body fonts;editable
PDF/SVG and300dpi preview. Boxes and profile bands show trajectory IQR, not CI.
"""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import sp_u2d_diagnostic as d
d.install_guard()
os.environ.setdefault('MPLCONFIGDIR',str(d.OUT/'sp_u2d_mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
SKILL=Path('C:/Users/yuhua/.codex/skills/nature-figure/scripts')
sys.path.insert(0,str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment

STYLE={'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,
       'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
       'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':.8,
       'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'SP-U2D-LOFO-v1','legend.frameon':False}
COLORS={10:'#527CA5',20:'#BD794D'};MARKERS={10:'o',20:'^'}


def simple(letter,title):
    fig,ax=plt.subplots(figsize=(183/25.4,125/25.4))
    fig.subplots_adjust(left=.17,right=.96,bottom=.22,top=.79)
    fig.text(.17,.94,f'Fig {letter} | {title}',fontsize=10)
    fig.text(.17,.07,'20 trajectory clusters per SNR; 6 windows x 20 reused noises; matched four-frame LOFO',fontsize=8)
    return fig,ax


def export(letter,fig,source,count):
    path=d.OUT/f'sp_u2d_figures/fig_{letter}';path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.with_suffix('.pdf').exists()
    report=require_matplotlib_panel_alignment(fig,json_out=str(path)+'.alignment.json',overlay_svg=str(path)+'.alignment.svg',strict=True)
    fig.savefig(str(path)+'.png',dpi=300)
    fig.savefig(str(path)+'.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(str(path)+'.svg',metadata={'Date':None});plt.close(fig)
    return dict(figure=letter,pdf=str(path)+'.pdf',png=str(path)+'.png',svg=str(path)+'.svg',
          source=str(source),source_sha256=d.sha(source),script_sha256=d.sha(Path(__file__)),count=count,
          unit='trajectory cluster mean;20trajectories/SNR',excluded=0,interval='trajectory IQR, no inferential CI',
          alignment=report['verdict'],visual_QA='pending')


def main():
    ages=d.read_rows(d.OUT/'sp_u2d_trajectory_ages.csv')
    contrasts=d.read_rows(d.OUT/'sp_u2d_trajectory_contrasts.csv')
    profiles=d.read_rows(d.OUT/'sp_u2d_normalized_profiles.csv')
    assert len(ages)==200 and len(contrasts)==40 and len(profiles)==600
    manifest=[]
    with plt.rc_context(STYLE):
        for letter,key,title,label in [('A','influence_norm','Age-wise coordinate influence','Mean coordinate influence per trajectory'),
                                      ('B','C','Age-wise oracle alignment term','Mean frame alignment term C'),
                                      ('C','delta_R','Age-wise diagnostic risk contribution','Mean full-minus-LOFO risk change')]:
            fig,ax=simple(letter,title)
            for snr,offset in ((10,-.16),(20,.16)):
                positions=np.arange(5)+offset
                data=[]
                for age in d.AGES:
                    group=sorted([a for a in ages if a['snr_db']==snr and a['age']==age],key=lambda a:a['trajectory_id'])
                    values=np.array([a[key] for a in group]);assert len(values)==20 and np.isfinite(values).all()
                    if letter=='A':assert np.all(values>0)
                    data.append(values)
                    jitter=np.linspace(-.06,.06,20)
                    ax.scatter(positions[age]+jitter,values,s=8,color=COLORS[snr],marker=MARKERS[snr],alpha=.6,edgecolors='none')
                ax.boxplot(data,positions=positions,widths=.24,manage_ticks=False,showfliers=False,patch_artist=True,
                    boxprops=dict(facecolor=COLORS[snr],alpha=.18,edgecolor=COLORS[snr]),
                    medianprops=dict(color=COLORS[snr],linewidth=1.3),whiskerprops=dict(color=COLORS[snr]),capprops=dict(color=COLORS[snr]))
                ax.plot([],[],color=COLORS[snr],marker=MARKERS[snr],linestyle='none',label=f'{snr} dB (20 trajectories)')
            if letter=='A':ax.set_yscale('log')
            else:ax.axhline(0,color='.6',linewidth=.8)
            ax.set_xticks(range(5));ax.set_xticklabels(['0: Current','1','2','3','4: Oldest'])
            ax.set_xlabel('Removed frame age');ax.set_ylabel(label)
            ax.legend(loc='upper center',bbox_to_anchor=(.5,1.25),ncol=2)
            manifest.append(export(letter,fig,d.OUT/'sp_u2d_trajectory_ages.csv',200))
        fig,ax=simple('D','Current versus mean-past risk by trajectory')
        for i in d.IDS:
            pair=sorted([a for a in contrasts if a['trajectory_id']==i],key=lambda a:a['snr_db'])
            assert len(pair)==2
            ax.plot([i,i],[a['D_R'] for a in pair],color='.8',linewidth=.7)
        for snr in d.SNRS:
            group=sorted([a for a in contrasts if a['snr_db']==snr],key=lambda a:a['trajectory_id'])
            ax.scatter([a['trajectory_id'] for a in group],[a['D_R'] for a in group],s=22,
                       color=COLORS[snr],marker=MARKERS[snr],edgecolors='none',label=f'{snr} dB')
        ax.axhline(0,color='.5',linewidth=.8);ax.set_xticks([0,4,9,14,19])
        ax.set_xlabel('Fixed trajectory ID');ax.set_ylabel('Trajectory mean current-minus-past risk contrast')
        ax.legend(loc='upper center',bbox_to_anchor=(.5,1.25),ncol=2)
        manifest.append(export('D',fig,d.OUT/'sp_u2d_trajectory_contrasts.csv',40))
        fig,axes=plt.subplots(1,3,figsize=(183/25.4,125/25.4))
        fig.subplots_adjust(left=.13,right=.96,bottom=.23,top=.76,wspace=.6)
        fig.text(.13,.94,'Fig E | Age profiles retain nonmonotonicity',fontsize=10)
        fig.text(.13,.06,'Each trajectory divided by its five-age mean absolute metric; median and IQR across 20',fontsize=8)
        legend=[]
        for letter,ax,key,title in zip('abc',axes,['influence_norm','C','delta_R'],['Influence','Alignment C','Risk change']):
            ax.annotate(letter,xy=(0,1),xytext=(-17,12),textcoords='offset points',xycoords='axes fraction',fontweight='bold')
            for snr in d.SNRS:
                data=np.array([[next(a['value'] for a in profiles if a['snr_db']==snr and a['trajectory_id']==i and a['age']==age and a['quantity']==key)
                                for age in d.AGES] for i in d.IDS])
                assert data.shape==(20,5) and np.isfinite(data).all()
                for values in data:ax.plot(range(5),values,color=COLORS[snr],alpha=.12,linewidth=.6)
                med=np.median(data,axis=0);lo=np.quantile(data,.25,axis=0);hi=np.quantile(data,.75,axis=0)
                ax.fill_between(range(5),lo,hi,color=COLORS[snr],alpha=.16,linewidth=0)
                line,=ax.plot(range(5),med,color=COLORS[snr],marker=MARKERS[snr],markersize=3,linewidth=1.4,label=f'{snr} dB')
                if key=='influence_norm':legend.append(line)
            ax.axhline(0,color='.6',linewidth=.6);ax.set_xticks(range(5));ax.set_xlabel('Removed frame age')
            ax.set_title(title,fontsize=8,pad=15);ax.set_ylabel('Normalized age metric')
        fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.89),ncol=2)
        manifest.append(export('E',fig,d.OUT/'sp_u2d_normalized_profiles.csv',600))
    d.save(d.OUT/'sp_u2d_figure_manifest.json',manifest)
    print('Five figures exported from frozen CSV; all trajectory profiles preserved')


if __name__=='__main__':main()
