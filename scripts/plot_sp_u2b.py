"""Six single-panel mechanism diagnostics; all 600 windows/SNR retained.

Figure contract: A/B separate excitation and coherence geometry; C/D test joint
perturbation descriptors; E verifies output propagation; F retains signed
noise-versus-surrogate MSE. Quantitative mechanism archetype, development only.
Raw scatter and eight equal-count binned medians, no inferential intervals:
overlapping windows are repeated observations within 100 independent trajectories.
Python only, 183x125 mm, editable PDF/SVG and 300 dpi PNG, minimum 8 pt body text.
"""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import sp_u2b_diagnostic as d
d.install_guard()
os.environ.setdefault('MPLCONFIGDIR',str(d.OUT/'sp_u2b_mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.analyze_sp_u2b import read_rows
SKILL=Path('C:/Users/yuhua/.codex/skills/nature-figure/scripts')
sys.path.insert(0,str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment

SPECS=[('A','clean_sigma_min','delta_noise_norm','Clean design sigma min','Noise-induced coordinate norm',True),
       ('B','clean_kappa_G','delta_noise_norm','Clean design condition (G)','Noise-induced coordinate norm',True),
       ('C','relative_E_spectral','delta_noise_norm','Relative design perturbation (spectral)','Noise-induced coordinate norm',True),
       ('D','delta_noise_norm','joint_bound','Actual coordinate change norm','Deterministic perturbation bound',True),
       ('E','coord_output_quadratic','coord_output_energy','Predicted coordinate output energy','Measured coordinate output energy',True),
       ('F','delta_R_noise','delta_R_surrogate','Noise-coordinate MSE penalty (signed)','Pilot-surrogate MSE penalty',False)]


def main():
    rows=read_rows(d.OUT/'sp_u2b_development_windows.csv')
    source=d.OUT/'sp_u2b_figure_source.csv'
    fields=['trajectory_id','frame_id','snr_db']+sorted({key for _,x,y,*_ in SPECS for key in (x,y)})
    d.csvsave(source,[{k:r[k] for k in fields} for r in rows])
    style={'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':9,
           'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
           'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':.8,
           'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'SP-U2B-frozen-mechanism-v1',
           'legend.frameon':False}
    manifest=[]
    for letter,x,y,xlabel,ylabel,log in SPECS:
        path=d.OUT/f'sp_u2b_fig_{letter}'
        if path.with_suffix('.png').exists():raise RuntimeError('existing figure protected')
        with plt.rc_context(style):
            fig,ax=plt.subplots(figsize=(183/25.4,125/25.4))
            fig.subplots_adjust(left=.15,right=.96,bottom=.2,top=.82)
            for s,color,marker in ((10,'#527CA5','o'),(20,'#BD794D','^')):
                group=[r for r in rows if r['snr_db']==s]
                a=np.array([r[x] for r in group]);b=np.array([r[y] for r in group])
                assert len(a)==600 and np.isfinite(a).all() and np.isfinite(b).all()
                if log:assert np.all(a>0) and np.all(b>0), 'do not silently drop nonpositive log data'
                ax.scatter(a,b,s=7,alpha=.38,color=color,marker=marker,edgecolors='none',rasterized=True,label=f'{s} dB')
                if letter not in ('E','F'):
                    order=np.argsort(a,kind='stable')
                    groups=np.array_split(order,8)
                    ax.plot([np.median(a[idx]) for idx in groups],[np.median(b[idx]) for idx in groups],
                            color=color,linewidth=1.2,marker=marker,markersize=3)
            if log:ax.set_xscale('log');ax.set_yscale('log')
            else:ax.set_xscale('symlog',linthresh=.001)
            if letter in ('D','E'):
                allvalues=[r[k] for r in rows for k in (x,y)]
                lo,hi=min(allvalues),max(allvalues)
                ax.plot([lo,hi],[lo,hi],color='.45',linestyle='--',linewidth=.8)
            if letter=='F':ax.axvline(0,color='.6',linewidth=.8)
            ax.set_xlabel(xlabel);ax.set_ylabel(ylabel)
            fig.text(.15,.93,f'Fig {letter} | SP-U2B development mechanism',fontsize=10)
            fig.text(.15,.06,'100 independent trajectories; 600 repeated windows per SNR; all observations retained',fontsize=8)
            ax.legend(loc='upper center',bbox_to_anchor=(.5,1.2),ncol=2)
            require_matplotlib_panel_alignment(fig,json_out=str(path)+'.alignment.json',
                                               overlay_svg=str(path)+'.alignment.svg',strict=True)
            fig.savefig(str(path)+'.png',dpi=300)
            fig.savefig(str(path)+'.pdf',metadata={'CreationDate':None,'ModDate':None})
            fig.savefig(str(path)+'.svg',metadata={'Date':None})
            plt.close(fig)
        manifest.append(dict(figure=letter,png=str(path)+'.png',pdf=str(path)+'.pdf',svg=str(path)+'.svg',
                             raw_data=str(source),raw_data_sha256=d.sha(source),script=str(Path(__file__).resolve()),
                             statistical_unit='trajectory',independent_trajectories=100,
                             windows_per_snr=600,paired_SNR_points=1200,excluded_rows=0,
                             x=x,y=y,x_scale='log' if log else 'symlog (linear threshold 0.001)',
                             y_scale='log' if log else 'linear',
                             interval='none: mechanism scatter with dependent windows, descriptive binned medians only',
                             alignment='single panel, not applicable',visual_self_check='pending'))
    d.save(d.OUT/'sp_u2b_figure_manifest.json',manifest)
    print('Six figures exported from frozen CSV; visual/PDF QA pending',flush=True)


if __name__=='__main__':main()
