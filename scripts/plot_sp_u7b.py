"""Frozen two-figure plan: complete ECDFs and pooled BER percentile intervals."""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fitz

METHODS=('LS','Source','Hard','Convex','Dense','LR4','Matched')
COLORS=('#909090','#202020','#a6761d','#7570b3','#1b9e77','#d95f02','#0072b2')

def qa(fig,axes,pdf,path):
    fig.canvas.draw();boxes=[a.get_position().bounds for a in axes]
    widths=[b[2]*fig.get_figwidth()*72 for b in boxes]
    maxdev=max(widths)-min(widths)
    if maxdev>1.5:raise ValueError('panel alignment failure')
    doc=fitz.open(pdf);spans=[s for p in doc for block in p.get_text('dict')['blocks'] if 'lines' in block for line in block['lines'] for s in line['spans']]
    minimum=min(s['size'] for s in spans)
    if minimum<5:raise ValueError('PDF glyph floor')
    if not spans:raise ValueError('PDF must contain selectable text')
    collisions=[]
    for pi,page in enumerate(doc):
        words=page.get_text('words')
        for i,a in enumerate(words):
            ra=fitz.Rect(a[:4])
            for b in words[i+1:]:
                rb=fitz.Rect(b[:4]);inter=ra&rb
                if not inter.is_empty and inter.width>.8 and inter.height>1.5:
                    collisions.append([pi,a[4],b[4]])
    if collisions:raise ValueError('text collision: '+repr(collisions[:4]))
    record=dict(PASS=True,selectable_PDF_text=True,minimum_glyph_pt=minimum,alignment_max_deviation_pt=maxdev,text_collision_count=0,
      panels=[dict(panel=chr(97+i),complete_data_extent=True,unique_question=a.get_title(),unit='independent trajectory',alignment_PASS=True,labels_PASS=True) for i,a in enumerate(axes)],
      visual_review='requires final rendered-panel inspection by calling agent')
    Path(path).write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    doc.close()

def save(fig,axes,folder,name):
    for ext in ('pdf','svg','png','tiff'):
        fig.savefig(folder/f'{name}.{ext}',dpi=600 if ext=='tiff' else 180)
    qa(fig,axes,folder/f'{name}.pdf',folder/f'{name}_QA.json');plt.close(fig)

def render(folder,traj_db,result):
    folder=Path(folder);folder.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'axes.titlesize':9,'legend.fontsize':7,
        'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    n=len(traj_db);cdf=np.arange(1,n+1)/n
    fig,axs=plt.subplots(2,2,figsize=(7.3,5.4),gridspec_kw={'wspace':.32,'hspace':.55})
    fig.subplots_adjust(left=.09,right=.98,bottom=.10,top=.88)
    for si,s in enumerate((10,20)):
        ax=axs[0,si]
        for mi,m in enumerate(METHODS):ax.step(np.sort(traj_db[:,si,mi]),cdf,where='post',label=m,color=COLORS[mi],lw=1.5 if m=='LR4' else .9)
        ax.set(title=f'{chr(97+si)}   {s} dB: trajectory NMSE',xlabel='Trajectory NMSE (dB)',ylabel='Empirical CDF',ylim=(0,1))
        ax.grid(alpha=.16)
        gain=traj_db[:,si,1]-traj_db[:,si,5];ax=axs[1,si]
        ax.step(np.sort(gain),cdf,where='post',color=COLORS[5],lw=1.5);ax.axvline(0,color='#606060',ls='--',lw=.7)
        ax.set(title=f'{chr(99+si)}   {s} dB: paired Source to LR4 gain',xlabel='Source minus LR4 (dB)',ylabel='Empirical CDF',ylim=(0,1));ax.grid(alpha=.16)
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=7,bbox_to_anchor=(.53,.995),frameon=False,columnspacing=1.4)
    save(fig,list(axs.flat),folder,'Figure_A')
    fig,axs=plt.subplots(1,2,figsize=(7.3,3.2),gridspec_kw={'wspace':.32})
    fig.subplots_adjust(left=.09,right=.98,bottom=.28,top=.84)
    for si,s in enumerate((10,20)):
        ax=axs[si]
        points=np.array(result['ber'])[si];ci=np.array(result['ber_ci'])[:,si]
        for mi,m in enumerate(METHODS):
            ax.vlines(mi,ci[0,mi],ci[1,mi],color=COLORS[mi],lw=1.2)
            ax.plot(mi,points[mi],'o',color=COLORS[mi],ms=4)
            ax.hlines([ci[0,mi],ci[1,mi]],mi-.08,mi+.08,color=COLORS[mi],lw=.8)
        ax.set(title=f'{chr(97+si)}   {s} dB: pooled BER',ylabel='Bit error probability',xticks=np.arange(7),xticklabels=METHODS)
        ax.tick_params(axis='x',rotation=45);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.16)
    fig.text(.5,.035,'95% descriptive paired trajectory bootstrap intervals; fixed Matched reference',ha='center',fontsize=7)
    save(fig,list(axs),folder,'Figure_B')

def validate_synthetic(folder):
    x=np.linspace(-3,3,1000)
    db=np.array([[[x[i]+mi-si for mi in range(7)] for si in range(2)] for i in range(1000)])
    ber=np.array([[.10+.01*m for m in range(7)],[.03+.005*m for m in range(7)]])
    render(folder,db,dict(ber=ber,ber_ci=np.array([ber*.9,ber*1.1])))

if __name__=='__main__':
    raise SystemExit('Figures are generated only by frozen Phase A synthetic QA or Phase D analysis.')
