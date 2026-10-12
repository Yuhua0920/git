"""Plot the complete frozen comparison CSV, never infer or fit a model."""
from pathlib import Path
import argparse,csv,json,hashlib,sys,subprocess
def render(csv_path,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    qa=Path(__file__).resolve().parent/'figure_qa';sys.path.insert(0,str(qa))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    rows=list(csv.DictReader(Path(csv_path).open(encoding='utf8')))
    conditions=('ID','Profile','Mobility','Composed','Boundary');snrs=(0,5,10,15,20,25,30)
    keys={(r['condition'],int(r['snr_db'])) for r in rows}
    if len(rows)!=35 or keys!={(c,s) for c in conditions for s in snrs}:raise RuntimeError('COMPLETE_35_CELL_FIGURE_REQUIRED')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    manifest={'input_csv_sha256':hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),'omitted_cells':0,'figures':[],'human_visual_QA':'REQUIRED_AFTER_REAL_RENDER'}
    for condition in conditions:
        cells=sorted([r for r in rows if r['condition']==condition],key=lambda r:int(r['snr_db']))
        fig,(a,d)=plt.subplots(1,2,figsize=(7.1,3.1));fig.subplots_adjust(left=.10,right=.97,bottom=.22,top=.92,wspace=.48)
        a.plot(snrs,[float(r['LR4_NMSE_dB']) for r in cells],'o-',color='#176A9A',label='LR4')
        a.plot(snrs,[float(r['MFA_NMSE_dB']) for r in cells],'s--',color='#A85E2E',label='MFA')
        a.set(xlabel='SNR (dB)',ylabel='Data-domain NMSE (dB)');a.legend(fontsize=8);a.set_xticks(snrs)
        value=np.array([float(r['MFA_minus_LR4_dB']) for r in cells]);lo=np.array([float(r['simultaneous_CI_lower_dB']) for r in cells]);hi=np.array([float(r['simultaneous_CI_upper_dB']) for r in cells])
        # Percentile intervals need not contain their point estimate: draw interval independently.
        d.vlines(snrs,lo,hi,color='#444444',linewidth=1);d.plot(snrs,value,'o',color='#176A9A');d.axhline(0,color='#888888',linewidth=.7)
        d.set(xlabel='SNR (dB)',ylabel='MFA − LR4 NMSE (dB)');d.set_xticks(snrs)
        a.text(0,1.02,'(a) '+condition,transform=a.transAxes,fontweight='bold');d.text(0,1.02,'(b) Paired difference',transform=d.transAxes,fontweight='bold')
        fig.canvas.draw();rect=[ax.get_position().bounds for ax in (a,d)]
        widths=[v[2]*fig.get_figwidth()*72 for v in rect];heights=[v[3]*fig.get_figheight()*72 for v in rect]
        if max(widths)-min(widths)>1.5 or max(heights)-min(heights)>1.5:raise RuntimeError('PANEL_ALIGNMENT_FAIL')
        stem=destination/condition
        require_matplotlib_panel_alignment(fig,json_out=str(stem)+'.alignment.json',tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
        for ext in ('pdf','svg','png'):fig.savefig(str(stem)+'.'+ext,dpi=300)
        plt.close(fig)
        commands=[(qa/'audit_pdf_text.py',[str(stem)+'.pdf','--min-pt','5','--json']),
                  (qa/'audit_figure_collisions.py',[str(stem)+'.pdf','--json-out',str(stem)+'.collision.json','--overlay-pdf',str(stem)+'.collision.pdf'])]
        for program,args in commands:
            result=subprocess.run([sys.executable,'-B',str(program),*args],capture_output=True,text=True,encoding='utf8')
            Path(str(stem)+'.'+program.stem+'.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf8')
            if result.returncode!=0:raise RuntimeError('FIGURE_AUTOMATIC_QA_BLOCKED:'+program.name)
        caption=(f'{condition}: (a) pooled data-domain NMSE on 1000 paired trajectories at each SNR, six scored frames per trajectory. Curves show point estimates. (b) MFA-minus-LR4 NMSE difference, positive favoring LR4. Vertical intervals use 100000 paired trajectory bootstrap replicates and a two-sided Bonferroni correction across 35 comparisons (familywise level 95%). No frame is treated as an independent trajectory. All planned SNR points are shown.\n')
        stem.with_suffix('.caption.txt').write_text(caption,encoding='utf8')
        manifest['figures'].append({'condition':condition,'panel_rectangles':rect,'equal_width_and_height_tolerance_pt':1.5,'source_cells':7})
    (destination/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',type=Path,required=True);p.add_argument('--out',type=Path,required=True);x=p.parse_args();render(x.csv,x.out)
