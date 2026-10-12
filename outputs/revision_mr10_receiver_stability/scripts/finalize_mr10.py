"""Render final PDF figures and package already completed results; no new inference."""
import sys,zipfile,json,hashlib,csv
import numpy as np
from PIL import Image,ImageDraw
from run_mr10 import OUT,ROOT,METHODS,npz,read,write,table,sha,need,csvrows
def render():
    import pymupdf
    names=('receiver_SNR_ZF_risk','same_input_detector_BER','coordinate_Gram_risk')
    pages=[]
    for name in names:
        with pymupdf.open(OUT/'figures'/f'{name}.pdf') as doc:
            need(len(doc)==1,'FIGURE_PAGES')
            pix=doc[0].get_pixmap(matrix=pymupdf.Matrix(2,2),alpha=False)
            p=OUT/'reports'/f'{name}_PDF_render.png';pix.save(p)
            pages.append(Image.open(p).convert('RGB'))
    width=max(p.width for p in pages);canvas=Image.new('RGB',(width,sum(p.height for p in pages)+30*(len(pages)-1)),'white');top=0
    for p in pages:canvas.paste(p,(0,top));top+=p.height+30
    canvas.save(OUT/'reports/final_PDF_contact_sheet.png')
    print('Three final PDF pages rendered for actual visual inspection')
def compact():
    masks=[];refs=[];metas=[];condition_checks=[]
    for i in range(100):
        p=OUT/f'data/receiver/{i:04d}.npz';meta=read(p.with_suffix('.json'));authsha=sha(p);need(authsha==meta['raw_sha256'],'RAW_HASH')
        with np.load(p,allow_pickle=False) as z:masks.append(z['bit_error_mask']);refs.append(z['reference_bits'])
        metas.append(meta)
        cp=OUT/f'data/conditional_ZF/{i:04d}.npz'
        with zipfile.ZipFile(cp) as zz:need(zz.testzip() is None,'CONDITIONAL_CRC')
        with np.load(cp,allow_pickle=False) as zz:
            need(zz['expected_bit_BER'].shape==(2,3,6,992),'CONDITIONAL_SHAPE')
            for name in ('expected_bit_BER','noise_gain','mismatch_symbol_MSE'):need(np.isfinite(zz[name]).all(),'CONDITIONAL_NONFINITE')
            need(np.all(zz['noise_gain']>0) and np.all(zz['mismatch_symbol_MSE']>=0),'CONDITIONAL_SIGN')
            need(int(zz['trajectory_id'])==i,'CONDITIONAL_IDENTITY')
            for si,s in enumerate((10,20)):
                for mi,m in enumerate(('Perfect','LS','LR4')):
                    condition_checks.append((s,m,float(zz['expected_bit_BER'][si,mi].mean())))
    for r in csvrows(OUT/'tables/estimated_channel_conditional_ZF_check.csv'):
        s=int(r['snr_db']);m=r['method'];actual=np.mean([v for ss,mm,v in condition_checks if (ss,mm)==(s,m)])
        need(actual==float(r['mean_conditional_expected_BER']),'CONDITIONAL_TABLE_RECOUNT')
    masks=np.array(masks);refs=np.array(refs)
    need(masks.shape==(100,2,7,6,992,4),'COMPACT_BIT_SHAPE')
    npz(OUT/'data/receiver_bit_ledger.npz',bit_error_mask=masks,reference_bits=refs,methods=np.array(METHODS),
        trajectory_ids=np.arange(100),snrs=np.array([10,20]),frames=np.arange(4,10))
    npz(OUT/'data/receiver_frame_error_counts.npz',bit_errors=masks.sum(axis=(4,5)),total_bits_per_frame=np.array(3968),
        trajectory_ids=np.arange(100),snrs=np.array([10,20]),frames=np.arange(4,10),methods=np.array(METHODS))
    write(OUT/'data/receiver_trajectory_metadata.json',{'trajectories':metas,'scope':'original IDs0–99, post-hoc; no selection by outcomes'})
    write(OUT/'reports/final_data_recount.json',{'PASS':True,'bit_decisions':int(masks.size),'unique_payload_bits':int(refs.size),
       'raw_conditional_CRC_checked':100,'conditional_tables_recount_match':True,'complete_receiver_trajectories':100,
       'statistical_unit':'trajectory','no_new_bootstrap':True})
    print('Compact ledger: all 100 trajectories, all methods/bits; conditional risk CRC/recount PASS')
def package():
    need(read(OUT/'reports/visual_qa.json')['PASS'],'VISUAL_QA_REQUIRED')
    need(read(OUT/'reports/preservation_after.json')['PASS'],'PRESERVATION_REQUIRED')
    # Include the complete bit ledger and coordinate sufficient arrays; per-RE raw files
    # remain at their original new-output paths and are authenticated in the full manifest.
    excluded={'package.txt','delivery.json'}  # stdout is still being captured while packaging runs.
    files=[p for folder in ('scripts','tests','tables','reports','logs','figures') for p in (OUT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in excluded]
    files.extend([OUT/'README.md',OUT/'data/coordinate_windows.npz',OUT/'data/receiver_bit_ledger.npz',OUT/'data/receiver_frame_error_counts.npz',OUT/'data/receiver_trajectory_metadata.json'])
    full=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ('results_review.zip','results_review.zip.sha256','results_review_file_manifest.csv','full_output_file_manifest.csv','delivery.json','package.txt')]
    fullrows=[{'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(full)]
    def manifest(path,content):
        with path.open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=list(content[0]));w.writeheader();w.writerows(content)
    manifest(OUT/'full_output_file_manifest.csv',fullrows);files.append(OUT/'full_output_file_manifest.csv')
    rows=[{'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]
    manifest(OUT/'results_review_file_manifest.csv',rows);files.append(OUT/'results_review_file_manifest.csv')
    target=OUT/'results_review.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):z.write(p,p.relative_to(OUT).as_posix())
    hh=sha(target)
    with (OUT/'results_review.zip.sha256').open('w',encoding='ascii') as f:f.write(hh+'  results_review.zip\n')
    with zipfile.ZipFile(target) as z:
        need(z.testzip() is None,'REVIEW_ZIP_CRC')
        for r in rows:need(hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256'],'REVIEW_ZIP_ENTRY_HASH')
    receipt={'verdict':'PASS','ZIP':str(target),'ZIP_sha256':hh,'ZIP_bytes':target.stat().st_size,
        'ZIP_files':len(files),'entry_hashes_verified':len(rows),'ZIP_CRC_PASS':True,'raw_output_files':len(full),
        'scope':'review archive includes complete bit/error/coordinate sufficient evidence; larger per-RE blocks retained in output, not duplicated',
        'STOP':True}
    (OUT/'reports/delivery.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps(read(OUT/'reports/delivery.json'),indent=2))
if __name__=='__main__':globals()[sys.argv[1]]()
