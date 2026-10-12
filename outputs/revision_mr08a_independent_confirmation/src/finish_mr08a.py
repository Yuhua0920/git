"""Evidence-grounded confirmation report from saved tables, no new inference."""
from pathlib import Path
import sys,csv,json,hashlib,math,importlib.metadata as md
import numpy as np
ROOT=Path(r'D:\Users\yuhua\Desktop\Project');OUT=ROOT/'outputs/revision_mr08a_independent_confirmation'
sys.path.insert(0,str(OUT/'src'))
from run_mr08a import verify

def js(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def text(p,s):
    with Path(p).open('x',encoding='utf8') as f:f.write(s)
def csvout(p,z):
    with Path(p).open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(z[0]));w.writeheader();w.writerows(z)
def table(h,z):return '| '+' | '.join(h)+' |\n| '+' | '.join(['---']*len(h))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in z)

def main():
    protocol=js(OUT/'protocol.json');run=js(OUT/'reports/formal_receipt.json');analysis=js(OUT/'reports/formal_analysis.json');independent=js(OUT/'reports/independent_numerical_review.json');visual=js(OUT/'reports/visual_qa.json')
    if not all(x['PASS'] for x in (run,analysis,independent,visual)):raise ValueError('INCOMPLETE_REQUIRED_RESULT_REVIEW')
    nm=rows(OUT/'tables/formal_nmse.csv');ct=rows(OUT/'tables/formal_primary_contrasts.csv');methods=protocol['methods'];conds=list(protocol['conditions']);snrs=protocol['snrs']
    ix={(z['condition'],int(z['snr_db']),z['method']):z for z in nm};cx={(z['family'],z['condition'],int(z['snr_db'])):z for z in ct}
    assert len(ix)==len(nm)==len(conds)*len(snrs)*len(methods) and len(cx)==len(ct)==3*len(conds)*len(snrs)
    wide=[];dw=[];sign=[];neg=[]
    for c in conds:
        for s in snrs:
            z=dict(condition=c,snr_db=s,trajectory_count=1000,scored_frames_per_trajectory=6)
            for m in methods:
                for k in ('nmse_linear','nmse_db'):z[m+'_'+k]=float(ix[c,s,m][k])
            wide.append(z);w=dict(condition=c,snr_db=s)
            for f in ('A','B','C'):
                for k in ('estimate_db','CI_lower','CI_upper'):w[f+'_'+k]=float(cx[f,c,s][k])
                w[f+'_classification']=cx[f,c,s]['classification']
            dw.append(w)
    for f in ('A','B','C'):
        rr=[z for z in ct if z['family']==f]
        sign.append(dict(family=f,first=rr[0]['first'],second=rr[0]['second'],first_supported=sum(float(z['CI_upper'])<0 for z in rr),second_supported=sum(float(z['CI_lower'])>0 for z in rr),indeterminate=sum(float(z['CI_lower'])<=0<=float(z['CI_upper']) for z in rr),comparisons=len(rr),scope='three separately interpreted35comparison families; not105combined95%'))
    csvout(OUT/'tables/formal_nmse_wide.csv',wide);csvout(OUT/'tables/formal_primary_wide.csv',dw);csvout(OUT/'tables/family_direction_summary.csv',sign)
    # Descriptive differences only; never create new confirmatory intervals.
    descriptive=[]
    for c in conds:
        for s in snrs:
            for a,b in (('GMM_PREPARED_3','POOLED_NATURAL'),('PPCA_DIRECT_R32','POOLED_NATURAL'),('HARD_TW5','LR4_TW5'),('CONVEX_TW5','LR4_TW5'),('LR4_TW5','DENSE_TW5'),('WHOLE_POOL_M2_R24','DENSE_TW5'),('SOURCE','LR4_TW5')):
                delta=float(ix[c,s,a]['nmse_db'])-float(ix[c,s,b]['nmse_db'])
                descriptive.append(dict(condition=c,snr_db=s,first=a,second=b,point_difference_db=delta,scope='descriptive point estimate; no newly declared primary family or interval',sign='positive favors second'))
    csvout(OUT/'tables/descriptive_comparisons.csv',descriptive)
    oldroot=ROOT/'outputs/revision_mr03c_prepared_gmm';old=rows(oldroot/'tables/nmse.csv');oi={(z['dataset'],z['condition'],int(z['snr_db']),z['method']):z for z in old};reversal=[]
    pairs=(('POOLED_NATURAL','LR4_TW5'),('GMM_PREPARED_3','LR4_TW1'),('LR4_TW5','WHOLE_POOL_M2_R24'),('GMM_PREPARED_3','POOLED_NATURAL'),('WHOLE_POOL_M2_R24','DENSE_TW5'))
    for ds in ('V_new','D_eval'):
        for c in conds:
            for s in (10,20,30):
                for a,b in pairs:
                    prior=float(oi[ds,c,s,a]['nmse_db'])-float(oi[ds,c,s,b]['nmse_db']);now=float(ix[c,s,a]['nmse_db'])-float(ix[c,s,b]['nmse_db'])
                    reversal.append(dict(development_dataset=ds,condition=c,snr_db=s,first=a,second=b,development_point_difference_db=prior,confirmation_point_difference_db=now,point_order_reversal=bool(prior*now<0),absolute_confirmation_gap_db=abs(now),scope='descriptive crossdataset ordering; not paired estimates between different datasets; no mechanism or newCI'))
    csvout(OUT/'tables/development_order_comparison.csv',reversal)
    for z in descriptive:
        if ((z['first']=='SOURCE' and z['point_difference_db']<0) or (z['first']=='GMM_PREPARED_3' and z['second']=='POOLED_NATURAL' and z['point_difference_db']>0) or (z['first']=='LR4_TW5' and z['second']=='DENSE_TW5' and z['point_difference_db']>0)):
            neg.append(z)
    csvout(OUT/'tables/negative_result_index.csv',neg)
    short={'SOURCE':'Source','POOLED_NATURAL':'Pooled','PPCA_DIRECT_R32':'Direct32','GMM_PREPARED_3':'GMM3','HARD_TW5':'Hard5','CONVEX_TW5':'Convex5','LR4_TW1':'LR4Tw1','LR4_TW5':'LR4Tw5','DENSE_TW5':'Dense5','WHOLE_POOL_M2_R24':'Whole24'}
    report=['# MR-08A 独立确认科研审查\n\n']
    report.append(f'真实执行：{run["physical_trajectories"]}条独立物理trajectory；每方法{run["scored_positions_per_method"]}位置；十方法共{run["total_method_scored_positions"]}位置。失败{run["failures"]}、非有限{run["nonfinite"]}、缺失{run["missing"]}、替换{run["substituted"]}。\n\n')
    report.append('三个family各35项、100000次trajectory-level配对bootstrap；tail=.05/(2×35)。每条件共享全部SNR和方法抽样。分别解释，不宣称105项统一95%保证。新轨迹来自五种既定已研究分布，不是新profile/OOD证明。\n\n')
    report.append('## 完整350方法单元\n\n单位dB，越低越好；显示4位，CSV保留原精度。\n\n'+table(['condition','SNR']+[short[m] for m in methods],[[c,s]+[f'{float(ix[c,s,m]["nmse_db"]):.4f}' for m in methods] for c in conds for s in snrs]))
    report.append('\n## 主要确认族\n\nA=Pooled−LR4Tw5；B=GMM3−LR4Tw1；C=LR4Tw5−Whole24。正值表示后者更好。\n\n'+table(['family','first','second','first支持','second支持','跨零'],[[z['family'],z['first'],z['second'],z['first_supported'],z['second_supported'],z['indeterminate']] for z in sign]))
    report.append('\n所有主比较点估计及调整区间：\n\n'+table(['condition','SNR','A [adjusted CI]','B [adjusted CI]','C [adjusted CI]'],[[c,s]+[f'{float(cx[f,c,s]["estimate_db"]):+.4f} [{float(cx[f,c,s]["CI_lower"]):+.4f},{float(cx[f,c,s]["CI_upper"]):+.4f}]' for f in ('A','B','C')] for c in conds for s in snrs]))
    report.append('\n## 必须回答的科学问题\n\n')
    for f in ('A','B','C'):
        zz=next(z for z in sign if z['family']==f)
        report.append(f'Family{f}：支持{zz["first"]} {zz["first_supported"]}格，支持{zz["second"]} {zz["second_supported"]}格，跨零{zz["indeterminate"]}格。跨零不表示等效。\n\n')
    for c in ('Composed','Boundary','Profile','Mobility'):
        report.append(c+' 10/20/30dB：\n\n'+table(['SNR','A','B','C'],[[s]+[f'{float(cx[f,c,s]["estimate_db"]):+.4f} ({cx[f,c,s]["classification"]})' for f in ('A','B','C')] for s in (10,20,30)]))
    report.append('\nGMM与Pooled、Direct32与Pooled的全SNR点估计：\n\n'+table(['condition','SNR','GMM−Pooled','Direct32−Pooled'],[[c,s]+[f'{float(ix[c,s,a]["nmse_db"])-float(ix[c,s,"POOLED_NATURAL"]["nmse_db"]):+.6f}' for a in ('GMM_PREPARED_3','PPCA_DIRECT_R32')] for c in conds for s in snrs]))
    flips=[z for z in reversal if z['point_order_reversal']]
    report.append(f'\n与两套开发数据的5组预列描述性排序对照共有{len(flips)}个点估计反转（涉及两次开发记录）。不能将近零反转自动当作实质反驳；逐项幅度与主族区间在CSV中保留。\n\n')
    if flips:report.append(table(['development','condition','SNR','first−second','old','new'],[[z['development_dataset'],z['condition'],z['snr_db'],z['first']+'−'+z['second'],f'{z["development_point_difference_db"]:+.6f}',f'{z["confirmation_point_difference_db"]:+.6f}'] for z in flips]))
    report.append('''
## 科学解释边界

Source-relative改善主要说明可用准备统计的收益，不能单独证明online refinement。
FamilyA直接回答Pooled固定统计之外的原LR4完整系统增量，但时间历史预算不同。
FamilyB只对齐当前帧信息，不隔离目标函数、自然均值、统计模型结构或全部开发预算。
FamilyC在共同原Tw5坐标下比较两种压缩部署，保留权重、rank及完整Source与whole三算子压缩的区别；它不单独隔离压缩位置的因果作用。
Whole24比LR4 Tw5的dominantMAC proxy高1.2327%，不将近预算称为完全相同成本。
GMM的准备来源后验与误差关联仅作观察；不凭T-new排序作因果解释或改先验。
Direct32使用完整dense条件算子缓存，rank32是生成模型loadingrank。
原EM16/B5C历史比较仍保留，此轮未将其当作强MFA模型族竞争证明。
任何高SNR压缩损失、ID损失、Pooled优于在线系统或Hard/Convex优势都保留。
独立确认仅针对固定准备资产下的新trajectory，不验证准备资产随机重建的变异性。

## 对论文主张的影响

中心问题应为有限prepared统计空间中的固定准备收益、条件性坐标细化和部署精度成本折中。
主族支持/反向/不确定格按上表判定，不为保持旧主张选择性省略condition/SNR。
cross-pilot构造可用性不等于其目标独立最优或普遍优于合理统计后验。
低秩压缩谱尾不直接等于最终NMSE损失；Whole24/Dense对照必须进入压缩解释。
本轮不改论文、不选新模型。导师应据完整结果决定V5主张与独立确认后的范围。

## 复核和交付

完整原始NPZ/JSON先落盘后统计。项目保留5×100000×1000真实bootstrapindices，
每个索引无损uint16保存；originalint32流未改变。独立复核使用不同直接索引路线，
不是仅重复主counts矩阵。All350points、105CI、35cell全部SNR和方法完整。
实际图像检查记录reports/visual_qa.json。原正式/开发文件没有覆盖或改写。
轻量ZIP可省去巨量模型/cache与1GBindices，但保留主要差值抽样分布、
原drawseed/hash/路径及充分能量；不要声称解压包重新核验未携带的完整indices。
''')
    text(OUT/'scientific_review_cn.md',''.join(report))
    text(OUT/'negative_results_and_scope.md','''# Negative evidence and applicability

See tables/negative_result_index.csv and the complete350cell/105contrast tables.
The index is a descriptive navigation aid, not sample filtering. All outcomes
remain in the fulltables. ID degradation, fixedPooled superiority, extra
direction risk, highSNR correction-rank loss and anyHard/Convex advantage remain.

Independent trajectories from already-studied5distributions do not establish
unseenprofile or arbitraryOOD robustness. Comparisons share a single5000
trajectory population. Each of3families has its own35comparison interpretation;
no105comparison pooled95%guarantee. Crosszero intervals are indeterminate,
not equivalence. Newpoint-order reversals near0 need their actualmagnitude,
not just sign, reported. Mechanistic claims need evidence beyond performance.
''')
    text(OUT/'README_REPRODUCE.md',r'''# Completed independent confirmation

protocol.json and reports/execution_freeze.json predate every T-new trajectory.
data/formal contains 5000 NPZ/JSON identity pairs. Each NPZ stores frame_error
(7,10,6), frame_truth (6), score axes, coordinates/Gram/rhs and small GMM
posterior arrays. JSON records seeds, H/pilot/noise/symbol/truth hashes and
the protocol binding. Common truth is shared across all methods and SNRs.

data/bootstrap/formal/<condition>/indices.npy stores the original 100000x1000
draws, losslessly cast to uint16 from the frozen int32 generation sequence.
bootstrap_linear.npy and primary_differences.npy retain resampling statistics.
No historical experiment draws were reused.

Tables contain 350 method cells, 105 primary contrasts and trajectory energies.
A=Pooled-LR4Tw5; B=GMM-LR4Tw1; C=LR4Tw5-Whole24. Positive favors the second
method. Each family separately contains 35 two-sided Bonferroni intervals.
The RNG stream is SeedSequence([3246293542, condition ordinal]); all methods
and SNRs share each trajectory draw. Standardized noise is shared across SNR.

Do not overwrite results or rerun inference to alter conclusions. For a
read-only check of complete project records:
python -B src/independent_review_mr08a.py --help

The package review entry checks retained energies and primary distributions.
It explicitly references the completed full-project certification and does
not pretend to read omitted indices, full linear arrays or model caches.

The run uses the existing .venv-cdl, the original CUDA CDL generator and
complex128 CPU estimators. Necessary preflight tests: 69 actual passes.
There was no EM/MFA training, model search, new rank/weight/prior selection,
historical experiment rerun or manuscript modification.
''')
    preservation=verify();write(OUT/'reports/preservation_after.json',preservation)
    write(OUT/'reports/completion.json',dict(PASS=True,protocol_SHA256=sha(OUT/'protocol.json'),trajectory_count=run['physical_trajectories'],scored_positions_per_method=run['scored_positions_per_method'],total_method_positions=run['total_method_scored_positions'],NMSE_cells=len(nm),primary_contrasts=len(ct),failures=0,nonfinite=0,new_models_or_training=0,old_results_modified=0,manuscript_modified=0,source_modified=0,primary_families=sign,independent_review=independent,visual_QA=visual['PASS'],preservation=preservation,STOP_after_delivery=True))
    print(json.dumps(dict(PASS=True,primary_families=sign,development_point_order_reversals=len(flips),preservation=preservation),ensure_ascii=False))

if __name__=='__main__':main()
