"""Read persisted MR02 development diagnostics and make a review-only export.

No generator, fit, new bootstrap draw, or production-method modification.
"""
import argparse, csv, hashlib, json, sys, zipfile, re
from pathlib import Path
import numpy as np

CONDITIONS=('ID','Profile','Mobility','Composed','Boundary')
SNRS=(10,20,30)
METHODS=('Source','Pooled','Hard','Convex','Dense-online','LR4-online','Dense-cp','Dense-cd','Dense-oracle','LR4-cp','LR4-cd','LR4-oracle','Matched')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for v in iter(lambda:f.read(8*1024*1024),b''):h.update(v)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def table(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)
def csvwrite(p,rows):
    with Path(p).open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def fnum(v):return f'{float(v):.6f}'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--out',required=True);a=parser.parse_args()
    root=Path(a.root).resolve();out=Path(a.out).resolve();mr=root/'outputs/revision_mr01_pooled_lmmse'
    if out!=root/'outputs/revision_mr02_coordinate_mechanism':raise ValueError('Output boundary')
    sys.path[:0]=[str(out/'src_or_scripts'),str(mr/'src_or_scripts'),str(root)]
    import run_mr02 as runner
    runner.init(root,out);runner.frozen();pres=runner.preserve()
    analysis=read(out/'reports/analysis_report.json');evr=read(out/'reports/evaluation_receipt.json');refr=read(out/'reports/reference_receipt.json');fit=read(out/'reports/reference_fit.json');toy=read(out/'reports/theory_toy_results.json')
    if not all(v['PASS'] for v in (analysis,evr,refr,fit,toy)):raise ValueError('Incomplete diagnostics')
    nm=table(out/'tables/nmse.csv');co=table(out/'tables/contrasts.csv');comp=table(out/'tables/compression.csv');tr=table(out/'tables/risk_transfer.csv');tail=table(out/'tables/coordinate_diagnostics.csv')
    if (len(nm),len(co),len(comp),len(tr),len(tail))!=(195,300,15,30,15):raise ValueError('Scientific table identities')
    nd={(r['condition'],int(r['snr_db']),r['method']):r for r in nm}
    cd={(r['condition'],int(r['snr_db']),r['space'],r['contrast']):r for r in co}
    if len(nd)!=195 or len(cd)!=300:raise ValueError('Duplicate scientific table identity')
    def n(c,s,m):return float(nd[c,s,m]['nmse_db'])
    wide=[]
    for c in CONDITIONS:
        for s in SNRS:
            wide.append(dict(condition=c,snr_db=s,**{m+'_NMSE_dB':n(c,s,m) for m in METHODS}))
    csvwrite(out/'tables/nmse_wide_all15.csv',wide)
    # Independently recompute all saved new diagnostic points and intervals from
    # persisted error/truth and the original MR01 resampling indices.
    with np.load(out/'data/eval_summary.npz',allow_pickle=False) as ev, np.load(mr/'data/exploratory_bootstrap_indices.npz',allow_pickle=False) as draws:
        errors=ev['error'].sum(3);truth=ev['truth'].sum(2);mx=0.;cmx=0.
        for ci,c in enumerate(CONDITIONS):
            ids=draws[c];db=10*np.log10(errors[ci].sum(0)/truth[ci].sum())
            bdb=10*np.log10(errors[ci][ids].sum(1)/truth[ci][ids].sum(1)[:,None,None])
            lo,hi=np.quantile(bdb,[.025,.975],axis=0,method='linear')
            for si,s in enumerate(SNRS):
                for mi,m in enumerate(METHODS):
                    r=nd[c,s,m]
                    mx=max(mx,abs(db[si,mi]-float(r['nmse_db'])),abs(lo[si,mi]-float(r['CI_lower'])),abs(hi[si,mi]-float(r['CI_upper'])))
                for space in ('Dense','LR4'):
                    for name in ('online-cp','online-cd','cp-cd','online-oracle','cp-oracle','cd-oracle','online-Pooled','cp-Pooled','cd-Pooled','oracle-Pooled'):
                        r=cd[c,s,space,name];i=METHODS.index(r['first']);j=METHODS.index(r['second']);l,h=np.quantile(bdb[:,si,i]-bdb[:,si,j],[.025,.975],method='linear')
                        cmx=max(cmx,abs(db[si,i]-db[si,j]-float(r['estimate_db'])),abs(l-float(r['CI_lower'])),abs(h-float(r['CI_upper'])))
        if max(mx,cmx)>1e-10:raise ValueError('Persisted statistical summary recheck')
        oracle_ranks={str(k):int(np.sum(ev['oracle_rank']==k)) for k in (0,1,2)}
    # CRC/hash every small block; no count-only completion claim.
    verified=0
    for receipt in (evr,refr):
        for r in receipt['blocks']:
            p=out/r['path']
            if sha(p)!=r['sha256'] or sha(p.with_suffix('.json'))!=r['json_sha256']:raise ValueError('Raw block hash')
            with zipfile.ZipFile(p) as z:
                if z.testzip():raise ValueError('Raw block CRC')
            verified+=1
    write(out/'reports/independent_result_recheck.json',dict(PASS=True,raw_blocks_verified=verified,point_interval_max_db_error=mx,contrast_interval_max_db_error=cmx,new_evaluation_draws=0,eval_oracle_rank_counts=oracle_ranks))
    paragraphs=[
      '# MR-02 科学机制诊断报告\n',
      '本轮为已揭示的开发数据上的诊断，不是最终确认，不是新估计器。D_ref新生成500条（每条件100），D_eval复用MR-01的500条；每条10帧，评分4–9，三档SNR10/20/30。P、均值、完整cross-pilot核、rank-four factors与在线直接求解均不改变。',
      '只有一个共享pilot参照c_p；Dense、LR4分别有data参照c_d,D、c_d,L和逐帧oracle。固定参照使用D_ref标签和clean truth，oracle使用当前data truth，均仅进入诊断层。Matched复用既有五条件B2 MATCHED_REF的有限统计，重新在本轮D_eval评价，未拼接旧性能曲线；它具有额外的condition-specific统计信息。',
      f'45个固定参照的有效秩：{fit["ranks"]}。D_eval oracle秩计数：{oracle_ranks}。独立核验1000个小型block的SHA256/CRC；原MR01复放及输入/代码保全见receipt和MR01_numerical_recheck.json。',
      '五帧窗口在六个评分位置重复使用原始帧的次数为[1,2,3,4,5,5,4,3,2,1]；data风险只给帧4–9等权。c_p和c_d的差异包含目标域、时间权重、有限样本与部署输出空间差异，不能全部归因为pilot/data位置几何。',
      '所有NMSE先sum(error)/sum(truth)再转dB。评价CI复用MR01已有2000次paired trajectory draws，探索性未调整95%，条件于本次固定D_ref；不含prepared corpus或参照拟合不确定性。另500次D_ref trajectory bootstrap仅重算小型参照和风险；不是新的正式family。0.5dB仅为开发实际差异参考，不是事后显著性阈值。',
      '\n## 全15-cell风险（dB，lower is better）\n',
      '|condition|SNR|Pooled|Dense online|Dense c_p|Dense c_d|Dense oracle|LR4 online|LR4 c_p|LR4 c_d|LR4 oracle|Matched|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for c in CONDITIONS:
        for s in SNRS:
            names=('Pooled','Dense-online','Dense-cp','Dense-cd','Dense-oracle','LR4-online','LR4-cp','LR4-cd','LR4-oracle','Matched')
            paragraphs.append('|'+c+'|'+str(s)+'|'+'|'.join(fnum(n(c,s,m)) for m in names)+'|')
    paragraphs+=['\n原Source/Hard/Convex和完整区间仍在nmse.csv（195行）及nmse_wide_all15.csv；本表不隐藏这些背景方法。', '\n## 全15-cell目标及在线风险差\n','差值=first minus second；正值表示second更好。下表括号为探索性未调整95%CI。','|condition|SNR|space|online−c_p|c_p−c_d|online−oracle|oracle−Pooled|\n|---|---:|---|---:|---:|---:|---:|']
    for c in CONDITIONS:
        for s in SNRS:
            for space in ('Dense','LR4'):
                values=[]
                for name in ('online-cp','cp-cd','online-oracle','oracle-Pooled'):
                    r=cd[c,s,space,name];values.append(f'{float(r["estimate_db"]):.4f} [{float(r["CI_lower"]):.4f}, {float(r["CI_upper"]):.4f}]')
                paragraphs.append('|'+c+'|'+str(s)+'|'+space+'|'+'|'.join(values)+'|')
    paragraphs+=['\n## 原压缩风险闭合\n','A=||h_L−h_D||²≥0，B=2Re(e_Dᴴ(h_L−h_D))可为负。表内均在线性能量域按共同truth归一化；A+B=NMSE_L−NMSE_D，不能分别取dB后相加。','|condition|SNR|A/truth|B/truth|sum|LR4−Dense dB|negative B fraction|\n|---|---:|---:|---:|---:|---:|---:|']
    for r in comp:paragraphs.append('|'+r['condition']+'|'+r['snr_db']+'|'+'|'.join(fnum(r[k]) for k in ('A_over_truth','B_over_truth','sum_over_truth','LR4_minus_Dense_db','negative_cross_term_fraction'))+'|')
    paragraphs+=['\n风险迁移精确关系R_eval(c_p)−R_eval(c_d)=δᵀHδ+2δᵀ(Hc_d−b)逐cell闭合，非零梯度cross项已完整记录于risk_transfer.csv。原在线随机坐标的误差使用逐帧真实向量计算；不套用固定坐标总体公式。Gram最小特征值为λ_min/trace归一化；条件数与坐标尾部仅作描述关联，不能证明NMSE因果。',
      '\n## 理论及合成检查\n',
      f'固定c、冻结核且fold held-out噪声独立于该fold输入和clean信道时，E[J_noisy(c)]=E[J_clean-target(c)]+320σ²。输入仍含噪。σ²=0.4，解析128，2048个独立窗口Monte Carlo均值{toy["heldout_noise"]["monte_carlo_mean"]:.12f}，SE={toy["heldout_noise"]["monte_carlo_standard_error"]:.12f}。跨fold复用不妨碍固定c期望相加；该式不能给同窗口拟合的随机c_hat去偏。',
      '正定总体固定目标对齐当且仅当H_d c_p*=b_d；在data总体最优c_d*处风险差为Gram距离。独立D_eval上的经验参照一般不满足零梯度；toy显式得到1.060=1.248−0.188。',
      '合法Gaussian成功toy中精确prepared P目标令pilot/Dense固定最优均为(1,0)，压缩toy最优不自动继承。失败toy保持相同noisy-pilot边缘而改变pilot–data关系，使同一pilot最优对应不同data最优；完整joint covariance正定。两toy是逻辑与适用性检查，不是CDL Composed30已证成因。参见theory_notes.md和theory_toy_results.json的可重建原始矩阵。',
      '\n## 审查范围、失败与下一步\n',
      '所有实际科学输入、坐标、充分量、oracle及风险身份已按预先容差核验。部署pytest先出现命令行参数和模块路径错误，均在新D_ref生成前，原始日志保留；随后42项回归实际通过。未删除样本、扩大容差、改变方法或生成最终确认集。',
      'results_review.zip含本轮小型充分量、原始blocks/receipts、抽样记录、代码、表和图，以及MR01关键数字/矩核验材料。完整H/noise、72MB准备算子与文献PDF留本地；轻包可核验代数、风险、汇总和统计区间，不能独立完成全部向量级外部复现。',
      'MR01根data实际保存了exploratory_bootstrap_indices.npz和verified_trajectory_energies.npz，未发现单独持久化的bootstrap抽样分数数组；本包附实际索引、原能量和原统计表。本轮使用这些原索引独立复核点估计及区间，没有为补包生成新随机draws或改写MR01。',
      '未找到独立SP-U17大修总计划；下一步命名按本次MR02任务书，MR03为外部统计基线容量/优化核验，不沿用MR01旧Tw=1任务名。保留MR01对Tong2017已有观测预测代理/低维组合拟合的全文对照，不把它们包装为新增原创。具体优先级依据独立结果审查报告，完成后STOP。',
      '\n## 实际成本与保全\n',
      f'D_ref {refr["seconds"]:.3f}s，D_eval重放{evr["seconds"]:.3f}s，分析{analysis["seconds"]:.3f}s。D_ref peak working set {refr["peak_working_set_bytes"]/2**30:.3f}GiB，D_eval {evr["peak_working_set_bytes"]/2**30:.3f}GiB。原受保护文件{pres["files"]}个，修改0；science字节冻结检查通过。',
    ]
    markdown=re.sub(r'\n\n(?=\|)', '\n', '\n\n'.join(paragraphs))+'\n'
    (out/'reports/chinese_scientific_report.md').write_text(markdown,encoding='utf8')
    qa=read(out/'reports/visual_qa.json')
    write(out/'reports/final_decision.json',dict(verdict='MR02_DIAGNOSTIC_COMPLETE',meaning='Completed exploratory mechanism evidence, not original-method superiority or confirmation',D_ref=500,D_eval=500,fixed_references=45,eval_oracles=18000,reference_oracles=18000,pytest_passed=42,scientific_failures=0,pre_science_wrapper_failures_preserved=True,own_scientific_code_drift=0,original_method_drift=0,manuscript_drift=0,formal_outcome_performance_reads=0,original_input_drift=0,preservation=pres,visual_qa=qa['verdict'],new_V_T_generated=False,MFA_trained=False,future_tasks_started=False,stop=True))
    # Small review bundle, never copy full H/noise or old formal outcomes.
    items={}
    for folder in ('src_or_scripts','tests','tables','figures','reports'):
        for p in (out/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:items[p.relative_to(out).as_posix()]=p
    for folder in ('reference','evaluation','smoke_ref','smoke_eval'):
        for p in (out/'data'/folder).glob('*'):
            if p.is_file():items[p.relative_to(out).as_posix()]=p
    for name in ('fixed_references.npz','reference_summary.npz','eval_summary.npz','reference_bootstrap.npz','reference_seed_registry.json','smoke_references.npz','smoke_summary.npz'):
        p=out/'data'/name;items[p.relative_to(out).as_posix()]=p
    oldnames=('tables/core_results.csv','tables/nmse_all_methods.csv','tables/paired_contrasts.csv','tables/trajectory_energies.csv','tables/preparation_moment_comparison.csv','tables/pooled_operator_audit.csv',
              'data/exploratory_bootstrap_indices.npz','data/verified_trajectory_energies.npz','data/seed_registry.json','reports/pooled_construction.json','reports/protocol.json','reports/independent_result_recheck.json','reports/analysis_integrity.json','reports/tong_bounded_comparison.md','reports/tong_primary_source_metadata.json','reports/preparation_readonly_lineage.json','reports/operator_cache_authentication.json','src_or_scripts/run_mr01.py','src_or_scripts/seed_union.py')
    for name in oldnames:
        p=mr/name
        if not p.is_file():raise ValueError('MR01 review key missing:'+name)
        items['MR01_key/'+name]=p
    items['taskbook/SP_U17_MR02_Mechanism_Taskbook.md']=root/'SP_U17_MR02_Mechanism_Taskbook.md'
    manifest=[dict(path=k,bytes=p.stat().st_size,sha256=sha(p)) for k,p in sorted(items.items())]
    mp=out/'results_review_file_manifest.csv';csvwrite(mp,manifest)
    zp=out/'results_review.zip'
    with zipfile.ZipFile(zp,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for k,p in sorted(items.items()):z.write(p,k)
        z.write(mp,'results_review_file_manifest.csv')
    with zipfile.ZipFile(zp) as z:
        if z.testzip():raise ValueError('Review ZIP CRC')
        for row in manifest:
            if hashlib.sha256(z.read(row['path'])).hexdigest()!=row['sha256']:raise ValueError('Review ZIP SHA')
    zh=sha(zp);zp.with_suffix('.zip.sha256').write_text(zh+'  results_review.zip\n',encoding='ascii')
    write(out/'review_export.json',dict(path=str(zp),bytes=zp.stat().st_size,sha256=zh,files=len(manifest)+1,CRC_and_all_member_SHA256='PASS',excluded='full H/noise, operators, literature PDFs, historical formal outcomes',role='lightweight independent numeric review, not complete vector reproduction'))
    print(json.dumps(dict(review='PASS',sha256=zh,bytes=zp.stat().st_size,files=len(manifest)+1)))
if __name__=='__main__':main()
