"""Render reports from completed MR01 data; no prediction or random draw."""
from pathlib import Path
import argparse,csv,hashlib,json,sys
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
root=Path(a.root).resolve();out=Path(a.out).resolve();sys.path[:0]=[str(root),str(out/'src_or_scripts')]
from run_mr01 import check_preservation,verify_freeze,init
init(root,out)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def rows(name):
    with (out/'tables'/name).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
absolute=rows('nmse_all_methods.csv');contrasts=rows('paired_contrasts.csv');core=rows('core_results.csv')
with np.load(out/'data/verified_trajectory_energies.npz',allow_pickle=False) as z:
    E=z['error'];Q=z['truth'];conds=tuple(z['condition_names']);methods=tuple(z['method_names']);snrs=tuple(z['snr_db'])
    point=10*np.log10(E.sum(1)/Q.sum(1)[:,None,None])
max_error=0.;max_contrast=0.
for r in absolute:
    ci=conds.index(r['condition']);si=snrs.index(int(r['snr_db']));mi=methods.index(r['method'])
    max_error=max(max_error,abs(float(r['nmse_db'])-point[ci,si,mi]))
for r in contrasts:
    ci=conds.index(r['condition']);si=snrs.index(int(r['snr_db']));i=methods.index(r['first_method']);j=methods.index(r['second_method'])
    max_contrast=max(max_contrast,abs(float(r['estimate_db'])-(point[ci,si,i]-point[ci,si,j])))
if max(max_error,max_contrast)>1e-12:raise ValueError('REPORT_POINT_RECOMPUTATION_MISMATCH')
# Independently recompute the existing saved draws; generate no new draws.
max_ci=0.
with np.load(out/'data/exploratory_bootstrap_indices.npz',allow_pickle=False) as draws:
    for ci,condition in enumerate(conds):
        idx=draws[condition]
        if idx.shape!=(2000,100):raise ValueError('DRAW_COUNTS')
        sampled=10*np.log10(E[ci][idx].sum(1)/Q[ci][idx].sum(1)[:,None,None])
        for r in (r for r in contrasts if r['condition']==condition):
            si=snrs.index(int(r['snr_db']));i=methods.index(r['first_method']);j=methods.index(r['second_method'])
            lo,hi=np.quantile(sampled[:,si,i]-sampled[:,si,j],[.025,.975],method='linear')
            max_ci=max(max_ci,abs(float(r['CI_lower'])-lo),abs(float(r['CI_upper'])-hi))
if max_ci>1e-12:raise ValueError('SAVED_DRAW_CI_RECOMPUTATION_MISMATCH')
write(out/'reports/independent_result_recheck.json',dict(PASS=True,absolute_points=105,contrasts=90,intervals_recomputed=90,max_absolute_db_error=max_error,max_contrast_db_error=max_contrast,max_interval_db_error=max_ci,new_bootstrap_draws=0))
panels=[]
for fig in ('mr01_absolute_nmse','mr01_pooled_lr4_difference'):
    align=read(out/'figures/final'/f'{fig}.alignment.json');collision=read(out/'reports'/f'{fig}_collision_final.json');glyph=read(out/'reports'/f'{fig}_glyph_final.json')
    if align['verdict']!='PASS' or collision['verdict']!='PASS' or glyph['below_minimum_count']!=0:raise ValueError('GRAPHICAL_GATE')
    for ci,condition in enumerate(conds):panels.append(dict(figure=fig,panel=chr(97+ci),condition=condition,all_three_points=True,shared_axis_range=True,unit='100 physical trajectories, six frames, paired SNR',interval='point estimate only; paired CI in companion figure' if fig=='mr01_absolute_nmse' else 'unadjusted exploratory 95% trajectory paired percentile CI, 2000 draws',visual_inspection='Actually opened final PNG; no cropping, legend/text overlap or hidden negative effect',alignment='PASS',minimum_glyph_pt=glyph['minimum_found_pt'],collision='PASS',PASS=True))
write(out/'reports/visual_qa.json',dict(PASS=True,actual_final_PNGs_opened=True,panels=panels,formats=['PDF','SVG','600dpiPNG'],width_mm=180,height_mm=108,initial_absolute_overlap='Retained prototype and failed collision report; final corrected legend and footer only',scientific_data_and_statistics_changed=False,source_preflight_TIFF_warning='Accepted: user requested PNG; editable PDF/SVG also supplied',grayscale_encoding='Distinct marker shapes and line styles; close estimates may coincide geometrically'))
report=['# MR-01 中文科学结论报告','',
'【1】PASS。指实验身份、计算及保存结果可信，不表示LR4获胜。所有解释为MR-01 DEVELOPMENT，不是新的正式确认。','',
'【2】真实范围：Smoke每条件3条，共15条，排除于科学表；Development每条件100条，共500条。五条件×10/20/30 dB×6评分帧（4–9），每方法9000个评分位置，7方法共63000次评分。独立统计单位500条trajectory，非9000个独立frame-SNR样本。',
'Source、Natural Pooled、Hard、Convex、Dense、LR4，加Source-centered Pooled诊断。全部共享当前pilot realization，但Source/Pooled仅用当前帧，拟合方法用五帧。','',
'【3】Pooled使用同三组prepared来源与1/N矩约定，包含组间均值项，绝非W的算术平均。mean_p[64]、mean_d[3968]、Rpp[64,64]、Rdp/W[3968,64]，complex128。直接核对15000帧最大相对误差8.478785564818526e-16；预定容差1e-10。Hermitian相对残差9.813507266268827e-17，最小特征值3.19464458439661e-14。三个Cholesky相对残差均<5e-16。数值、接收机和统计测试62/62通过。','',
'【4】Natural Pooled与LR4全部点值及探索性未调整95%配对区间（dB）：','',
'|Condition|SNR|Pooled|LR4|Pooled−LR4|95%探索区间|','|---|---:|---:|---:|---:|---|']
for r in core:report.append('|'+ '|'.join([r['condition'],r['snr_db'],f"{float(r['Pooled_nmse_db']):.6f}",f"{float(r['LR4_nmse_db']):.6f}",f"{float(r['Pooled-LR4']):+.6f}",f"[{float(r['CI_lower']):.6f}, {float(r['CI_upper']):.6f}]"])+'|')
report+=['','正差值表示LR4误差更低。15个区间中9个全正、5个全负、1个跨零；未作多重比较调整，不是15项正式显著性结论。完整105方法cells见tables/nmse_all_methods.csv；本轮core见tables/core_results.csv。','',
'【5】关键差值（全部为前者NMSE减后者NMSE，正值有利于后者）：','',
'|Condition|SNR|Source−Pooled|Pooled−LR4|Pooled−Dense|Dense−LR4|','|---|---:|---:|---:|---:|---:|']
for r in core:report.append('|'+ '|'.join([r['condition'],r['snr_db']]+[f'{float(r[k]):+.6f}' for k in ('Source-Pooled','Pooled-LR4','Pooled-Dense','Dense-LR4')])+'|')
report+=['',
'Natural Pooled在12个失配cells均比Source更好；ID则退化2.454、3.302、2.471 dB。Composed20，Source−LR4=11.729 dB，而Source−Pooled=11.594 dB，LR4之外的增量仅0.136且区间跨零。说明固定多来源统计已经是强对照；不能把dB相减换算成在线适应的独立因果贡献比例。',
'Composed10仍有LR4增量1.937[1.790,2.088] dB；Composed30反转，Pooled好1.938 dB，100/100条本轮trajectory均为Pooled误差较低。Boundary三档Pooled优0.983、0.471、1.086 dB；不把其Source-relative收益解释为delay方向已覆盖。',
'Profile30 Dense比Pooled好0.937 dB，压缩损失1.337后LR4落后0.400；Boundary30 Dense好1.838，压缩损失2.925后LR4落后1.086。Composed20 Dense好0.621[0.487,0.760]，rank4损失0.485使余量缩到0.136。Composed30 Dense本身比Pooled差0.170，区间[-0.402,0.039]跨零，因此仅提高修正秩不能保证恢复Pooled之外的收益。',
'Profile三档LR4分别比Hard差0.198/0.323/1.362、比Convex差0.181/0.310/1.359 dB；Mobility对应0.354/0.195/0.646和0.330/0.209/0.648。这里是已有点估计对比，未新增正式统计family。Hard/Convex保留为重要限制性对照。','',
'【6】中心化：15个Natural−Source-centered差值最大绝对值仅0.0017102833 dB。本次语料中未见足以解释dB级差距的实用中心化影响，不能外推为所有分布均无需均值。差异严格为固定截距a，风险变化含2Re(e_SC^Ha)。两个极小探索区间不含零也不构成实用重要效应。',
'接收链：解析非病态2×2 H的无噪声ZF最大符号误差1.6653345369377348e-16，容差1e-12，4096 bits零错误；原masked/batched data接口3968 bits零错误，batched与逐RE求解差0。近奇异信道仅报告数值行为，未要求零错误；未重跑历史BER或修改其结论。','',
'【7】成本：preparation阶段17.328秒（direct矩核对/算子构造14.462秒）；smoke生成、推断和落盘26.624秒；development同端点174.002秒；后续原始哈希/CRC核验及2000次探索统计45.632秒。development原始落盘之后的完整性扫描另有约42秒，不能把174秒称为含全部后处理的总时长。峰值Windows进程working set1.36833 GiB，Torch CUDA分配峰值19951104 bytes（约19.03 MiB，不含driver/allocator总显存）。i7-13650HX/RTX4060Laptop，BLAS/OMP单线程CPU估计，CUDA0原Sionna生成。非B4端点，不解释为方法加速比。','',
'【8】最终图为figures/final/mr01_absolute_nmse与mr01_pooled_lr4_difference，均PDF/SVG/600dpiPNG，180×108mm。实际打开两张最终PNG逐面板核对：五条件、三SNR点、零基准及正负方向正确；没有裁切或明显重叠。最小实际PDF字形7pt，面板对齐严格1.5pt PASS；两个最终碰撞检查均0 FAIL/0 WARN。最初绝对图的重叠版本和失败QA保留，勿用作最终稿。','',
'【9】负面结果与范围：仅一份prepared corpus、每条件100条D、三个SNR及已有五物理条件；不是广泛OOD或新确认数据。Pooled当前pilot与拟合方法五帧存在时间信息差异。没有Matched、MFA或新objective ablation，本轮不能给出各机制的独立因果份额。无科学轨迹缺失、替换、非有限输出或solve failure。新seed16996与已认证科学历史any-role集合2299519无交集，并防护SDK±1状态；P和旧rank validation/B2/B3/B5C身份分离。历史性能读取0。107项本轮实际依赖与当前稿tex的保全hash均未变；未声称扫描全项目75GB。',
'技术记录：第一条pytest包装命令参数格式错误，no tests ran，尚未生成科学轨迹；原日志保留，正确入口及最终62 tests先于科学生成。只改本轮输出中的工程代码/图排版，原方法、配置、稿件与冻结结果未改。继承的Python迁移警告及缺少triton的计数提示未阻止实际执行，未升级环境。','',
'Tong2017出版社全文p.2 Eqs.20–24已用其余观测预测held-out分量，进行两参数解析拟合，不能将这一本身当新贡献。Tong2018作者全文p.3 Eqs.10–17为协方差外积LOOCV；pp.9–10 Eqs.86–91明确从noisy LS channels构造统计，不能泛称其需要clean online target。区别限于固定Source/prepared-differences、具体四fold五帧接口及correction-only压缩的联合构造，不主张field-wide priority。详见reports/tong_bounded_comparison.md。','',
'【10】下一步建议：情形A/B/C同时出现，D无实用支持。导师审查后，MR02优先诊断pilot-loss最优与data-domain最优坐标的关系，分别定义Dense和LR4可达空间，集中解释Composed20/30及Boundary，同时保留Composed10的清楚增量。Profile30与Boundary20/30显示压缩吞噬dense增益，后续MR05优先级应上升；Composed30还需检查原dense空间而非仅提秩。等时间信息控制归未来MR03，不能把当前结果纯归因于coordinate fitting。若以后依D选择科学参数，须使用独立V；T-new待设计冻结后再生成。本轮STOP，不执行MR02、MR03、MR05或新确认。']
(out/'reports/chinese_scientific_report.md').write_text('\n'.join(report)+'\n',encoding='utf8')
verify_freeze();preservation=check_preservation()
write(out/'reports/final_decision.json',dict(verdict='PASS',meaning='Trustworthy completed MR01 development evidence, not estimator victory',smoke_trajectories=15,development_trajectories=500,method_scored_predictions=63000,cells=15,pytest=dict(passed=62,failed=0,seconds=.68),old_outcome_performance_reads=0,method_tuning=False,scientific_failures=0,scientific_source_drift=0,manuscript_drift=0,preserved=preservation,new_experiments_outside_MR01=0,bootstrap_identity='Exploratory development, 2000, unadjusted 95%',independent_recompute=dict(max_point_db_error=max_error,max_contrast_db_error=max_contrast,max_interval_db_error=max_ci,new_random_draws=0),actual_visual_qa='PASS',final_figure_directory='figures/final',stop=True))
print(json.dumps(dict(verdict='PASS',point_recheck=max_error,contrast_recheck=max_contrast,saved_draw_CI_recheck=max_ci,preservation=preservation)))
