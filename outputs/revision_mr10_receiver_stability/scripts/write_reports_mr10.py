"""Assemble evidence-linked reports; do not edit manuscript or historical results."""
from run_mr10 import OUT,ROOT,read,write,csvrows,table,sha
def txt(path,text):
    with path.open('x',encoding='utf8') as f:f.write(text.strip()+'\n')
def main():
    d=csvrows(OUT/'tables/detector_BER.csv');hist=csvrows(OUT/'tables/historical_BER.csv')
    ci=read(OUT/'reports/coordinate_integrity.json');num=read(OUT/'reports/independent_numeric_review.json')
    power=csvrows(OUT/'tables/power_summary.csv');cond=csvrows(OUT/'tables/estimated_channel_conditional_ZF_check.csv')
    def md(rows,cols):return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(str(r[k]) for k in cols)+' |' for r in rows)
    dettable=md(d,['snr_db','method','bit_errors','total_bits','BER'])
    histtable=md(hist,['snr_db','method','bit_errors','total_bits','BER'])
    report=f'''# MR-10 中文科学审查报告

执行判定：PASS。含义是输入、计算与结果可追溯，不是证明所有异常均已获得唯一机制解释。

## 真实范围与身份

历史结果为有效 SP-U7B-R1-CDLA120-V2 与 SP-U8A，未使用已作废的初次 SP-U7B 科学输出。物理条件为 CDL-A、120 km/h、100 ns、2×2 MIMO、64 子载波、每帧16符号、30 kHz间隔、3.5 GHz载频；symbol-wise quasi-static，ideal synchronization，不含显式ICI。数据RE每帧992个，两流Gray QPSK每RE共4bit。单流符号功率0.5，总发射数据功率1；owner pilots为单位模、非owner静默。sigma²固定为0.9941134822523973/0.09941134822523973，不对target重新归一化。

1000条历史BER账目只读核对；新增接收机诊断预先固定原轨迹0–99，两个SNR、frames4–9，共1200个评分frame。每检测路径/SNR为2,380,800 bit。原H、bits、symbols、noise、Y、pilots均复用；LR4只使用已保存的原坐标，不重新拟合。输入数组哈希12,000项、预测哈希2,400项和原Perfect/LS/LR4逐帧错误数3,600项全部相同。第一条真实固定轨迹先完成全链smoke落盘，再执行余下99条，无替换与筛选。7个接收路径和两个SNR共33,331,200个方法-bit判决，底层唯一payload仅2,380,800 bit，同一bit跨方法/SNR是配对重复测量。

坐标诊断复用MR-02的原500条D_eval物理轨迹（五条件每条件100条），三个SNR、六窗口，共9,000窗口。500个小型块的SHA/CRC/JSON身份和汇总字段逐项相同；没有新生成信道、训练、正式确认集或新统计family。

## 1. 历史BER异常确实存在

原Supplement表S19精确未四舍五入结果如下；每方法/SNR为1000条轨迹、23,808,000 bit：

{histtable}

10dB的LS−Perfect差值为−0.006426033266129032，trajectory SD=0.036435466056056305，描述性SE=0.0011521906034689014；580条LS更好、420条更差。20dB差值+0.0359906754032258，70条LS更好、930条更差。SE不是预注册或多重校正区间，本轮不产生显著性结论。历史排序并非只凭表中的舍入数字产生，也不能把所有bit当成独立重复来夸大证据。

固定100条重放同样保留10dB的LS-ZF优于Perfect-ZF，差值−0.004024697580645162；该子集59/100条LS更好。20dB97/100条Perfect-ZF更好。完整账目见tables/historical_paired_fluctuation.csv及detector_paired_differences.csv。

## 2. 接收机实现与功率核验

实际pytest为18项接收机sanity加2项条件误码公式检查，均通过。检查了16个QPSK发送向量、Gray dibit/发送天线次序、幅度1/2、complex AWGN总方差sigma²（实/虚各sigma²/2）、一般复2×2无噪声恢复、TX/RX/共轭/RE次序负面哨兵、病态与奇异输入、MMSE的sigma²/0.5系数、bit-MAP与vector-ML的区别。独立批处理线性检测与原src.detectors输出一致；所有历史LS/LR4预测哈希与原bit错误数相同。

未发现被检查路径中的真实接收机实现错误。矩阵条件数较高导致的是统计噪声增强，不等于浮点求解错误或实现bug。实际unit-noise实/虚二阶矩分别为{num['mean_unit_noise_real_second_moment']:.9f}/{num['mean_unit_noise_imag_second_moment']:.9f}，complex二阶矩{num['mean_unit_noise_complex_second_moment']:.9f}，实虚交叉二阶矩{num['mean_unit_noise_real_imag_cross_moment']:.9f}，与原proper-complex生成代码一致。未修改原噪声。

## 3. 实际接收功率与ZF风险

数据RE上的期望接收功率为P_rx(H)=P_s mean_RE,RX(sum_TX |H|²)，P_s=0.5；另记录真实已发送payload的mean|Hx|²。两者均每接收天线计。原source参考P_ref={power[0]['P_source_reference']}。

固定100条平均实际payload接收功率{float(power[0]['mean_P_rx_realized']):.9f}，平均符号期望功率{float(power[0]['mean_P_rx_expected']):.9f}。实际汇总SNR分别{float(power[0]['pooled_realized_SNR_db']):.6f}/{float(power[1]['pooled_realized_SNR_db']):.6f}dB。轨迹中位数分别{float(power[0]['median_trajectory_realized_SNR_db']):.6f}/{float(power[1]['median_trajectory_realized_SNR_db']):.6f}dB；5–95百分位分别[{float(power[0]['realized_SNR_q05']):.6f},{float(power[0]['realized_SNR_q95']):.6f}]/[{float(power[1]['realized_SNR_q05']):.6f},{float(power[1]['realized_SNR_q95']):.6f}]dB。使用gamma_actual=gamma_source+10log10(P_rx,target/P_ref)，没有重标噪声。不能用轨迹SNR中位数替代汇总SNR。

595,200个原始data RE的sigma_min中位数0.287037900、5百分位0.104393220，最小0.001689971；H条件数中位数20.549125、95百分位80.315740，最大1868.547556，无精确奇异或非有限RE。ZF噪声增强trace中位数12.199307、95百分位91.801445，最大350139.871374；这是inverse row squared norms之和，输出实际噪声协方差还应乘sigma²。两流/RE的post-ZF Es/N0中位数为−10.665345/−0.665345dB，低于0dB占{100*float(power[0]['post_ZF_below_0dB_fraction']):.6f}%/{100*float(power[1]['post_ZF_below_0dB_fraction']):.6f}%。这些RE统计不是独立轨迹统计。

完美CSI情况下v_q=sigma²[H^(-1)H^(-H)]_qq，已知proper QPSK下条件bit BER=0.5 erfc(sqrt(P_s/(2v_q)))。该解析式不需要重新抽噪声。两个SNR平均预测0.368413887791/0.180876304695，实测0.368676495296/0.180758988575，轨迹Spearman 0.997009/0.999220，实测−预测仅+0.0002626075/−0.0001173161。此证据超过仅列出条件数：原固定噪声通过真实inverse产生的条件bit风险可预测高BER。实际接收功率差异很宽，但汇总功率并未降低，不能把高BER单独归因于target更弱。

## 4. 同输入检测器对照

{dettable}

Perfect-bit-MAP用16个等先验发送向量、score=−||y−Hx||²/sigma²，对每bit分别logsumexp后验判决；不是vector-ML。MMSE对所有估计器统一使用(Hhat^H Hhat+sigma²/P_s I)^(-1)Hhat^H y。它是本轮接收机对照，不改变LR4估计器。两个SNR下Perfect-MMSE和bit-MAP分别比Perfect-ZF降低0.169352739/0.087117776与0.251144573/0.166331485的绝对BER；bit-MAP相对降幅约68.1%/92.0%，仅指这100条事后诊断。

同MMSE策略下LR4 BER为0.208660534/0.105231015，低于LS的0.234477907/0.130881216；两个SNR均100/100条轨迹LR4更好。同子集既有channel NMSE为LS −12.568556/−18.583214dB，LR4 −20.183131/−25.783394dB。聚合排序在MMSE下同向，但各方法的trajectory NMSE–BER相关性并不统一，不能推出一般单调映射。

## 5. LS-ZF为何可以低于Perfect-ZF

固定H与由pilots构成的Hhat，A=Hhat^(-1)H，输出噪声方差v_q=sigma²||row_q(Hhat^(-1))||²。针对16个QPSK向量及每个real/imag bit，其准确条件风险由signed mean margin除以sqrt(v_q/2)决定。它同时包含信号失配和噪声传播，不由channel NMSE或inverse norm单独决定。data噪声与估计器所用pilot噪声独立，因而可按原生成约定条件化。

在相同100条实际pilot估计上，10dB该联合条件式预测Perfect/LS/LR4 BER=0.368413888/0.363575649/0.376457298，实测0.368676495/0.364651798/0.376557460；20dB预测0.180876305/0.219253665/0.197323733，实测0.180758989/0.219895413/0.197363911。10dB的排序方向在条件期望中也出现，因此不能只说它来自一次随机误bit波动。条件预测是平均16个等先验payload的期望，实测是原来一份有限payload和噪声，二者不要求逐项相等。

10dB mean inverse-noise-gain trace为Perfect44.123917、LS22.852451、LR497.840173，同时LS/LR4引入信号失配。不能只把较低gain叫作“LS的隐式正则化优势”：新增负面sanity证明Hhat整体正缩放会改变gain与失配，但硬ZF BER不变。当前证据解释固定ZF的联合判决风险，未把插值的哪一部分、pilot噪声或奇异值扰动分解为单独因果贡献。

## 6. Gram与坐标诊断

完整账目9000/9000，非有限、非正定、数值秩不足窗口均0。数值rank判据为lambda_min<=2 eps lambda_max，只用于描述，没有丢弃或替换。最小lambda_min/trace={ci['min_normalized_eigenvalue']:.12f}；最大kappa(S)={ci['max_condition_number']:.9f}。归一化backward error最大{ci['max_backward_error']:.4e}；最高条件数100个窗口以60位Decimal独立求2×2方程，相对坐标差最大{ci['max_high_precision_relative_difference']:.4e}。直接2×2公式复核全部9000窗口最大绝对坐标差{num['coordinate_explicit_2x2_solution_max_abs_difference']:.4e}。

max|alpha|={ci['max_alpha_abs']:.9f}，max|beta|={ci['max_beta_abs']:.9f}；alpha总范围[−1.384525618,1.390379261]，beta总范围[−0.555713240,1.401609839]；最大坐标norm=1.725421980。没有|单坐标|>2窗口。超过1是unrestricted coordinates允许的外推，不是违反约束。ID10 alpha下尾−1.384526及Boundary10 beta下尾−0.555713均完整保留，不能宣称没有统计尾部风险。这里的1/2只表示坐标尺度，不是新阈值或方法停止规则。

Composed30 kappa中位数2.946142、最大16.632882，Boundary30中位数2.172399、最大4.078613，而ID30最大61.526707。因此主要高SNR失效并不与极端Gram病态共同出现。描述性trajectory rho(kappa,LR4 NMSE)在Composed30为−0.758740，Boundary30为−0.103678；coordinate norm的对应rho为+0.721764/−0.409529，符号与大小均依条件改变。相关性不区分pilot/data目标差异、response-space限制、压缩cross term及输入能量等影响，不能作为因果解释。

结论：这些观察支持“在所查窗口上数值可辨识且求解准确”，不保证coordinates统计最优或最终channel-risk最小。当前证据不足以把主要失败归因于数值不稳定；没有理由为修复浮点问题自动加ridge或裁剪。window/ridge若将来研究，只能作为另一个统计bias–variance/目标问题，不是本轮必需修复。本轮不启动。

## 7. 论文决策

历史BER表保持有效，建议保留在Supplement并补充本次100轨迹条件风险与检测器对照；不用诊断重算或替换1000轨迹历史表。未发现动摇独立NMSE实验的实现错误，channel NMSE是在检测器之前对Hhat与H评分，未用检测bit结果。不能将本结论升级为全接收机、所有信道或显式ICI链路验证。

V5可把“机制全部未解决”细化为：Perfect-ZF高BER被原功率与条件噪声传播式准确解释；LS/Perfect排序与联合mismatched ZF bit-risk一致，但没有识别插值的独立因果机制。坐标稳定性写入Supplement，主稿仅一句，保留现有output-risk cross term、noisy design与identifiability范围。精确位置和英文候选见v5_revision_recommendations.md。本轮未修改论文。

## 8. 证据路径与限制

核心表：tables/historical_BER.csv、detector_BER.csv、receive_power_actual_vs_reference.csv、ZF_risk_distribution.csv、estimated_channel_conditional_ZF_check.csv、coordinate_distribution.csv、coordinate_risk_association.csv。每trajectory原始bit masks和对应reference_bits在data/receiver；解析条件风险在data/conditional_ZF；原Gram/rhs/coords/error/truth在data/coordinate_windows.npz。reports/protocol.json记录完整固定范围；preservation_before/after记录使用文件及原V5/source字节保全。

本轮是事后有限诊断；100接收机轨迹不是新独立确认。MR02数据已揭示。没有bootstrap、显著性新声明或multiple-comparison family。1000条历史表与100条新检测器表必须严格区分。未给Perfect-MMSE/bit-MAP重新生成完整1000条表；未识别LS插值独立因果机制；未证明Gram条件良好在其他conditions永远成立。

STOP：不修改估计器、不加ridge、不更换receiver作为主方法、不启动下一实验。
'''
    txt(OUT/'reports/MR10_scientific_review_zh.md',report)
    math='''# Receiver mathematics and code mapping

Saved input contract: H[RX,TX,K,T], x[TX,K,T], y[RX,K,T]; y=Hx+n, n~CN(0,sigma² I). QPSK x_q=(±1±i)/2, E|x_q|²=0.5. Data REs are shared by both streams; the 16 transmit vectors have uniform prior.

ZF: xhat=Hhat^(-1)y. The frozen original code is src/detectors.py:31; selection is src/link.py:154 onward. MR10's vectorized np.linalg.solve uses exactly the same per-RE matrix orientation, and all historical bit counts match.

Linear MMSE detector: (Hhat^H Hhat+sigma²/0.5 I)^(-1)Hhat^H y, evaluated by Cholesky. src/detectors.py:55–114 provides the same convention. No per-estimator tuning or channel-estimator regularizer was added.

Bit-MAP: for vector c, score_c=−||y−Hx_c||²/sigma². The common proper-complex density factor (pi sigma²)^(-2) and equal prior1/16 cancel. For each bit b, compute logsumexp(score_c:b_c=1)−logsumexp(score_c:b_c=0); choose1 iff positive. Vector argmax is not a BER-optimal reference in general; a deterministic regression demonstrates their difference.

Useful receive signal power, per RX: P_rx(H)=0.5 mean_RE,RX(sum_TX|H_rq|²). The actual fixed payload power is mean|Hx|² on the same useful data REs. gamma_actual=gamma_source+10log10(P_rx/P_ref). Data power, received power, and post-ZF per-stream Es/N0 are different quantities. Original noise remains unchanged.

Perfect ZF covariance: sigma² H^(-1)H^(-H). Let gain_q=||row_q(H^(-1))||². Es/N0_q=0.5/(sigma² gain_q); conditional Gray-QPSK BER_q=0.5 erfc(sqrt(Es/N0_q/2)). This is exact for each fixed nonsingular H and the stated proper AWGN model; correlations between output streams do not change marginal bit BER.

Mismatched ZF conditional risk: A=Hhat^(-1)H, mean m_q(c)=(Ax_c)_q, v_q=sigma²||row_q(Hhat^(-1))||². For real/imag bit sign s in {+1,−1}, expected error=Q(s m_real_or_imag/sqrt(v_q/2)). Average all4bits over16 uniform vectors. Hhat uses pilots only, so the current data noise is independent under the frozen generation model. Numerical agreement is compared against the original finite noise/bits realization, not assumed exact equality to an ensemble expectation.

Positive-scaling negative control: Hhat→a Hhat for a>0 changes A and noise gain but leaves all hard ZF bit decisions and conditional BER unchanged. Consequently an inverse-noise-gain reduction alone cannot establish an LS regularization explanation.

Observed Gram: S=G^T G, Sc=b, c=(alpha,beta) unrestricted. Numerical uniqueness, relative excitation, backward error and statistical coordinate variation are distinct. backward_error=||Sc−b||/(||S||₂||c||+||b||); rank tolerance2 eps lambda_max is reported only, not used for removing windows. High-precision checks solve the already saved equation, not a new fitting objective. Theory scope remains fixed observed windows, noisy G,z, and output-risk cross terms.
'''
    txt(OUT/'reports/mathematics_and_code_mapping.md',math)
    recommendations='''# V5 precise revision recommendations (not applied)

## 1. Supplement si_transfer_ber.tex, lines64–82, Table S19 and following explanation

KEEP the original1000-trajectory table and its raw numbers. Add a separate fixed100-trajectory post-hoc detector table/figure after the existing ZF error equation; never replace the history table with the subset.

Replace only the last sentence, currently “The recorded associations between channel conditioning and BER leave the mechanism of these ordering reversals unresolved.” with the following candidate:

“On a fixed post-hoc subset of100 original trajectories, the true-channel ZF noise covariance predicts BERs of0.368414 and0.180876 at the two source-referenced SNRs, compared with observed values of0.368676 and0.180759. Perfect-CSI linear MMSE and bit-MAP detection reduce the corresponding BERs to0.199324/0.093641 and0.117532/0.014428. A mismatched-ZF conditional bit-risk calculation also reproduces the direction of the LS-LIN/Perfect-ZF reversal at10dB. These receiver-specific results separate the high ZF bit risk from channel-estimation NMSE; they do not identify a standalone interpolation mechanism.”

Add exact detector definitions, source-referenced variance and stream power0.5 in a compact paragraph. Define Perfect bit-MAP as marginal posterior bit decisions over16 uniform QPSK vectors, not vector ML. Specify IDs0–99, original saved inputs, scoreframes4–9, no new random generation, and post-hoc scope. Present raw bit errors/denominators in the extended table. No newly adjusted/significant interval should be claimed.

KEEP the statement “These orderings characterize the fixed receiver. Its BER ordering differs from the channel-NMSE ordering.” Qualify any current phrase “Perfect has lowest BER” as “among the original fixed-ZF paths”, so it does not conflict with the new MAP/MMSE detector comparison.

Do not add “LS regularization causes the advantage” or “ill-conditioning causes the ranking reversal”. Conditional true-channel noise propagation explains the large perfect-ZF risk; estimated-channel ordering uses joint signal/noise margins.

## 2. Supplement si_math_note.tex, after the opening identifiability/noisy-design paragraph

ADD a compact observed-window diagnostic subsection, with the following candidate:

“All9000 five-frame windows in the previously revealed development corpus had positive-definite observed Grams. The smallest trace-normalized eigenvalue was0.0159932 and the largest Gram condition number was61.5267. Maximum normalized solve backward error was1.69×10^(-16); a60-digit check on the100 highest-condition windows changed coordinates by at most1.73×10^(-15) on the normalized scale. Coordinate magnitudes and their associations with data risk remained condition dependent. These observations identify accurate numerical solves in this corpus, rather than statistical optimality of pilot-fit coordinates.”

Report all15 condition/SNR rows in an auxiliary table or linked CSV. Add alpha/beta distribution and complete failure ledger. The maximumabsolute alpha/beta1.390379/1.401610 and coordinate-norm1.725422 are descriptive observations, not thresholds. Keep all existing mathematical scope and cross terms.

## 3. Main sections/07_discussion.tex, final paragraph

KEEP the current single-condition transfer, symbol-wise quasi-static/noICI and separate-receiver-diagnostic limits. Optionally append one sentence:

“The receiver diagnostic attributes the large perfect-ZF bit risk to the measured conditional noise propagation, while observed-window checks show accurate coordinate solves in the examined development corpus.”

Refer to the two Supplement subsections. This is a measured local statement; do not turn it into universal stability, robust BER or broader physical realism.

## 4. Main sections/04_method.tex:100, Algorithm1 output and Proposition2

NO mathematical change is needed. Keep identifiable-window direct solve and propagated singular-system exception. Do not add ridge, clipping, pseudoinverse or fallback. The new empirical stability check belongs next to the Supplement identifiability discussion, not as a theorem strengthening.

## 5. Figures and scope

Supplement placement recommended for allthree MR10 figures: receiver_SNR_ZF_risk, same_input_detector_BER, coordinate_Gram_risk. Main paper needs at most the one discussion sentence. CDFs and trajectory scatter are descriptive distributions, not confidence bands; the detector chart has point BER only. Keep the1000-trajectory original BER and100-trajectory new comparison separated. No Results/MainNMSE table, contribution statement, statisticalfamily or formal model selection needs alteration.
'''
    for a,b in [('original1000-trajectory','original 1000-trajectory'),('fixed100-trajectory','fixed 100-trajectory'),('of100','of 100'),('of0.','of 0.'),('values of0.','values of 0.'),('to0.','to 0.'),('at10dB','at 10 dB'),('over16','over 16'),('power0.5','power 0.5'),('IDs0–99','IDs 0–99'),('scoreframes4–9','score frames 4–9'),('All9000','All 9000'),('was0.','was 0.'),('was61.','was 61.'),('was1.','was 1.'),('a60-digit','a 60-digit'),('the100','the 100'),('all15','all 15'),('maximumabsolute','maximum absolute'),('alpha/beta1.','alpha/beta 1.'),('coordinate-norm1.','coordinate norm 1.'),('allthree','all three'),('original1000','original 1000'),('and100','and 100'),('newMAP','new MAP'),('Results/MainNMSE','Results/Main NMSE')]:recommendations=recommendations.replace(a,b)
    txt(OUT/'reports/v5_revision_recommendations.md',recommendations)
    captions='''# Figure sources, captions and reproduction

Allfigures have PDF/SVG editable labels and600dpi PNG. Rasterized dense scatter marks in the coordinate figure preserve vectoraxes/text. Source: scripts/plot_mr10.py. Run with the same Python environment; the script reads raw/tables only and does not generate scientific samples. Auxiliary audit_panel_alignment.py is a dependency-free plotting QA helper copied from the installed nature-figure skill, not scientific inference code.

1.receiver_SNR_ZF_risk: fixed original CDL-A IDs0–99, source-referenced SNR10/20, frames4–9. (a)Useful received payload SNR per trajectory, from tables/receive_power_actual_vs_reference.csv. (b)Per-stream post-ZF Es/N0 under true H. (c)trace(H^(-1)H^(-H)) at data REs; multiply by sigma² for actual output-noise covariance trace. (d)True-channel conditional BER prediction versus original observed trajectory BER, from tables/receive_power_SNR_trajectory.csv. Data raw: data/receiver/0000–0099.npz. The ECDF render uses1001 empirical quantile nodes, including full min/max, for compact output; no sample is removed from calculations. AllCDFs are descriptive and have different observational units, explicitly named on their axes.

2.same_input_detector_BER: post-hoc same100 trajectories, both source-referenced SNRs, same H/dataQPSK/noise/REs. Each bar uses2,380,800 bits. From tables/detector_BER.csv; all raw errors and denominators included there. Perfect-CSI bit-MAP marginalizes16 equal-prior transmit vectors. MMSE usesstream power0.5 and originalsigma² for allchannel estimates. Directlabels andcircle/square/diamond endpoint marks distinguish ZF/MMSE/bit-MAP in grayscale. No interval or new familywise claim is plotted. It must not replace the original1000-trajectory table.

3.coordinate_Gram_risk: originalMR02 D_eval, fiveconditions,100 trajectories/condition, SNR10/20/30, frames4–9, all9000 saved windows. (a)Coordinate-norm ECDF. (b)ObservedGram condition ECDF. (c)alpha/beta scatter. (d)Per-trajectory/SNR medianGram condition versus trajectory pooledLR4 NMSE,1500points; each physicaltrajectory repeats atthreeSNRs. Source:data/coordinate_windows.npz andtables/coordinate_windows.csv. Plotting includes everywindow; sample pairing is retained. Associations are descriptive and neither numerical-conditioning causation nor coordinate-magnitude causation is asserted.

Power and analytic conditional diagnostics are original-law calculations on saved samples, not new channel/noise draws. Final visual inspection includes allthree PNGs and renderedPDFpages; programmatic font, alignment and collision checks are stored underreports. The firstplotQA identified panel-label andlegend crossings; corrected figures retain allscientific values and data, with the earlierQA findings preserved underreports/figure_qa_first_pass.
'''
    for a,b in [('Allfigures','All figures'),('and600dpi','and 600 dpi'),('vectoraxes','vector axes'),('original CDL-A IDs0–99','original CDL-A IDs 0–99'),('SNR10/20','SNR 10/20'),('frames4–9','frames 4–9'),('uses1001','uses 1001'),('AllCDFs','All CDFs'),('same100','same 100'),('dataQPSK','data QPSK'),('uses2,','uses 2,'),('marginalizes16','marginalizes 16'),('usesstream','uses stream'),('power0.5','power 0.5'),('originalsigma','original sigma'),('allchannel','all channel'),('Directlabels','Direct labels'),('andcircle','and circle'),('original1000','original 1000'),('originalMR02','original MR02'),('fiveconditions','five conditions'),('SNR10/20/30','SNR 10/20/30'),('all9000','all 9000'),('pooledLR4','pooled LR4'),('NMSE,1500points','NMSE, 1500 points'),('physicaltrajectory','physical trajectory'),('atthreeSNRs','at three SNRs'),('everywindow','every window'),('allthree','all three'),('allscientific','all scientific'),('earlierQA','earlier QA'),('underreports','under reports')]:captions=captions.replace(a,b)
    txt(OUT/'reports/figure_sources_and_captions.md',captions)
    guide='''# Reproduction of MR10 diagnostics

Requires the existing project .venv-cdl Python, NumPy, SciPy, matplotlib and pytest. Existing frozeninputs must remain present. Do not rerun the tasks into this existing resultdirectory: writes are exclusive and will refuse to replace saved raw evidence.

Read-only tests (PowerShell):
$root='D:\\Users\\yuhua\\Desktop\\Project'
$out=Join-Path $root 'outputs\\revision_mr10_receiver_stability'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH="$out\\scripts;$root"
& "$root\\.venv-cdl\\Scripts\\python.exe" -B -m pytest "$out\\tests" -q -p no:cacheprovider

Plotting-only reproduction:
& "$root\\.venv-cdl\\Scripts\\python.exe" -B "$out\\scripts\\plot_mr10.py"

Executed diagnostics are documented inlogs in order:prepare,receiver,coordinate,analyze,independent_review,conditional_ZF,plot_final. Existing productionfunctions and historicalsavedcoordinates are reused. Nofuturetask/experiment is authorized by thisguide.

results_review.zip contains complete tables,reports,scripts,tests,finalfigures, all9000window sufficient quantities and a compactall100-trajectory bit ledger. Larger per-RE rawreceiver/conditional-risk blocks remain in the outputdata directories, authenticated by full_output_file_manifest.csv. The originalR1corpus andfrozenoperators are not duplicated. ZIPentry hashes are inresults_review_file_manifest.csv; ZIPsidecar usesSHA256.
'''
    for a,b in [('resultdirectory','result directory'),('inlogs','in logs'),('productionfunctions','production functions'),('historicalsavedcoordinates','historical saved coordinates'),('Nofuturetask/experiment','No future task/experiment'),('thisguide','this guide'),('all9000window','all 9000 window'),('compactall100-trajectory','compact all 100-trajectory'),('rawreceiver','raw receiver'),('outputdata','output data'),('originalR1corpus','original R1 corpus'),('andfrozenoperators','and frozen operators'),('ZIPentry','ZIP entry'),('ZIPsidecar','ZIP sidecar'),('usesSHA256','uses SHA256')]:guide=guide.replace(a,b)
    txt(OUT/'README.md',guide)
    write(OUT/'reports/scientific_decision.json',{'verdict':'PASS','receiver_implementation_error_found':False,
        'historical_BER_ordering_reproduced':True,'NMSE_invalidation_evidence':False,
        'Perfect_ZF_high_BER_conditional_risk_explained':True,'LS_interpolation_causal_mechanism_identified':False,
        'Gram_numerical_instability_major_failure_explanation_supported':False,
        'ridge_or_clipping_required':False,'window_statistical_study_required_by_this_diagnostic':False,
        'BER_placement':'SUPPLEMENT','all_results_post_hoc_diagnostic':True,'new_formal_family':False,
        'manuscript_modified':False,'new_channel_noise_generation':False,'models_trained':False,
        'formal_experiments_rerun':False,'next_action':'mentor review; STOP'})
    print('REPORTS WRITTEN: evidence-linked Chinese review, mathematics, V5 suggestions, figure captions')
if __name__=='__main__':main()
