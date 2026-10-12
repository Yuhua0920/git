"""Assemble MR03C review material from persisted results; no fits/draws/inference."""
from pathlib import Path
import csv, json, hashlib, sys, math, importlib.metadata as md
import numpy as np

ROOT=Path(r'D:\Users\yuhua\Desktop\Project')
OUT=ROOT/'outputs/revision_mr03c_prepared_gmm'
sys.path[:0]=[str(OUT/'src')]
from run_gmm3 import verify

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def text(p,x):
    with Path(p).open('x',encoding='utf8') as f:f.write(x)
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csvout(p,x):
    with Path(p).open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(x[0]));w.writeheader();w.writerows(x)
def table(headers,values):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,v))+' |\n' for v in values)

nm=rows(OUT/'tables/nmse.csv');ct=rows(OUT/'tables/paired_contrasts.csv');dg=rows(OUT/'tables/posterior_diagnostics.csv');tr=rows(OUT/'tables/trajectory_energies.csv')
cfg=read(OUT/'config.json');prep=read(OUT/'reports/preparation.json');timing=read(OUT/'reports/timing.json');qa=read(OUT/'reports/visual_qa.json');review=read(OUT/'reports/independent_numerical_review.json')
assert all(z['PASS'] for z in (prep,timing,qa,review))
methods=cfg['methods'];conds=cfg['conditions'];snrs=cfg['snrs']
ix={(z['dataset'],z['condition'],int(z['snr_db']),z['method']):z for z in nm}
cx={(z['dataset'],z['condition'],int(z['snr_db']),z['second']):z for z in ct}
dx={(z['dataset'],z['condition'],int(z['snr_db'])):z for z in dg}
wide=[];contrastwide=[];signs=[]
for ds in cfg['datasets']:
    for c in conds:
        for s in snrs:
            r=dict(dataset=ds,condition=c,snr_db=s,trajectory_count=100,scored_frames_per_trajectory=6)
            for m in methods:
                for v in ('nmse_linear','nmse_db'):r[m+'_'+v]=float(ix[ds,c,s,m][v])
            wide.append(r)
            r=dict(dataset=ds,condition=c,snr_db=s)
            for m in ('POOLED_NATURAL','LR4_TW1','LR4_TW5','WHOLE_POOL_M2_R24','DENSE_TW5'):
                for v in ('estimate_db','CI_lower','CI_upper','GMM_worse_trajectory_fraction'):r['GMM_minus_'+m+'_'+v]=float(cx[ds,c,s,m][v])
            contrastwide.append(r)
    for m in ('POOLED_NATURAL','LR4_TW1','LR4_TW5','WHOLE_POOL_M2_R24','DENSE_TW5'):
        rr=[x for x in ct if x['dataset']==ds and x['second']==m]
        signs.append(dict(dataset=ds,contrast='GMM minus '+m,GMM_better_interval=int(sum(float(x['CI_upper'])<0 for x in rr)),comparator_better_interval=int(sum(float(x['CI_lower'])>0 for x in rr)),crosses_zero=int(sum(float(x['CI_lower'])<=0<=float(x['CI_upper']) for x in rr)),interval='unadjusted exploratory 95%; previously disclosed development data'))
csvout(OUT/'tables/nmse_wide.csv',wide);csvout(OUT/'tables/paired_contrasts_wide.csv',contrastwide);csvout(OUT/'tables/contrast_direction_summary.csv',signs)

ti={(z['dataset'],z['condition'],int(z['trajectory_id']),int(z['snr_db']),z['method']):z for z in tr}
diff=[]
for ds in cfg['datasets']:
    for c in conds:
        for t in range(100):
            for s in snrs:
                a=ti[ds,c,t,s,'GMM_PREPARED_3'];ea=float(a['error_energy']);q=float(a['truth_energy'])
                for m in ('POOLED_NATURAL','LR4_TW1','LR4_TW5','WHOLE_POOL_M2_R24','DENSE_TW5'):
                    b=ti[ds,c,t,s,m];eb=float(b['error_energy']);assert q==float(b['truth_energy']) and ea>0 and eb>0
                    diff.append(dict(dataset=ds,condition=c,trajectory_id=t,snr_db=s,first='GMM_PREPARED_3',second=m,GMM_error_energy=ea,comparator_error_energy=eb,truth_energy=q,GMM_NMSE_linear=ea/q,comparator_NMSE_linear=eb/q,GMM_minus_comparator_db=10*math.log10(ea/eb),scope='trajectory descriptive paired difference; not cell pooled difference'))
csvout(OUT/'tables/trajectory_paired_differences.csv',diff)

receipts=[read(OUT/'reports'/f'{ds}_receipt.json') for ds in cfg['datasets']]
assert all(z['failures']==0 and z['nonfinite']==0 and z['GMM_scored_positions']==9000 for z in receipts)
allgammaerror=0
for ds in cfg['datasets']:
    for r in read(OUT/'reports'/f'{ds}_receipt.json')['receipts']:
        with np.load(OUT/r['path'],allow_pickle=False) as z:
            allgammaerror=max(allgammaerror,float(np.max(abs(z['posterior_weights'].sum(-1)-1))))

env=dict(Python=sys.version,packages={n:md.version(n) for n in ('numpy','scipy','pytest','matplotlib')},CPU='13th Gen Intel(R) Core(TM) i7-13650HX',physical_cores=14,logical_processors=20,OS='Microsoft Windows 11 家庭版 中文版',visible_RAM_KiB=16387288,hardware_source='Actual Get-CimInstance Win32_Processor/Win32_OperatingSystem output; read only',threads=timing['threads'])
write(OUT/'reports/environment.json',env)
setuprows=[]
for entry in prep['setup']:
    for co in entry['setup_audit']['components']:
        setuprows.append(dict(snr_db=entry['snr_db'],**co))
csvout(OUT/'tables/component_setup.csv',setuprows)

mapping=[
dict(object='Saved centered empirical moments',equation='Rpp=E[(hp-mup)(hp-mup)^H]; Rdp=E[(hd-mud)(hp-mup)^H]',code='src/cdl_benchmark.py:149-179; run_gmm3.py:42-51',meaning='read authenticated saved moments only; no re-estimation'),
dict(object='Cq and Cholesky',equation='Cq=Rppq+sigma2 I; Cq=Lq Lq^H',code='src/gmm3.py:158-213',meaning='same Source-referenced noise added once; no explicit inverse'),
dict(object='Conditional map',equation='Wq=Rdpq Cq^-1',code='src/gmm3.py:216',meaning='Cholesky adjoint solve; same actual64pilot ordering'),
dict(object='Gaussian score',equation='lq=log(1/3)-logdet(Cq)-||Lq^-1(p-mupq)||^2',code='src/gmm3.py:116-126',meaning='current received complex pilot only; natural per-origin means'),
dict(object='Posterior',equation='gammaq=softmax(lq)',code='src/gmm3.py:131-137',meaning='center before logsumexp; all3components included; no labels'),
dict(object='Conditional mixture mean',equation='sumq gammaq [mudq+Wq(p-mupq)]',code='src/gmm3.py:138-155',meaning='complex128; row-transpose batch uses W.T not W.conj().T'),
dict(object='Raw energy/identity',equation='error=||estimate-truth||^2; truth=||truth||^2',code='src/run_gmm3.py:75-128',meaning='save pertrajectory raw NPZ/JSON before aggregate; old8method energies copied exactly'),
dict(object='Cell NMSE and paired intervals',equation='sum e/sum q; delta=GMMdB-comparatordB',code='src/run_gmm3.py:132-155; src/independent_review_gmm.py',meaning='same2000 persisted trajectory draws; unadjusted exploratory95%'),
dict(object='Cost timing endpoint',equation='posterior+all3maps+weighted output; adaptive fit+deployment',code='src/benchmark_gmm3.py:22-97',meaning='1thread; samewindows/warmup/order; no entropy/logging/finite guards inGMMtimedkernel')]
csvout(OUT/'tables/equation_code_mapping.csv',mapping)

text(OUT/'mathematical_definition.md',r'''# Prepared-origin Gaussian posterior mixture

This baseline uses three authenticated empirical preparation laws (Source,
Profile, Mobility), each500 independent trajectories and5000 clean frames.
No new clean statistics, covariance refit, EM or evaluation-data model fitting
is performed. It is a fixed prepared3 Gaussian estimator, not a reproduction
of the author's MFA and not the cross-pilot Convex coordinate estimator.

For q=1,2,3, C_q=R_pp,q+sigma^2 I is the64x64 noisy pilot covariance and
W_q=R_dp,q C_q^{-1} is3968x64. Given current p inC^64, the component mean is
m_q(p)=mu_d,q+W_q(p-mu_p,q). Noise is proper complex with covariance sigma^2 I.
The pilot density is exp[-(p-mu_p,q)^H C_q^{-1}(p-mu_p,q)]/
(pi^64 det C_q). Equal priors are exactly1/3; the common -64log(pi) cancels.

L_q L_q^H=C_q; logdet=2sum log diag(L_q); quadratic=||L_q^{-1}delta||^2.
The Cholesky solve computes W via its adjoint, without forming an inverse.
Scores are centered before logsumexp. Gamma_q is finite, nonnegative and
sums to1. The final output is sum_q gamma_q m_q(p). Each origin uses its
own natural mean, rather than Source-centering every component.

For ordinary row-transpose complex batches, delta @ W.T gives the output;
using W.conj().T would change the conditional mean. The6-frame batch in the
production call is vectorization of independent current-frame functions,
not history fusion. Inference receives no condition label, clean target CSI,
target covariance or future pilot. The weighted conditional mean is exact
for the defined Gaussian mixture model; empirical moments do not establish
that this mixture equals the target population distribution.

Shapes: mu_p(64),mu_d(3968),R_pp(64,64),R_dp(3968,64),allcomplex128.
Nine component maps (3origins x3noise levels) match the previous natural
LMMSE maps exactly. sigma^2=0.9941134822523973/0.09941134822523973/
0.009941134822523972 at10/20/30dB, using frozen Source power9.941134822523972.

Numerical tolerances: model/toy1e-10 absolute+1e-10relative; posterior sum
1e-12; pipeline identity numerical check1e-10absolute+1e-9relative;
persisted-statistics independent review1e-10absolute+1e-10relative.
Actual posterior summation errors are at machine precision. Smallcomplex
Gaussian toys independently verify conditional maps, means, conjugation,
noncontiguous pilot/data indexing, noise-once, identicalcomponent limit and
single-prior limit. The latter is a synthetic test, never a tuned prior.

Current-frame comparison: Source,Pooled,DirectPPCA,GMM3,LR4Tw1.
Full-system comparison: LR4Tw5,DenseTw5,Whole24Tw5 additionally use4past
frames. Offline model and storage budgets also differ. Same evaluation
inputs do not imply identical temporal or computational budgets.

See tables/equation_code_mapping.csv and src/gmm3.py for implementation.
''')

text(OUT/'policy_clarification.md',r'''# Frozen-author max_condition_number policy clarification

Verdict: initialization-scale parameter; not an optimized-covariance hard cap.

Frozen source reference/sp_u12b5a/paper_period/cplx-mfa/MFA_cplx.py:
- line46 declares max_condition_number; line64 stores it.
- lines122-124 initialize complex random loading with scale
  1/(sqrt(2)*sqrt(max_condition_number)), giving variance1/max_condition_number.
- lines186-189 clip residual variances to1e-6 and, forPPCA=True, average them.
- lines198-205 form the covariance FF^H+diag(psi) without a condition-number cap.
- lines286-296 form the Woodbury/pseudoinverse computation without referencing
  this field. MFA_estimator.py:70-80 adds observation noise once and uses its
  actual Hermitian conditional solve; this field does not govern that solve.

The original final training used the frozen original author fit; prototype
low-memory paths are not substituted into the historical final fit merely
because they are recorded among frozen dependencies.

Original reconstructed prior kappa≈3,192,446.1707 exceeds1e6 but violates no
constraint expressed by this field. It must not be called an illegal model
or evidence that Direct16 bypassed a condition-number feasible-set constraint.
Both final K1/r16PPCA=True andDirect16 have free means, proper-complex rank16
loading and isotropic residual variance with the1e-6floor. Initialization,
EM updates, convergence criterion, finite precision and spectral construction
differ. Same final statistical family does not mean identical algorithms.

This note supersedes any inference of a hard1e6 covariance cap in MR03B
interpretation. Historical MR03B files, model parameters and B5C outcomes
remain byte-preserved. No training, replacement or retrospective correction
has occurred. Existing MR03B performance/fidelity results retain their
original status; this task only clarifies the meaning of one policy field.
''')

cost=rows(OUT/'tables/costs.csv');ts=rows(OUT/'tables/timing_summary.csv');pr=rows(OUT/'tables/timing_paired_ratios.csv')
fullbytes={'GMM_PREPARED_3':12777520,'POOLED_NATURAL':4128768,'PPCA_DIRECT_R32':4128768,'LR4_TW1':4794432,'LR4_TW5':4798528,'DENSE_TW5':12408896,'WHOLE_POOL_M2_R24':4864064}
csvout(OUT/'tables/full_core_storage.csv',[dict(method=z['method'],dominant_complex_MAC=int(z['total_dominant_complex_MAC']),minimal_operator_means_Cholesky_bytes=int(z['deployment_arrays_bytes']),full_core_array_payload_plus_input_bytes=fullbytes[z['method']],Pooled_Source_centered_wrapper_extra_bytes=1024 if z['method']=='POOLED_NATURAL' else 0,scope='oneSNR independentdeployment; excludesPythonobjects,temporaries,cleanstats,unusedbanks,multipleSNRcache') for z in cost])
text(OUT/'cost_interpretation.md',r'''# Computation and memory endpoint

Fixed dimensions: Q=3,np=64,nd=3968,M=2,F=4,no=48,nh=16,r=4.
GMM dominant complex-MAC proxy:
Q nd np + Q np(np-1)/2 =761856+6048=767904.
The triangular solve also has real diagonal divisions. Mixture output has
Q nd=11904complex-by-real products; norms,real divisions,exp/logsumexp,
additions and control/validation are not counted as complexMACs.

Pooled and the ACTUAL cached Direct32 implementation each use a dense
conditional W:253952MACs. Loading rank32 does not imply a rank32 deployed
pilot-to-data cache. LR4Tw1=295424;LR4Tw5=332288;DenseTw5=807936;
Whole24Tw5=336384. Adaptive counts include full original cross-pilot fitting.
Gram/2real-coordinate solve terms are excluded consistently from this proxy.

Clean3-origin statistics:786240complex entries=12579840B. These are model
construction assets, distinct from noise-dependent deployment maps.
At eachSNR,GMM maps alone=761856complex coefficients=12189696B;
means193536B;3full-array lower Cholesky factors196608B.
Minimal map/mean/L payload=12579840B. Actual component objects also retain
the3noisy covariance arrays196608B, yielding12776448B, plus48B for prior and
log-prior arrays. Scalars, Python objects and temporary outputs are extra.

tables/costs.csv deployment_arrays_bytes is MINIMAL array payload, not total
deployment storage or RSS. Adaptive full-core payload additionally retains
9216complex fold-kernel entries147456B, fold indices2048B, realGram/RHS48B,
coordinates16B andnp*Twcomplex pilot buffer1024/5120B. Including these gives
LR4Tw1=4794432B,LR4Tw5=4798528B,DenseTw5=12408896B,Whole24Tw5=4864064B.
Pooled/Direct32 map+naturalmeans+currentpilot=4128768B. The original Pooled
source-centered-to-natural wrapper also allocates1024B conversion storage,
giving4129792B for that particular wrapper. GMM actual retained arrays plus
currentpilot=12777520B. These exclude Python overhead and dynamic workspaces.

Actual timer: prevalidated full mathematical estimator core. GMM includes
posterior triangular solves, stableweights,all3conditionalmeans and weighted
output; its entropy/diagnostic arrays and heavyfinite/shapeguards are outside
the timer. Adaptive methods include unchanged fitting plus deployment.
Frozen config wording 'posterior diagnostics included in GMM production API'
describes the full scoring API, NOT the timed kernel. reports/timing.json
same_endpoint and the frozen benchmark code are authoritative for actual
timed work. The diagnostic distinction is clarified here without changing
the frozen code/config or repeating the measurement.

Windowsi7-13650HX;actualNumPy/SciPyBLASthreads1;100samewindows/SNR,
30warmupcalls/method/SNR,10blocks,randomizedmethodorderseed170505;
21000measuredcalls. Same currentpilot and same available5-framewindow;
current-only methods deliberately use only the last frame. Disk,generation,
hashes,logging,plotting,setup and output serialization are excluded.
Paired block ratios are descriptive, not latency confidence guarantees.

Peak processRSS contains runtime, shared banks/data and all resident models.
V_new scoring includes generator runtime; D_eval reads saved inputs. Neither
is a standalone model memory estimate. The timing process also contains all
comparators. Separate persistent array bytes, scoring-processRSS and timing
processRSS. No bottleneck attribution, hardware/energy or real-time claim.
Noise-dependent construction timings and eigendiagnostics are reported
separately in tables/component_setup.csv; these are single construction
observations, not a formal setup benchmark.
''')

parts=['# MR-03C 中文科研审查报告\n\n执行状态：完成。两套已揭示开发数据均15格、每格100条物理轨迹，每条6个评分帧；GMM新增18000位置，旧8方法能量直接复用。未生成确认集、未重训、未调先验。\n']
parts.append('## 1. 完整绝对NMSE\n\n单位dB，越低越好；所有数值来自本轮真实pooled-energy CSV。表内四舍五入至4位，CSV保留原精度。\n')
short={'SOURCE':'Source','POOLED_NATURAL':'Pooled','GMM_PREPARED_3':'GMM3','PPCA_DIRECT_R16':'Direct16','PPCA_DIRECT_R32':'Direct32','LR4_TW1':'LR4Tw1','LR4_TW5':'LR4Tw5','DENSE_TW5':'Dense5','WHOLE_POOL_M2_R24':'Whole24'}
for ds in cfg['datasets']:
    parts.append('\n### '+ds+'\n\n'+table(['condition','SNR']+[short[m] for m in methods],[[c,s]+[f"{float(ix[ds,c,s,m]['nmse_db']):.4f}" for m in methods] for c in conds for s in snrs]))
parts.append('\n## 2. 配对区间和负面边界\n\n差值GMM3−对照，正值对照更好。复用2000个原始trajectory配对draws；以下全部是已揭示开发数据的未调整探索性95%区间，不是新的familywise正式确认。\n'+table(['dataset','comparison','GMM区间更低','对照区间更低','跨零'],[[z['dataset'],z['contrast'],z['GMM_better_interval'],z['comparator_better_interval'],z['crosses_zero']] for z in signs]))
parts.append('\n关键联合失配和边界：\n\n'+table(['dataset','condition','SNR','GMM−Pooled [95%]','GMM−LR4Tw1 [95%]','GMM−LR4Tw5 [95%]'],[[ds,c,s]+[f"{float(cx[ds,c,s,m]['estimate_db']):+.4f} [{float(cx[ds,c,s,m]['CI_lower']):+.4f},{float(cx[ds,c,s,m]['CI_upper']):+.4f}]" for m in ('POOLED_NATURAL','LR4_TW1','LR4_TW5')] for ds in cfg['datasets'] for c in ('Composed','Boundary') for s in snrs]))
parts.append('''
GMM相对Pooled、LR4Tw1、LR4Tw5在两套数据均11格区间全负、4格全正。
反向4格为Composed10/20/30和Boundary10，均完整保留。相对Dense/Whole24
支持GMM的很多幅度小于0.24dB，不能以获胜格数替代实际性能幅度。
Individualtrajectory并非全部具有相同排序；tables/trajectory_paired_differences.csv
保留15000个逐轨迹差值，paired_contrasts.csv保留逐格GMM较差轨迹比例。

## 3. 科学问题逐项回答

1. 相对Pooled有实质但有条件的增量。ID/Profile/Mobility和Boundary20/30
均支持GMM，Composed三个SNR及Boundary10反向，因此不能推广为一般混合优势。
2. Composed10/20/30，V_new的GMM−LR4Tw5为4.2963/6.0008/2.8510dB，
D_eval为4.5324/6.2224/3.0738dB。GMM甚至弱于Pooled；联合失配是明确失败边界。
3. Profile/Mobility三个SNR，GMM均优于Pooled和两种LR4。对Dense/Whole24
有微幅优势、近似相等和反向情况，不能统称所有对照均胜出。
4. Boundary10反向，20/30优于Pooled/LR4但Dense/Whole24接近或更好。
没有对应Boundary的唯一真实prepared来源，不能把Profile后验大称为错误选择。
5. 当前帧公平口径下，GMM较LR4Tw1在11格更好、4格更差；Composed的
LR4Tw1优势说明全系统Composed差距不能全部归因于额外四帧历史。
6. 相对LR4Tw5也为11/4；其额外历史预算必须明确，不能将两类比较混为一谈。
7. GMM较强部分削弱“在线坐标细化普遍比统计后验混合更好”的主张。
当前论文价值应落到有限准备空间中有条件的细化与部署精度成本折中，保留
准备来源Gaussian posterior mixture作为真正强对照，而不是弱外部实例替代物。
8. Composed和Boundary10弱于Pooled与有限prepared-origin模型在该条件下的
边界一致；不能凭此断言具体原因是过拟合、后验错误或噪声增强。
9. 同机成本高于Pooled、Direct32、LR4和Whole24，低于Dense；部分精度
优势需要对照较高成本评价，不存在统一最优。MAC和实际计时分别报告。

## 4. 后验诊断

三分量后验仅由当前接收导频生成。condition标签只用于离线汇总。
Composed的平均Mobility权重约0.83–0.94，20/30dB多数位置maxgamma≥.95；
低entropy与较高误差共同出现，但没有干预证据证明前者导致后者。
Boundary10后验在Source与Profile之间分配较多，20/30dB转为Profile占主导，
同时其对Pooled/LR4的排序改善，说明来源物理标签差异不能简单等同误差机制。
ID/Profile/Mobility可定义离线准备来源correspondence；Composed/Boundary
不存在单一正确来源，相关mismatch字段留空。Profile10的trajectory-level
对应来源不一致率与GMM−Pooled误差差值Spearman约V=.3114,D=.5004；
其他condition差异不稳定或没有不一致变化，不能提出普遍因果解释。
H采用自然对数，单位nats；图仅为统一0–1显示除以ln3，不改原数据。
''')
parts.append('\n后验摘要（meanγSource/γProfile/γMobility，Hnats，近硬选择比）：\n\n'+table(['dataset','condition','SNR','γS','γP','γM','H(nats)','maxγ≥.95'],[[z['dataset'],z['condition'],z['snr_db']]+[f"{float(z[k]):.6f}" for k in ('mean_weight_Source','mean_weight_Profile','mean_weight_Mobility','mean_entropy_nats','hardlike_fraction_gamma_ge_095')] for z in dg]))
parts.append('\n## 5. 实际成本\n\n完整estimator-core同机单线程中位时间（微秒/帧），现有D_eval同窗口、100次/块、10块、每法30次预热；当前帧法只读取末帧，窗口法保持其真实历史。\n\n'+table(['method','10dB','20dB','30dB'],[[m]+[f"{float(next(z for z in ts if z['method']==m and int(z['snr_db'])==s)['median_us']):.3f}" for s in snrs] for m in timing['methods']]))
parts.append('\n'+table(['method','dominantcomplexMAC','map/mean/Lminimalbytes'],[[z['method'],z['total_dominant_complex_MAC'],z['deployment_arrays_bytes']] for z in cost]))
parts.append(f'''
GMM保留C后的实际noise-dependent组件数组每SNR12776448B（另priors48B），
与仅maps/means/L的12579840B区别明确。统计矩阵本身12579840B，与部署
缓存不是同一资产。参见cost_interpretation.md的完整foldkernel/buffer存储。
V_new评分峰值RSS={receipts[0]['resource']['lifetime_peak_RSS_bytes']/2**20:.3f}MiB
（含信道generator运行时）；D_eval评分峰值RSS={receipts[1]['resource']['lifetime_peak_RSS_bytes']/2**20:.3f}MiB；
同机计时进程峰值RSS={timing['resource']['lifetime_peak_RSS_bytes']/2**20:.3f}MiB
（含全部比较方法）。这不是每个方法独立RSS或部署容量。
每SNR Cholesky/logdet与conditionalmap构造在component_setup.csv分开记录；
谱诊断检查开销单列，没有按MAC猜测runtime瓶颈。

## 6. 数值、图像和保全

34个实际pytest通过；10条现有真实轨迹smoke共180位置通过，并直接复用
到完整评价。九个分量W与原LMMSE缓存精确一致；三个SNR全部9个C均HPD。
18000位置后验和最大误差{allgammaerror:.4g}；失败样本0，非有限0。
独立198015次数值比较最大绝对误差{review['max_absolute_error']:.6g}；
8个旧方法逐轨迹能量原样复用，原抽样数组原样复用，未生成新draws。
5图共24面板实际打开；PDF/SVG+600dpiPNG，字形≥7pt、无裁切/冲突。
早期绘图字段适配和色标位置问题只改展示代码，原错误日志/初版保留。
首次pytest缺PYTHONPATH导致collection失败，补齐运行路径后34项通过；
没有以失败测试为科学成功，相关原始日志全部保留。
max_condition_number源码核对及MR03B含义澄清见policy_clarification.md。
所有科学输入/模型/旧结果/论文的保全哈希在preservation_after.json复核。

## 7. 研究决策

本次完成最后一个优先强统计开发对照。建议导师冻结独立确认对象和
明确的准备统计/当前帧/完整系统口径。GMM3、Pooled/Direct32、LR4/Whole24
具有不同边界，不自动选新主方法或继续扩大列表。所有结论目前属于已
揭示开发数据；不将本次探索性区间升级为独立确认。完成后停止。
''')
text(OUT/'scientific_review_cn.md',''.join(parts))

text(OUT/'reports/lessons_learned.md','''# Task-local research lessons

- Runpytest with explicit sourcePYTHONPATH when the tests exercise a new standalone
  module; retain the original collection failure and verify the corrected command.
- Figure input schema must be checked before export. Truth is in raw/traj tables,
  not an assumed field in the NMSEsummary; no scientific values may be fabricated.
- Review actual rendered colorbar/textgeometry; preserve earlier draft evidence.
- Report timedkernel contents from actual frozen code. Full diagnosticAPI and
  estimatorcore are different endpoints; avoid interpreting descriptiveconfig
  wording as proof that entropy/finiteguards were timed.
- Sourcepolicy names are not mathematical constraints. Verify every use of an
  argument before declaring a trained model outside the feasible set.
''')

text(OUT/'README.md',r'''# MR03C completed development review

Raw→statistics→figures chronology is retained. data/V_new anddata/D_eval each
have500authenticatedJSON/NPZ pairs. All9methods' frameenergies and commontruth
are persisted; only GMMpredictions were newly calculated. EachNPZ records
pilots,3posteriorweights,entropy,maxweight,scores,quadratics andfailureflags.
No full3968-dimensionalpredictiontensor is required for energy/CI reproduction.

tables/nmse.csv is270rows;nmse_wide.csv is30cells x9methods.
paired_contrasts.csv is150contrasts with existing2000trajectorydraws.
trajectory_energies.csv is27000rows;trajectory_paired_differences.csv is15000rows.
Allintervals areunadjustedexploratory95% onalreadydiscloseddevelopmentdata.

scientific_review_cn.md gives complete9method tables and positive/negative
findings. mathematical_definition.md,equation_code_mapping.csv,gmm3.py define
the estimator. policy_clarification.md corrects the interpretation of the
author initializationfield without modifying historicalMR03B.

Review package includes smallrawstatistics/posteriors/pilots,originaldraws,
alltables,figures,sourcecode,tests,logs,hashesandreports. ExistinglargeH,
preparationcorpora,originalMFAparameters andoldformaloutputs are not duplicated.
The fullscientific inference drivers rely on their authenticated original
project inputs; they are not rerunnable into an existing immutable output.
Independent packaged review can run without any largeprojectinput.

For local numeric unit tests in PowerShell:
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH=(Join-Path $PWD 'src')
python -B -m pytest .\tests\test_gmm3.py -q -p no:cacheprovider

For package-only persisted statistic review:
python -B .\src\independent_review_gmm_standalone.py
This performs no predictions,training ornewrandomsampling andwrites nooldfile.
''')

after=verify();write(OUT/'reports/preservation_after.json',after)
completion=dict(PASS=True,task='SP-U17/MR-03C',new_scored_GMM_positions=18000,physical_trajectories=1000,cells=30,methods=methods,pytest_passed=34,smoke_positions=180,smoke_positions_reused=True,independent_review=review,visual_QA=qa['PASS'],timing=timing['PASS'],failure_samples=0,nonfinite_outputs=0,posterior_sum_error_max=allgammaerror,model_retraining=0,new_moments=0,new_validation_sets=0,new_confirmation_sets=0,new_bootstrap_draws=0,model_or_prior_tuned=False,old_outputs_or_manuscript_modified=0,preservation=after,scope='Previously revealed development diagnostics; unadjusted exploratory95%')
write(OUT/'reports/completion.json',completion)
print(json.dumps(dict(PASS=True,wide_rows=len(wide),trajectory_differences=len(diff),preservation=after),ensure_ascii=False))
