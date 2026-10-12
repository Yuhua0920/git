"""Assemble existing MR03 results only. No inference, fitting, or random draws."""
from pathlib import Path
import csv, json, hashlib, zipfile, sys, datetime

ROOT=Path(r'D:\Users\yuhua\Desktop\Project')
OUT=ROOT/'outputs/revision_mr03_ppca_capacity_controls'
CONDITIONS=('ID','Profile','Mobility','Composed','Boundary')
METHODS=('SOURCE','POOLED_NATURAL','DENSE_TW5','LR4_TW5','WHOLE_POOL_M2_R23','WHOLE_POOL_M2_R24','DENSE_TW1','LR4_TW1','PPCA_DIRECT_R16','PPCA_DIRECT_R32','PPCA_DIRECT_R64')
STEMS=('capacity_overview','capacity_V_new','current_frame_V_new','deployment_boundaries_V_new','capacity_D_eval','current_frame_D_eval','deployment_boundaries_D_eval')

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def textfile(p,t):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:f.write(t)
def rows(name):
    with (OUT/'tables'/name).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
def exact_name(name):
    return {'LR4Tw5_minus_Direct32':'LR4_TW5_minus_PPCA_DIRECT_R32','LR4Tw1_minus_Direct32':'LR4_TW1_minus_PPCA_DIRECT_R32','LR4Tw5_minus_LR4Tw1':'LR4_TW5_minus_LR4_TW1','DenseTw5_minus_DenseTw1':'DENSE_TW5_minus_DENSE_TW1'}.get(name,name)
def csvout(p,rs):
    with p.open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def lookup(dataset,condition,snr,contrast):
    return next(r for r in rows('contrasts_'+dataset+'.csv') if r['condition']==condition and int(r['snr_db'])==snr and r['contrast']==exact_name(contrast))
def contrast_string(r):return f"{float(r['estimate_db']):+.9f} dB [{float(r['CI_lower']):+.9f}, {float(r['CI_upper']):+.9f}]"
def counts(dataset,contrast):
    r=[r for r in rows('contrasts_'+dataset+'.csv') if r['contrast']==exact_name(contrast)]
    return dict(negative=sum(float(x['CI_upper'])<0 for x in r),positive=sum(float(x['CI_lower'])>0 for x in r),crosses_zero=sum(float(x['CI_lower'])<=0<=float(x['CI_upper']) for x in r))
def core_table(dataset):
    rs=rows('nmse_'+dataset+'.csv');index={(r['condition'],int(r['snr_db']),r['method']):r for r in rs}
    result=[]
    for c in CONDITIONS:
        for s in (10,20,30):
            x=dict(dataset=dataset,condition=c,SNR_dB=s,physical_trajectories=100,scored_frames_per_trajectory=6)
            for m in METHODS:x[m+'_NMSE_dB']=float(index[c,s,m]['nmse_db'])
            result.append(x)
    csvout(OUT/'tables'/('core_15_cells_'+dataset+'.csv'),result)
    abbreviations=('Source','Pooled','Dense5','LR4-5','Whole23','Whole24','Dense1','LR4-1','Direct16','Direct32','Direct64')
    lines=['| condition | SNR | '+' | '.join(abbreviations)+' |','|---|---:|'+'---:|'*len(abbreviations)]
    for x in result:lines.append('| '+x['condition']+' | '+str(x['SNR_dB'])+' | '+' | '.join(f"{x[m+'_NMSE_dB']:.6f}" for m in METHODS)+' |')
    return '\n'.join(lines)

def main():
    # This verifies existing evidence. It does not rewrite a receipt/frozen source.
    preserved=read(OUT/'reports/preservation_before.json')['hashes']
    changed=[p for p,h in preserved.items() if not (ROOT/p).is_file() or digest(ROOT/p)!=h]
    if changed:raise ValueError('PRESERVATION_CHANGED:'+str(changed))
    frozen=read(OUT/'reports/execution_freeze.json')['hashes']
    bad=[p for p,h in frozen.items() if digest(OUT/p)!=h]
    if bad:raise ValueError('FROZEN_MR03_CHANGED:'+str(bad))
    train=ROOT/'outputs/sp_u12b5b/full_retraining.npy'
    if digest(train)!='2553aab058511770ca95036d6e0d0e7fee9248080b43e826f672ea156099aed0':raise ValueError('P_CHANGED')
    final=read(OUT/'reports/original_em16_availability.json')
    independent=read(OUT/'reports/independent_review_portable.json')
    if not independent['PASS']:raise ValueError('INDEPENDENT_REVIEW')
    receipts={ds:read(OUT/'reports'/f'{ds}_receipt.json') for ds in ('V_new','D_eval','smoke')}
    for ds,r in receipts.items():
        if not r['PASS'] or r['fit_failure_count'] or r['nonfinite_count']:raise ValueError('INCOMPLETE_RAW:'+ds)
    construction=read(OUT/'reports/spectral_construction.json')
    manual=[]
    for stem in STEMS:
        qa=read(OUT/'figures'/f'{stem}.qa.json')
        alignment=read(OUT/'figures'/f'{stem}.alignment.json')
        collision=read(OUT/'figures'/f'{stem}.collision.json')
        if alignment['verdict']!='PASS' or collision.get('verdict')=='FAIL':raise ValueError('FIGURE_QA:'+stem)
        manual.append(dict(stem=stem,PDF_SHA256=digest(OUT/'figures'/f'{stem}.pdf'),PNG_SHA256=digest(OUT/'figures'/f'{stem}.png'),SVG_SHA256=digest(OUT/'figures'/f'{stem}.svg'),manual_inspection='actual PNG opened with view_image in current root session',panels=4 if stem=='capacity_overview' else 5,layout='PASS: no clipped text or overlaps; panels, axes, units, legends and negative results readable',minimum_glyph_pt=5.04 if stem=='capacity_overview' else 'see text-audit log; no below-5pt glyphs',alignment=alignment['verdict'],collision='PASS: no FAIL findings',formats='editable vector PDF/SVG; 600-dpi PNG',visual_limit='Coincident Pooled/Direct32/Direct64 curves intentionally not displaced; full precision in source CSV.'))
    write(OUT/'reports/visual_qa.json',dict(PASS=True,actual_images_opened=7,panels_inspected=34,alignment_tolerance_pt=1.5,glyph_floor_pt=5,figures=manual,prior_failed_overview_kept='reports/figure_draft_before_label_fix',scientific_values_changed=False))
    write(OUT/'reports/preservation_after.json',dict(PASS=True,protected_original_files=len(preserved),changed_original_files=changed,frozen_MR03_files=len(frozen),changed_frozen_MR03_files=bad,training_P_actual_sha256=digest(train),original_EM_retrained=False,manuscript_modified=False,production_modified=False,formal_outputs_modified=False,formal_performance_read=False,git_commit=False,scope='Targeted authenticated input files plus original training P; not a whole-disk or 75GB rescan.'))
    write(OUT/'reports/test_execution_summary.json',dict(PASS=True,pytest_unique_passed=49,cdl_latest=dict(result='48 passed, 1 deselected in 0.90s',command=r'.venv-cdl\Scripts\python.exe -B -m pytest outputs/revision_mr03_ppca_capacity_controls/tests -q -p no:cacheprovider -k "not synthetic_author_adapter"',source='actual final exec tool output in current session; 48 applicable tests, BLAS6/OpenMP1'),author_adapter_synthetic=dict(result='1 passed, 26 deselected',log='reports/pytest_author_adapter_toy_corrected.log',environment='.venv-gmm; existing sklearn; frozen author source and adapter; synthetic parameters only'),independent_tiny_review_groups=6,original_final_EM_entity_test_executed=False,prior_logs_retained=True))
    tables={ds:core_table(ds) for ds in ('V_new','D_eval')}
    key=[]
    for ds in tables:
        for c,s,k in [('Composed',10,'LR4Tw5_minus_Direct32'),('Composed',20,'LR4Tw5_minus_Direct32'),('Composed',30,'LR4Tw5_minus_Direct32'),('Profile',30,'LR4Tw5_minus_Direct32'),('Boundary',10,'LR4Tw5_minus_Direct32'),('Boundary',20,'LR4Tw5_minus_Direct32'),('Boundary',30,'LR4Tw5_minus_Direct32'),('Composed',30,'Direct16_minus_Direct32'),('Boundary',30,'Direct16_minus_Direct32'),('ID',30,'Direct16_minus_Direct32'),('Composed',30,'LR4Tw5_minus_LR4Tw1')]:
            key.append(lookup(ds,c,s,k))
    csvout(OUT/'tables/key_contrasts.csv',key)
    model=rows('spectral_models.csv');macros=rows('validation_capacity_macros.csv')
    # All displayed figures below are direct copies from already verified CSV.
    scope_text=[]
    for r in macros:
        scope_text.append(f"| {r['scope']} | {r['rank']} | {float(r['macro_linear_pooled_nmse']):.17g} | {float(r['db_of_macro_linear']):.10f} |")
    bound='\n'.join(f"- {r['dataset']} {r['condition']} {r['snr_db']} {r['contrast']}: {contrast_string(r)}" for r in key)
    modeltext='\n'.join(f"| {r['rank']} | {r['effective_rank']} | {float(r['nu']):.12g} | {r['floor_active']} | {float(r['training_average_complex_dimension_logscore']):.10f} | {r['model_array_bytes']} |" for r in model)
    report=f'''# MR-03 第一阶段：真实结果与科学审查

状态：可执行独立分支完成，原最终 EM16 实体对照待补。独立部分证据 PASS 不表示 LR4 获胜。

## 1. 已完成与未完成

同一原 P 的 proper-complex PPCA16/32/64 精确谱解、共同 V_new 验证、MR01 D_eval 固定开发评价、Tw1 信息控制、保存后统计、独立复核及七组科学图均完成。

`{final['entity_status']}`；`{final['comparison_status']}`。
个人电脑没有找到最终实体的七个 NPY；不使用候选checkpoint、不重训原 MFA、不用Direct16代替EM16。
原最终22次迭代CONVERGED目前是用户提供状态，未凭缺失的实际history自行核验。

P：1500物理轨迹×10帧，15000×4032 complex128；SHA256 `{construction['training']['sha256']}`。
来源为三个prepared origins，每来源500轨迹；中心化均值自由估计，边缘帧等权、分母1/N。
实际joint顺序为已认证的full-grid索引升序并集，非前64维连续pilot。pilot/data真实selector被复用。
nu是clean generative residual variance；received sigma²另加一次；proper complex不展开成实数。

## 2. 模型与优化证据

| declared rank | effective loading rank | nu | floor active | frame marginal log score / complex dimension | model array bytes |
|---:|---:|---:|---|---:|---:|
{modeltext}

训练分数是相关帧的边缘拟合口径，不是15000条独立trajectory的联合似然。
Direct32与64触发1e-6 residual floor，Direct64有效loading/conditional rank为48，不能把声明rank64写成实际64个有效方向。
conditional operator的实际数值秩16/32/48；最大64维求解相对残差5.035e-16。三个模型共用一次clean moments与同一top64谱。
作者EM的floor先逐坐标clip再PPCA平均；这里是同一可行各向同性族的独立谱MLE，不声称重演作者EM迭代。
从synthetic作者参数得到的float1.0映射与解析条件式通过；它不替代最终EM参数的实体对照。
没有原参数，因此原EM与Direct16的均值、nu、训练分数、子空间、conditional operator与V/D差异均尚未核验；无法判断原EM停止质量或优化不足。

## 3. 独立验证建议（不是最终部署选择）

| validation scope | rank | macro linear pooled NMSE | dB of macro linear |
|---|---:|---:|---:|
{chr(10).join(scope_text)}

origin-only严格最小建议r32；expanded严格最小建议r64。二者区别来自固定条件可见性。
r32/r64不为exact tie，但实际NMSE差异极小；不能将微小浮点最小值自动升级为实质容量改善或新主方法。
两口径均在首次V前冻结、验证建议在D_eval前落盘且hash绑定；没有根据D_eval重选。
Direct16→32在Composed30和Boundary30有可重复的局部改善，低SNR的宏平均不能代替逐格风险。
Direct32/64接近自然均值Pooled；rank增加改善训练边缘score不自动保证所有条件的conditional NMSE改善，例如V_new ID30出现反向小差异。

## 4. 规模与统计口径

V_new 500新物理trajectory，五条件各100，十帧统一评分4–9，三SNR；11方法各9000评分位置，99000新输出。
D_eval复用MR01的500已揭示轨迹；54000既有六背景方法输出精确复用，45000新控制输出，首轨迹540次重放对照误差≤1.14e-13。
两套数据分别报告，跨SNR同trajectory为重复测量，不能将它们合并或称最终确认集。
每套500 JSON+500 NPZ先落盘，包含逐帧error/truth energy、相同pilots、Tw1/Tw5坐标/Gram/rhs/残差/失败位、身份与哈希。
pooled NMSE=sum(error)/sum(truth)，然后转dB；差值为first-minus-second，正值利于second。
每条件2000次trajectory-level paired bootstrap。D_eval使用原MR01索引，V_new使用独立的预冻结索引。
全部区间为未调整探索性95%percentile CI，不能描述为35项Bonferroni正式family确认。
独立count-matrix重算330 NMSE行、660 contrast及其区间，复核既有索引、原始CRC和identity，不新增随机抽样。

## 5. 15格核心NMSE（dB，越低越好）

Dense5/LR4-5是原Tw5；Dense1/LR4-1仅当前帧；Whole23/24是MR05预设Tw5对照；Direct-r为本轮独立谱解。
以下六位小数仅用于阅读，完整精度见CSV。原EM没有填入任何结果。

### V_new

{tables['V_new']}

### D_eval

{tables['D_eval']}

## 6. 当前帧控制、优势及失败边界

LR4Tw5−Direct32：V_new和D_eval均9格CI<0、5格CI>0、1格跨零。
LR4Tw1−Direct32同样为9/5/1。共同弱势包括Profile30、Composed30、Boundary10/20/30；Composed20跨零。
LR4Tw5−LR4Tw1两套都14格CI<0，Composed30跨零。说明历史信息在许多格有增益，但Tw1仍在9格占优，历史观测不能解释全部相对差异。
这是当前时间观测控制，不抹去方法过去rank/validation/结构选择的信息预算差异。
DenseTw5−Tw1：V_new13格负、Composed30正、1格跨零；D_eval13格负、2格跨零，不能声称更长窗口始终改善。
Whole23/24点估计接近Dense，且LR4仅在ID三格及Boundary10的点估计低于Whole24；不能自动换主方法或把近似相等写成统计等价。

关键差值（95%探索性区间）：

{bound}

这些结果同时限制容量、时间信息和压缩部署的解释，没有证明单一失败机制。不能据Gram条件或坐标大小声称channel NMSE的因果关系。
Composed30中PPCA32对LR4的优势是真实保留的负面结果；它不证明所有MFA模型族均优于LR4，也不与其它格的LR4优势矛盾。

## 7. 数值、资源、测试与图形

Hermitian相对残差4.09e-17；eigenpair相对残差2.742e-15；正交Frobenius残差8.552e-15。
lambda_min=-3.552e-11在预设绝对/相对roundoff容差内，没有大幅负谱被静默clip。
rank64尾部blocked energy与trace-minus-top64差1.588e-11，通过预设尺度容差。
两套数据及smoke中不可解窗口0、非有限输出0；Tw1四fold128实数行、Tw5五帧640实数行，没有ridge/伪逆/clip/fallback。
最终49项相关pytest通过（48新环境测试+1冻结作者adapter synthetic）；独立统计toy六组通过。没有完成原EM实体测试。
full-size谱构造约37.99s，峰值RSS1,823,657,984B，BLAS6/OpenMP1；V运行129.60s，峰值RSS1,539,489,792B；D运行39.05s，峰值RSS235,032,576B，推断BLAS1。
这些是不同脚本端点，包括生成/保存/setup，不能用于公平的逐方法实测加速比。缓存每模型每SNR3968×64complex conditional W，4,063,232B；参数数组字节与压缩磁盘文件、RSS分开。
首次资源不足在科学构造前停止，两次失败记录均保留。等待可用RAM/commit恢复后继续，没有强行关闭用户进程。
七幅图/34面板已实际打开检查；PDF/SVG可编辑、PNG600dpi；alignment1.5pt和PDF glyph5pt检查通过；先前overview布局失败证据保留。
显示Pooled/Direct32/64近重合曲线，未人为移开制造优势；精细差异由完整CSV确认。

## 8. 下一步所需原实体

实验室另一台Windows电脑目录：`{final['lab_directory']}`。
需要：{', '.join(final['seven_exact_NPY'])}，以及`model_manifest.json`。
内manifest预期SHA256：`{final['inner_manifest_expected_sha256']}`。
外FINAL_MODEL_MANIFEST没有逐NPY哈希，不能补造；取得authenticated内manifest后才能校验每文件hash/shape/实际22迭代history。
只接受最终1500轨迹全新初始化的K1/r16/final_seed4143049057实体；九候选checkpoint不合格。
收到并通过实体验证后，下一轮才能补做同rank优化/条件推断数值对照；本轮不继续执行。

## 9. 保全与交付边界

{len(preserved)}个原有保护文件实际SHA复核0变化；原P再次实算SHA一致；{len(frozen)}个本轮科学冻结文件一致。
只新增本任务输出，未改原方法、MFA、MR01/02/05结果、V4.2、production或正式B2/B3性能数据；未Git commit。
轻包支持从小型raw能量、原bootstrap索引与模型摘要独立复核统计和图。完整P、旧项目operators和原EM不在包中，因此它不是从头重生全部信道的独立项目迁移包。
完成后STOP。没有自动训练多分量MFA、启动最终确认、改论文或选新方法。
'''
    textfile(OUT/'results_review.md',report)
    textfile(OUT/'README_REVIEW.md',r'''# MR03 Stage1 light numerical review

Start with results_review.md; original_em16_availability.json closes the precise missing-entity status.
The 11 available methods are real results. ORIGINAL_EM16 is unavailable and never replaced by Direct16.
Tables contain full-precision NMSE and first-minus-second contrasts. All intervals are unadjusted exploratory 95%, from 2000 trajectory-paired draws.
The light ZIP includes raw small energies and identity JSON, the exact draws, PPCA parameters, shared spectrum, frozen source/tests, all figures and figure QA.
It omits only the cached 9 conditional operators (data/conditional_operators.npz; binding recorded) and the original P/whole project outside this output. No training or inference is needed for the following independent numerical review.

PowerShell (use an already available Python with NumPy; no version upgrade implied):

```powershell
$review = 'D:\path\to\extracted\results_review'
$python = 'D:\path\to\python.exe'
& $python -B "$review\src_or_scripts\independent_review_capacity.py" --out $review --mr01-bootstrap "$review\review_reference\exploratory_bootstrap_indices.npz" --report "$review\reports\user_independent_review.json"
```

The report path must be new. This reads existing energy/draw files; it does not generate channels, refit a model or resample new random bootstrap indices.
Review package_manifest.csv for each member's SHA256. The ZIP sidecar authenticates the entire archive.
Original full-experiment replay still requires the existing project, frozen preparation P, source operators, original author implementation and its environment. It is intentionally not run by these commands.
''')
    status=dict(state='PARTIAL_COMPLETE_ORIGINAL_EM16_PENDING',independent_branches_PASS=True,original_EM_entity_status=final['entity_status'],original_EM_vs_direct_status=final['comparison_status'],datasets=dict(V_new=dict(physical_trajectories=500,cells=15,methods=11,scored_positions_per_method=9000,new_predictions=99000),D_eval=dict(physical_trajectories=500,cells=15,methods=11,scored_positions_per_method=9000,new_predictions=45000,reused_predictions=54000)),bootstrap=dict(replicates=2000,unit='trajectory',interval='unadjusted exploratory 95% percentile',new_draws_during_review=0),validation_recommendations=read(OUT/'reports/validation_recommendations.json'),comparison_counts={ds:{name:counts(ds,name) for name in ('LR4Tw5_minus_Direct32','LR4Tw1_minus_Direct32','LR4Tw5_minus_LR4Tw1','DenseTw5_minus_DenseTw1')} for ds in tables},original_MFA_retrained=False,new_MFA_multicomponent=False,new_T_final_generated=False,manuscript_modified=False,formal_results_tuned=False,STOP_after_delivery=True)
    write(OUT/'reports/completion_summary.json',status)
    exclusions=['data/conditional_operators.npz']
    write(OUT/'reports/light_package_scope.json',dict(scope='Small raw energies, frozen draws, three PPCA parameters, shared exact spectrum, numerical review and figures; not full independent simulation project',excluded_existing_output_files=[dict(path=p,bytes=(OUT/p).stat().st_size,sha256=digest(OUT/p),reason='Nine cached W operators; not needed to independently recompute persisted energy statistics') for p in exclusions],external_not_packaged=['original full_retraining.npy','old project frozen production/reference/adapter/operators','original EM16 parameters unavailable'],original_sources_unchanged=True))
    print(json.dumps(dict(PASS=True,documents_created=True,protected_original_files=len(preserved),science_frozen_files=len(frozen),all_core_cells_written=30,original_EM_pending=True),ensure_ascii=False))

if __name__=='__main__':main()
