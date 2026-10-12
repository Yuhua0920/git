"""Text-only presentation and review package of completed MR05 data."""
from pathlib import Path
import csv,json,hashlib,zipfile,platform,os,winreg
import numpy as np
from run_mr05 import OUT,ROOT,CONDITIONS,SNRS,METHODS,CONTROL,verify_sources,write,csvout,sha

def rows(name):
    with (OUT/'tables'/name).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
def text(path,value):
    with path.open('x',encoding='utf8') as f:f.write(value)
def main():
    assert json.loads((OUT/'reports/analysis_integrity.json').read_text())['PASS']
    assert json.loads((OUT/'reports/independent_review.json').read_text())['PASS']
    assert json.loads((OUT/'reports/timing.json').read_text())['PASS']
    assert json.loads((OUT/'reports/visual_qa.json').read_text())['PASS']
    n=rows('nmse.csv');v={(r['condition'],int(r['snr_db']),r['method']):float(r['nmse_db']) for r in n}
    contrasts=rows('contrasts.csv');cs={(r['condition'],int(r['snr_db']),r['contrast'],r['first'],r['second']):r for r in contrasts}
    decomp=rows('decomposition.csv');ds={(r['condition'],int(r['snr_db']),r['method']):r for r in decomp}
    factor=rows('factor_audit.csv');timing=rows('timing_summary.csv');ct=rows('costs.csv')
    development=json.loads((OUT/'reports/development_receipt.json').read_text());integrity=json.loads((OUT/'reports/analysis_integrity.json').read_text())
    m2max=max(abs(v[c,s,'CORR_ORIGIN_R4']-v[c,s,'CORR_ORIGIN_M2_R4']) for c in CONDITIONS for s in SNRS)
    rankmax=max(abs(v[c,s,METHODS[7]]-v[c,s,METHODS[8]]) for c in CONDITIONS for s in SNRS)
    densemax=max(abs(v[c,s,m]-v[c,s,'DENSE_TW5']) for c in CONDITIONS for s in SNRS for m in METHODS[7:])
    headings=['condition','SNR','原LR4','I权重','来源M2','CorrPool','Whole23','Whole24']
    core=[];lines=['| '+' | '.join(headings)+' |','|'+'|'.join(['---']*8)+'|']
    for c in CONDITIONS:
        for s in SNRS:
            vals=[v[c,s,m] for m in METHODS[3:]]
            lines.append('| '+' | '.join([c,str(s)]+[f'{x:.6f}' for x in vals])+' |')
            core.append(dict(condition=c,snr_db=s,**{m:v[c,s,m] for m in METHODS[3:]}))
    csvout(OUT/'tables/core_15_cells.csv',core)
    report=['# MR-05 第一阶段中文实验结论','',
        '执行状态：PASS。全规模开发诊断、统计、两端点计时和四图视觉检查完成；没有选择最终设计。',
        '五条件×100条物理trajectory×3 SNR×6评分帧，共500物理trajectory、15单元、每方法9000位置。认证复用四方法36000位置；新五控制45000输出；为风险分解重放原四输出36000位置。另行小型全链5条trajectory不进入主统计。',
        '额外检查单列：通过的小型全链新增450个控制输出、360个原方法重放；第一次JSON失败前完成ID一条的90个控制输出和72个原方法重放，原始记录保留，均不加入主统计。计时54000调用及1620预热调用单独计量。',
        '原输入500/500哈希、CRC、种子与pilot/noise绑定通过。原坐标和复用能量逐位一致；重放数值有浮点路径差，保留原能量没有覆盖。',
        f"重放能量最大绝对差 {development['audit']['replay_max_abs']:.12g}，scaled差 {development['audit']['replay_scaled']:.12g}；逐frame signed恒等式最大绝对残差 {integrity['frame_signed_identity_max_abs']:.12g}，scaled残差 {integrity['frame_signed_identity_scaled']:.12g}。",
        '绝对和相对容差预定均为1e−10，具体按算子/能量尺度。采用same saved 2000次trajectory indices，无新随机抽样；285行配对区间均未调整、探索性，非新独立确认。',
        '', '## 全部15格结果', '以下为 Pooled-energy NMSE (dB)，越低越好。完整135行在 tables/nmse.csv，全部285个差值/区间在 tables/contrasts.csv。',*lines,'',
        '## 同rank权重回答',
        f'来源二阶矩增补与原权重的最大点估计差只有 {m2max:.9f} dB；本参考均值项未表现为重要实际改进。这不是普遍等效检验。',
        '原协方差权重相较单位权重的作用明显且有条件：Mobility30单位权重更差8.829820 dB，Profile30更差4.826766 dB；Boundary20单位权重反而更好0.563134 dB（探索区间0.492875至0.633987）。不能声称原权重是全条件最优。',
        'CorrPool在15格点估计均比原LR4差，其中11格探索区间不跨零；Mobility30差6.581966 dB。更完整/合并的Preparation二阶矩并不自动改善rank4最终风险。没有追加无界weight搜索。',
        '', '## 近预算位置回答',
        '严格位置对照使用共同C_pool：Whole23/24在12个非ID格均优于CorrPool，探索区间全部位于Whole−CorrPool的负侧；ID三格CorrPool小幅更好，约0.012487/0.025690/0.029363 dB。这一结果明确削弱correction-only压缩位置的普遍优越性主张。',
        'Whole23较CorrPool：Composed10/20/30改进1.188611/1.573805/2.263351 dB；Profile30改进3.914164 dB；Mobility30改进7.176245 dB；Boundary20/30改进1.012650/3.972323 dB。点估计、完整CI及15格反向/正向结果全部保留。',
        f'Whole23/24之间最大点估计差 {rankmax:.12g} dB；两者相对Dense的最大点估计差 {densemax:.12g} dB。图中曲线重叠是实际数值很接近，不代表做了等效检验，也没有据此只选其中一个rank。',
        '原LR4 vs Whole同时涉及weight与位置，不能作为单独位置归因。实际budget分别为332288/324288/336384 MAC，whole23低2.408%、whole24高1.233%，不是成本严格相等。',
        '', '## 关键问题与风险交叉项',
        'Composed10：原LR4较Pooled增益1.937392 dB；CorrPool降至0.989642 dB；Whole23为2.178253 dB。原有效增量并非所有设计同样保留。',
        f"Composed20：原LR4−Dense代价{v['Composed',20,'CORR_ORIGIN_R4']-v['Composed',20,'DENSE_TW5']:.6f} dB，CorrPool增至{v['Composed',20,'CORR_POOL_M2_R4']-v['Composed',20,'DENSE_TW5']:.6f} dB。改weight没有消除原损失，不能把全部差异归因于pilot目标迁移。",
        'Composed30：CorrPool的A/Q为0.003320246，比原LR4的0.005772694更小，但B/Q从−0.003527409减弱至−0.000262588，故总linear风险差反而从0.002245285升至0.003057658。A下降不等于最终NMSE下降。',
        'ID及Boundary10原LR4存在负compression代价：ID10 A/Q=3.389457e−5、B/Q=−5.661758e−5，总和=−2.272301e−5；Boundary10 A/Q=0.001311374、B/Q=−0.002911305，总和=−0.001599931。这是同一输出/残差上的精确代数关系，不能称噪声有益或物理去噪机制已证明。',
        'Boundary10的CorrPool A更小（0.000515883），却失去原更强的负B，总风险差变正（0.000198882）。Boundary20/30也必须与Dense及Pooled共同看，而不是只看Source-relative收益。',
        '', '## 成本与实际计时',
        '一般MAC：fit=(M+1) F Tw no nh；Source=nd np；correction部署=nd np+M r(nd+np)；whole部署=(M+1) r(nd+np)。当前公共fit46080。Gram、solve、scalar权重和加法不包含于dominant proxy。',
        '每operating SNR最小部署矩阵/因子+必要均值：Corr4 290240 complex /4643840B；Whole23 282240 complex /4515840B；Whole24 294336 complex /4709376B。完整cross-pilot核、buffer、fold索引、小矩阵和实验复查多SNR资产分别在costs/timing报告列出，不用RSS冒充因子存储。',
        '每SNR100固定window、三SNR共300；每方法每端点30次预热、10个随机打散成对块，每块100调用，共54000测量调用。full-core对七个适应控制独立执行原两方向五帧拟合；Source/Pooled仅用当前帧。大矩阵验证、IO、日志和绘图排除计时。',
        '下表是同机单线程块中位数μs；离线构造时间见 tables/offline_construction.csv，进程RSS见 reports/timing.json。rank23在此机比rank24慢，未以没有profiling的原因解释，也不与旧B4拼接速度结论。',
        '|端点|SNR|原LR4|CorrPool|Whole23|Whole24|','|---|---|---|---|---|---|']
    for endpoint in ('deployment-only','full-estimator-core'):
        for s in SNRS:
            vals=[float(next(r for r in timing if r['endpoint']==endpoint and int(r['snr_db'])==s and r['method']==m)['median_us']) for m in (METHODS[3],METHODS[6],METHODS[7],METHODS[8])]
            report.append('| '+' | '.join([endpoint,str(s)]+[f'{x:.3f}' for x in vals])+' |')
    report += ['', '## 获得与未获得的证据',
        '新证据：同坐标下来源权重相对I的条件作用；mean-shift增补很小；pooled权重不是改善保证；近预算whole-bank能够明显改变压缩代价；有符号残差交叉项解释A与风险排序不一致；两端点运行时间和部署资产分离。',
        '未测：重锚定、联合风险最优权重、自适应rank、新拟合目标、全rank曲线、更多独立target、MFA容量或同信息检验。所有结果来自已揭示开发集，不能直接选择最终weight/rank/anchor，不能升级为新确认结论。',
        '科学主方法、原rank选择、论文和历史B2/B3/B5C均未修改或运行。V4.2主文/Supplement来源链作为稿件基线；main(1).pdf/supplement(1).pdf别名未找到，未据此回退V4.1。',
        '', '## 技术记录',
        '第一条小型检查在JSON布尔序列化失败，原NPZ/部分JSON/traceback/初始源码保留；仅改显式bool及独立smoke输出标签后全链PASS。完整development只执行一次。计时预检发现原fit接口必需显式modules=(P,M)，仅修调用；一个synthetic夹具使用错误导频geometry，改为真实冻结几何后31项pytest通过。各初始日志/源码快照均保留，未补造成功状态或混入失败数据。',
        'STOP：没有替换原方法，没有启动MR03或MR05后续阶段。']
    body='\n\n'.join(report)+'\n'
    while '|\n\n|' in body: body=body.replace('|\n\n|','|\n|')
    text(OUT/'chinese_experiment_report.md',body)
    maxes={k:max(float(r[k]) for r in factor) for k in ('hermitian_relative_residual','cholesky_relative_residual','right_solve_relative_residual','full_rank_relative_residual','weighted_tail_error_over_total')}
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:cpu=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
    write(OUT/'reports/final_summary.json',dict(PASS=True,pytest_passed=31,scientific_preservation=verify_sources(),factor_audits=len(factor),factor_maxima=maxes,mean_shift_max_abs_difference_db=m2max,whole23_24_max_abs_difference_db=rankmax,whole_max_abs_difference_from_dense_db=densemax,model_and_rank_selection_changes=0,manuscript_changes=0,old_formal_runs=0,new_channel_or_noise_generation=0,scientific_development_runs=1,new_control_positions=45000,reused_positions=36000,check_replay_positions=36000,bootstrap_new_draws=0,CPU=cpu,OS=platform.platform(),logical_cores=os.cpu_count(),host=platform.node(),no_further_task_started=True))
    files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('results_review.zip','results_review.zip.sha256','results_review_file_manifest.csv','package_report.json')]
    manifest=[dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    csvout(OUT/'results_review_file_manifest.csv',manifest);files.append(OUT/'results_review_file_manifest.csv')
    zpath=OUT/'results_review.zip'
    with zipfile.ZipFile(zpath,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(zpath) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(files)
        for r in manifest:assert hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256']
    h=sha(zpath);text(OUT/'results_review.zip.sha256',h+'  results_review.zip\n')
    write(OUT/'reports/package_report.json',dict(PASS=True,path=str(zpath),sha256=h,bytes=zpath.stat().st_size,file_count=len(files),all_members_independently_SHA256_checked=True,ZIP_CRC=True))
    print(json.dumps(dict(PASS=True,ZIP=str(zpath),sha256=h,bytes=zpath.stat().st_size,files=len(files)),indent=2))
if __name__=='__main__':main()
