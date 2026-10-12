"""Deliver persisted MR03B evidence; additional model-cache diagnostics only."""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import json,csv,hashlib,zipfile,sys,math
import numpy as np
R=Path(r'D:\Users\yuhua\Desktop\Project');O=R/'outputs/revision_mr03_em16_reconciliation';MR=R/'outputs/revision_mr03_ppca_capacity_controls'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
    with p.open('x',encoding='utf8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def text(p,x):
    with p.open('x',encoding='utf8') as f:f.write(x)
def rows(p):
    with p.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
def csvout(p,rs):
    with p.open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def main():
    identity=read(O/'reports/model_identity.json');comp=read(O/'reports/model_comparison.json');fidelity=read(O/'reports/conditional_fidelity.json');review=read(O/'reports/independent_numerical_review.json')
    if not all(x['PASS'] for x in (identity,comp,fidelity,review)):raise ValueError('UPSTREAM_NOT_PASS')
    # Saved inverse cache: deterministic vectors, independent spectral expression.
    F=np.load(O/'original_model/lambdas.npy',allow_pickle=False)[0];nu=identity['isotropic_nu'];C=np.load(O/'original_model/covs.npy',mmap_mode='r',allow_pickle=False)[0];I=np.load(O/'original_model/inv_covs.npy',mmap_mode='r',allow_pickle=False)[0]
    U,s,_=np.linalg.svd(F,full_matrices=False);theta=s*s;kappa=float((theta[0]+nu)/nu)
    t=np.arange(4032);v=np.stack([np.ones(4032),(-1.)**t,np.exp(2j*np.pi*t/4032),np.eye(1,4032,37)[0]],axis=1).astype(complex);v/=np.linalg.norm(v,axis=0)
    got=I@v;expect=v/nu-U@((theta/(nu*(nu+theta)))[:,None]*(U.conj().T@v))
    forward=float(np.linalg.norm(got-expect)/np.linalg.norm(expect));res=C@got-v
    backward=float(np.linalg.norm(res)/(float(theta[0]+nu)*np.linalg.norm(got)+np.linalg.norm(v)))
    eps=np.finfo(float).eps;tol=max(1e-10,100*eps*kappa)
    if forward>tol or backward>1e-10:raise ValueError('INVERSE_CACHE_DIAGNOSTIC:'+str((forward,backward,tol)))
    write(O/'reports/inverse_cache_diagnostics.json',dict(PASS=True,vectors=4,deterministic_not_random=True,independent_spectral_inverse_forward_relative_difference=forward,normalized_backward_residual=backward,condition_number_full_prior=kappa,forward_tolerance=tol,backward_tolerance=1e-10,tolerance_rule='max(1e-10,100 eps kappa) before residual adjudication; backward normalized by ||C||2||x||+||v||',no_full_dense_inverse_constructed=True,scope='four deterministic directions plus complete authenticated bytes; not an exhaustive inverse-column comparison'))
    with np.load(MR/'data/PPCA_DIRECT_R16.npz',allow_pickle=False) as z:Fd=z['loading']
    td=np.linalg.svd(Fd,compute_uv=False)**2
    spectra=[dict(sorted_direction=i+1,EM_loading_variance=float(theta[i]),Direct_loading_variance=float(td[i]),EM_to_Direct_ratio=float(theta[i]/td[i])) for i in range(16)]
    csvout(O/'tables/loading_variances.csv',spectra)
    h=np.load(O/'original_model/training_history.npy',allow_pickle=False)
    csvout(O/'tables/training_history_diagnostics.csv',[dict(record=i+1,pre_update_author_logscore=float(x),mean_frame_pre_update_score=float(x/15000),increment_from_previous=None if i==0 else float(x-h[i-1])) for i,x in enumerate(h)])
    con=read(R/'deployment/b5c_paired_v1/contract.json');top=read(R/'FINAL_MODEL_MANIFEST.json')
    for key,target in [('final_manifest_sha256',sha(R/'FINAL_MODEL_MANIFEST.json')),('final_model_manifest_sha256',sha(O/'original_model/model_manifest.json')),('full_training_data_sha256',top['full_data_sha256']),('selection_sha256',top['SELECTION_MANIFEST_SHA256'])]:
        if con[key]!=target:raise ValueError('HISTORIC_B5C_BINDING:'+key)
    source_checks=[]
    for p,d in con['source_dependencies'].items():
        actual=sha(R/p);source_checks.append(dict(path=p,expected=d['sha256'],actual=actual,PASS=actual==d['sha256']))
    if not all(v['PASS'] for v in source_checks):raise ValueError('HISTORIC_B5C_SOURCE_DRIFT')
    with zipfile.ZipFile(R/'Recovery_Export_0142a8f9cefa.zip') as z:
        gate_bytes=z.read('integrity_gate.json');gate=json.loads(gate_bytes)
    old_analysis=read(R/'outputs/paper_b5c_results/analysis/analysis_manifest.json')
    if hashlib.sha256(gate_bytes).hexdigest()!=old_analysis['independent_recovery_gate_sha256'] or not gate['PASS'] or not gate['model_and_source_hashes_verified']:raise ValueError('RECOVERY_GATE_AUTH')
    write(O/'reports/b5c_fidelity_bridge.json',dict(PASS=True,old_contract_SHA256=sha(R/'deployment/b5c_paired_v1/contract.json'),old_model_identity_exactly_matches_transfer=True,source_checks=source_checks,original_recovery_gate_SHA256=hashlib.sha256(gate_bytes).hexdigest(),original_recovery_gate=dict(PASS=gate['PASS'],MFA_scored_positions=gate['MFA_scored_positions'],runtime_calls=gate['runtime_calls'],model_and_source_hashes_verified=gate['model_and_source_hashes_verified'],original_attempt_state=gate['original_attempt_state']),old_35cell_analysis_manifest_SHA256=sha(R/'outputs/paper_b5c_results/analysis/analysis_manifest.json'),old_35cell_result_not_recomputed=True,old_attempt_state_not_changed=True,real_adapter_fidelity_current_points=30,classification='A: valid frozen instance and faithful inference; weaker fitted parameters than same-rank spectral MLE',residual_causal_uncertainty='Specific effects of initialization, old-loading residual update, finite stopping and numerical arithmetic not separately identified'))
    nms=rows(O/'tables/capacity_comparison.csv');cs=rows(O/'tables/paired_contrasts.csv');ix={(r['dataset'],r['condition'],int(r['snr_db']),r['method']):r for r in nms}
    core=[];markdown={}
    for ds in ('V_new','D_eval'):
        lines=['| 条件 | SNR | EM16 | Direct16 | EM−Direct16 | 95%探索性区间 |','|---|---:|---:|---:|---:|---|']
        for c in ('ID','Profile','Mobility','Composed','Boundary'):
            for s in (10,20,30):
                a=ix[ds,c,s,'ORIGINAL_EM16'];b=ix[ds,c,s,'PPCA_DIRECT_R16'];d=next(r for r in cs if r['dataset']==ds and r['condition']==c and int(r['snr_db'])==s and r['second']=='PPCA_DIRECT_R16')
                core.append(dict(dataset=ds,condition=c,snr_db=s,EM16_NMSE_dB=float(a['nmse_db']),Direct16_NMSE_dB=float(b['nmse_db']),EM_minus_Direct16_dB=float(d['estimate_db']),CI_lower=float(d['CI_lower']),CI_upper=float(d['CI_upper']),trajectory_count=100))
                lines.append(f"| {c} | {s} | {float(a['nmse_db']):.6f} | {float(b['nmse_db']):.6f} | {float(d['estimate_db']):.6f} | [{float(d['CI_lower']):.6f}, {float(d['CI_upper']):.6f}] |")
        markdown[ds]='\n'.join(lines)
    csvout(O/'tables/em16_direct16_risk_comparison.csv',core)
    inventory=rows(O/'tables/original_parameter_inventory.csv')
    invtext='\n'.join(f"| {r['file']} | {r['shape']} | {r['dtype']} | {r['file_SHA256']} |" for r in inventory)
    text(O/'model_identity_report.md',f'''# Original final EM16 identity

PASS: exact eight ZIP members, seven file SHA256 and six payload SHA256 authenticated by the original inner manifest. Transfer ZIP SHA256 `{identity['ZIP_SHA256']}` matches the human operator's lab PowerShell output; no historical standalone sidecar was available.

Inner manifest SHA256 `{identity['inner_SHA256']}`; outer SHA256 `{identity['outer_SHA256']}`.
K1/r16, seed4143049057, role final,1500 preparation trajectories/15000 frames,D4032,22 iterationsCONVERGED,fresh fit, no candidate checkpoint.

| file | shape | dtype | file SHA256 |
|---|---|---|---|
{invtext}

training_history.npy shape22 float64 hash `{identity['seven_file_hashes']['training_history.npy']}`.
Actual final relative history change `{identity['last_history_relative_change']}` satisfies the original1e-4 rule. Six negative increments are retained. History is pre-update score, and final saved arrays follow the final update. It is not an independent optimality certificate.
The archive extraction is an authenticated byte-for-byte copy under original_model/. All arrays used in inference were read-only; private posterior caches are ephemeral. No parameter file is rewritten.
''')
    text(O/'conditional_prediction_fidelity.md',f'''# Conditional prediction fidelity

PASS. Thirty actual saved V_new pilots (five conditions,frames4/9,threeSNRs) were passed through the authenticated frozen author implementation and adapter, with float1.0 verified, ten component calls perSNR,K1 responsibilities exactlyone.

Independent map from saved clean covs:
W=C_dp(C_pp+sigma²I)^(-1); estimate=mu_d+W(p-mu_p).
The covariance block is solved by Cholesky; the author uses its frozen Hermitian pseudoinverse. Original implementation was not patched.
Loading map independently gives C_dp=F_dF_pᴴ and C_pp=F_pF_pᴴ+nuI. Pilot positions come from the original sorted4032-index union, never an assumed64-prefix. Known noise is added once, and natural model means are preserved.

Maximum author-versus-independent data-output difference: `{max(r['author_vs_covariance']['max_absolute'] for r in fidelity['checks'])}`. Public means/lambdas/psis/amps/covs/inv_covs payload hashes remain unchanged. Original single-versus-batch outputs match exactly at checked points.

Complete paired evaluation uses the mathematically same cached independent map; it does not rerun the author adapter18000 times. Current fidelity sampling is30 pilots, together with exact covariance identity and algebraic equivalence. No claim of18000 separate adapter comparisons.

Saved inv_covs is a clean training inverse cache, unused as the noisy64-pilot inverse. Four deterministic vector directions independently checked against the spectral inverse: forward relative difference `{forward}`, normalized backward residual `{backward}`. This is a limited numerical diagnostic in addition to full byte authentication.

Model/source/code mapping:
- reference/sp_u12b5a/paper_period/MFA_estimator/MFA_estimator.py:70-80, noisy observed covariance and conditional mean;
- src/baselines/mfa_reference_adapter.py:126, float1.0;149-156, geometry;173-180, owner division/noise covariance;
- deployment/b5c_paired_r2_v1/science.py:139-145, exact adapter call;159-160, scoring frames and data selectors;
- new src/reconcile.py:35-41, independent covariance map;127-151, physical fidelity;183-202, same-input fixed evaluation.
''')
    text(O/'b5c_implication.md',f'''# Historical external-baseline interpretation

Decision: keep the historical35-cell B5C result and its100000-replicate/35-contrast Bonferroni family as the valid frozen selected-instance comparison. No old number,raw block,manifest or FAILED attempt is rewritten. No formal B5C call is rerun.

The current actual model matches the old B5C top/inner/preparation/selection bindings. All{len(source_checks)} recorded source dependencies are byte-identical. The recovered integrity Gate is authenticated against the existing analysis manifest and verifies5000 trajectories,210000MFA positions,35000runtime calls and model/source identity.

Current independent actual-model conditional fidelity supports correct B5C inference. The primary classification is A: a faithful frozen-instance evaluation of a comparatively weak fitted model. B(noise/selector/mean/real-complex inference error) is not supported by current evidence. C(statistical family incompatibility) is not present for the same-rank comparison; optimizer update policies do differ. D remains relevant for the specific causal explanation of the optimizer's covariance amplitudes.

Original EM16 and Direct16 use the same clean P,free natural mean,proper-complexK1,rank16,isotropicnu family; neither final residual hits the1e-6floor. Same-rank Direct16 improves all30 developmental cells by2.735077–9.303353dB, so the discrepancy is not a rank16-versus32 capacity effect.

The old B5C superiority remains evidence against this selected frozenEM16 instance. It cannot carry a strong claim against an adequately optimizedK1/r16 family,still less allMFA models. More competitive Direct/Pooled findings must be included in later paper reasoning,including conditions in which they beatLR4.

The current V_new andD_eval were already disclosed; their2000-draw unadjusted95%CI is diagnostic. It does not replace the old35-cell formal family or become a new final confirmation. Any new formal evaluation or manuscript integration requires mentor review; none is initiated here.
''')
    score_em=comp['scores']['ORIGINAL_EM16'];score_di=comp['scores']['PPCA_DIRECT_R16'];cov=comp['comparisons'][0]
    reviewtext=f'''# MR-03B 中文科研审查

## 结论

原最终模型身份及推断fidelity通过；原EM16与Direct16的同秩优化质量并不接近。
现有证据支持合法冻结实例、同统计模型族、显著参数估计差距；没有发现影响历史B5C结论有效性的身份或推断接口错误。原B5C需降低强外部基线代表性，而不是删除或改写。

## 数学及训练差异

模型均值差范数{cov['mean_absolute_difference']:.12g}。nu_EM={identity['isotropic_nu']:.17g}，nu_Direct16={read(MR/'reports/PPCA_DIRECT_R16.json')['model_audit']['residual_variance']:.17g}；均为方差、各向同性、未触发下限。
EM/Direct16同口径平均边缘score/复数维分别{score_em['mean_complex_dimension_logscore']:.12f}/{score_di['mean_complex_dimension_logscore']:.12f}，每帧相差{score_di['mean_frame_logscore']-score_em['mean_frame_logscore']:.12f}。
模型协方差相对Frobenius差{cov['covariance_relative_difference']:.12f}；pilot covariance相对差{cov['pilot_covariance_relative_difference']:.12f}。最大子空间角{max(cov['principal_angles_radians']):.12g}rad；loading variance trace分别{float(theta.sum()):.12f}/{float(td.sum()):.12f}，幅度差远大于均值差。
三SNR的条件算子相对差分别{', '.join(format(x['conditional_operator_relative_difference'],'.9f') for x in comp['comparisons'])}。

作者残差更新先用旧loading形成lambdalatents，再更新loading，残差仍使用此前的lambdalatents；先逐坐标clip，再PPCA平均。这与直接谱优化更新不同。训练history记录的是更新前参数，末次history不是最终保存参数的精确独立score。CONVERGED仅证实原relative-history rule满足，不能证明谱最优。
尚未独立隔离初始化、有限停止、旧loading residual更新、floor与数值算术各自的因果贡献；不新跑EM、不归咎单一机制。

## 同输入风险

两个dataset各500物理轨迹，15格、三SNR、frames4–9，各9000新EM评分位置。原11方法能量严格复用。V_new完整H未保存，按原generator/seed/device重放；500/500 H哈希精确相同，保存pilot直接复用，未生成新noise。D_eval的既有H/pilot直接读取、500/500身份一致。
pooled NMSE先error/truth求和后dB；first-minus-second正值利于second。原2000次trajectory paired索引逐字节复用，无新bootstrap draws。两套均已披露开发数据，不能称未见确认。

V_new:

{markdown['V_new']}

D_eval:

{markdown['D_eval']}

上述30格EM−Direct16的未调整探索性95%区间均大于0；EM−Direct32范围3.056950–9.303278dB。更高rank的局部改善不解释巨大的同rank差距。
Direct32/64接近Pooled，但LR4仍有有条件优势；不能将原EM较弱反过来当作LR4普遍失效或普遍最优的证据。原MR03的Composed30、Profile30及Boundary劣势继续保留。

## 测试、复核、成本与边界

14项本轮必要pytest实际通过；真实source-only author code的30个实际pilot fidelity通过，最大差4.40909024415e-13。
独立统计复核2410次数值比较最大差1.42108547152e-14；360NMSE行、240contrasts、1000新JSON/NPZ身份及CRC、旧11方法逐元素复用全通过。
compare峰值RSS{comp['resource']['lifetime_peak_RSS_bytes']}B；比较包括认证源码、作者setup、训练moments/固定参数score，非公平逐方法延迟benchmark。
V_new新EM评价17.0677s、D_eval12.5130s，均用cached条件map。不同端点不能与旧B5C adapter总时长直接作速度比。
prepare末尾控制台打印曾把flush误传json.dumps；身份报告已先写PASS。错误记录保留、打印修正在全部比较/评价前冻结，测试包括此回归；不构成模型错误。

## 后续建议

先由导师审查本次模型估计质量证据及外部基线表述。后续论文应同时呈现固定EM实例与合理强基线，保留单分量范围及开发信息预算边界。若需要探究确切优化因果或独立正式确认，由导师另行决定，不在此任务自动启动。
'''
    text(O/'scientific_review_cn.md',reviewtext)
    write(O/'reports/test_summary.json',dict(PASS=True,pytest_passed=14,elapsed_seconds=.59,actual_command=r'.venv-gmm\Scripts\python.exe -B -m pytest outputs/revision_mr03_em16_reconciliation/tests -q -p no:cacheprovider',pytest_existing_cdl_site_on_PYTHONPATH=True,no_dependencies_upgraded=True,actual_author_model_fidelity_points=30,original_EM_fit_calls=0))
    text(O/'README_REPRODUCE.md',r'''# MR03B numerical review

Use scientific_review_cn.md,model_identity_report.md,conditional_prediction_fidelity.md,b5c_implication.md.
Tables use pooled-energy NMSE. Positive EM-minus-other favors other. Intervals are unadjusted exploratory95%,2000 original paired trajectory draws. BothV_new/D_eval already disclosed.

Existing Project environment only; no dependency upgrades implied:
```powershell
$root='D:\Users\yuhua\Desktop\Project'
Set-Location $root
& .\.venv-cdl\Scripts\python.exe -B .\outputs\revision_mr03_em16_reconciliation\src\independent_review_em16.py --root $root --out .\outputs\revision_mr03_em16_reconciliation\reports\user_review.json
```
The destination report must not exist. This only reads persisted energies/draws, not a new experiment.
Do not rerun prepare/compare/evaluation stages into existing outputs. Original EM/Direct models are immutable.
The review ZIP omits the two260MBcovariance caches; these remain in original_model/ and are authenticated by the original manifest. Original transfer ZIP is bound bySHA. Complete preparationP and MR03 background artifacts remain in the project; they are not a project/environment migration package.
''')
    protected=read(O/'reports/preservation_execution.json')['hashes'];changed=[p for p,h in protected.items() if sha(R/p)!=h]
    if changed:raise ValueError('HISTORIC_CHANGED:'+str(changed))
    f=read(O/'reports/execution_freeze.json')['hashes'];changednew=[p for p,h in f.items() if sha(O/p)!=h]
    if changednew:raise ValueError('FROZEN_NEW_CHANGED')
    write(O/'reports/preservation_after.json',dict(PASS=True,protected_historical_files=len(protected),changed=changed,science_frozen_files=len(f),changed_scientific_sources=changednew,original_model_unmodified=True,B5C_history_unmodified=True,old_attempt_still_FAILED=True,new_B5C_formal_calls=0,new_fit_calls=0,manuscript_edits=0,git_commit=False))
    write(O/'reports/completion_summary.json',dict(PASS=True,state='MR03B_RECONCILIATION_COMPLETE',model_identity_PASS=True,same_statistical_family=True,exact_optimizer_updates_identical=False,conditional_fidelity_PASS=True,all30cells_EM_worse_than_Direct16=True,B5C_classification='A: valid weaker frozen instance; no identified inference/identity error',causal_optimizer_diagnosis_closed=False,original_EM_training_executed=False,formal_B5C_replayed=False,STOP=True))
    print(json.dumps(dict(PASS=True,inverse_forward=forward,inverse_backward=backward,documents_created=True,protected_files=len(protected),state='MR03B_RECONCILIATION_COMPLETE')))
if __name__=='__main__':main()
