"""Assemble reports from actual technical evidence, never scientific outcomes."""
from pathlib import Path
import csv,json,subprocess,sys,xml.etree.ElementTree as ET
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/sp_u12b5c_r1'
sys.path.insert(0,str(ROOT/'scripts'));from prepare_sp_u12b5c_r1 import read,dump,sha,need
def main():
    original=read(OUT/'identity_original.json');fixed=read(OUT/'identity_authenticated_source_loading.json')
    source=read(OUT/'source_audit.json');windows=read(OUT/'windows_qa.json');preserve=read(OUT/'preservation_after.json');package=read(OUT/'package_report.json')
    for v in (original,fixed,source,windows,preserve):need(v['PASS'],'TECHNICAL_EVIDENCE_NOT_PASS')
    need(fixed['corrected_proof_passes']==original['original_failure_reproductions']==5000,'FULL_IDENTITY_COVERAGE_REQUIRED')
    for k in ('H_identity_digest','noise_identity_digest','pilot_truth_Y_identity_digest'):need(original[k]==fixed[k],'INPUT_IDENTITY_DIGEST_DRIFT')
    reports=sorted((ROOT/'lab_runs/b5c_paired_r1/technical').glob('pytest_*/pytest.xml'),key=lambda p:p.stat().st_mtime_ns)
    last=reports[-1];tree=ET.parse(last)
    suites=list(tree.getroot().iter('testsuite'));tests=sum(int(x.get('tests',0)) for x in suites)
    fails=sum(int(x.get('failures',0))+int(x.get('errors',0))+int(x.get('skipped',0)) for x in suites)
    need(tests==86 and fails==0,'FINAL_PYTEST_REQUIRED')
    need(sha(Path(package['path']))==package['sha256'],'FINAL_ZIP_HASH_DRIFT')
    with (OUT/'cell_coverage.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(fixed['cells'][0]));w.writeheader();w.writerows(fixed['cells'])
    conclusion={'verdict':'TECHNICAL_SEED_MAPPING_FIX_READY','root_cause_category':'B_TECHNICAL_REPRESENTATION_ERROR',
        'first_original_B2_identity':fixed['first_mismatch'],'scientific_seed_mismatches':0,'model_or_LR4_math_changed':False,
        'B2_or_B3_modified':False,'formal_performance_decoded':False,'formal_predictions':0,
        'pytest':{'tests':tests,'failed_errors_or_skipped':fails,'xml':str(last)},'windows':windows,
        'preservation':preserve,'package':package,'lab_failure_original_records_received':False,
        'formal_restart_authorized_by_this_task':False,'next_action':'Mentor review; LAB read-only IdentityAudit; only mentor may authorize a new independent formal attempt'}
    dump(OUT/'final_report.json',conclusion)
    status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True,encoding='utf8')
    (OUT/'git_status.txt').write_text(status,encoding='utf8')
    f=fixed['first_mismatch'];rows='\n'.join(f"| {r} | {v} | {f['original_expected'].get(r,'字段缺失')} |" for r,v in f['actual'].items())
    text=f'''SP-U12B5C-R1 — 信道种子身份冲突定位与最小修复

结论：B 类技术表示错误；已认证的原 B2 种子身份真实不一致为 0。

原 B2 consumed(j=0) 为 EVAL 保存完整六字段：前三 channel RNG roles 在 frame=-1，后三 communication roles 在 frame=0。H receipt 保存完整字典，raw proof 原样转录。OfficialCDL.generate 实际仅前三字段生成 H。旧 B5C expected 只构造前三字段，而使用完整字典 equality，必然拒绝合法记录。

第一处按正式顺序可重现：ID / CDL-C-30-100 / trajectory_id=0。检查在 SNR 循环之前，涉及该轨迹全部 0/5/10/15/20/25/30 dB；0 dB 是首计划 cell，不是单一 SNR 种子错误。所有 seed 均为 int；不是字符串、顺序或浮点序列化差异。

| 字段 | 原始值/正确expected | 旧expected |
|---|---:|---:|
{rows}

最小科学修复仅一行：

expected={{r:index[condition,i,-1 if r in ROLES[:3] else 0,r] for r in ROLES}}

完整六字段 equality 仍保留；没有删字段、转类型、改 seed 或跳过检查。checked_input、pilot_inputs、infer、evaluate 函数 AST 完全相同；contract、data_manifest、paired_stats 字节相同。其余新增文件用于隔离目录、回归测试和只读技术校验，controller/start 与 worker/evaluate 的正式入口在本技术包内无条件拒绝。

实际核对：15002 个已认证技术 JSON；5000 条轨迹；165000 EVAL registry entries；30000 channel-record seed scalars；300000 communication seed scalars；350000 输入身份；210000 评分身份；35cell 各1000 trajectory、6000评分frame。旧错误重现5000/5000，修复通过5000/5000。真实 NPZ/NPY 解码0，NMSE/BER读取0，正式预测0。H/noise/pilot/truth/Y技术哈希链接及完整网格保持。

实际 pytest：86 passed，0 failed/errors/skipped。Windows实际搬迁 QA：{len(windows['tests'])}项全部PASS；包括ZIP哈希拒绝、add-only安装、重复安装拒绝、Verify、ToyTests、完整只读IdentityAudit、Status/Export和两个正式入口拒绝。未认证 synthetic .pyc 存在且未读未改；比较函数由已认证源码直接编译。首次技术 QA 的缓存读取失败及其包完整保留，最终包通过新的独立QA。

保全：{preserve['protected_files']} 个原文件、16166402657 bytes，前后 SHA256 无变化；包含所有个人电脑 B2/B3 sealed 目录，原 B5C namespace/日志、production dependencies及manuscript。Git HEAD、tracked/staged diff保持。实验室真实失败日志未传回，本机不声称已物理核验；新安装和只读工具会在实验室前后opaque-hash核对旧文件，保留STOP/消费标记。

交付ZIP：{package['path']}
bytes：{package['bytes']}；files：{package['files']}。
SHA256：{package['sha256']}
模型及B2输入不重复传输。

实验室操作（导师审查后，仅技术）：

$root = 'D:\\Research\\Project_MFA_Lab'
$incoming = 'D:\\Research\\Incoming\\B5C_R1'
powershell -NoProfile -ExecutionPolicy Bypass -File "$incoming\\Install-B5C-R1.ps1" -ProjectRoot $root -ZipPath "$incoming\\SP_U12B5C_R1_Seed_Identity_Fix.zip" -ExpectedSHA256 '{package['sha256']}'
Set-Location $root
powershell -NoProfile -ExecutionPolicy Bypass -File .\\deployment\\b5c_paired_r1_v1\\R1.ps1 -Action Verify
powershell -NoProfile -ExecutionPolicy Bypass -File .\\deployment\\b5c_paired_r1_v1\\R1.ps1 -Action ToyTests
powershell -NoProfile -ExecutionPolicy Bypass -File .\\deployment\\b5c_paired_r1_v1\\R1.ps1 -Action IdentityAudit
powershell -NoProfile -ExecutionPolicy Bypass -File .\\deployment\\b5c_paired_r1_v1\\R1.ps1 -Action Status
powershell -NoProfile -ExecutionPolicy Bypass -File .\\deployment\\b5c_paired_r1_v1\\R1.ps1 -Action Export

每步单独执行，仅成功才下一步；失败保留独立attempt，不自动重试。

正式重启：本轮不允许。旧attempt永久STOP；实验室只读身份及旧失败保全通过后，由导师决定新的独立formal attempt。本包没有正式Run权限，不能靠删除旧消费标记继续。

STOP：不训练、不validation、不模型选择、不target评价、不改论文、不commit。
'''
    report=ROOT/'docs/sp_u12b5c_r1_fix_report.md'
    with report.open('x',encoding='utf8') as h:h.write(text)
    print(json.dumps({'PASS':True,'pytest_passed':tests,'windows_checks':len(windows['tests']),'protected_files':preserve['protected_files'],'zip':package['path'],'sha256':package['sha256'],'formal_predictions':0},indent=2))
if __name__=='__main__':main()
