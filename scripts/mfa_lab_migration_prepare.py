"""Explicit allowlist migration packaging; captures working bytes, not git HEAD."""
from pathlib import Path
import sys,os,json,csv,hashlib,shutil,subprocess,ast,zipfile,time,re,struct
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.mfa_lab_migration_common import OUT,DEPLOY,read,write,sha
STAGE=OUT/'package'/'Project_MFA_Lab'

GUIDE=r'''# Windows MFA 实验室迁移操作指南

本包只支持完整性、环境/主机、Fidelity、日志/监测自检，以及预备 synthetic smoke。
没有正式 candidate training、validation、final retraining 或 target evaluation 启动选项。
原作者/reference、adapter、R3 优化和 R4 Fidelity 数学文件保持原字节。
个人 venv、.git、密钥、论文及 B2/B3 封存结果不在本包。

## 1. 传输与解压

把 Project_MFA_Lab_v1.zip 与 Project_MFA_Lab_v1.zip.sha256 一起复制到实验室。
以下假定复制到 D:\Transfer，可替换为实验室实际目录。先打开 PowerShell：

```powershell
Set-Location 'D:\Transfer'
$expected = ((Get-Content -LiteralPath '.\Project_MFA_Lab_v1.zip.sha256' -Raw).Trim() -split '\s+')[0]
$actual = (Get-FileHash -LiteralPath '.\Project_MFA_Lab_v1.zip' -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $expected) { throw 'ZIP SHA256 mismatch: STOP' }
New-Item -ItemType Directory -Path 'D:\Research' -Force | Out-Null
Expand-Archive -LiteralPath '.\Project_MFA_Lab_v1.zip' -DestinationPath 'D:\Research'
Set-Location 'D:\Research\Project_MFA_Lab'
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\deployment\mfa_lab_v1\Verify-Package.ps1
```

目标目录必须是新的空目录；不要与旧工程混合。外部 ZIP hash 是传输校验锚点，
包内 manifest 不包含自己的 hash，避免自引用。所有可执行路径从实际项目根推导。
无需安装 Git；不得初始化/提交此迁移目录来冒充原 Git 历史。

## 2. 安装独立 CPU 环境

```powershell
.\deployment\mfa_lab_v1\Install-Environment.ps1
```

脚本使用包内官方 Python3.13.15 x64 安装器，在迁移后根目录下安装
.python-runtime，并新建 .venv-lab；随后使用包内14个 wheels 和 hash lock
离线安装。它不修改系统 PATH，不复制个人虚拟环境，不安装 CUDA/PyTorch。
Python 安装器可能受实验室 Windows 策略约束；若失败，保存退出码并 STOP。
不要自行替换版本后继续声称 Fidelity 已通过。

## 3. 日志、PID 与资源监测自检

```powershell
.\deployment\mfa_lab_v1\Start-LabJob.ps1 -Mode self-test -Threads 1
.\deployment\mfa_lab_v1\Get-LabStatus.ps1
```

启动器隐藏后台运行，关闭远程控制会话无需个人电脑继续在线。不要让实验室电脑
睡眠或重启。只在 status.state=COMPLETE 且 PASS=true 后进入下一项。

## 4. 检查实际 CPU、内存、环境与 BLAS

```powershell
.\deployment\mfa_lab_v1\Start-LabJob.ps1 -Mode host -Threads 1
.\deployment\mfa_lab_v1\Get-LabStatus.ps1
```

报告在 lab_runs 中：host_inventory.json、environment.json、resources.jsonl。
用户报告的硬件为128GB RAM、i9-10900KF、RTX3090，尚未由本轮实测确认。
使用 CPU；RTX3090 不参与本任务。线程候选以实际 physical cores 为限；
不照搬个人 i7 的12-thread建议。正式配置仍是1-thread。

## 5. Fidelity 检查（先单线程）

```powershell
.\deployment\mfa_lab_v1\Start-LabJob.ps1 -Mode fidelity -Threads 1
.\deployment\mfa_lab_v1\Get-LabStatus.ps1
```

该入口运行原 B5A unit tests 和被冻结 R4 九结构短轨迹、代表性完整 convergence、
中规模对照、serialization 与 float1.0 author inference。所有数据均 synthetic。
原始容差 atol1e-10、rtol1e-9 不可放宽；历史长度不截断，不重新排列 components。
失败时查看 exception.json/worker.log，STOP；禁止训练或选择幸存候选。
通过后若导师需要，可单独指定4/8/10 threads重复 Fidelity，但不是本轮默认步骤。

## 6. 预备 R5 synthetic smoke（导师审查本迁移包后才能启动）

本轮未执行该项；它也不是未提供任务书的正式 R5 资源资格证书。
预备形状为D4032/N128/K4/r16，reference/low分别隔离执行一次synthetic EM，
随后比较参数。不是N12000/15000训练，不访问preparation/validation数值。

```powershell
.\deployment\mfa_lab_v1\Start-LabJob.ps1 -Mode smoke -Threads 1
.\deployment\mfa_lab_v1\Get-LabStatus.ps1
```

资源配置继续为RSS6GiB / physical headroom2GiB；在5.5GiB或2.25GiB附近预先STOP。
128GB硬件不能自动解除原门槛。若被停止，不自动提高门槛、不重复挑选成功运行。

## 7. 状态与错误处理

```powershell
.\deployment\mfa_lab_v1\Get-LabStatus.ps1
Get-ChildItem -LiteralPath '.\lab_runs' -Directory
```

每个job保存status.json、worker.log、resources.jsonl、environment.json，
异常保存exception.json或supervisor_exception.json。Start-Process使用Hidden，
monitor使用worker自报actual PID而非venv launcher PID。
STARTING/RUNNING不是PASS；STOPPED意味着必须检查。不要删除失败日志。

## 8. 数据与历史记录边界

包内selection_train.npy=(12000,4032)、full_retraining.npy=(15000,4032)，
并带prepared-origin validation输入及identity/CRN metadata，供未来授权阶段使用。
本部署入口禁止读取这些NPY数值，仅允许完整性hash；selection仍NONE。
Validation count erratum是单candidate3600、九candidate32400；210000 formal
predictions规划不改，实际target数据不随包迁移。

旧B5A/B5B/R3/R4完整manifests作为原始证据保留，其中引用未搬迁的历史logs/outputs。
不要在实验室执行旧check_frozen/seals脚本要求它们全部存在；当前包验证使用
migration_manifest.json中完整payload子集，每个冻结源文件核对原始hash。
科学配置/旧审计记录中原电脑的路径字符串仅为历史元数据，不是部署运行路径。

原正式B5B runner保存为冻结科学入口，但依赖原电脑历史治理/封存目录，且本任务
禁止正式执行。以后正式训练需要导师授权的部署适配/资源协议；本包不偷改旧runner。
完成以上获准检查后 STOP，返回完整logs给导师审查，不自行运行9-candidate grid。
'''

def selected():
    names={};add=lambda p,c:names.update({Path(p).as_posix():c})
    for p in(ROOT/'src').rglob('*.py'):
        if '__pycache__'not in p.parts:add(p.relative_to(ROOT),'PROJECT_SOURCE_ARCHIVE'if p.name!='mfa_reference_adapter.py'else'FROZEN_ADAPTER')
    for p in(ROOT/'reference/sp_u12b5a/paper_period').rglob('*'):
        if p.is_file()and not any((x.startswith('.')and x not in('.gitignore','.gitmodules'))or x=='__pycache__'for x in p.relative_to(ROOT/'reference').parts):add(p.relative_to(ROOT),'FROZEN_AUTHOR_REFERENCE')
    for directory in ('experiments/mfa_low_memory_prototype/portable','experiments/mfa_low_memory_fidelity_r4'):
        for p in(ROOT/directory).glob('*.py'):add(p.relative_to(ROOT),'FROZEN_R3_OPTIMIZATION'if'prototype'in directory else'FROZEN_FIDELITY_HARNESS')
    for p in(ROOT/'experiments/mfa_low_memory_prototype/portable').rglob('*'):
        if p.is_file()and'__pycache__'not in p.parts:add(p.relative_to(ROOT),'FROZEN_R3_PORTABLE_SUPPORT')
    # Current working bytes, including essential untracked research files.
    for pattern in ('sp_u12b5a*.py','sp_u12b5b*.py','run_sp_u12b5a*.py','run_sp_u12b5b*.py'):
        for p in(ROOT/'scripts').glob(pattern):add(p.relative_to(ROOT),'FROZEN_RESEARCH_ENTRY_ARCHIVE')
    for n in ('scripts/sp_u12b4a_preserve.py','scripts/validate_mfa_fidelity.py','tests/test_mfa_reference.py',
        'tests/test_sp_u12b5b_r4_fidelity.py','tests/test_sp_u12b5b_r4_canonical_contract.py','tests/test_sp_u12b5b_r4_delivery.py',
        'configs/sp_u12b5b_mfa_selection.yaml','configs/sp_u12b5b_r4_synthetic.json','configs/sp_u3a_protocol.json','configs/sp_u4a_protocol.json',
        'AGENTS.md','pyproject.toml'):
        add(n,'FROZEN_FIDELITY_TEST'if n.startswith('tests/')else'FROZEN_SCIENTIFIC_CONFIG'if n.startswith('configs/')else'PROJECT_CONTEXT_ARCHIVE')
    metadata=('outputs/sp_u12b5a/historical_reference.json','outputs/sp_u12b5a_mfa_fidelity_manifest.json',
        'outputs/sp_u12b5b_candidate_seed_registry.json','outputs/sp_u12b5b_preparation_split_manifest.json',
        'outputs/sp_u12b5b/input_manifest.json','outputs/sp_u12b5b/validation_noise_registry.json','outputs/sp_u12b5b/preparation_channel_proofs.json',
        'outputs/sp_u12b5b_r3_manifest.json','outputs/sp_u12b5b_r4_fidelity_manifest.json',
        'outputs/sp_u12b5b_r4_validation_cardinality_erratum.json','outputs/sp_u12b5b_r4_validation_contract_audit.json',
        'outputs/sp_u12b5b_r4_complex_conjugation_audit.json','outputs/sp_u12b5b_r4_update_order_audit.json',
        'outputs/sp_u12b5b_r4_nine_structure_parity.json','outputs/sp_u12b5b_r4_stopping_parity.json',
        'outputs/sp_u12b5b_r4_medium_scale_parity.json','outputs/sp_u12b5b_r4_multithread_parity.json')
    for n in metadata:add(n,'FROZEN_PROVENANCE_METADATA')
    for n in ('selection_train.npy','full_retraining.npy','validation_truth.npy','validation_pilots.npy'):add('outputs/sp_u12b5b/'+n,'PREPARATION_DATA_NOT_AUTHORIZED_FOR_EXECUTION')
    for n in ('docs/sp_u12b5b_selection_protocol.md','docs/sp_u12b5b_r4_mathematical_fidelity_report.md',
        'docs/sp_u12b5b_r4_validation_erratum.md','docs/sp_u12b5b_r4_next_stage_requirements.md'):add(n,'RESEARCH_PROTOCOL_ARCHIVE')
    # Static dependency closure for selected src/scripts Python, without imports/execution.
    changed=True
    while changed:
        changed=False
        for n in list(names):
            if not n.endswith('.py'):continue
            for node in ast.walk(ast.parse((ROOT/n).read_text(encoding='utf8'))):
                modules=[a.name for a in node.names]if isinstance(node,ast.Import)else[node.module]if isinstance(node,ast.ImportFrom)and node.module else[]
                for module in modules:
                    if module.startswith(('src.','scripts.','experiments.')):
                        p=ROOT/(module.replace('.','/')+'.py')
                        if p.is_file()and p.relative_to(ROOT).as_posix()not in names:add(p.relative_to(ROOT),'INTERNAL_IMPORT_DEPENDENCY_ARCHIVE');changed=True
    return names

def npy_header(path):
    with path.open('rb')as f:
        assert f.read(6)==b'\x93NUMPY';version=f.read(2);length=int.from_bytes(f.read(2 if version[0]==1 else 4),'little')
        return ast.literal_eval(f.read(length).decode('latin1').strip())

def prepare():
    assert read(OUT/'preservation_before.json');assert read(DEPLOY/'environment_source.json')['wheels']
    assert not STAGE.exists(),'FRESH_PACKAGE_STAGE_REQUIRED';STAGE.mkdir(parents=True)
    config={'purpose':'DEPLOYMENT_PREPARATION_ONLY','user_reported_target':{'OS':'Windows','RAM_GB':128,'CPU':'Intel Core i9-10900KF','GPU':'RTX3090 unused'},
        'target_hardware_verified':False,'resource_limits':{'max_process_bytes':6*2**30,'min_system_available_physical_bytes':2*2**30},
        'near_limit_stop':{'RSS_bytes':int(5.5*2**30),'available_bytes':int(2.25*2**30)},'timeout_seconds':3600,
        'default_threads':1,'formal_threads_changed':False,'formal_training_authorized':False,
        'proposed_R5_smoke':{'D':4032,'N':128,'K':4,'r':16,'steps':1,'data_seed':42494000,'initialization_seed':42494001,
            'status':'PREPARATORY_CANDIDATE_NOT_EXECUTED_OR_FORMALLY_CERTIFIED'},
        'prepared_validation_predictions_per_candidate':3600,'prepared_validation_predictions_nine_candidates':32400,
        'formal_target_predictions_plan':210000,'GPU_dependency':False}
    write(DEPLOY/'deployment_config.json',config);(DEPLOY/'WINDOWS_GUIDE.md').write_text(GUIDE,encoding='utf8')
    names=selected()
    for p in DEPLOY.rglob('*'):
        if p.is_file()and'__pycache__'not in p.parts:names[p.relative_to(ROOT).as_posix()]='DEPLOYMENT_ADDITION'if p.suffix not in('.whl','.exe')else'OFFICIAL_OFFLINE_INSTALLER_OR_WHEEL'
    status=subprocess.check_output(['git','-c','core.quotepath=false','status','--porcelain=v1','-uall'],cwd=ROOT,text=True,encoding='utf8')
    tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode('utf8').split('\0'))
    statuses={line[3:]:line[:2]for line in status.splitlines()};rows=[];payload={};path_notes=[]
    secret_pattern=re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\b(?:sk|ghp|github_pat)-[A-Za-z0-9_]{20,}|https?://[^\s/@]+:[^\s/@]+@')
    inputs=read(ROOT/'outputs/sp_u12b5b/input_manifest.json')
    for n,c in sorted(names.items()):
        p=ROOT/n;assert p.is_file(),n;assert not p.is_symlink(),n
        assert not any(x in n for x in ('sp_u12b2b_sealed/','sp_u12b3g_next_sealed/','.venv','__pycache__','/.git/'))
        if p.suffix in('.py','.json','.yaml','.md','.toml','.ps1','.lock'):
            text=p.read_text(encoding='utf-8-sig');assert not secret_pattern.search(text),'POSSIBLE_SECRET_STOP:'+n
            if re.search(r'[A-Za-z]:[\\/]|Users[\\/]+yuhua',text):path_notes.append({'path':n,'classification':'ARCHIVED_PATH_STRINGS_NOT_DEPLOYMENT_EXECUTION'if not n.startswith('deployment/')else'DOCUMENTED_EXAMPLE_TARGET_PATH'})
        target=STAGE/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        h=sha(target);assert h==sha(p),'COPY_BYTE_DRIFT:'+n
        if p.suffix=='.npy'and p.parent==ROOT/'outputs/sp_u12b5b':assert h==inputs['files'][p.name],'PREPARATION_INPUT_DRIFT'
        git='UNTRACKED'if n not in tracked else'MODIFIED'if n in statuses else'TRACKED_CLEAN'
        row={'path':n,'category':c,'bytes':p.stat().st_size,'sha256':h,'git_state':git};rows.append(row);payload[n]={k:v for k,v in row.items()if k!='path'}
    metadata={'selection_train.npy':{'shape':[12000,4032],'descr':'<c16'},'full_retraining.npy':{'shape':[15000,4032],'descr':'<c16'},
        'validation_truth.npy':{'shape':[3,100,6,3968],'descr':'<c16'},'validation_pilots.npy':{'shape':[3,2,100,6,64],'descr':'<c16'}}
    for n,expected in metadata.items():
        header=npy_header(ROOT/'outputs/sp_u12b5b'/n);assert list(header['shape'])==expected['shape']and header['descr']==expected['descr']and header['fortran_order']is False
        expected['file_sha256']=inputs['files'][n]
    write(OUT/'file_inventory.json',rows)
    with(OUT/'necessary_files.csv').open('w',encoding='utf-8-sig',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    shutil.copyfile(OUT/'necessary_files.csv',STAGE/'necessary_files.csv')
    h=sha(STAGE/'necessary_files.csv');payload['necessary_files.csv']={'category':'GENERATED_PAYLOAD_INVENTORY','bytes':(STAGE/'necessary_files.csv').stat().st_size,'sha256':h,'git_state':'NEW_DEPLOYMENT_RECORD'}
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    audit={'Git_HEAD':head,'tracked_files_total':len(tracked)-(''in tracked),'status_entries_total':len(statuses),
        'source_payload_files':len(rows),'payload_untracked':sum(r['git_state']=='UNTRACKED'for r in rows),'payload_modified':sum(r['git_state']=='MODIFIED'for r in rows),
        'untracked_and_modified_research_files_included':[r['path']for r in rows if r['git_state']!='TRACKED_CLEAN'and r['category']!='OFFICIAL_OFFLINE_INSTALLER_OR_WHEEL'],
        'directory_inventory':[{ 'name':p.name,'type':'directory'if p.is_dir()else'file','included_policy':'ALLOWLIST_ONLY'}for p in ROOT.iterdir()],
        'excluded':['.git and all account Git config','all personal .venv* and caches','B2/B3 scientific outcomes and target inputs','paper/ and PDFs','unrelated historical scientific outputs','old candidate failure intermediates/models'],
        'preparation_headers_only_no_values':metadata,'formal_entry_status':'archived frozen runner depends on excluded governance history; not a supported migration launch command',
        'portable_supported_entries':['verify','host','fidelity','self-test','smoke prepared only','status'],'absolute_path_audit':path_notes,
        'deployment_executable_absolute_old_paths':0,'GPU_path_added':False,'R5_running':False,'R5_executed_during_migration':False,
        'Python_process_check':'only the migration inspection interpreter and its venv launcher were present',
        'preservation_before_files':len(read(OUT/'preservation_before.json')['hashes'])}
    write(OUT/'engineering_audit.json',audit)
    # Sanitized snapshot provenance, not the unrelated full working tree or account .git.
    write(STAGE/'migration_provenance.json',{'origin_Git_HEAD':head,'snapshot':'CURRENT_WORKING_BYTES_INCLUDING_UNCOMMITTED_UNTRACKED_FILES',
        'not_a_clean_Git_checkout':True,'formal_training_authorized':False,'excluded_B2_B3_outcomes':True,
        'data_contract':metadata,'original_manifest_limit':'original full manifests reference historical files deliberately not transferred; this bundle has its own complete subset manifest',
        'MFA_model_semantics_changed':False,'source_file_count':len(rows)})
    payload['migration_provenance.json']={'category':'GENERATED_PROVENANCE','bytes':(STAGE/'migration_provenance.json').stat().st_size,'sha256':sha(STAGE/'migration_provenance.json'),'git_state':'NEW_DEPLOYMENT_RECORD'}
    write(STAGE/'migration_manifest.json',{'format':'MFA_WINDOWS_MIGRATION_V1','files':payload,'self_reference_excluded':['migration_manifest.json','SHA256SUMS.txt'],
        'hash_anchor':'external ZIP SHA256 sidecar','original_Git_HEAD':head,'formal_training_authorized':False,'CPU_only':True})
    (STAGE/'SHA256SUMS.txt').write_text(''.join(f"{v['sha256']}  {n}\n"for n,v in sorted(payload.items()))+
        sha(STAGE/'migration_manifest.json')+'  migration_manifest.json\n',encoding='utf8')
    write(OUT/'stage_ready.json',{'PASS':True,'root':str(STAGE),'payload_files':len(payload),'archive_expected_files':len(payload)+2,
        'payload_bytes':sum(v['bytes']for v in payload.values()),'manifest_sha256':sha(STAGE/'migration_manifest.json'),'scientific_value_reads':0})
    print(json.dumps({'stage':'PACKAGE_STAGED','files':len(payload)+2,'payload_GiB':sum(v['bytes']for v in payload.values())/2**30,'untracked_included':audit['payload_untracked']}))

def pack():
    assert read(OUT/'stage_ready.json')['PASS'];assert read(OUT/'relocation_validation.json')['PASS']
    assert read(OUT/'powershell_launch_validation.json')['PASS']
    manifest=read(STAGE/'migration_manifest.json')
    assert all(sha(STAGE/n)==v['sha256']for n,v in manifest['files'].items())
    archive=OUT/'Project_MFA_Lab_v1.zip';assert not archive.exists()
    files=sorted(p for p in STAGE.rglob('*')if p.is_file());assert len(files)==read(OUT/'stage_ready.json')['archive_expected_files']
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED,allowZip64=True)as z:
        for p in files:z.write(p,'Project_MFA_Lab/'+p.relative_to(STAGE).as_posix())
    digest=sha(archive);(OUT/'Project_MFA_Lab_v1.zip.sha256').write_text(digest+'  '+archive.name+'\n',encoding='ascii')
    with zipfile.ZipFile(archive)as z:
        assert len(z.infolist())==len(files)
        assert not any('sp_u12b2b_sealed/'in n or'sp_u12b3g_next_sealed/'in n or '/.venv'in n or '/.git/'in n for n in z.namelist())
        # Stream each archived byte for SHA256; CRC alone is insufficient.
        for n,v in manifest['files'].items():
            h=hashlib.sha256()
            with z.open('Project_MFA_Lab/'+n)as f:
                for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
            assert h.hexdigest()==v['sha256'],'ZIP_BYTE_DRIFT:'+n
        for name in ('migration_manifest.json','SHA256SUMS.txt'):
            h=hashlib.sha256(z.read('Project_MFA_Lab/'+name)).hexdigest()
            assert h==sha(STAGE/name),'ZIP_CONTROL_FILE_DRIFT:'+name
    result={'PASS':True,'archive':str(archive),'bytes':archive.stat().st_size,'GiB':archive.stat().st_size/2**30,
        'files':len(files),'sha256':digest,'manifest_payload_files':len(manifest['files']),
        'zip_stream_SHA256_all_payload_PASS':True,'zip_control_file_SHA256_PASS':True,
        'B2_B3_outcome_files':0,'personal_venv_files':0,'account_keys':0,
        'CPU_only':True,'offline_installer_and_wheels_included':True,'formal_execution':0}
    write(OUT/'package_report.json',result);print(json.dumps(result))

if __name__=='__main__':prepare()if sys.argv[1]=='stage'else pack()
