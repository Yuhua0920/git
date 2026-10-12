param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('Snapshot','Repair','Probe','Minimal','Full','Math')]
    [string]$Mode
)
$ErrorActionPreference = 'Stop'
$taskProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$taskCacheRoot = Join-Path $taskProjectRoot '.pytest_cache'
$taskTempRoot = Join-Path $taskCacheRoot 'pytest_workspace_temp'
$taskAuditRoot = Join-Path $taskProjectRoot 'outputs/t17a_permission_recovery'
$taskIdentity = 'ROG\yuhua'
Set-Location -LiteralPath $taskProjectRoot
if ((whoami).Trim() -ine $taskIdentity) { throw 'Execution identity differs from requested account' }

function Save-TaskRecord($Name, $Value) {
    New-Item -ItemType Directory -Force -Path $taskAuditRoot | Out-Null
    $taskRecordPath = Join-Path $taskAuditRoot $Name
    if (Test-Path -LiteralPath $taskRecordPath) { throw "Existing audit record protected: $taskRecordPath" }
    $Value | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $taskRecordPath -Encoding UTF8
}
function Read-TaskAcl($Path) {
    $taskAcl = Get-Acl -LiteralPath $Path
    return [ordered]@{
        path=$Path; owner=$taskAcl.Owner; inheritance_protected=$taskAcl.AreAccessRulesProtected
        sddl=$taskAcl.Sddl
        entries=@($taskAcl.Access | ForEach-Object {
            [ordered]@{identity=$_.IdentityReference.Value;rights=$_.FileSystemRights.ToString()
                inheritance=$_.InheritanceFlags.ToString();propagation=$_.PropagationFlags.ToString()
                inherited=$_.IsInherited;type=$_.AccessControlType.ToString()}
        })
    }
}
function Read-TaskCacheAcls {
    $taskPaths = @($taskCacheRoot) + @(Get-ChildItem -LiteralPath $taskCacheRoot -Force -Recurse |
        ForEach-Object {$_.FullName})
    return @($taskPaths | ForEach-Object {Read-TaskAcl $_})
}
function Read-TaskContent($RootPath) {
    $taskMap = [ordered]@{}
    foreach ($taskFile in (Get-ChildItem -LiteralPath $RootPath -Force -Recurse -File | Sort-Object FullName)) {
        $taskRelative = $taskFile.FullName.Substring($taskProjectRoot.Length+1).Replace('\','/')
        $taskMap[$taskRelative] = (Get-FileHash -LiteralPath $taskFile.FullName -Algorithm SHA256).Hash.ToLower()
    }
    return $taskMap
}
if ($Mode -eq 'Snapshot') {
    $taskBefore = [ordered]@{
        account=(whoami).Trim();git_head=(git rev-parse HEAD).Trim()
        cache_acl=Read-TaskCacheAcls;native_temp_acl=Read-TaskAcl $env:TEMP
        intended_workspace_temp_root=$taskTempRoot
        cache_sha256=Read-TaskContent $taskCacheRoot
        outputs_sha256=Read-TaskContent (Join-Path $taskProjectRoot 'outputs')
        preparation_sha256=[ordered]@{}
    }
    $taskPreparationPaths = @(git ls-files --others --exclude-standard | Where-Object {
        $_ -match '^(docs/t17a_|scripts/(run_wiener_mechanism|check_wiener_preflight|repair_pytest_workspace_permissions)|src/wiener_perturbation|tests/test_wiener_perturbation)'
    })
    foreach ($taskPath in $taskPreparationPaths) {
        $taskBefore.preparation_sha256[$taskPath]=(Get-FileHash -LiteralPath (Join-Path $taskProjectRoot $taskPath)).Hash.ToLower()
    }
    $taskBefore.cache_file_count=$taskBefore.cache_sha256.Count
    $taskBefore.outputs_file_count=$taskBefore.outputs_sha256.Count
    Save-TaskRecord 'before.json' $taskBefore
    Write-Output "Before: $($taskBefore.cache_file_count) cache files; $($taskBefore.outputs_file_count) outputs files"
    exit 0
}
if (-not (Test-Path -LiteralPath (Join-Path $taskAuditRoot 'before.json'))) {throw 'Before snapshot required'}
if ($Mode -eq 'Repair') {
    $taskBeforeAcl = Read-TaskAcl $taskCacheRoot
    $taskAcl = Get-Acl -LiteralPath $taskCacheRoot
    $taskInheritance = [Security.AccessControl.InheritanceFlags]::ContainerInherit -bor [Security.AccessControl.InheritanceFlags]::ObjectInherit
    $taskRule = [Security.AccessControl.FileSystemAccessRule]::new($taskIdentity,
        [Security.AccessControl.FileSystemRights]::Modify,$taskInheritance,
        [Security.AccessControl.PropagationFlags]::None,[Security.AccessControl.AccessControlType]::Allow)
    $taskAcl.AddAccessRule($taskRule)
    Set-Acl -LiteralPath $taskCacheRoot -AclObject $taskAcl
    $taskAfterAcl=Read-TaskAcl $taskCacheRoot
    if ($taskBeforeAcl.owner -ne $taskAfterAcl.owner -or
        $taskBeforeAcl.inheritance_protected -ne $taskAfterAcl.inheritance_protected) {throw 'Owner/protection drift'}
    Save-TaskRecord 'acl_after.json' ([ordered]@{account=$taskIdentity;cache_acl=Read-TaskCacheAcls
        change='Add inheritable Modify ACE for ROG\yuhua at cache root; preserve owner/protection/other explicit entries'})
    Write-Output 'Cache inheritable Modify ACE applied; owner/protection preserved'
    exit 0
}
if ($Mode -eq 'Probe') {
    New-Item -ItemType Directory -Force -Path $taskTempRoot | Out-Null
    $taskProbeDir=[IO.Path]::GetFullPath((Join-Path $taskTempRoot ('permission_probe_'+[guid]::NewGuid().ToString('N'))))
    if (-not $taskProbeDir.StartsWith($taskTempRoot+'\')) {throw 'Unsafe probe directory'}
    $taskProbeFile=Join-Path $taskProbeDir 'probe.txt'
    New-Item -ItemType Directory -Path $taskProbeDir | Out-Null
    [IO.File]::WriteAllText($taskProbeFile,'pytest workspace permission probe')
    if ([IO.File]::ReadAllText($taskProbeFile) -ne 'pytest workspace permission probe') {throw 'Probe read mismatch'}
    Remove-Item -LiteralPath $taskProbeFile
    Remove-Item -LiteralPath $taskProbeDir
    Save-TaskRecord 'minimal_temp_permission.json' ([ordered]@{account=$taskIdentity
        temp_root=$taskTempRoot;temp_acl=Read-TaskAcl $taskTempRoot
        created=$true;written=$true;read=$true;deleted=$true;existing_cache_deleted=$false})
    Write-Output 'Temporary child directory/file create-write-read-delete PASS'
    exit 0
}
# Fixed workspace TEMP/TMP is inherited by pytest and its subprocesses. No test,
# algorithm, pytest configuration, cache path, or seed is changed.
$env:TEMP=$taskTempRoot
$env:TMP=$taskTempRoot
$taskPytestArgs = switch ($Mode) {
    'Minimal' {@('-m','pytest','tests/test_interpolation.py','tests/test_statistics_information.py::test_result_writer_serializes_numpy_diagnostic_scalars','-q')}
    'Full' {@('-m','pytest','-q')}
    'Math' {@('-m','pytest','tests/test_wiener_perturbation.py','-q')}
}
$taskLogName = switch ($Mode) {
    'Minimal' {'minimal_pytest.txt'}
    'Full' {'t17a_startup_accepted_pytest.txt'}
    'Math' {'t17a_postcommit_math_pytest.txt'}
}
$taskLogPath=Join-Path $taskAuditRoot $taskLogName
if (Test-Path -LiteralPath $taskLogPath) {throw 'Existing test transcript protected'}
$taskRuntimeBefore=@{}
foreach ($taskKey in @('nodeids','lastfailed')) {
    $taskMetadataPath=Join-Path $taskCacheRoot "v/cache/$taskKey"
    $taskRuntimeBefore[$taskKey]=[ordered]@{sha256=(Get-FileHash -LiteralPath $taskMetadataPath).Hash
        mtime_utc=(Get-Item -LiteralPath $taskMetadataPath).LastWriteTimeUtc.ToString('o')}
}
& (Join-Path $taskProjectRoot '.venv/Scripts/python.exe') @taskPytestArgs *> $taskLogPath
$taskExit=$LASTEXITCODE
$taskRuntimeAfter=@{}
foreach ($taskKey in @('nodeids','lastfailed')) {
    $taskMetadataPath=Join-Path $taskCacheRoot "v/cache/$taskKey"
    $taskRuntimeAfter[$taskKey]=[ordered]@{sha256=(Get-FileHash -LiteralPath $taskMetadataPath).Hash
        mtime_utc=(Get-Item -LiteralPath $taskMetadataPath).LastWriteTimeUtc.ToString('o')}
}
Save-TaskRecord ($Mode.ToLower()+'_execution.json') ([ordered]@{account=$taskIdentity
    temp_root=$taskTempRoot;cache_root=$taskCacheRoot;argv=$taskPytestArgs;exit_code=$taskExit
    metadata_before=$taskRuntimeBefore;metadata_after=$taskRuntimeAfter
    metadata_classification='runtime cache metadata change; written by pytest only'
    summary=(Get-Content -LiteralPath $taskLogPath -Tail 1)})
Get-Content -LiteralPath $taskLogPath -Tail 8
exit $taskExit
