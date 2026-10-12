param([Parameter(Mandatory=$true)][ValidateSet('Verify','ToyTests','AuthorizationBindings','SelectionPreflight','CompleteSelectionOnce','MonitorToy','Status','Export','Stop')][string]$Action,
      [string]$AuditEvidenceZip,[string]$R7ArchiveZip,[string]$AuthorizationPath,[string]$InstructionPath,[string]$ApprovalRecordSha256,
      [string]$PythonPath,[string]$JobPath,[switch]$Background)
$ErrorActionPreference='Stop'
Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Utility\Microsoft.PowerShell.Utility.psd1') -ErrorAction Stop
Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Management\Microsoft.PowerShell.Management.psd1') -ErrorAction Stop
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if (-not $PythonPath) {$PythonPath=Join-Path $root '.venv-lab\Scripts\python.exe'}
if (-not (Test-Path -LiteralPath $PythonPath)) {throw 'Existing pinned laboratory Python required; no installation or upgrade.'}
$map=@{Verify='verify';ToyTests='toy-tests';AuthorizationBindings='authorization-bindings';SelectionPreflight='selection-preflight';CompleteSelectionOnce='complete-selection-once';MonitorToy='monitor-toy';Status='status';Export='export';Stop='stop'}
$argsList=@('-B','-X','utf8',(Join-Path $PSScriptRoot 'cli.py'),$map[$Action])
foreach ($pair in @(@('--evidence',$AuditEvidenceZip),@('--archive',$R7ArchiveZip),@('--approval',$AuthorizationPath),@('--instruction',$InstructionPath),@('--job',$JobPath))) {
    if ($pair[1]) {$argsList+=@($pair[0],[IO.Path]::GetFullPath($pair[1]))}
}
if ($ApprovalRecordSha256) {$argsList+=@('--approval-sha',$ApprovalRecordSha256)}
$env:PYTHONDONTWRITEBYTECODE='1';$env:OPENBLAS_NUM_THREADS='6';$env:MKL_NUM_THREADS='6';$env:NUMEXPR_NUM_THREADS='6';$env:OMP_NUM_THREADS='1'
if ($Background) {
    if ($Action -notin @('SelectionPreflight','CompleteSelectionOnce','MonitorToy')) {throw 'Background allowed only for role-scoped worker actions.'}
    & $PythonPath -B -X utf8 (Join-Path $PSScriptRoot 'cli.py') verify
    if ($LASTEXITCODE -ne 0) {throw 'Tool verification failed: STOP.'}
    $logs=Join-Path $root ('lab_runs\r8_selection\launchers\'+[guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $logs | Out-Null
    $quoted=$argsList | ForEach-Object {'"'+($_ -replace '"','\"')+'"'}
    $child=Start-Process -FilePath $PythonPath -ArgumentList $quoted -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logs 'stdout.log') -RedirectStandardError (Join-Path $logs 'stderr.log')
    @{launcher_PID=$child.Id;action=$Action;logs=$logs;automatic_retry=$false;approval_generated=$false} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logs 'launcher.json') -Encoding UTF8
    Write-Output "R8 launcher PID $($child.Id); query Status. FAILED/STOP: preserve logs, no retry."
} else {
    & $PythonPath @argsList
    if ($LASTEXITCODE -ne 0) {throw "R8 $Action FAILED/STOP/refused. Preserve logs; no retry."}
}
